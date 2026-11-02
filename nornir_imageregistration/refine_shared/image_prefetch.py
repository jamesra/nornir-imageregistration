"""Load the next STOS refine pair's images while the current pair computes.

Grid refine runs one section pair at a time, and each pair starts by decoding its
control and mapped images (and masks) from the volume, usually over the NAS. On
srv (2026-11-01, RC3 Grid32 on CuPy) that load was about 18% of a 4 s pair, with
the GPU idle meanwhile. The pipeline knows the next pair, so it calls
:func:`prefetch_stos_images` with it before refining the current one, and
``RefineStosFile`` collects the arrays with :func:`take_image`.

The pipeline prefetches ahead of the pair it is about to refine, so the three most
recent prefetches are kept: the current pair's, which ``RefineStosFile`` is about to
claim, and up to two ahead. The pipeline calls refine once per section mapping, so
it prefetches the next mapping's image files with :func:`prefetch_image_files` (the
next mapping's input stos does not exist yet), and a mapping with several pairs also
prefetches its next pair's stos. Anything older and still unclaimed is discarded.
Files are decoded to host (NumPy) arrays on a few background threads, one file each
(a pair is two images and two masks; on srv Grid16 one thread decoding them in turn
left refine waiting on fast pairs);
:func:`take_image` moves them to the device when CuPy is active, which is what
``LoadImage`` does on the calling thread. Any miss, error or file changed since
the prefetch returns the path instead, so the caller loads it the usual way.

``NORNIR_REFINE_PREFETCH=0`` turns prefetching off.
"""

from __future__ import annotations

import collections
import concurrent.futures
import os
import threading
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

import nornir_imageregistration
from nornir_imageregistration import cp
import nornir_pools

_FileSignature = tuple[int, int]
_Key = tuple[str, str]

_executor: concurrent.futures.ThreadPoolExecutor | None = None
_lock = threading.Lock()
_pending: dict[_Key, concurrent.futures.Future[tuple[_FileSignature, NDArray]]] = {}
# Unclaimed keys of each prefetch call, oldest first. Consecutive pairs share a
# section, so a key can sit in both batches; it is loaded once and dropped from
# _pending only when no batch still wants it.
_batches: collections.deque[set[_Key]] = collections.deque()
_KEEP_PREFETCHES = 3
# One per file of a pair (two images, two masks). PIL and NumPy release the GIL while
# decoding and scaling, so these overlap; peak memory is the same batch either way.
_DECODE_THREADS = 4


def prefetch_enabled() -> bool:
    """Return False when ``NORNIR_REFINE_PREFETCH`` is set to 0/false/off."""
    return os.environ.get('NORNIR_REFINE_PREFETCH', '1').strip().lower() not in ('0', 'false', 'off', 'no')


def _key(path: str, dtype: DTypeLike) -> _Key:
    return os.path.normcase(os.path.abspath(path)), np.dtype(dtype).str


def _signature(path: str) -> _FileSignature:
    stat = os.stat(path)
    return stat.st_mtime_ns, stat.st_size


def _load_host(path: str, dtype: DTypeLike) -> tuple[_FileSignature, NDArray]:
    """Background thread: decode *path* to a NumPy array, recording the file it came from."""
    signature = _signature(path)
    image = nornir_imageregistration.LoadImage(path, dtype=dtype, backend='numpy')
    return signature, image


def stos_image_requests(stos_path: str) -> list[tuple[str, DTypeLike]]:
    """Return the ``(path, dtype)`` loads ``RefineStosFile`` makes for *stos_path*."""
    stos = nornir_imageregistration.files.StosFile.Load(stos_path)
    stos.TryConvertRelativePathsToAbsolutePaths(os.path.dirname(stos_path))
    image_dtype = nornir_imageregistration.default_image_dtype()
    requests: list[tuple[str, DTypeLike]] = []
    for image_path, mask_path in ((stos.ControlImageFullPath, stos.ControlMaskFullPath),
                                  (stos.MappedImageFullPath, stos.MappedMaskFullPath)):
        if image_path:
            requests.append((image_path, image_dtype))
        if mask_path:
            requests.append((mask_path, bool))
    return requests


def prefetch_stos_images(stos_path: str | None) -> None:
    """Start loading the images of the STOS refine pair at *stos_path* in the background.

    Keeps the previous prefetch (the pair about to be refined) and discards anything
    older that was never claimed. Never raises: a prefetch that cannot start just
    leaves the caller to load normally.
    """
    if not stos_path or not prefetch_enabled():
        return
    try:
        requests = stos_image_requests(stos_path)
    except Exception:  # noqa: BLE001 - best effort; the refine itself reports a bad stos
        return
    _submit_batch(requests)


def prefetch_image_files(requests: list[tuple[str, DTypeLike]]) -> None:
    """Start loading these ``(path, dtype)`` files in the background as one prefetch.

    For callers that know a pair's image files before its stos exists. Use the dtypes
    ``RefineStosFile`` loads with (:func:`refine_image_dtype` for images, ``bool`` for
    masks), or the prefetch is never claimed.
    """
    if not requests or not prefetch_enabled():
        return
    _submit_batch([(path, dtype) for path, dtype in requests if path])


def refine_image_dtype() -> np.dtype:
    """The dtype ``RefineStosFile`` loads images with."""
    return np.dtype(nornir_imageregistration.default_image_dtype())


def _submit_batch(requests: list[tuple[str, DTypeLike]]) -> None:
    """Queue one prefetch batch, discarding the oldest unclaimed batches beyond the kept number."""
    global _executor
    with _lock:
        while len(_batches) >= _KEEP_PREFETCHES:
            evicted = _batches.popleft()
            for key in evicted:
                if not any(key in batch for batch in _batches):
                    future = _pending.pop(key, None)
                    if future is not None:
                        future.cancel()
        if _executor is None:
            _executor = nornir_pools.InstrumentedThreadPoolExecutor('refine-prefetch', 
                max_workers=_DECODE_THREADS, thread_name_prefix='refine-prefetch')
        keys: set[_Key] = set()
        for path, dtype in requests:
            key = _key(path, dtype)
            if key not in _pending:
                _pending[key] = _executor.submit(_load_host, path, dtype)
            keys.add(key)
        _batches.append(keys)


def take_image(path: str | None, dtype: DTypeLike) -> Any:
    """Return the prefetched array for *path*, or *path* itself when there is none.

    The array is on the active computation backend, as ``LoadImage`` would return it,
    and is the caller's own: a file the next prefetch also wants is handed out as a
    copy. Waits for a load still in progress. ``None`` passes through.
    """
    if path is None:
        return None
    key = _key(path, dtype)
    with _lock:
        claimed_by = next((batch for batch in _batches if key in batch), None)
        if claimed_by is None:
            return path
        claimed_by.discard(key)
        shared = any(key in batch for batch in _batches)
        future = _pending[key] if shared else _pending.pop(key)
    try:
        signature, image = future.result()
        if signature != _signature(path):
            return path
    except Exception:  # noqa: BLE001 - the caller's own load reports the error properly
        return path
    if nornir_imageregistration.UsingCupy():
        return cp.asarray(image)  # a new device array either way
    return image.copy() if shared else image


def clear_prefetched() -> None:
    """Drop every unclaimed prefetch (tests, or a pipeline that stops early)."""
    with _lock:
        for future in _pending.values():
            future.cancel()
        _pending.clear()
        _batches.clear()
