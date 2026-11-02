"""Write output files on background threads, so slow network storage overlaps compute.

Stages that write one or more large images per item (``AssembleStosOverlays``
writes three PNG previews per section pair) spend most of their time writing when
the output is on the NAS. On srv (2026-11-02, RC3 Grid32 overlays to /storage2 over
CIFS) a pair's three PNGs took 11-12 s to land while encoding them took 1.5 s:
Pillow streams each compressed chunk to the open file, so every chunk is a network
round trip.

:class:`BackgroundFileWriter` runs each save on a worker thread. The save function
writes to a local temporary file with the usual code (so output bytes do not
change); the file is then copied to a ``.partial`` name beside the destination in
large blocks and renamed into place, so an interrupted run never leaves a
truncated file under the final name. Submissions block once ``max_pending`` files
are queued, which bounds the memory held by queued images.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from types import TracebackType
from typing import Self

import nornir_pools

_COPY_BUFFER_BYTES = 8 * 1024 * 1024
# Retries of the copy onto the destination (the share), not of the local save. On RC2
# (2026-11-02) one ConnectionAbortedError on /storage4 after six hours of overlays
# failed the whole stage.
_COPY_ATTEMPTS = 5
_COPY_RETRY_BASE_S = 0.5
_logger = logging.getLogger(__name__)


def write_via_local_temp(dest_path: str, save: Callable[[str], None]) -> None:
    """Call ``save(local_path)``, then move the result to *dest_path* in large blocks.

    The file appears at *dest_path* only once it is complete. The copy is retried with
    backoff on ``OSError`` (network shares drop connections now and then); the error from
    the last attempt is raised.
    """
    _, ext = os.path.splitext(dest_path)
    handle, local_path = tempfile.mkstemp(suffix=ext, prefix='nornir_write_')
    os.close(handle)
    partial_path = dest_path + '.partial'
    try:
        save(local_path)
        dest_dir = os.path.dirname(dest_path)
        for attempt in range(_COPY_ATTEMPTS):
            try:
                if dest_dir:
                    os.makedirs(dest_dir, exist_ok=True)
                with open(local_path, 'rb') as src, open(partial_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst, _COPY_BUFFER_BYTES)
                os.replace(partial_path, dest_path)
                break
            except OSError as e:
                if attempt == _COPY_ATTEMPTS - 1:
                    raise
                delay = _COPY_RETRY_BASE_S * (2 ** attempt)
                _logger.warning('Writing %s failed (%s); retrying in %.1f s', dest_path, e, delay)
                time.sleep(delay)
    finally:
        for path in (local_path, partial_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


class BackgroundFileWriter:
    """Run file saves on worker threads with a bounded queue.

    Use as a context manager; leaving the block waits for every write. By default it
    then raises the first error any of them hit. With ``raise_errors=False`` failed
    writes are logged and listed in :attr:`failed_paths` instead, for outputs a stage can
    regenerate later (previews) and should not fail over. Pillow and zlib release the GIL
    while encoding and the copy waits on I/O, so a few threads overlap well with the
    caller's compute.
    """

    _executor: concurrent.futures.ThreadPoolExecutor
    _slots: threading.BoundedSemaphore
    _futures: list[tuple[str, concurrent.futures.Future[None]]]
    _lock: threading.Lock
    raise_errors: bool
    failed_paths: list[str]

    def __init__(self, max_workers: int = 3, max_pending: int = 6, *, raise_errors: bool = True) -> None:
        self._executor = nornir_pools.InstrumentedThreadPoolExecutor('background-file-writer', 
            max_workers=max_workers, thread_name_prefix='background-file-writer')
        self._slots = threading.BoundedSemaphore(max(max_pending, max_workers))
        self._futures = []
        self._lock = threading.Lock()
        self.raise_errors = raise_errors
        self.failed_paths = []

    def submit(self, dest_path: str, save: Callable[[str], None]) -> None:
        """Queue ``save`` for *dest_path* (see :func:`write_via_local_temp`); blocks while the queue is full."""
        self._slots.acquire()
        try:
            future = self._executor.submit(write_via_local_temp, dest_path, save)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _f: self._slots.release())
        with self._lock:
            self._futures = [(p, f) for p, f in self._futures if not f.done() or f.exception() is not None]
            self._futures.append((dest_path, future))

    def drain(self) -> None:
        """Wait for every queued write; raise the first error, or log failures when ``raise_errors`` is off."""
        with self._lock:
            futures = list(self._futures)
            self._futures = []
        first_error: BaseException | None = None
        for path, future in futures:
            error = future.exception()
            if error is None:
                continue
            if self.raise_errors:
                if first_error is None:
                    first_error = error
            else:
                self.failed_paths.append(path)
                _logger.error('Could not write %s: %s', path, error)
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        """Wait for queued writes, then stop the worker threads."""
        try:
            self.drain()
        finally:
            self._executor.shutdown(wait=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self,
                 exc_type: type[BaseException] | None,
                 exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        if exc_type is None:
            self.close()
            return
        # Already failing: finish the writes, but keep the original exception.
        try:
            self.close()
        except Exception:  # the caller's exception takes precedence
            logging.getLogger(__name__).exception('Background file write also failed')
