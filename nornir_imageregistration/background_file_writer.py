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
from collections.abc import Callable
from types import TracebackType
from typing import Self

_COPY_BUFFER_BYTES = 8 * 1024 * 1024


def write_via_local_temp(dest_path: str, save: Callable[[str], None]) -> None:
    """Call ``save(local_path)``, then move the result to *dest_path* in large blocks.

    The file appears at *dest_path* only once it is complete.
    """
    _, ext = os.path.splitext(dest_path)
    handle, local_path = tempfile.mkstemp(suffix=ext, prefix='nornir_write_')
    os.close(handle)
    partial_path = dest_path + '.partial'
    try:
        save(local_path)
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        with open(local_path, 'rb') as src, open(partial_path, 'wb') as dst:
            shutil.copyfileobj(src, dst, _COPY_BUFFER_BYTES)
        os.replace(partial_path, dest_path)
    finally:
        for path in (local_path, partial_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


class BackgroundFileWriter:
    """Run file saves on worker threads with a bounded queue.

    Use as a context manager; leaving the block waits for every write and raises
    the first error any of them hit. Pillow and zlib release the GIL while
    encoding and the copy waits on I/O, so a few threads overlap well with the
    caller's compute.
    """

    _executor: concurrent.futures.ThreadPoolExecutor
    _slots: threading.BoundedSemaphore
    _futures: list[concurrent.futures.Future[None]]
    _lock: threading.Lock

    def __init__(self, max_workers: int = 3, max_pending: int = 6) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix='background-file-writer')
        self._slots = threading.BoundedSemaphore(max(max_pending, max_workers))
        self._futures = []
        self._lock = threading.Lock()

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
            self._futures = [f for f in self._futures if not f.done() or f.exception() is not None]
            self._futures.append(future)

    def drain(self) -> None:
        """Wait for every queued write; raise the first error."""
        with self._lock:
            futures = list(self._futures)
            self._futures = []
        first_error: BaseException | None = None
        for future in futures:
            error = future.exception()
            if error is not None and first_error is None:
                first_error = error
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
