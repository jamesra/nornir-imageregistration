"""Memmap assemble buffers must clean up without raising at shutdown.

__CreateOutputBufferForArea registers weakref.finalize(buffer, os.remove, path)
for its memmap backing file. weakref._exitfunc can run that finalizer before
NumPy releases the mapping, and on Windows os.remove then raises WinError 32
from inside interpreter shutdown, where no caller can handle it:

    PermissionError: [WinError 32] The process cannot access the file because
    it is being used by another process: '...\\Temp\\image_64x48_....npy'

The memmap path is currently dead code (_use_memmap() returns False), so these
tests force it on. They are the groundwork for re-enabling it; see issue #13.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration.assemble_tiles as assemble_tiles


class TestRemoveMemmapBackingFile(unittest.TestCase):

    def test_removes_an_unmapped_file(self) -> None:
        fd, path = tempfile.mkstemp(suffix='.npy')
        os.close(fd)
        self.assertTrue(os.path.exists(path))

        assemble_tiles._remove_memmap_backing_file(path)

        self.assertFalse(os.path.exists(path), "Backing file should be deleted when unmapped")

    def test_missing_file_is_not_an_error(self) -> None:
        path = os.path.join(tempfile.gettempdir(), 'nornir-does-not-exist-12345.npy')
        self.assertFalse(os.path.exists(path))

        assemble_tiles._remove_memmap_backing_file(path)  # must not raise

    def test_open_mapping_does_not_raise(self) -> None:
        """The case that broke shutdown: mapping still held when cleanup runs."""
        path = os.path.join(tempfile.gettempdir(), 'nornir-memmap-cleanup-test.npy')
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))

        mapped = np.memmap(path, dtype=np.float32, mode='w+', shape=(16, 16))
        mapped.fill(0)
        try:
            # Plain os.remove is what the code used to register; on Windows this
            # is the call that raised out of weakref._exitfunc.
            assemble_tiles._remove_memmap_backing_file(path)
        finally:
            del mapped

    def test_permission_error_is_swallowed_on_every_platform(self) -> None:
        """Pin the behavior even where the OS would allow the unlink."""
        path = os.path.join(tempfile.gettempdir(), 'nornir-memmap-perm-test.npy')

        with mock.patch('os.remove', side_effect=PermissionError(32, 'in use')) as removed:
            assemble_tiles._remove_memmap_backing_file(path)

        removed.assert_called_once_with(path)


class TestCreateOutputBufferForAreaWithMemmap(unittest.TestCase):
    """End-to-end: forcing the memmap path must not leave a raising finalizer."""

    def _create_buffer_fn(self):
        return getattr(assemble_tiles, '__CreateOutputBufferForArea')

    def test_memmap_buffer_finalizer_tolerates_open_mapping(self) -> None:
        create = self._create_buffer_fn()

        with mock.patch.object(assemble_tiles, '_use_memmap', return_value=True):
            full_image, z_buffer = create(64, 48, np.float32)

            self.assertIsInstance(full_image, np.memmap)
            self.assertEqual(full_image.shape, (64, 48))
            self.assertEqual(full_image.dtype, np.float32)
            self.assertEqual(z_buffer.shape, (64, 48))

            path = full_image.filename
            self.assertIsNotNone(path)

            # Invoke cleanup while the mapping is still open -- exactly the
            # shutdown ordering that raised before.
            assemble_tiles._remove_memmap_backing_file(path)

            del full_image, z_buffer


if __name__ == "__main__":
    unittest.main()
