"""create_shared_memory_array falls back to a temp-file memmap when /dev/shm is too small, as npArrayToSharedArray does."""
from __future__ import annotations

import gc
import os
import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.core import _core
from nornir_imageregistration.mmap_metadata import memmap_metadata


class TestCreateSharedMemoryArrayFallback(unittest.TestCase):
    def test_small_dev_shm_gives_a_writable_attachable_memmap(self) -> None:
        with mock.patch.object(_core, "_posix_dev_shm_avail_bytes", return_value=0):
            meta, array = nornir_imageregistration.create_shared_memory_array((3, 4), dtype=np.complex64,
                                                                              read_only=False)
        self.assertIsInstance(meta, memmap_metadata)
        self.assertEqual(np.complex64, array.dtype)
        self.assertFalse(array.any())
        array[1, 2] = 1 + 2j
        attached = nornir_imageregistration.ImageParamToNumpyImageArray(meta)
        self.assertEqual(1 + 2j, attached[1, 2])
        attached[0, 0] = 5
        self.assertEqual(5, array[0, 0])
        del attached
        path = meta.path
        del array  # Windows cannot delete a file that is still mapped
        gc.collect()
        nornir_imageregistration.unlink_shared_memory(meta)
        self.assertFalse(os.path.exists(path))

    def test_read_only_sets_how_others_attach(self) -> None:
        with mock.patch.object(_core, "_posix_dev_shm_avail_bytes", return_value=0):
            meta, array = nornir_imageregistration.create_shared_memory_array((2, 2), dtype=np.float32)
        array[:] = 7  # the creator can always fill it
        attached = nornir_imageregistration.ImageParamToNumpyImageArray(meta)
        self.assertFalse(attached.flags.writeable)
        np.testing.assert_array_equal(attached, 7)
        del attached, array
        gc.collect()
        nornir_imageregistration.unlink_shared_memory(meta)


if __name__ == "__main__":
    unittest.main()
