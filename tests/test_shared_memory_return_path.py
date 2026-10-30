"""Shared-memory return path must stay attachable after the local array is dropped (#257)."""
from __future__ import annotations

import gc
import unittest
from multiprocessing.shared_memory import SharedMemory

import numpy as np

import nornir_imageregistration
import nornir_imageregistration.assemble as assemble
from nornir_imageregistration.transforms import factory as transform_factory


class TestSharedMemorySurvivesArrayGC(unittest.TestCase):
    def test_create_shared_memory_array_stays_attachable_after_array_gc(self) -> None:
        meta, array = nornir_imageregistration.create_shared_memory_array((8, 8), dtype=np.float32)
        array.fill(0.25)
        name = meta.name
        del array
        gc.collect()
        attached = SharedMemory(name=name, create=False)
        try:
            view = np.ndarray(meta.shape, dtype=meta.dtype, buffer=attached.buf)
            self.assertAlmostEqual(0.25, float(view[0, 0]))
        finally:
            attached.close()
            nornir_imageregistration.unlink_shared_memory(meta)

    def test_np_array_to_shared_survives_array_gc(self) -> None:
        source = np.arange(16, dtype=np.float32).reshape(4, 4)
        meta, array = nornir_imageregistration.npArrayToSharedArray(source, read_only=False)
        name = meta.name
        del array
        gc.collect()
        attached = SharedMemory(name=name, create=False)
        try:
            view = np.ndarray(meta.shape, dtype=meta.dtype, buffer=attached.buf)
            np.testing.assert_array_equal(view, source)
        finally:
            attached.close()
            nornir_imageregistration.unlink_shared_memory(meta)


class TestReturnSharedMemoryWarp(unittest.TestCase):
    def test_identity_warp_shared_memory_is_attachable(self) -> None:
        image = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
        shape = np.asarray(image.shape, dtype=np.float64)
        transform = transform_factory.CreateRigidTransform(
            (0.0, 0.0), 0.0, shape, shape)
        meta = assemble.SourceImageToTargetSpace(
            transform, image, output_botleft=(0, 0), output_area=image.shape,
            return_shared_memory=True)
        self.assertIsInstance(meta, nornir_imageregistration.Shared_Mem_Metadata)
        gc.collect()
        attached = SharedMemory(name=meta.name, create=False)
        try:
            view = np.ndarray(meta.shape, dtype=meta.dtype, buffer=attached.buf)
            self.assertEqual(view.shape, image.shape)
            self.assertTrue(np.isfinite(view).all())
        finally:
            attached.close()
            nornir_imageregistration.unlink_shared_memory(meta)

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), 'CuPy required')
    def test_cupy_warp_into_shared_memory_does_not_raise(self) -> None:
        import cupy as cp

        image = cp.asarray(np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8))
        shape = np.asarray((8, 8), dtype=np.float64)
        transform = transform_factory.CreateRigidTransform(
            (0.0, 0.0), 0.0, shape, shape)
        meta = assemble.SourceImageToTargetSpace(
            transform, image, output_botleft=(0, 0), output_area=(8, 8),
            return_shared_memory=True)
        self.assertIsInstance(meta, nornir_imageregistration.Shared_Mem_Metadata)
        attached = SharedMemory(name=meta.name, create=False)
        try:
            view = np.ndarray(meta.shape, dtype=meta.dtype, buffer=attached.buf)
            self.assertEqual((8, 8), view.shape)
        finally:
            attached.close()
            nornir_imageregistration.unlink_shared_memory(meta)


if __name__ == '__main__':
    unittest.main()
