'''
Created on Sep 14, 2018

@author: u0490822
'''
import unittest

import numpy as np

import nornir_imageregistration
import nornir_imageregistration.overlapmasking

try:
    from . import setup_imagetest
except (ImportError, ModuleNotFoundError):
    import setup_imagetest


class TestOverlapMask(setup_imagetest.ImageTestBase):

    def testSquareOverlapMask(self):
        FixedImageSize = np.asarray((128, 128), dtype=np.int32)
        MovingImageSize = np.asarray((128, 128), dtype=np.int32)
        CorrelationImageSize = FixedImageSize + MovingImageSize

        mask = nornir_imageregistration.GetOverlapMask(FixedImageSize, MovingImageSize, CorrelationImageSize,
                                                       MinOverlap=0.25, MaxOverlap=0.75)
        np.testing.assert_equal(CorrelationImageSize, mask.shape)
        self.assertTrue(nornir_imageregistration.ShowGrayscale([mask],
                                                               title="Square Overlap Mask: 25% Min overlap, 75% max overlap",
                                                               PassFail=True))

    def testSquareOddOverlapMask(self):
        FixedImageSize = np.asarray((127, 127), dtype=np.int32)
        MovingImageSize = np.asarray((128, 128), dtype=np.int32)
        CorrelationImageSize = FixedImageSize + MovingImageSize

        mask = nornir_imageregistration.GetOverlapMask(FixedImageSize, MovingImageSize, CorrelationImageSize,
                                                       MinOverlap=0.25, MaxOverlap=0.75)
        np.testing.assert_equal(CorrelationImageSize, mask.shape, "Dimensions of output mask is incorrect")
        self.assertTrue(nornir_imageregistration.ShowGrayscale([mask],
                                                               title="Square Overlap Mask with odd dimensions: 25% Min overlap, 75% max overlap",
                                                               PassFail=True))

    def testSquareOddWidthOverlapMask(self):
        FixedImageSize = np.asarray((64, 31), dtype=np.int32)
        MovingImageSize = np.asarray((64, 32), dtype=np.int32)
        CorrelationImageSize = FixedImageSize + MovingImageSize

        mask = nornir_imageregistration.GetOverlapMask(FixedImageSize, MovingImageSize, CorrelationImageSize,
                                                       MinOverlap=0.25, MaxOverlap=0.75)
        np.testing.assert_equal(CorrelationImageSize, mask.shape, "Dimensions of output mask is incorrect")
        self.assertTrue(nornir_imageregistration.ShowGrayscale([mask],
                                                               title="Square Overlap Mask with odd width dimension: 25% Min overlap, 75% max overlap",
                                                               PassFail=True))

    def testSquareOddHeightOverlapMask(self):
        FixedImageSize = np.asarray((31, 64), dtype=np.int32)
        MovingImageSize = np.asarray((32, 64), dtype=np.int32)
        CorrelationImageSize = FixedImageSize + MovingImageSize

        mask = nornir_imageregistration.GetOverlapMask(FixedImageSize, MovingImageSize, CorrelationImageSize,
                                                       MinOverlap=0.25, MaxOverlap=0.75)
        np.testing.assert_equal(CorrelationImageSize, mask.shape, "Dimensions of output mask is incorrect")
        self.assertTrue(nornir_imageregistration.ShowGrayscale([mask],
                                                               title="Square Overlap Mask with odd height dimension: 25% Min overlap, 75% max overlap",
                                                               PassFail=True))

    def testSquarePaddedOverlapMask(self):
        FixedImageSize = np.asarray((128, 128), dtype=np.int32)
        MovingImageSize = np.asarray((128, 128), dtype=np.int32)
        CorrelationImageSize = FixedImageSize + MovingImageSize + np.asarray((64, 64), dtype=np.int32)

        mask = nornir_imageregistration.GetOverlapMask(FixedImageSize, MovingImageSize, CorrelationImageSize,
                                                       MinOverlap=0.25, MaxOverlap=0.75)
        np.testing.assert_equal(CorrelationImageSize, mask.shape)
        self.assertTrue(nornir_imageregistration.ShowGrayscale([mask],
                                                               title="Square Padded Overlap Mask: 25% Min overlap, 75% max overlap",
                                                               PassFail=True))

    def testSquareOddPaddedOverlapMask(self):
        FixedImageSize = np.asarray((128, 128), dtype=np.int32)
        MovingImageSize = np.asarray((128, 128), dtype=np.int32)
        CorrelationImageSize = FixedImageSize + MovingImageSize + np.asarray((63, 63), dtype=np.int32)

        mask = nornir_imageregistration.GetOverlapMask(FixedImageSize, MovingImageSize, CorrelationImageSize,
                                                       MinOverlap=0.25, MaxOverlap=0.75)
        np.testing.assert_equal(CorrelationImageSize, mask.shape)
        self.assertTrue(nornir_imageregistration.ShowGrayscale([mask],
                                                               title="Square Padded Overlap Mask: 25% Min overlap, 75% max overlap",
                                                               PassFail=True))

    def testMismatchedOverlapMask_FixedLarger(self):
        FixedImageSize = np.asarray((64, 256), dtype=np.int32)
        MovingImageSize = np.asarray((64, 64), dtype=np.int32)
        CorrelationImageSize = FixedImageSize + MovingImageSize

        mask = nornir_imageregistration.GetOverlapMask(FixedImageSize, MovingImageSize, CorrelationImageSize,
                                                       MinOverlap=0.25, MaxOverlap=0.75)
        np.testing.assert_equal(CorrelationImageSize, mask.shape)
        self.assertTrue(nornir_imageregistration.ShowGrayscale([mask],
                                                               title="Mismatched Overlap Mask (Fixed Larger): 25% Min overlap, 75% max overlap",
                                                               PassFail=True))

        return

    def testMismatchedPaddedOverlapMask_FixedLarger(self):
        FixedImageSize = np.asarray((64, 256), dtype=np.int32)
        MovingImageSize = np.asarray((64, 64), dtype=np.int32)
        CorrelationImageSize = FixedImageSize + MovingImageSize + np.asarray((64, 64), dtype=np.int32)

        mask = nornir_imageregistration.GetOverlapMask(FixedImageSize, MovingImageSize, CorrelationImageSize,
                                                       MinOverlap=0.25, MaxOverlap=0.75)
        np.testing.assert_equal(CorrelationImageSize, mask.shape)
        self.assertTrue(nornir_imageregistration.ShowGrayscale([mask],
                                                               title="Mismatched Padded Overlap Mask (Fixed Larger): 25% Min overlap, 75% max overlap",
                                                               PassFail=True))

        return

    def testMismatchedOverlapMask_MovingLarger(self):
        FixedImageSize = np.asarray((64, 64), dtype=np.int32)
        MovingImageSize = np.asarray((64, 256), dtype=np.int32)
        CorrelationImageSize = FixedImageSize + MovingImageSize

        mask = nornir_imageregistration.GetOverlapMask(FixedImageSize, MovingImageSize, CorrelationImageSize,
                                                       MinOverlap=0.25, MaxOverlap=0.75)
        np.testing.assert_equal(CorrelationImageSize, mask.shape)
        self.assertTrue(nornir_imageregistration.ShowGrayscale([mask],
                                                               title="Mismatched Overlap Mask (Moving Larger): 25% Min overlap, 75% max overlap",
                                                               PassFail=True))

        return

    def testMismatchedPaddedOverlapMask_MovingLarger(self):
        FixedImageSize = np.asarray((64, 64), dtype=np.int32)
        MovingImageSize = np.asarray((64, 256), dtype=np.int32)
        CorrelationImageSize = FixedImageSize + MovingImageSize + np.asarray((64, 64), dtype=np.int32)

        mask = nornir_imageregistration.GetOverlapMask(FixedImageSize, MovingImageSize, CorrelationImageSize,
                                                       MinOverlap=0.25, MaxOverlap=0.75)
        np.testing.assert_equal(CorrelationImageSize, mask.shape)
        self.assertTrue(nornir_imageregistration.ShowGrayscale([mask],
                                                               title="Mismatched Padded Overlap Mask (Moving Larger): 25% Min overlap, 75% max overlap",
                                                               PassFail=True))

        return

    def testOverlapMaskPopulation(self):
        FixedImageSize = np.asarray((128, 128), dtype=np.int32)
        MovingImageSize = np.asarray((128, 128), dtype=np.int32)
        CorrelationImageSize = FixedImageSize + MovingImageSize + np.asarray((64, 64), dtype=np.int32)

        QuadrantSize = CorrelationImageSize // 2

        BruteForceMask = np.zeros(QuadrantSize, dtype=bool)
        BruteForceMaskOptimized = np.zeros(QuadrantSize, dtype=bool)

        BruteForceMask = nornir_imageregistration.overlapmasking._PopulateMaskQuadrantBruteForce(BruteForceMask,
                                                                                                 FixedImageSize,
                                                                                                 MovingImageSize,
                                                                                                 MinOverlap=0.25,
                                                                                                 MaxOverlap=0.75)
        BruteForceMaskOptimized = nornir_imageregistration.overlapmasking._PopulateMaskQuadrantBruteForceOptimized(
            BruteForceMaskOptimized, FixedImageSize, MovingImageSize, MinOverlap=0.25, MaxOverlap=0.75)

        self.assertTrue(
            nornir_imageregistration.ShowGrayscale([BruteForceMask, BruteForceMaskOptimized], title="Two equal masks",
                                                   PassFail=True))

        self.assertTrue(np.array_equal(BruteForceMask, BruteForceMaskOptimized),
                        "Masks should be equal regardless of how they are made")

        return


class TestOverlapMaskOnDevice(unittest.TestCase):
    """Verify device-resident overlap mask caching."""

    def setUp(self) -> None:
        nornir_imageregistration.overlapmasking.clear_overlap_mask_caches()

    def tearDown(self) -> None:
        # Restore default budgets after tests that override the env var.
        nornir_imageregistration.overlapmasking.clear_overlap_mask_caches()

    def test_device_mask_reuses_upload(self) -> None:
        """GetOverlapMaskOnDevice should upload each geometry to CuPy at most once."""
        if not nornir_imageregistration.HasCupy():
            self.skipTest("CuPy not available")

        import cupy as cp

        fixed = np.asarray((64, 64), dtype=np.int32)
        moving = np.asarray((64, 64), dtype=np.int32)
        corr = fixed + moving

        mask_a = nornir_imageregistration.GetOverlapMaskOnDevice(
            fixed, moving, corr, MinOverlap=0.25, MaxOverlap=0.75, xp=cp)
        mask_b = nornir_imageregistration.GetOverlapMaskOnDevice(
            fixed, moving, corr, MinOverlap=0.25, MaxOverlap=0.75, xp=cp)

        self.assertIs(mask_a, mask_b)
        np.testing.assert_array_equal(
            nornir_imageregistration.GetOverlapMask(fixed, moving, corr, 0.25, 0.75),
            cp.asnumpy(mask_a),
        )
        stats = nornir_imageregistration.overlapmasking.overlap_mask_cache_stats()
        self.assertEqual(stats["device_entries"], 1)

    def test_device_lru_evicts_oldest(self) -> None:
        """Byte-capped LRU should drop the oldest device entry while live refs stay valid."""
        if not nornir_imageregistration.HasCupy():
            self.skipTest("CuPy not available")

        import os
        import cupy as cp

        # Each 128x128 bool mask is 16 KiB; budget of ~20 KiB keeps at most one entry.
        os.environ["NORNIR_OVERLAP_MASK_CACHE_MB"] = str(20 / 1024)
        try:
            nornir_imageregistration.overlapmasking.clear_overlap_mask_caches()
            shapes = [
                (np.asarray((64, 64), dtype=np.int32), 0.20),
                (np.asarray((64, 64), dtype=np.int32), 0.30),
                (np.asarray((64, 64), dtype=np.int32), 0.40),
            ]
            held = []
            for fixed, min_overlap in shapes:
                corr = fixed + fixed
                mask = nornir_imageregistration.GetOverlapMaskOnDevice(
                    fixed, fixed, corr, MinOverlap=min_overlap, MaxOverlap=0.9, xp=cp)
                held.append(mask)
                self.assertEqual(mask.shape, (128, 128))

            stats = nornir_imageregistration.overlapmasking.overlap_mask_cache_stats()
            self.assertLessEqual(stats["device_entries"], 1)
            # Live references remain usable after eviction.
            for mask in held:
                self.assertEqual(int(cp.asnumpy(mask).sum()) > 0, True)
        finally:
            os.environ.pop("NORNIR_OVERLAP_MASK_CACHE_MB", None)
            nornir_imageregistration.overlapmasking.clear_overlap_mask_caches()

    def test_host_lru_evicts_oldest(self) -> None:
        """Host cache also respects its byte budget."""
        import os

        os.environ["NORNIR_OVERLAP_MASK_HOST_CACHE_MB"] = str(20 / 1024)
        try:
            nornir_imageregistration.overlapmasking.clear_overlap_mask_caches()
            held = []
            for min_overlap in (0.20, 0.30, 0.40):
                fixed = np.asarray((64, 64), dtype=np.int32)
                corr = fixed + fixed
                mask = nornir_imageregistration.GetOverlapMask(
                    fixed, fixed, corr, MinOverlap=min_overlap, MaxOverlap=0.9)
                held.append(mask)
            stats = nornir_imageregistration.overlapmasking.overlap_mask_cache_stats()
            self.assertLessEqual(stats["host_entries"], 1)
            for mask in held:
                self.assertTrue(mask.sum() > 0)
        finally:
            os.environ.pop("NORNIR_OVERLAP_MASK_HOST_CACHE_MB", None)
            nornir_imageregistration.overlapmasking.clear_overlap_mask_caches()

    def test_find_peak_scalar_export(self) -> None:
        """find_peak should return host floats without requiring EnsureNumpyArray on the offset."""
        image = np.zeros((32, 32), dtype=np.float32)
        image[16, 16] = 1.0
        result = nornir_imageregistration.phasecorrelation.find_peak(image)
        self.assertIsInstance(result.scaled_offset[0], float)
        self.assertIsInstance(result.scaled_offset[1], float)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
