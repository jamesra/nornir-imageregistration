"""Regression tests for CreateExtremaMask image/mask shape handling."""
from __future__ import annotations

import unittest
import warnings

import numpy as np

import nornir_imageregistration


class TestCreateExtremaMaskShapes(unittest.TestCase):
    """Image/mask off-by-one and valid-mask NaN handling."""

    def test_ensure_matching_crops_to_overlap(self) -> None:
        image = np.zeros((10, 1918), dtype=np.float32)
        mask = np.ones((10, 1919), dtype=bool)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cropped_image, cropped_mask = nornir_imageregistration.EnsureMatchingImageMaskShape(image, mask)
        self.assertEqual(cropped_image.shape, (10, 1918))
        self.assertEqual(cropped_mask.shape, (10, 1918))
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))

    def test_create_extrema_mask_mismatched_shapes(self) -> None:
        """Off-by-one mask (as seen in AlignSections) must not raise IndexError."""
        rng = np.random.default_rng(1)
        image = rng.random((8, 1918), dtype=np.float32)
        image[2:4, 2:4] = 0.0
        mask = np.ones((8, 1919), dtype=bool)
        mask[:, -1] = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = nornir_imageregistration.CreateExtremaMask(image, mask, size_cutoff=1)

        self.assertEqual(result.shape, (8, 1918))
        self.assertEqual(result.dtype, bool)

    def test_create_extrema_mask_excludes_invalid_from_minmax(self) -> None:
        """Invalid (False) mask pixels must not define min/max for extrema."""
        image = np.full((6, 6), 0.4, dtype=np.float32)
        image[0, 0] = 0.0  # only in invalid region — must not become global min
        image[2, 2] = 0.0  # valid minimum
        image[3, 3] = 1.0  # valid maximum
        mask = np.ones((6, 6), dtype=bool)
        mask[0, 0] = False

        result = nornir_imageregistration.CreateExtremaMask(image, mask, size_cutoff=None)
        # size_cutoff=None returns the raw extrema-candidate mask (True = extrema/invalid).
        self.assertTrue(bool(result[2, 2]))
        self.assertTrue(bool(result[3, 3]))
        self.assertTrue(bool(result[0, 0]))  # invalid region marked as extrema candidate
        self.assertFalse(bool(result[1, 1]))

    def test_image_permutation_helper_mismatched_shapes(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.random((8, 1918), dtype=np.float32)
        mask = np.ones((8, 1919), dtype=bool)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            helper = nornir_imageregistration.ImagePermutationHelper(image, mask)
        self.assertEqual(helper.Image.shape, helper.Mask.shape)
        self.assertEqual(helper.Image.shape, helper.BlendedMask.shape)
        self.assertGreater(int(np.count_nonzero(helper.BlendedMask)), 0)

    def test_image_permutation_helper_defers_extrema_until_needed(self) -> None:
        """Constructor must not build blended mask/stats until registration properties are used."""
        image = np.full((16, 16), 0.4, dtype=np.float32)
        helper = nornir_imageregistration.ImagePermutationHelper(image, mask=None)
        self.assertIsNone(helper._blended_mask)
        self.assertIsNone(helper._stats)
        _ = helper.BlendedMask
        self.assertIsNotNone(helper._blended_mask)
        self.assertIsNotNone(helper._stats)

    def test_image_permutation_helper_prefetch_completes(self) -> None:
        """prefetch_extrema_async fills blended mask/stats without a direct property access first."""
        from concurrent.futures import ThreadPoolExecutor

        image = np.full((16, 16), 0.4, dtype=np.float32)
        helper = nornir_imageregistration.ImagePermutationHelper(image, mask=None)
        self.assertIsNone(helper._blended_mask)
        with ThreadPoolExecutor(max_workers=1) as pool:
            helper.prefetch_extrema_async(executor=pool)
            future = helper._extrema_future
            self.assertIsNotNone(future)
            assert future is not None
            future.result(timeout=5.0)
        self.assertIsNotNone(helper._blended_mask)
        self.assertIsNotNone(helper._stats)
        self.assertGreater(int(np.count_nonzero(helper.BlendedMask)), 0)

    def test_image_permutation_helper_prefetch_join_from_accessor(self) -> None:
        """Reading BlendedMask while prefetch is in flight must join without double-failure."""
        from concurrent.futures import ThreadPoolExecutor

        image = np.full((32, 32), 0.35, dtype=np.float32)
        image[0, 0] = 0.0
        image[-1, -1] = 1.0
        helper = nornir_imageregistration.ImagePermutationHelper(image, mask=None)
        with ThreadPoolExecutor(max_workers=1) as pool:
            helper.prefetch_extrema_async(executor=pool)
            blended = helper.BlendedMask
        self.assertEqual(blended.shape, image.shape)
        self.assertIsNotNone(helper.Stats)
        # Second prefetch is a no-op once results exist.
        helper.prefetch_extrema_async()
        self.assertIs(helper.BlendedMask, blended)

    def test_image_permutation_helper_constant_roi_does_not_raise(self) -> None:
        """Entire-ROI extrema (constant pad) must not raise Image has no data."""
        image = np.full((32, 32), 0.25, dtype=np.float32)
        helper = nornir_imageregistration.ImagePermutationHelper(image, mask=None)
        self.assertGreater(int(np.count_nonzero(helper.BlendedMask)), 0)
        self.assertIsNotNone(helper.Stats)

    def test_image_permutation_helper_extrema_wipe_falls_back_to_mask(self) -> None:
        """When large extrema exclude all tissue, fall back to the tissue mask."""
        image = np.full((32, 32), 0.0, dtype=np.float32)
        image[8:24, 8:24] = 0.5
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        # Force every min/max island into the excluded set (cutoff above ROI area).
        helper = nornir_imageregistration.ImagePermutationHelper(
            image, mask=mask, extrema_mask_size_cuttoff=1)
        self.assertGreater(int(np.count_nonzero(helper.BlendedMask)), 0)
        self.assertIsNotNone(helper.Stats)


if __name__ == "__main__":
    unittest.main()
