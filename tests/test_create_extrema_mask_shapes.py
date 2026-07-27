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


if __name__ == "__main__":
    unittest.main()
