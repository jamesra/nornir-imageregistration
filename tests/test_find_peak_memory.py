"""Permanent tests for find_peak memory-oriented refactor."""
from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.phasecorrelation import find_peak

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp


def _synthetic_peak(xp, *, out_of_mask_high: bool = False, in_mask_zeros: bool = False):
    image = xp.full((32, 32), 0.1, dtype=xp.float32)
    if in_mask_zeros:
        image[8:24, 8:24] = 0.0
    image[14:18, 14:18] = 0.9
    if out_of_mask_high:
        image[0:3, 0:3] = 5.0
    mask = xp.zeros((32, 32), dtype=bool)
    mask[4:28, 4:28] = True
    return image, mask


class TestFindPeakMemory(unittest.TestCase):
    """allow_in_place parity, mutation semantics, and masked-peak correctness."""

    def test_allow_in_place_matches_copy_path(self) -> None:
        image, mask = _synthetic_peak(np, in_mask_zeros=True)
        copy_result = find_peak(image.copy(), mask.copy(), allow_in_place=False)
        in_place_image = image.copy()
        in_place_result = find_peak(in_place_image, mask.copy(), allow_in_place=True)

        self.assertAlmostEqual(copy_result.peak_strength, in_place_result.peak_strength, places=5)
        self.assertAlmostEqual(copy_result.scaled_offset[0], in_place_result.scaled_offset[0], places=4)
        self.assertAlmostEqual(copy_result.scaled_offset[1], in_place_result.scaled_offset[1], places=4)
        self.assertAlmostEqual(copy_result.cutoff_value, in_place_result.cutoff_value, places=5)

    def test_allow_in_place_mutates_input(self) -> None:
        image, mask = _synthetic_peak(np)
        pristine = image.copy()
        find_peak(image, mask, allow_in_place=True)
        self.assertFalse(np.allclose(image, pristine),
                         "allow_in_place=True should overwrite the correlation image")

    def test_copy_path_preserves_input(self) -> None:
        image, mask = _synthetic_peak(np)
        pristine = image.copy()
        find_peak(image, mask, allow_in_place=False)
        np.testing.assert_allclose(image, pristine)

    def test_out_of_mask_high_ignored(self) -> None:
        image, mask = _synthetic_peak(np, out_of_mask_high=True)
        result = find_peak(image, mask)
        cy, cx = np.asarray(image.shape, dtype=np.float32) / 2.0
        self.assertAlmostEqual(result.scaled_offset[0], cy - 16.0, delta=2.0)
        self.assertAlmostEqual(result.scaled_offset[1], cx - 16.0, delta=2.0)
        # SNR must use in-mask mean, not the out-of-mask 5.0 blob.
        reference_mean = float(np.mean(image[mask]))
        work = image.copy()
        work[~mask] = 0
        work[work < result.cutoff_value] = 0
        expected_snr = float(np.max(work)) / reference_mean
        self.assertAlmostEqual(result.peak_strength, expected_snr, places=5)

    def test_in_mask_zeros_counted_in_mean(self) -> None:
        """count_nonzero(overlap_mask) must include in-mask zeros."""
        image, mask = _synthetic_peak(np, in_mask_zeros=True)
        result = find_peak(image, mask)
        reference_mean = float(np.mean(image[mask]))
        # Wrong denominator would exclude zeros and inflate the mean.
        wrong_mean = float(np.sum(image[mask]) / np.count_nonzero(image[mask]))
        self.assertLess(reference_mean, wrong_mean)
        work = image.copy()
        work[~mask] = 0
        work[work < result.cutoff_value] = 0
        expected_snr = float(np.max(work)) / reference_mean
        self.assertAlmostEqual(result.peak_strength, expected_snr, places=5)

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "CuPy not available")
    def test_cupy_parity_with_numpy(self) -> None:
        image_np, mask_np = _synthetic_peak(np, in_mask_zeros=True, out_of_mask_high=True)
        image_cp = cp.asarray(image_np)
        mask_cp = cp.asarray(mask_np)
        result_np = find_peak(image_np, mask_np)
        result_cp = find_peak(image_cp, mask_cp)
        self.assertAlmostEqual(result_np.peak_strength, result_cp.peak_strength, places=4)
        self.assertAlmostEqual(result_np.scaled_offset[0], result_cp.scaled_offset[0], places=3)
        self.assertAlmostEqual(result_np.scaled_offset[1], result_cp.scaled_offset[1], places=3)
        self.assertIsInstance(image_cp, cp.ndarray)
        self.assertIsInstance(image_np, np.ndarray)


if __name__ == "__main__":
    unittest.main()
