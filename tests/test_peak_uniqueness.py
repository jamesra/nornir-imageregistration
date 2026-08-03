"""Unit tests for masked peak uniqueness (false-peak Phase 1)."""
from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.batched_phase_correlation import batched_find_peak
from nornir_imageregistration.peak_uniqueness import (
    DEFAULT_PEAK_RATIO_EXCLUSION_RADIUS,
    _UNIQUE_PEAK_RATIO,
    batched_masked_peak_ratios,
    masked_peak_ratio,
)
from nornir_imageregistration.phasecorrelation import find_peak


class TestMaskedPeakRatio(unittest.TestCase):
    """Synthetic correlation surfaces for uniqueness scoring."""

    def test_single_sharp_peak_is_unique(self) -> None:
        image = np.full((32, 32), 0.05, dtype=np.float64)
        image[16, 16] = 1.0
        ratio = masked_peak_ratio(image, 16, 16, exclusion_radius=3)
        self.assertGreater(ratio, 10.0)

    def test_two_equal_distant_peaks_near_one(self) -> None:
        image = np.full((32, 32), 0.01, dtype=np.float64)
        image[8, 8] = 1.0
        image[24, 24] = 1.0
        ratio = masked_peak_ratio(image, 8, 8, exclusion_radius=3)
        self.assertAlmostEqual(ratio, 1.0, places=5)

    def test_adjacent_lobe_without_mask_would_look_ambiguous(self) -> None:
        """Neighbor of primary is almost as bright; masking keeps ratio high."""
        image = np.full((32, 32), 0.01, dtype=np.float64)
        image[16, 16] = 1.0
        image[16, 17] = 0.95  # same lobe
        image[16, 15] = 0.90
        unmasked_second = float(np.partition(image.ravel(), -2)[-2])
        naive = 1.0 / unmasked_second
        self.assertLess(naive, 1.2)
        ratio = masked_peak_ratio(image, 16, 16, exclusion_radius=3)
        self.assertGreater(ratio, 10.0)

    def test_no_competing_peak_returns_unique_sentinel(self) -> None:
        image = np.zeros((16, 16), dtype=np.float64)
        image[8, 8] = 0.7
        ratio = masked_peak_ratio(image, 8, 8, exclusion_radius=2)
        self.assertEqual(ratio, _UNIQUE_PEAK_RATIO)

    def test_batched_matches_serial(self) -> None:
        rng = np.random.default_rng(0)
        stack = rng.random((5, 24, 24), dtype=np.float64)
        # Plant distinct primary peaks.
        peaks_r = np.array([4, 10, 18, 6, 12], dtype=np.int64)
        peaks_c = np.array([5, 11, 7, 20, 3], dtype=np.int64)
        for i, (r, c) in enumerate(zip(peaks_r, peaks_c)):
            stack[i, r, c] = 1.0
            stack[i, (r + 8) % 24, (c + 8) % 24] = 0.4 + 0.1 * i

        batched = np.asarray(
            batched_masked_peak_ratios(stack, peaks_r, peaks_c, exclusion_radius=3),
            dtype=np.float64)
        serial = np.asarray([
            masked_peak_ratio(stack[i], int(peaks_r[i]), int(peaks_c[i]), exclusion_radius=3)
            for i in range(stack.shape[0])
        ], dtype=np.float64)
        np.testing.assert_allclose(batched, serial, rtol=1e-6, atol=1e-6)


class TestFindPeakPeakRatio(unittest.TestCase):
    """Serial find_peak exposes peak_ratio."""

    def test_find_peak_sets_peak_ratio(self) -> None:
        image = np.full((40, 40), 0.02, dtype=np.float64)
        image[20, 20] = 1.0
        image[5, 5] = 0.3
        result = find_peak(image)
        self.assertGreater(result.peak_strength, 0.0)
        self.assertGreater(result.peak_ratio, 2.0)

    def test_batched_find_peak_returns_ratios(self) -> None:
        stack = np.full((3, 32, 32), 0.02, dtype=np.float64)
        stack[0, 16, 16] = 1.0
        stack[1, 8, 8] = 1.0
        stack[1, 24, 24] = 1.0
        stack[2, 10, 10] = 1.0
        stack[2, 10, 11] = 0.9  # adjacent lobe — should stay unique after mask
        peaks, weights, ratios = batched_find_peak(
            stack, peak_ratio_exclusion_radius=DEFAULT_PEAK_RATIO_EXCLUSION_RADIUS)
        ratios = np.asarray(ratios, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        self.assertEqual(peaks.shape, (3, 2))
        self.assertTrue(np.all(weights > 0))
        self.assertGreater(float(ratios[0]), 5.0)
        self.assertAlmostEqual(float(ratios[1]), 1.0, places=5)
        self.assertGreater(float(ratios[2]), 5.0)


if __name__ == '__main__':
    unittest.main()
