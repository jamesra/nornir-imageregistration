"""peak_ratio must not depend on allow_in_place.

``find_peak`` applied the cutoff threshold in place when ``allow_in_place=True``,
zeroing every sub-cutoff pixel in the caller's correlation surface.  The
uniqueness measurement that follows reads that same surface, so the competing
peaks it is supposed to find had already been erased.  ``masked_peak_ratio`` then
saw nothing above zero outside the exclusion box and returned its
"nothing competes" sentinel, making every peak look perfectly unique.

Both production callers pass ``allow_in_place=True``
(``phasecorrelation.find_offset`` and ``stos_brute``), so the ambiguity gate was
effectively dead on the paths that matter: ``is_ambiguous_peak`` never fired.

The fix evaluates the cutoff into a boolean mask and labels that, leaving the
surface readable.  Weighted label statistics are unaffected because every pixel
inside a label is above the cutoff by construction.
"""
from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.batched_phase_correlation import batched_find_peak
from nornir_imageregistration.peak_uniqueness import _UNIQUE_PEAK_RATIO
from nornir_imageregistration.phasecorrelation import find_peak
from nornir_imageregistration.refine_shared.peak_ratio_gates import is_ambiguous_peak


def _two_peak_surface(secondary: float, *, n: int = 64, primary: float = 0.90):
    """A primary peak plus a genuine competing peak far enough away to survive exclusion."""
    image = np.full((n, n), 0.05, dtype=np.float32)
    image[10:14, 10:14] = primary
    image[46:50, 46:50] = secondary
    mask = np.ones((n, n), dtype=bool)
    return image, mask


class TestRatioIsIndependentOfAllowInPlace(unittest.TestCase):
    """The core invariant: allow_in_place is a memory optimization, not a semantic one."""

    def _both_paths(self, image, mask):
        copy_result = find_peak(image.copy(), mask.copy(), allow_in_place=False)
        in_place_result = find_peak(image.copy(), mask.copy(), allow_in_place=True)
        return copy_result, in_place_result

    def test_ratio_matches_the_copy_path(self):
        for secondary in (0.86, 0.70, 0.50, 0.20):
            with self.subTest(secondary=secondary):
                image, mask = _two_peak_surface(secondary)
                copy_result, in_place_result = self._both_paths(image, mask)

                self.assertAlmostEqual(copy_result.peak_ratio, in_place_result.peak_ratio, places=4)

    def test_every_field_matches_the_copy_path(self):
        image, mask = _two_peak_surface(0.86)
        copy_result, in_place_result = self._both_paths(image, mask)

        self.assertAlmostEqual(copy_result.scaled_offset[0], in_place_result.scaled_offset[0], places=4)
        self.assertAlmostEqual(copy_result.scaled_offset[1], in_place_result.scaled_offset[1], places=4)
        self.assertAlmostEqual(copy_result.peak_strength, in_place_result.peak_strength, places=5)
        self.assertAlmostEqual(copy_result.cutoff_value, in_place_result.cutoff_value, places=5)
        self.assertAlmostEqual(copy_result.peak_ratio, in_place_result.peak_ratio, places=4)

    def test_close_second_peak_is_not_reported_as_unique(self):
        """0.90 vs 0.86 is ambiguous; the sentinel would claim nothing competes."""
        image, mask = _two_peak_surface(0.86)

        result = find_peak(image, mask, allow_in_place=True)

        self.assertLess(result.peak_ratio, _UNIQUE_PEAK_RATIO)
        self.assertLess(result.peak_ratio, 1.2)

    def test_ambiguity_gate_actually_fires_in_place(self):
        """The regression that mattered: the gate was dead for in-place callers."""
        image, mask = _two_peak_surface(0.86)

        result = find_peak(image, mask, allow_in_place=True)

        self.assertTrue(is_ambiguous_peak(result.peak_ratio),
                       f'ratio {result.peak_ratio} should be flagged ambiguous')

    def test_ratio_falls_as_the_competitor_weakens(self):
        """A monotone response, rather than a constant sentinel."""
        ratios = []
        for secondary in (0.86, 0.70, 0.50, 0.20):
            image, mask = _two_peak_surface(secondary)
            ratios.append(find_peak(image, mask, allow_in_place=True).peak_ratio)

        self.assertEqual(ratios, sorted(ratios),
                         f'ratio should rise as the competitor weakens, got {ratios}')
        self.assertTrue(all(r < _UNIQUE_PEAK_RATIO for r in ratios))

    def test_genuinely_unique_peak_still_reports_the_sentinel(self):
        """The fix must not destroy the sentinel where it is correct."""
        n = 64
        image = np.zeros((n, n), dtype=np.float32)
        image[32, 32] = 1.0

        result = find_peak(image, np.ones((n, n), dtype=bool), allow_in_place=True)

        self.assertEqual(result.peak_ratio, _UNIQUE_PEAK_RATIO)
        self.assertFalse(is_ambiguous_peak(result.peak_ratio))


class TestSurfaceIsReadableForTheRatio(unittest.TestCase):
    """allow_in_place may still overwrite via the overlap multiply, but not the cutoff."""

    def test_sub_cutoff_detail_survives(self):
        image, mask = _two_peak_surface(0.86)
        original_secondary = float(image[47, 47])

        find_peak(image, mask, allow_in_place=True)

        self.assertAlmostEqual(float(image[47, 47]), original_secondary, places=5,
                               msg='the cutoff must not zero the competing peak')

    def test_copy_path_still_leaves_input_pristine(self):
        image, mask = _two_peak_surface(0.86)
        pristine = image.copy()

        find_peak(image, mask, allow_in_place=False)

        np.testing.assert_allclose(image, pristine)


class TestSerialMatchesBatched(unittest.TestCase):
    """This bug was a serial/batched divergence, and batched had it right.

    ``batched_find_peak`` never thresholds; it measures uniqueness on the raw
    stack.  The serial path's in-place cutoff made it disagree wildly -- 1e6
    against a true ratio near 1.05 -- so the two mirrors disagreed on whether a
    peak was ambiguous at all.
    """

    def test_ratios_agree_with_the_batched_mirror(self):
        secondaries = (0.86, 0.70, 0.50, 0.20)
        mask = np.ones((64, 64), dtype=bool)
        stack = np.stack([_two_peak_surface(s)[0] for s in secondaries])

        _, _, batched_ratios = batched_find_peak(stack.copy(), overlap_mask=mask.copy())

        for i, secondary in enumerate(secondaries):
            with self.subTest(secondary=secondary):
                image, _ = _two_peak_surface(secondary)
                serial = find_peak(image, mask.copy(), allow_in_place=True).peak_ratio
                batched = float(batched_ratios[i])

                self.assertAlmostEqual(serial, batched, delta=0.15 * max(serial, batched),
                                       msg=f'serial {serial} vs batched {batched}')

    def test_both_mirrors_agree_on_ambiguity(self):
        mask = np.ones((64, 64), dtype=bool)
        image, _ = _two_peak_surface(0.86)
        stack = image[None, :, :].copy()

        _, _, batched_ratios = batched_find_peak(stack, overlap_mask=mask.copy())
        serial_ratio = find_peak(image.copy(), mask.copy(), allow_in_place=True).peak_ratio

        self.assertEqual(is_ambiguous_peak(serial_ratio),
                         is_ambiguous_peak(float(batched_ratios[0])))


class TestLabelStatisticsUnchanged(unittest.TestCase):
    """Labeling a boolean instead of a thresholded buffer must not move the peak.

    Every pixel inside a label is above the cutoff, so sums, center of mass and
    maximum are identical.  These cases pin the corners where that reasoning is
    least obvious.
    """

    def _assert_paths_agree(self, image, mask):
        copy_result = find_peak(image.copy(), mask.copy(), allow_in_place=False)
        in_place_result = find_peak(image.copy(), mask.copy(), allow_in_place=True)
        np.testing.assert_allclose(copy_result.scaled_offset, in_place_result.scaled_offset, atol=1e-4)
        self.assertAlmostEqual(copy_result.peak_strength, in_place_result.peak_strength, places=5)
        self.assertAlmostEqual(copy_result.peak_ratio, in_place_result.peak_ratio, places=4)

    def test_all_zero_surface(self):
        n = 32
        self._assert_paths_agree(np.zeros((n, n), np.float32), np.ones((n, n), bool))

    def test_constant_surface(self):
        n = 32
        self._assert_paths_agree(np.full((n, n), 0.5, np.float32), np.ones((n, n), bool))

    def test_negative_surface_exercises_the_non_positive_cutoff_branch(self):
        """With a cutoff <= 0, exact zeros must stay background or components merge."""
        n = 32
        image = np.full((n, n), -0.3, dtype=np.float32)
        image[16:18, 16:18] = 0.8
        self._assert_paths_agree(image, np.ones((n, n), bool))

    def test_zero_background_keeps_components_separate(self):
        """Two peaks separated by exact zeros must not be labeled as one blob."""
        n = 32
        image = np.zeros((n, n), dtype=np.float32)
        image[6:9, 6:9] = 0.9
        image[24:27, 24:27] = 0.4

        result = find_peak(image.copy(), np.ones((n, n), bool), allow_in_place=True)

        # Center of mass must sit on the strong peak, not midway between the two.
        centre = n / 2.0
        self.assertAlmostEqual(result.scaled_offset[0], centre - 7.0, delta=1.5)
        self.assertAlmostEqual(result.scaled_offset[1], centre - 7.0, delta=1.5)

    def test_surface_with_nan(self):
        n = 32
        image = np.full((n, n), 0.1, dtype=np.float32)
        image[16:18, 16:18] = 0.9
        image[0, 0] = np.nan
        self._assert_paths_agree(image, np.ones((n, n), bool))

    def test_random_surfaces_agree(self):
        rng = np.random.default_rng(2024)
        n = 48
        for i in range(8):
            with self.subTest(i=i):
                image = rng.random((n, n), dtype=np.float32)
                mask = np.zeros((n, n), dtype=bool)
                mask[4:-4, 4:-4] = True
                self._assert_paths_agree(image, mask)


if __name__ == '__main__':
    unittest.main()
