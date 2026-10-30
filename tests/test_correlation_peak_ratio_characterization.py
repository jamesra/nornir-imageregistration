"""What ``_correlation_peak_ratio`` actually measures, pinned.

It takes a global top-2 with no exclusion radius, unlike
``peak_uniqueness.masked_peak_ratio`` which clears a radius-3 square around the primary
first. On a real correlation surface the runner-up is normally the pixel next to the
maximum, so the ratio falls as the peak *widens*, not only as a rival peak rises. It
reports sharpness and uniqueness together, whichever is worse.

Measured on the four surfaces the function is called on:

=========================  =========  =======  ======
surface                    top2 dist  as-is    masked
=========================  =========  =======  ======
translation, shift (13,21)         1   49.886  148.76
angle, rotate 0 deg                1    3.628   14.97
angle, rotate 30 deg              86    1.056    1.056
radial, scale 1.00                 1    3.845   16.45
=========================  =========  =======  ======

9 of 11 had the runner-up at Chebyshev distance 1.

Three properties matter, and this module locks each one down:

1. **Genuine ambiguity is still detected.** Two equal peaks far apart score 1.0000
   under either definition, so the metric is not broken for its primary job.
2. **The error is one-directional.** Excluding candidates can only lower the runner-up,
   so this ratio is always <= the masked one. Confidence is understated, never
   overstated, which costs extra fallback searching rather than a wrong answer.
3. **A unique but broad peak reads as ambiguous.** A lone Gaussian at sigma=6 scores
   1.014 here against 1.249 masked, straddling the 1.2 gate. This is the real cost.

The metric is *not* changed here. ``ambiguous`` gates whether the brute-force fallback
angle search runs at all, and its thresholds are calibrated against this definition, so
substituting the masked ratio would skip fallbacks that currently rescue bad alignments.
That needs a re-tune with the >=100-tile sign-off. See review issue #88.
"""

from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration import peak_uniqueness
from nornir_imageregistration import stos_brute as sb

_N = 256

# The gate in _find_angle_and_scale_with_logpolar that decides `ambiguous`.
_AMBIGUOUS_ANGLE_GATE = 1.2


def _lobes(peaks: list[tuple[int, int, float]], sigma: float) -> np.ndarray:
    """Surface built from Gaussian lobes at ``(row, col, height)``."""
    rows, cols = np.mgrid[0:_N, 0:_N].astype(np.float32)
    out = np.zeros((_N, _N), dtype=np.float32)
    for row, col, height in peaks:
        with np.errstate(under='ignore'):
            out += height * np.exp(
                -((rows - row) ** 2 + (cols - col) ** 2) / (2.0 * sigma ** 2))
    return out


def _masked(arr: np.ndarray) -> float:
    """The exclusion-radius ratio, for comparison only."""
    row, col = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return peak_uniqueness.masked_peak_ratio(
        arr, int(row), int(col),
        exclusion_radius=peak_uniqueness.DEFAULT_PEAK_RATIO_EXCLUSION_RADIUS)


def _runner_up_distance(arr: np.ndarray) -> int:
    """Chebyshev distance from the maximum to the second-highest pixel."""
    order = np.argsort(arr.ravel())[::-1]
    first = np.unravel_index(order[0], arr.shape)
    second = np.unravel_index(order[1], arr.shape)
    return max(abs(int(first[0]) - int(second[0])),
               abs(int(first[1]) - int(second[1])))


class TestGenuineAmbiguityIsStillDetected(unittest.TestCase):
    """The metric's primary job, which it does correctly."""

    def test_two_equal_peaks_far_apart_score_unity(self):
        for sigma in (1.0, 2.0, 4.0):
            with self.subTest(sigma=sigma):
                arr = _lobes([(64, 64, 1.0), (192, 192, 1.0)], sigma)

                ratio = sb._correlation_peak_ratio(arr)

                self.assertAlmostEqual(ratio, 1.0, places=3)
                self.assertLess(ratio, _AMBIGUOUS_ANGLE_GATE)

    def test_the_masked_ratio_agrees_when_the_rival_is_distant(self):
        """Where the two definitions must not diverge: a real competing peak."""
        arr = _lobes([(64, 64, 1.0), (192, 192, 1.0)], 2.0)

        self.assertAlmostEqual(sb._correlation_peak_ratio(arr), _masked(arr), places=3)

    def test_a_near_rival_is_flagged(self):
        arr = _lobes([(64, 64, 1.0), (192, 192, 0.95)], 1.0)

        self.assertLess(sb._correlation_peak_ratio(arr), _AMBIGUOUS_ANGLE_GATE)


class TestTheErrorIsOneDirectional(unittest.TestCase):
    """Confidence is understated, never overstated -- the safety argument."""

    def test_never_exceeds_the_masked_ratio(self):
        rng = np.random.default_rng(0)
        violations = []
        for _trial in range(120):
            peaks = [(int(rng.integers(10, _N - 10)), int(rng.integers(10, _N - 10)),
                      float(rng.uniform(0.3, 1.0)))
                     for _ in range(int(rng.integers(1, 4)))]
            arr = _lobes(peaks, float(rng.uniform(0.5, 20.0)))
            arr = arr + rng.normal(0, 0.01, arr.shape).astype(np.float32)

            as_is = sb._correlation_peak_ratio(arr)
            masked = _masked(arr)
            if as_is > masked + 1e-4:
                violations.append((as_is, masked))

        self.assertEqual(
            violations, [],
            'excluding candidates can only lower the runner-up, so the unmasked ratio '
            'must never exceed the masked one')

    def test_a_flagged_surface_is_never_wrongly_confident(self):
        """Anything this metric clears, the masked one clears too."""
        for sigma in (0.6, 1.0, 2.0, 6.0, 12.0):
            with self.subTest(sigma=sigma):
                arr = _lobes([(128, 128, 1.0)], sigma)
                if sb._correlation_peak_ratio(arr) >= _AMBIGUOUS_ANGLE_GATE:
                    self.assertGreaterEqual(_masked(arr), _AMBIGUOUS_ANGLE_GATE)


class TestSharpnessIsConflatedWithUniqueness(unittest.TestCase):
    """The actual defect: a unique peak can be flagged for being broad."""

    def test_the_runner_up_is_the_neighbouring_pixel(self):
        for sigma in (1.0, 2.0, 6.0, 12.0):
            with self.subTest(sigma=sigma):
                arr = _lobes([(128, 128, 1.0)], sigma)

                self.assertLessEqual(
                    _runner_up_distance(arr),
                    peak_uniqueness.DEFAULT_PEAK_RATIO_EXCLUSION_RADIUS,
                    'a single lobe should put its runner-up inside the exclusion square')

    def test_the_ratio_falls_as_a_lone_peak_widens(self):
        """Nothing about uniqueness changes across these surfaces; only the width."""
        ratios = [sb._correlation_peak_ratio(_lobes([(128, 128, 1.0)], sigma))
                  for sigma in (1.0, 2.0, 6.0, 12.0, 24.0)]

        for wider, narrower in zip(ratios[1:], ratios):
            self.assertLess(wider, narrower,
                            f'ratio should decay with peak width: {ratios}')

    def test_a_broad_unique_peak_is_flagged_ambiguous(self):
        arr = _lobes([(128, 128, 1.0)], 6.0)

        self.assertLess(sb._correlation_peak_ratio(arr), _AMBIGUOUS_ANGLE_GATE,
                        'a broad lone peak is flagged by this metric')
        self.assertGreater(_masked(arr), _AMBIGUOUS_ANGLE_GATE,
                           'and would not be flagged by the masked one -- this is the '
                           'decision that would flip if the metric were swapped')

    def test_a_lone_peak_can_score_far_below_its_masked_value(self):
        arr = _lobes([(128, 128, 1.0)], 1.0)

        self.assertGreater(_masked(arr) / sb._correlation_peak_ratio(arr), 100.0)


class TestDegenerateSurfacesAreUnchanged(unittest.TestCase):
    """Edge cases the function already handled; kept so a rewrite preserves them."""

    def test_a_lone_nonzero_pixel_reaches_the_same_sentinel_scale(self):
        """Both definitions hit ~1e6 here, so the degenerate case needs no special care."""
        arr = np.zeros((_N, _N), dtype=np.float32)
        arr[128, 128] = 1.0

        self.assertAlmostEqual(sb._correlation_peak_ratio(arr), 1e6, delta=1.0)
        self.assertAlmostEqual(_masked(arr), peak_uniqueness._UNIQUE_PEAK_RATIO,
                               delta=1.0)

    def test_too_small_to_have_a_runner_up(self):
        self.assertEqual(sb._correlation_peak_ratio(np.array([5.0])), 0.0)

    def test_all_non_finite(self):
        self.assertEqual(
            sb._correlation_peak_ratio(np.array([np.nan, np.inf, -np.inf])), 0.0)

    def test_a_flat_surface_scores_unity(self):
        self.assertAlmostEqual(
            sb._correlation_peak_ratio(np.ones((16, 16), dtype=np.float32)), 1.0)

    def test_complex_input_uses_magnitude(self):
        arr = np.array([[3 + 4j, 0 + 0j], [1 + 0j, 0 + 0j]])

        # |3+4j| = 5, next largest magnitude is 1.
        self.assertAlmostEqual(sb._correlation_peak_ratio(arr), 5.0, places=5)


class TestTheMetricIsStillTheUnmaskedOne(unittest.TestCase):
    """Trip if the definition is swapped without re-tuning the gates it feeds."""

    def test_a_broad_lone_peak_still_scores_near_unity(self):
        arr = _lobes([(128, 128, 1.0)], 12.0)

        ratio = sb._correlation_peak_ratio(arr)

        self.assertLess(
            ratio, 1.05,
            'this looks like an exclusion radius was introduced. That raises every '
            'ratio, and `ambiguous` gates the brute-force fallback search, so the '
            '1.2 / 1.12 / 1.35 thresholds and _logpolar_confidence must be re-tuned '
            'with a >=100-tile sign-off before this test is updated')


if __name__ == '__main__':
    unittest.main()
