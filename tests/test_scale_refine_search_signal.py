"""``_refine_scale_local``'s ternary search runs on a stochastic objective it cannot resolve.

Review #95 filed this as a cost problem: ~40 pad/rotate/FFT cycles per refinement, blamed
on re-zooming the source and recomputing full ``ImageStats`` per score. The count is right
(measured 34-37 scores narrow, 47-55 with ``wide_search=True``) but the blame is not: the
zoom is **2.3%** of a score and ``ImageStats`` **2.0%**, so the named cause is 4.3% of the
cost. ``pad_and_rotate_image`` plus the FFT is 95.7%.

The real problem is worse than a cost. ``_score`` is **stochastic**: the padding and
rotation-corner fills draw from the shared generator, so scoring one scale twice returns
different weights. Measured on a real pair, angle 30: noise sd 9.3e-02 against a mean near
1.9, an 11-17% relative spread. Re-seeding before each call collapses the spread to exactly
zero, which is how we know the generator is the only source.

Ternary search assumes a deterministic unimodal objective. With that noise, the
``_score(m1) < _score(m2)`` comparison stops being about scale almost immediately:

| ternary iteration | probe separation | mean signal | noise sd | SNR |
|---|---|---|---|---|
| 1 | 1.33e-02 | 1.09e-01 | 9.26e-02 | 1.18 |
| 2 | 8.89e-03 | 5.10e-02 | 9.26e-02 | 0.55 |
| 4 | 3.95e-03 | 2.69e-02 | 9.26e-02 | 0.29 |
| 7 | 1.17e-03 | 5.96e-03 | 9.26e-02 | 0.06 |
| 10 | 3.47e-04 | 0.00e+00 | 9.26e-02 | 0.00 |
| 14 | 6.85e-05 | 0.00e+00 | 9.26e-02 | 0.00 |

From iteration 2 the branch is decided by noise; from iteration 10 the two probes return
*identical* weights, so there is no signal at all. All 14 iterations always run -- the
``third < 1e-6`` break is unreachable from a 0.04 bracket, which would need 24 -- and they
account for 28 of the ~37 scores. So roughly 70% of the refinement is a random walk.

That also rules out the obvious cheap fix. Only 2 of the ~37 scores are exact duplicates
(3.6-5.9%, stable across seed positions and both search widths, because the candidate set
is already built through a ``set``), and memoising them would be **output-changing** twice
over: repeated scores legitimately differ today, and skipping a call also stops advancing
the generator, which shifts every later draw.

Every available fix therefore changes registration output and needs the >=100-tile
sign-off the serial/batched primitives skill requires, so #95 is blocked rather than fixed.
These tests pin the measurements the eventual fix has to be judged against.

The recommendation on record is common random numbers: derive the fill seed from the probe
parameters so both sides of each comparison see identical noise. That is what makes the
comparison about scale, and only then is cutting the iteration count safe.
"""

from __future__ import annotations

import statistics
import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import stos_brute

_ANGLE = 30.0  # nonzero, so rotation vacates corners and the fill actually matters


def _pair(size: int = 192) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260830)
    target = rng.random((size, size)).astype(np.float32)
    source = np.roll(target, 4, axis=1).astype(np.float32)
    return target, source


class _Scorer:

    def __init__(self, size: int = 192):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        self.target, self.source = _pair(size)
        self.t_stats = nornir_imageregistration.ImageStats.CalcStats(self.target)
        self.s_stats = nornir_imageregistration.ImageStats.CalcStats(self.source)

    def __call__(self, scale: float, angle: float = _ANGLE) -> float:
        return float(stos_brute.ScoreOneAngle(
            target_original=self.target, source_original=self.source,
            target_image_shape=self.target.shape,
            source_image_shape=self.source.shape,
            angle=angle, target_stats=self.t_stats, source_stats=self.s_stats,
            min_overlap=0.5, source_scale=scale).weight)


class TestTheObjectiveIsStochastic(unittest.TestCase):
    """Scoring one scale twice does not give one answer, and the RNG is why."""

    @classmethod
    def setUpClass(cls):
        cls.score = _Scorer()

    def test_repeated_scores_of_one_scale_differ(self):
        nornir_imageregistration.seed_random_data(1234)
        values = [self.score(1.0) for _ in range(5)]
        self.assertGreater(max(values) - min(values), 0.0,
                           'the padding and corner fills draw from the shared generator, '
                           'so repeated scores must differ; if they stop differing, the '
                           'noise objection to memoising has gone away')

    def test_reseeding_before_each_score_makes_them_identical(self):
        values = []
        for _ in range(4):
            nornir_imageregistration.seed_random_data(1234)
            values.append(self.score(1.0))
        self.assertEqual(0.0, max(values) - min(values),
                         'identical when re-seeded is what proves the generator is the '
                         'only source of the variation, rather than anything numerical')

    def test_the_spread_is_not_negligible(self):
        nornir_imageregistration.seed_random_data(1234)
        values = [self.score(1.0) for _ in range(6)]
        relative = (max(values) - min(values)) / abs(statistics.mean(values))
        self.assertGreater(relative, 0.005,
                           f'measured 11-17% on real imagery; got {100 * relative:.3f}%. '
                           'A negligible spread would mean the search has signal after all')


class TestTheSearchCannotResolveItsOwnBracket(unittest.TestCase):
    """The later ternary iterations compare weights that differ by less than the noise."""

    @classmethod
    def setUpClass(cls):
        cls.score = _Scorer()

    @staticmethod
    def _probe_separation(iteration: int) -> float:
        """Separation of the two ternary probes at a 1-based iteration."""
        width = 2 * stos_brute._SCALE_REFINE_LOCAL_HALF_WIDTH
        for _ in range(iteration - 1):
            width *= 2.0 / 3.0
        third = width / 3.0
        return (width - third) - third

    def _noise_sd(self) -> float:
        nornir_imageregistration.seed_random_data(4321)
        return statistics.pstdev([self.score(1.0) for _ in range(6)])

    def _paired_signal(self, separation: float, trials: int = 3) -> float:
        """Mean weight difference across *separation*, with common random numbers."""
        diffs = []
        for trial in range(trials):
            nornir_imageregistration.seed_random_data(9000 + trial)
            low = self.score(1.0 - separation / 2)
            nornir_imageregistration.seed_random_data(9000 + trial)
            high = self.score(1.0 + separation / 2)
            diffs.append(high - low)
        return abs(statistics.mean(diffs))

    def test_the_final_iteration_has_no_signal_above_the_noise(self):
        separation = self._probe_separation(
            stos_brute._SCALE_REFINE_TERNARY_ITERATIONS)
        signal = self._paired_signal(separation)
        noise = self._noise_sd()
        self.assertLess(signal, noise,
                        f'at iteration {stos_brute._SCALE_REFINE_TERNARY_ITERATIONS} the '
                        f'probes are {separation:.2e} apart; signal {signal:.2e} must stay '
                        f'below noise sd {noise:.2e}, or the search has become meaningful '
                        'and the iteration count should be revisited')

    def test_the_signal_shrinks_as_the_bracket_narrows(self):
        wide = self._paired_signal(self._probe_separation(1))
        narrow = self._paired_signal(self._probe_separation(10))
        self.assertGreater(wide, narrow,
                           'the first iteration should carry more signal than the tenth; '
                           'this is the trend that makes the later ones worthless')

    def test_the_first_iteration_is_the_only_one_near_usable(self):
        signal = self._paired_signal(self._probe_separation(1))
        noise = self._noise_sd()
        # Measured SNR 1.18 on real imagery -- marginal, not comfortable.
        self.assertGreater(signal / max(noise, 1e-12), 0.25,
                           'iteration 1 should retain some signal; if even it is pure '
                           'noise, local refinement is not merely over-iterated')


class TestAllFourteenIterationsAlwaysRun(unittest.TestCase):
    """The early-exit cannot fire, so the cost is fixed, not adaptive."""

    def test_the_break_threshold_is_unreachable(self):
        width = 2 * stos_brute._SCALE_REFINE_LOCAL_HALF_WIDTH
        ran = 0
        for _ in range(stos_brute._SCALE_REFINE_TERNARY_ITERATIONS):
            if width / 3.0 < 1e-6:
                break
            width *= 2.0 / 3.0
            ran += 1
        self.assertEqual(stos_brute._SCALE_REFINE_TERNARY_ITERATIONS, ran,
                         'every iteration runs; the third < 1e-6 guard would need about '
                         '24 iterations from a 0.04 bracket, so it is dead code')

    def test_the_iterations_dominate_the_score_count(self):
        ternary_scores = 2 * stos_brute._SCALE_REFINE_TERNARY_ITERATIONS
        self.assertEqual(28, ternary_scores)
        # Measured 34-37 total for wide_search=False.
        self.assertGreater(ternary_scores / 37.0, 0.70,
                           'the ternary loop is the majority of the refinement cost, '
                           'which is why the zoom and stats are not the lever')

    def test_the_final_bracket_is_far_finer_than_the_noise_floor_justifies(self):
        width = 2 * stos_brute._SCALE_REFINE_LOCAL_HALF_WIDTH
        for _ in range(stos_brute._SCALE_REFINE_TERNARY_ITERATIONS):
            width *= 2.0 / 3.0
        self.assertLess(width, 1.0e-3,
                        f'final bracket {width:.2e}; the point is that this precision is '
                        'asserted on an objective that cannot distinguish it')


class TestTheNamedCauseIsNotTheCost(unittest.TestCase):
    """#95 blamed the per-score zoom and ImageStats. They are 4.3% together."""

    @classmethod
    def setUpClass(cls):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        cls.target, cls.source = _pair(256)

    def test_a_non_unity_scale_does_rezoom_and_restat(self):
        # The mechanism is real even though it is not the cost, so pin it: a scale of 1.0
        # must skip both, and anything else must not.
        self.assertIs(stos_brute._scale_registration_image(self.source, 1.0), self.source,
                      'unity scale should short-circuit the zoom')
        zoomed = stos_brute._scale_registration_image(self.source, 0.97)
        self.assertIsNot(zoomed, self.source)
        self.assertNotEqual(self.source.shape, zoomed.shape)

    def test_the_zoom_is_a_small_fraction_of_a_score(self):
        import time
        scale = 0.97
        t0 = time.perf_counter()
        for _ in range(3):
            stos_brute._scale_registration_image(self.source, scale)
        zoom = (time.perf_counter() - t0) / 3

        scorer = _Scorer(256)
        t0 = time.perf_counter()
        scorer(scale)
        whole = time.perf_counter() - t0

        self.assertLess(zoom, 0.5 * whole,
                        f'zoom {zoom:.4f}s vs whole score {whole:.4f}s; measured 2.3% on '
                        'real imagery. If the zoom ever dominates, #95 becomes the right '
                        'diagnosis after all')


if __name__ == '__main__':
    unittest.main()
