"""A decimated brute sweep still ranks the winning angle and scale first (review #234).

#234 costs a full registration ~186 min serially (~348 before option 1's frame change),
because `_find_best_angle_with_scale_search` runs a complete angle sweep at each of the 11
`_SCALE_REFINE_TISSUE_GRID` candidates. Its option 2 -- sweep at scale 1.0 and refine -- is
blocked on #95, which measured `_refine_scale_local`'s ternary search as a random walk on a
stochastic objective. Option 3, coarse-to-fine on angle, was unmeasured.

Measured here on the ds32 pair (`0162_ds32.png` / `0164_ds32.png`, 4183x4309 target, 6000px
frame), numpy backend, 45 angles at 8 degree steps:

    LargestDimension   s/angle   sweep speedup   rank of full-res winner
    full (4309)        5.513     1.0x            1/45
    2048               1.206     4.6x            1/45
    1024               0.287     19.2x           1/45
    512                0.068     80.5x           1/45

and for the 11 scale candidates at the winning angle:

    LargestDimension   s/scale   speedup   rank of full-res best scale (1.0040)
    full               3.253     1.0x      1/11
    1024               0.183     17.8x     1/11
    512                0.044     73.8x     1/11

So **both** winners survive decimation all the way to 512. That matters more than it first
looks, because it means the 11x cross product does not have to be *cut* to be made affordable
-- it can be kept exhaustive and evaluated on decimated images, then the top few (angle,
scale) pairs re-scored at full resolution. That route does not touch `_refine_scale_local`, so
it is not blocked on #95.

The important caveat, and why this file asserts ranks rather than weights: the weight *curves*
correlate with full resolution only +0.74 to +0.83. Off the peak the objective is noise
(#95: sd 9.3e-02 on a mean near 1.9), so the coarse ranking below the top is not meaningful.
What carries is that the true peak stands clear of that floor -- 2.77 against a 2.11 runner-up
at full resolution, and the margin widens as resolution drops. A coarse-to-fine scheme must
therefore refine a top-K, and must not assume the coarse ordering below rank 1 means anything.

These are characterization tests, in the manner of `test_scale_refine_search_signal.py` (#95)
and `test_fft_budget_precision.py` (#227): they record a measurement that a behaviour change
depends on, so it does not have to be re-derived, and they fail if the property stops holding.
No production behaviour has changed for option 3 yet -- that needs the >=100-tile sign-off.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration import stos_brute

_IMAGE_DIR = r'D:\nornir-testdata\Images'
_TARGET = os.path.join(_IMAGE_DIR, '0162_ds32.png')
_SOURCE = os.path.join(_IMAGE_DIR, '0164_ds32.png')

# The winner on this pair, established by the full-resolution sweep.
_BEST_ANGLE = -180.0
_BEST_SCALE = 1.0040

_SEED = 1234


def _fixtures_present() -> bool:
    return os.path.exists(_TARGET) and os.path.exists(_SOURCE)


def _load(largest_dimension: int | None):
    target = nornir_imageregistration.ImageParamToImageArray(
        _TARGET, dtype=nornir_imageregistration.default_image_dtype())
    source = nornir_imageregistration.ImageParamToImageArray(
        _SOURCE, dtype=nornir_imageregistration.default_image_dtype())

    if largest_dimension is None:
        return target, source

    scale = float(largest_dimension) / max(*target.shape, *source.shape)
    if scale >= 1.0:
        return target, source
    return _resize(target, scale), _resize(source, scale)


def _resize(image, scale: float):
    # scipy's spline filter has no float16 kernel and these images load as float16.
    dtype = image.dtype
    return nornir_imageregistration.ResizeImage(
        image.astype(np.float32), scale).astype(dtype)


def _score(target, source, angle: float, frame_angles=None) -> float:
    target_stats = nornir_imageregistration.ImageStats.CalcStats(target)
    source_stats = nornir_imageregistration.ImageStats.CalcStats(source)
    frame = stos_brute._fixed_correlation_shape(
        target.shape, source.shape,
        [angle] if frame_angles is None else frame_angles, 0.75)
    padded = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
        target, min_overlap=1.0, image_median=target_stats.median,
        image_stddev=target_stats.std, new_height=frame[0], new_width=frame[1],
        power_of_two=False)
    record = stos_brute.ScoreOneAngle(
        padded, source, target.shape, source.shape, float(angle),
        target_stats=target_stats, source_stats=source_stats,
        target_image_prepadded=True, min_overlap=0.75, fixed_shape=frame)
    return float(record.weight)


def _sweep(target, source, angles) -> np.ndarray:
    return np.array([_score(target, source, a, frame_angles=angles) for a in angles])


@unittest.skipUnless(_fixtures_present(), 'ds32 registration pair not available')
class TestTheWinningAngleSurvivesDecimation(unittest.TestCase):
    """The cheap levels must agree with each other and keep the winner clear of the floor."""

    # A small set: the winner plus decoys drawn from the measured runners-up, so this runs in
    # seconds. The full 45-angle comparison against full resolution is the slow test below.
    ANGLES = np.array([-180.0, -92.0, -4.0, 84.0, 92.0, 172.0])

    def setUp(self):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        nornir_imageregistration.seed_random_data(_SEED)

    def test_512_and_1024_agree_on_the_winner(self):
        winners = {}
        for largest in (512, 1024):
            target, source = _load(largest)
            weights = self._sweep(target, source)
            winners[largest] = float(self.ANGLES[int(np.argmax(weights))])
        self.assertEqual(winners[512], winners[1024],
                         f'decimation levels disagree on the best angle: {winners}')
        self.assertEqual(_BEST_ANGLE, winners[512],
                         'the level that is 80x cheaper no longer finds the full-resolution '
                         'winner; coarse-to-fine for #234 would pick the wrong angle')

    def test_the_peak_stands_clear_of_the_noise_floor(self):
        """#95: off-peak weights are noise near 1.9, so the margin is what makes this work."""
        target, source = _load(512)
        weights = self._sweep(target, source)
        order = np.argsort(-weights)
        best, runner_up = weights[order[0]], weights[order[1]]
        self.assertGreater(best / runner_up, 1.2,
                           f'peak margin {best / runner_up:.3f} is inside the measured '
                           f'11-17% run-to-run spread of this objective, so rank 1 would '
                           f'not be reliable')

    def test_the_cheap_level_really_is_much_cheaper(self):
        """Guards the premise: if the frame stopped shrinking, coarse-to-fine buys nothing."""
        _, source_full = _load(None)
        target_full, _ = _load(None)
        target_small, source_small = _load(512)
        full_frame = stos_brute._fixed_correlation_shape(
            target_full.shape, source_full.shape, self.ANGLES, 0.75)
        small_frame = stos_brute._fixed_correlation_shape(
            target_small.shape, source_small.shape, self.ANGLES, 0.75)
        area_ratio = (full_frame[0] * full_frame[1]) / (small_frame[0] * small_frame[1])
        self.assertGreater(area_ratio, 25.0,
                           f'frame area ratio {area_ratio:.1f} is too small to be worth a '
                           f'coarse pass')

    def _sweep(self, target, source) -> np.ndarray:
        return _sweep(target, source, self.ANGLES)


@unittest.skipUnless(_fixtures_present(), 'ds32 registration pair not available')
class TestTheBestScaleSurvivesDecimation(unittest.TestCase):
    """If scale survives too, the 11x grid can stay exhaustive and just be evaluated cheaply.

    This is the part that matters for the #95 blocker: option 2 needs `_refine_scale_local`,
    and this route does not.
    """

    def setUp(self):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        nornir_imageregistration.seed_random_data(_SEED)

    def test_the_grid_still_has_eleven_candidates_with_no_hint(self):
        """The 11x the issue reports; if this changes, the arithmetic above is stale."""
        candidates = stos_brute._scale_search_candidates(1.0, None)
        self.assertEqual(11, len(list(candidates)))

    def test_512_ranks_the_full_resolution_best_scale_first(self):
        target, source = _load(512)
        scales = list(stos_brute._SCALE_REFINE_TISSUE_GRID)
        weights = []
        for scale in scales:
            scaled = source if scale == 1.0 else _resize(source, scale)
            weights.append(_score(target, scaled, _BEST_ANGLE))
        best = scales[int(np.argmax(weights))]
        self.assertAlmostEqual(_BEST_SCALE, best, places=3,
                               msg=f'the 74x-cheaper level ranks scale {best:.4f} first, not '
                                   f'the full-resolution best {_BEST_SCALE:.4f}')


@pytest.mark.slow
@unittest.skipUnless(_fixtures_present(), 'ds32 registration pair not available')
class TestAgainstFullResolution(unittest.TestCase):
    """The measurement in this module's docstring, re-derived. ~5 min: one full-res sweep."""

    ANGLES = np.arange(-180.0, 180.0, 8.0)

    def setUp(self):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        nornir_imageregistration.seed_random_data(_SEED)

    def test_every_decimation_level_ranks_the_full_resolution_winner_first(self):
        target, source = _load(None)
        full = _sweep(target, source, self.ANGLES)
        best_index = int(np.argmax(full))

        for largest in (2048, 1024, 512):
            with self.subTest(largest_dimension=largest):
                target, source = _load(largest)
                coarse = _sweep(target, source, self.ANGLES)
                rank = int(np.where(np.argsort(-coarse) == best_index)[0][0])
                self.assertEqual(0, rank,
                                 f'at LargestDimension={largest} the full-resolution winner '
                                 f'(angle {self.ANGLES[best_index]}) fell to rank {rank + 1}')

    def test_the_weight_curves_correlate_only_moderately(self):
        """Records why a top-K refine is required rather than trusting the coarse order."""
        target, source = _load(None)
        full = _sweep(target, source, self.ANGLES)
        target, source = _load(1024)
        coarse = _sweep(target, source, self.ANGLES)
        corr = float(np.corrcoef(full, coarse)[0, 1])
        self.assertGreater(corr, 0.6, f'correlation {corr:+.3f} is too low for the coarse '
                                      f'pass to be informative at all')
        self.assertLess(corr, 0.95, f'correlation {corr:+.3f} is higher than measured '
                                    f'(+0.805); if the surface is now faithful, a top-K '
                                    f'refine may be narrowed -- re-measure before relying '
                                    f'on the coarse order below rank 1')


if __name__ == '__main__':
    unittest.main()
