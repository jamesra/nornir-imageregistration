"""The two-stage angle x scale search: opt-in, exhaustive, and it must know when to give up.

#234: `_find_best_angle_with_scale_search` runs a complete angle sweep at each of the 11
`_SCALE_REFINE_TISSUE_GRID` candidates -- ~186 min serial for one registration. The cheap
route the issue proposed (sweep at scale 1.0, refine after) delegates to `_refine_scale_local`,
the ternary search #95 measured as a random walk on a stochastic objective, so it is blocked.

`test_brute_decimated_ranking.py` measured the alternative: both the winning angle and the
winning scale keep rank 1 when the grid is evaluated on decimated images. So the grid can stay
**exhaustive** and simply be scored cheaply, then the best few refined at full resolution --
which needs no scale search at all and is therefore not blocked on #95.

Measured end to end on the ds32 pair (45 angles x 11 scales, numpy, single thread):

    default (exhaustive)   angle -180.00  scale 1.0040  2320.3s
    coarse 1024            angle -180.00  scale 1.0040   148.7s   15.6x
    coarse  512            angle -180.00  scale 1.0040    53.1s   43.7x

Identical angle and scale at both levels. The peaks differ by (-0.60, +1.03) and
(-0.79, +0.53) px, which is *not* decimation error: holding the configuration fixed and
varying only the seed moves the peak by **1.434px in y and 1.028px in x** (weights 2.68-2.96),
so both deltas sit inside this objective's own run-to-run spread, exactly as #95 predicts.

These tests use synthetic images and mocks so they run in milliseconds; the expensive
comparison above lives in `test_brute_decimated_ranking.py`.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import stos_brute

_DIM_VAR = stos_brute._COARSE_GRID_DIM_VAR
_TOPK_VAR = stos_brute._COARSE_GRID_TOPK_VAR


def _pair(dim: int, min_overlap: float = 0.5) -> stos_brute._RegistrationPair:
    """Blank square source/target pair with placeholder stats."""
    return stos_brute._RegistrationPair(
        np.zeros((dim, dim), dtype=np.float32), np.zeros((dim, dim), dtype=np.float32),
        mock.Mock(), mock.Mock(), min_overlap)


class _EnvIsolated(unittest.TestCase):
    """Neither variable may leak between tests, or the default-off guarantee is untestable."""

    def setUp(self):
        self._saved = {var: os.environ.get(var) for var in (_DIM_VAR, _TOPK_VAR)}
        for var in (_DIM_VAR, _TOPK_VAR):
            os.environ.pop(var, None)

    def tearDown(self):
        for var, value in self._saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


class TestItIsOffUnlessAskedFor(_EnvIsolated):
    """It changes registration output, so nothing may change without an explicit opt-in."""

    def test_unset_means_disabled(self):
        self.assertIsNone(stos_brute._coarse_grid_settings())

    def test_empty_and_whitespace_mean_disabled(self):
        for value in ('', '   '):
            os.environ[_DIM_VAR] = value
            self.assertIsNone(stos_brute._coarse_grid_settings())

    def test_zero_and_negative_mean_disabled(self):
        for value in ('0', '-1', '-4096'):
            os.environ[_DIM_VAR] = value
            self.assertIsNone(stos_brute._coarse_grid_settings())

    def test_a_non_integer_disables_rather_than_raising(self):
        """A malformed knob must not fail a multi-hour registration."""
        os.environ[_DIM_VAR] = 'medium'
        with self.assertLogs(stos_brute.__name__, level='WARNING'):
            self.assertIsNone(stos_brute._coarse_grid_settings())

    def test_a_valid_dimension_enables_with_the_default_top_k(self):
        os.environ[_DIM_VAR] = '1024'
        self.assertEqual((1024, stos_brute._COARSE_GRID_DEFAULT_TOPK),
                         stos_brute._coarse_grid_settings())

    def test_top_k_can_be_overridden_and_is_floored_at_one(self):
        os.environ[_DIM_VAR] = '512'
        for raw, expected in (('5', 5), ('1', 1), ('0', 1), ('-3', 1)):
            os.environ[_TOPK_VAR] = raw
            self.assertEqual((512, expected), stos_brute._coarse_grid_settings())

    def test_a_non_integer_top_k_keeps_the_default(self):
        os.environ[_DIM_VAR] = '512'
        os.environ[_TOPK_VAR] = 'lots'
        with self.assertLogs(stos_brute.__name__, level='WARNING'):
            self.assertEqual((512, stos_brute._COARSE_GRID_DEFAULT_TOPK),
                             stos_brute._coarse_grid_settings())


class TestTheDefaultPathIsUnchanged(_EnvIsolated):
    """With the knob off, the entry point must behave exactly as it did before #234."""

    def test_the_entry_point_delegates_to_the_exhaustive_search(self):
        sentinel = (mock.sentinel.record, 1.25)
        with mock.patch.object(stos_brute, '_find_best_angle_exhaustive',
                               return_value=sentinel) as exhaustive:
            with mock.patch.object(stos_brute, '_find_best_angle_with_coarse_grid') as coarse:
                result = stos_brute._find_best_angle_with_scale_search(
                    _pair(8), [0.0, 10.0, 20.0], metadata_applied=1.0, scale_hint=None,
                    SingleThread=True, use_cluster=False)
        self.assertEqual(sentinel, result)
        exhaustive.assert_called_once()
        coarse.assert_not_called()

    def test_a_single_angle_does_not_use_the_coarse_grid(self):
        """One angle is already cheap, and the coarse pass has nothing to rank."""
        os.environ[_DIM_VAR] = '512'
        with mock.patch.object(stos_brute, '_find_best_angle_exhaustive',
                               return_value=(mock.sentinel.record, 1.0)) as exhaustive:
            with mock.patch.object(stos_brute, '_find_best_angle_with_coarse_grid') as coarse:
                stos_brute._find_best_angle_with_scale_search(
                    _pair(8), [0.0], metadata_applied=1.0, scale_hint=None,
                    SingleThread=True, use_cluster=False)
        exhaustive.assert_called_once()
        coarse.assert_not_called()

    def test_the_coarse_grid_is_used_when_enabled(self):
        os.environ[_DIM_VAR] = '512'
        sentinel = (mock.sentinel.coarse_record, 0.98)
        with mock.patch.object(stos_brute, '_find_best_angle_with_coarse_grid',
                               return_value=sentinel) as coarse:
            with mock.patch.object(stos_brute, '_find_best_angle_exhaustive') as exhaustive:
                result = stos_brute._find_best_angle_with_scale_search(
                    _pair(8), [0.0, 10.0, 20.0], metadata_applied=1.0, scale_hint=None,
                    SingleThread=True, use_cluster=False)
        self.assertEqual(sentinel, result)
        coarse.assert_called_once()
        exhaustive.assert_not_called()
        largest_dimension, top_k = coarse.call_args.args[-2:]
        self.assertEqual(512, largest_dimension)
        self.assertEqual(stos_brute._COARSE_GRID_DEFAULT_TOPK, top_k)


class TestTheGridStaysExhaustive(_EnvIsolated):
    """The saving must come from resolution, not from dropping candidates."""

    def test_every_scale_candidate_is_scored_in_the_coarse_pass(self):
        candidates = [0.9, 0.95, 1.0, 1.05, 1.1]
        angles = [0.0, 10.0, 20.0]
        seen = []

        def fake(pair, angle_range, scale, *args, **kwargs):
            seen.append(float(scale))
            record = mock.Mock()
            # A clear winner at 1.05 so the peak-ratio guard is satisfied.
            record.weight = 5.0 if scale == 1.05 else 1.0
            record.angle = 10.0
            return record

        with mock.patch.object(stos_brute, '_find_best_angle_at_scale', side_effect=fake):
            _, best_scale = stos_brute._find_best_angle_with_coarse_grid(
                _pair(2048), angles, candidates,
                True, False, 512, 2)

        coarse_calls = seen[:len(candidates)]
        self.assertEqual(sorted(candidates), sorted(coarse_calls),
                         'the coarse pass must score every candidate; the point of this '
                         'route is that the grid does not have to be cut')
        self.assertEqual(1.05, best_scale)

    def test_only_top_k_candidates_are_refined(self):
        candidates = [0.9, 0.95, 1.0, 1.05, 1.1]
        weights = {0.9: 1.0, 0.95: 2.0, 1.0: 5.0, 1.05: 3.0, 1.1: 1.5}
        calls = []

        def fake(pair, angle_range, scale, *args, **kwargs):
            calls.append((float(scale), tuple(float(a) for a in angle_range)))
            record = mock.Mock()
            record.weight = weights[scale]
            record.angle = 10.0
            return record

        with mock.patch.object(stos_brute, '_find_best_angle_at_scale', side_effect=fake):
            stos_brute._find_best_angle_with_coarse_grid(
                _pair(2048), [0.0, 10.0, 20.0], candidates,
                True, False, 512, 2)

        refine_calls = calls[len(candidates):]
        self.assertEqual(2, len(refine_calls))
        self.assertEqual([1.0, 1.05], sorted(scale for scale, _ in refine_calls),
                         'the two strongest candidates must be the ones refined')

    def test_the_refine_pass_covers_the_neighbours_of_the_coarse_angle(self):
        """Decimation can move the winner by a step, so the pick is not trusted exactly."""
        calls = []

        def fake(pair, angle_range, scale, *args, **kwargs):
            calls.append(tuple(float(a) for a in angle_range))
            record = mock.Mock()
            record.weight = 5.0 if scale == 1.0 else 1.0
            record.angle = 10.0
            return record

        with mock.patch.object(stos_brute, '_find_best_angle_at_scale', side_effect=fake):
            stos_brute._find_best_angle_with_coarse_grid(
                _pair(2048), [0.0, 10.0, 20.0, 30.0], [0.95, 1.0],
                True, False, 512, 1)

        self.assertEqual((0.0, 10.0, 20.0), calls[-1],
                         'the full-resolution refine should bracket the coarse winner')


class TestItFallsBackWhenTheSurfaceIsFlat(_EnvIsolated):
    """#235's lesson: on a flat objective, ranking cheaply just picks noise cheaply."""

    def test_a_flat_coarse_ranking_falls_back_to_the_full_search(self):
        candidates = [0.9, 0.95, 1.0, 1.05, 1.1]

        def flat(pair, angle_range, scale, *args, **kwargs):
            record = mock.Mock()
            record.weight = 1.9  # the measured noise floor, identical everywhere
            record.angle = 10.0
            return record

        sentinel = (mock.sentinel.full_record, 1.0)
        with mock.patch.object(stos_brute, '_find_best_angle_at_scale', side_effect=flat):
            with mock.patch.object(stos_brute, '_find_best_angle_exhaustive',
                                   return_value=sentinel) as exhaustive:
                with self.assertLogs(stos_brute.__name__, level='INFO'):
                    pair = _pair(2048)
                    result = stos_brute._find_best_angle_with_coarse_grid(
                        pair, [0.0, 10.0], candidates,
                        True, False, 512, 2)

        self.assertEqual(sentinel, result)
        exhaustive.assert_called_once()
        self.assertIs(pair, exhaustive.call_args.args[0],
                      'the fallback must search the full-resolution pair, not the decimated one')

    def test_a_margin_just_inside_the_noise_spread_falls_back(self):
        """1.2 is the threshold because #95 measured an 11-17% run-to-run spread."""
        self.assertGreater(stos_brute._COARSE_GRID_MIN_PEAK_RATIO, 1.17)

    def test_a_clear_peak_does_not_fall_back(self):
        def peaked(pair, angle_range, scale, *args, **kwargs):
            record = mock.Mock()
            record.weight = 5.0 if scale == 1.0 else 1.9
            record.angle = 10.0
            return record

        with mock.patch.object(stos_brute, '_find_best_angle_at_scale', side_effect=peaked):
            with mock.patch.object(stos_brute, '_find_best_angle_exhaustive') as exhaustive:
                _, scale = stos_brute._find_best_angle_with_coarse_grid(
                    _pair(2048), [0.0, 10.0], [0.95, 1.0, 1.05],
                    True, False, 512, 1)
        exhaustive.assert_not_called()
        self.assertEqual(1.0, scale)

    def test_images_already_smaller_than_the_target_skip_the_coarse_pass(self):
        """Decimating upward would cost time and add interpolation for nothing."""
        sentinel = (mock.sentinel.full_record, 1.0)
        with mock.patch.object(stos_brute, '_find_best_angle_exhaustive',
                               return_value=sentinel) as exhaustive:
            with mock.patch.object(stos_brute, '_find_best_angle_at_scale') as at_scale:
                result = stos_brute._find_best_angle_with_coarse_grid(
                    _pair(256), [0.0, 10.0], [0.95, 1.0],
                    True, False, 512, 2)
        self.assertEqual(sentinel, result)
        exhaustive.assert_called_once()
        at_scale.assert_not_called()


class TestRegistrationPairRouting(_EnvIsolated):
    """Each ``_RegistrationPair`` field must reach the matching argument one level down.

    Source and target get distinct shapes and distinct intensity levels, so a swapped image
    or stats object is visible.
    """

    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(0)
        self.source = rng.random((2048, 1024)).astype(np.float32)
        self.target = (rng.random((1024, 2048)) + 10.0).astype(np.float32)
        self.source_stats = nornir_imageregistration.ImageStats.CalcStats(self.source)
        self.target_stats = nornir_imageregistration.ImageStats.CalcStats(self.target)
        self.pair = stos_brute._RegistrationPair(
            self.source, self.target, self.source_stats, self.target_stats, 0.37)

    def test_at_scale_one_passes_every_field_unchanged(self):
        with mock.patch.object(stos_brute, '_find_best_angle') as inner:
            stos_brute._find_best_angle_at_scale(self.pair, [0.0, 1.0], 1.0, True, False)
        kw = inner.call_args.kwargs
        self.assertIs(self.source, kw['source_image'])
        self.assertIs(self.target, kw['target_image'])
        self.assertIs(self.source_stats, kw['source_stats'])
        self.assertIs(self.target_stats, kw['target_stats'])
        self.assertEqual(0.37, kw['min_overlap'])
        self.assertEqual(1.0, kw['source_scale'])

    def test_at_another_scale_only_the_source_is_rescaled(self):
        with mock.patch.object(stos_brute, '_find_best_angle') as inner:
            stos_brute._find_best_angle_at_scale(self.pair, [0.0, 1.0], 0.5, True, False)
        kw = inner.call_args.kwargs
        self.assertEqual((1024, 512), tuple(kw['source_image'].shape))
        self.assertIs(self.target, kw['target_image'])
        self.assertIs(self.target_stats, kw['target_stats'])
        self.assertIsNot(self.source_stats, kw['source_stats'])
        self.assertLess(kw['source_stats'].mean, 1.0, 'stats must come from the scaled source')
        self.assertEqual(0.37, kw['min_overlap'])

    def test_the_exhaustive_search_hands_the_pair_to_every_candidate(self):
        record = mock.Mock(weight=1.0, angle=0.0)
        with mock.patch.object(stos_brute, '_find_best_angle_at_scale',
                               return_value=record) as at_scale:
            stos_brute._find_best_angle_exhaustive(self.pair, [0.0], [0.9, 1.0], True, False)
        self.assertEqual(2, at_scale.call_count)
        for call in at_scale.call_args_list:
            self.assertIs(self.pair, call.args[0])

    def test_the_scale_search_hands_the_pair_on(self):
        for dim in (None, '512'):
            with self.subTest(coarse=dim):
                if dim is None:
                    os.environ.pop(_DIM_VAR, None)
                    target = '_find_best_angle_exhaustive'
                else:
                    os.environ[_DIM_VAR] = dim
                    target = '_find_best_angle_with_coarse_grid'
                with mock.patch.object(stos_brute, target,
                                       return_value=(mock.sentinel.r, 1.0)) as inner:
                    stos_brute._find_best_angle_with_scale_search(
                        self.pair, [0.0, 10.0], metadata_applied=1.0, scale_hint=None,
                        SingleThread=True, use_cluster=False, force_search=True)
                self.assertIs(self.pair, inner.call_args.args[0])

    def test_the_coarse_pass_scores_a_decimated_pair_and_the_refine_the_original(self):
        pairs = []

        def fake(pair, angle_range, scale, *args, **kwargs):
            pairs.append(pair)
            record = mock.Mock()
            record.weight = 5.0 if scale == 1.0 else 1.0
            record.angle = 0.0
            return record

        with mock.patch.object(stos_brute, '_find_best_angle_at_scale', side_effect=fake):
            stos_brute._find_best_angle_with_coarse_grid(
                self.pair, [0.0, 10.0], [0.95, 1.0], True, False, 512, 1)

        coarse, refine = pairs[:2], pairs[2:]
        self.assertEqual(1, len(refine))
        self.assertIs(self.pair, refine[0])
        for coarse_pair in coarse:
            self.assertEqual((512, 256), tuple(coarse_pair.source_image.shape))
            self.assertEqual((256, 512), tuple(coarse_pair.target_image.shape))
            self.assertLess(coarse_pair.source_stats.mean, 1.0)
            self.assertGreater(coarse_pair.target_stats.mean, 10.0)
            self.assertEqual(0.37, coarse_pair.min_overlap)


class TestDecimationScale(unittest.TestCase):

    def test_it_uses_the_largest_dimension_of_either_image(self):
        self.assertAlmostEqual(0.25, stos_brute._decimation_scale((1024, 512), (800, 900), 256))
        self.assertAlmostEqual(0.25, stos_brute._decimation_scale((512, 800), (1024, 900), 256))

    def test_it_returns_at_least_one_when_the_images_are_small(self):
        self.assertGreaterEqual(stos_brute._decimation_scale((100, 100), (100, 100), 512), 1.0)


class TestNeighbouringAngles(unittest.TestCase):

    def test_an_interior_angle_gets_both_neighbours(self):
        self.assertEqual([0.0, 10.0, 20.0],
                         stos_brute._neighbouring_angles([0.0, 10.0, 20.0, 30.0], 10.0))

    def test_the_first_angle_gets_only_the_one_after(self):
        self.assertEqual([0.0, 10.0],
                         stos_brute._neighbouring_angles([0.0, 10.0, 20.0], 0.0))

    def test_the_last_angle_gets_only_the_one_before(self):
        self.assertEqual([10.0, 20.0],
                         stos_brute._neighbouring_angles([0.0, 10.0, 20.0], 20.0))

    def test_a_single_angle_sweep_yields_that_angle(self):
        self.assertEqual([5.0], stos_brute._neighbouring_angles([5.0], 5.0))

    def test_an_empty_sweep_yields_the_requested_angle(self):
        self.assertEqual([7.0], stos_brute._neighbouring_angles([], 7.0))

    def test_it_snaps_to_the_nearest_listed_angle(self):
        """The coarse record's angle comes back as a float and may not compare equal."""
        self.assertEqual([0.0, 10.0, 20.0],
                         stos_brute._neighbouring_angles([0.0, 10.0, 20.0, 30.0], 10.0000001))

    def test_it_never_returns_duplicates(self):
        for angles in ([0.0], [0.0, 1.0], [0.0, 1.0, 2.0, 3.0]):
            for angle in angles:
                window = stos_brute._neighbouring_angles(angles, angle)
                self.assertEqual(len(window), len(set(window)))


if __name__ == '__main__':
    unittest.main()
