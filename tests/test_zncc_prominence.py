"""Tests for scale-free ZNCC prominence scoring used by the STOS lock gate.

Covers the same-cell cardinal fallback (``_zncc_prominence_stack``), the
neighbor-ROI null scored on the pass's own measured stacks
(``_zncc_scores_from_measured_rois``), the cross-pass null cache, and the
vectorized peak shift.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import local_distortion_correction as ldc
from nornir_imageregistration.refine_shared.cell_roles import ZnccScore, zncc_score_passes


def _loop_shift(moving, peaks, travel_eps, valid_mask):
    """The per-cell ``CropImage`` loop the vectorized shift replaced."""
    n, h, w = moving.shape
    out = np.empty_like(moving)
    out_mask = None if valid_mask is None else np.empty((n, h, w), dtype=bool)
    meds = np.median(moving, axis=(1, 2))
    for i in range(n):
        peak = peaks[i]
        if float(np.linalg.norm(peak)) < travel_eps:
            out[i] = moving[i]
            if out_mask is not None:
                out_mask[i] = valid_mask[i]
            continue
        dy = int(np.round(-peak[0]))
        dx = int(np.round(-peak[1]))
        out[i] = nornir_imageregistration.CropImage(moving[i], dx, dy, w, h, cval=float(meds[i]))
        if out_mask is not None:
            out_mask[i] = nornir_imageregistration.CropImage(
                valid_mask[i].astype(np.float32), dx, dy, w, h, cval=0.0) > 0.5
    return out, out_mask


class TestShiftMovingStackVectorized(unittest.TestCase):
    """The batched gather must reproduce the per-cell crop loop exactly."""

    def test_matches_per_cell_crop_loop(self) -> None:
        rng = np.random.default_rng(10)
        n, h, w = 7, 24, 20
        moving = rng.normal(size=(n, h, w)).astype(np.float32)
        valid = rng.random((n, h, w)) > 0.2
        peaks = np.array([
            [0.0, 0.0], [0.3, -0.2], [3.4, -2.6], [-30.0, 5.0],
            [25.0, 25.0], [1.5, 0.0], [0.0, -19.6]], dtype=np.float64)
        for mask in (None, valid):
            got, got_mask = ldc._shift_moving_stack_by_peaks(
                moving, peaks, travel_eps=0.5, valid_mask=mask)
            want, want_mask = _loop_shift(moving, peaks, 0.5, mask)
            self.assertEqual(got.dtype, want.dtype)
            np.testing.assert_array_equal(got, want)
            if mask is None:
                self.assertIsNone(got_mask)
            else:
                np.testing.assert_array_equal(got_mask, want_mask)

    def test_all_identity_returns_inputs(self) -> None:
        moving = np.zeros((2, 8, 8), dtype=np.float32)
        peaks = np.array([[0.1, 0.0], [0.0, -0.2]])
        out, out_mask = ldc._shift_moving_stack_by_peaks(moving, peaks, travel_eps=0.5)
        self.assertIs(out, moving)
        self.assertIsNone(out_mask)


class TestZnccProminenceStack(unittest.TestCase):
    """Batched prominence: true shift vs uncorrelated / masked."""

    def test_shifted_copy_high_prominence_any_cell_size(self) -> None:
        rng = np.random.default_rng(0)
        for cell in (32, 64, 128, 256):
            fixed = rng.normal(size=(cell, cell)).astype(np.float32)
            peak = np.array([3.0, -2.0], dtype=np.float64)
            # Moving is offset by +peak relative to fixed; scoring shifts by -peak.
            moving = np.roll(np.roll(fixed, -int(peak[0]), axis=0), -int(peak[1]), axis=1)
            from nornir_imageregistration.refine_shared.best_effort import zncc_decoy_radius_px
            radius = zncc_decoy_radius_px(cell, cell, base_radius=10.0)
            z_peak, _z_med, z_max, prom = ldc._zncc_prominence_stack(
                fixed[None], moving[None], peak.reshape(1, 2),
                decoy_radius=radius, travel_eps=0.5)
            self.assertGreater(float(z_peak[0]), float(z_max[0]), msg=f'cell={cell}')
            self.assertGreater(float(prom[0]), 4.0, msg=f'cell={cell} prom={prom[0]}')
            self.assertGreater(float(z_peak[0]), 0.9, msg=f'cell={cell}')

    def test_uncorrelated_prominence_near_zero(self) -> None:
        rng = np.random.default_rng(1)
        fixed = rng.normal(size=(64, 64)).astype(np.float32)
        moving = rng.normal(size=(64, 64)).astype(np.float32)
        peak = np.zeros((1, 2), dtype=np.float64)
        _z_peak, _z_med, _z_max, prom = ldc._zncc_prominence_stack(
            fixed[None], moving[None], peak, decoy_radius=5.0)
        self.assertLess(abs(float(prom[0])), 3.0)

    def test_invariant_under_gain_offset(self) -> None:
        rng = np.random.default_rng(2)
        fixed = rng.normal(size=(48, 48)).astype(np.float32)
        peak = np.array([2.0, 1.0], dtype=np.float64)
        moving = np.roll(np.roll(fixed, -int(peak[0]), axis=0), -int(peak[1]), axis=1)
        base = ldc._zncc_prominence_stack(
            fixed[None], moving[None], peak.reshape(1, 2), decoy_radius=5.0)
        remapped_fixed = (fixed * 3.5 + 12.0).astype(np.float32)
        remapped_moving = (moving * 3.5 + 12.0).astype(np.float32)
        remapped = ldc._zncc_prominence_stack(
            remapped_fixed[None], remapped_moving[None], peak.reshape(1, 2),
            decoy_radius=5.0)
        self.assertAlmostEqual(float(base[0][0]), float(remapped[0][0]), places=5)
        self.assertAlmostEqual(float(base[3][0]), float(remapped[3][0]), places=2)

    def test_invariant_under_partial_noise_mask(self) -> None:
        rng = np.random.default_rng(3)
        cell = 64
        fixed = rng.normal(size=(cell, cell)).astype(np.float32)
        peak = np.array([2.0, -1.0], dtype=np.float64)
        moving = np.roll(np.roll(fixed, -int(peak[0]), axis=0), -int(peak[1]), axis=1)
        valid = np.ones((cell, cell), dtype=bool)
        valid[: int(0.4 * cell), :] = False
        z_peak, _z_med, z_max, prom = ldc._zncc_prominence_stack(
            fixed[None], moving[None], peak.reshape(1, 2),
            valid_mask=valid[None], decoy_radius=5.0)
        self.assertGreater(float(z_peak[0]), float(z_max[0]))
        self.assertGreater(float(prom[0]), 4.0)
        self.assertGreater(float(z_peak[0]), 0.9)

    def test_round_shift_not_floor(self) -> None:
        """Fractional peaks must round; floor can over-shift by nearly 1 px."""
        rng = np.random.default_rng(4)
        fixed = rng.normal(size=(40, 40)).astype(np.float32)
        # True integer offset is 1 px. Claimed peak 1.4 → round(-1.4)=-1 aligns;
        # floor(-1.4)=-2 over-shifts.
        peak = np.array([1.4, 0.0], dtype=np.float64)
        moving = np.roll(fixed, -1, axis=0)
        shifted, _mask = ldc._shift_moving_stack_by_peaks(
            moving[None], peak.reshape(1, 2), travel_eps=0.01)
        zncc_round = float(ldc._masked_zncc_stack(fixed[None], shifted)[0])
        self.assertGreater(zncc_round, 0.95)
        self.assertEqual(int(np.round(-peak[0])), -1)
        self.assertEqual(int(np.floor(-peak[0])), -2)
        from nornir_imageregistration.refine_shared.cell_roles import masked_zncc
        import nornir_imageregistration as nir
        over = nir.CropImage(moving, 0, -2, 40, 40, cval=float(np.median(moving)))
        self.assertLess(masked_zncc(fixed, over), 0.2)


class TestComputeZnccUsesRoiCandidate(unittest.TestCase):
    """``_compute_zncc_for_candidates`` must score on the winning ROI kind."""

    def test_exact_roi_candidate_uses_exact_path(self) -> None:
        rng = np.random.default_rng(5)
        cell = 32
        fixed = rng.normal(size=(cell, cell)).astype(np.float32)
        moving_exact = np.roll(fixed, -2, axis=0)
        moving_rigid = rng.normal(size=(cell, cell)).astype(np.float32)
        rec = SimpleNamespace(
            ID=(1, 2),
            SourcePoint=np.array([10.0, 10.0]),
            TargetPoint=np.array([10.0, 10.0]),
            peak=np.array([2.0, 0.0], dtype=np.float64),
            roi_candidate='exact',
        )
        settings = SimpleNamespace(
            cell_size=(cell, cell),
            peak_ratio_exclusion_radius=5.0,
            ring_scale_fraction_max=0.05,
            ring_angle_max_degrees=5.0,
            ring_allow_flip_change=False,
            target_image=fixed,
            source_image=moving_exact,
            target_image_stats=None,
            source_image_stats=None,
        )

        def fake_batched(**_kwargs):
            nan = np.zeros((1, cell, cell), dtype=bool)
            return fixed[None], moving_rigid[None], nan

        def fake_exact(**_kwargs):
            nan = np.zeros((1, cell, cell), dtype=bool)
            return moving_exact[None], nan

        with patch.object(ldc, 'ApproximateRigidTransformBySourcePoints', return_value=[object()]), \
             patch.object(ldc, '_stos_settings_images', return_value=(fixed, moving_exact)), \
             patch.object(ldc, 'BuildAlignmentROIsBatched', side_effect=fake_batched), \
             patch.object(ldc, 'BuildExactMovingROIsBatched', side_effect=fake_exact):
            transform = object()
            scores = ldc._compute_zncc_for_candidates(
                [rec], {(1, 2)}, transform, settings, travel_eps=0.5)
        self.assertIn((1, 2), scores)
        score = scores[(1, 2)]
        self.assertIsInstance(score, ZnccScore)
        self.assertGreater(score.peak, 0.9)
        self.assertGreater(score.prominence, 4.0)


_CELL = 32
_TRUE_PEAK = np.array([2.0, -1.0], dtype=np.float64)
_RADIUS = 5.0


def _measured_grid(rng: np.random.Generator, ids: list[tuple[int, int]],
                   *, identity_ids: set[tuple[int, int]] = frozenset()) -> ldc.MeasuredCellRois:
    """A measured stack of independent textures, each moving ROI a shifted copy of its fixed ROI.

    Cells in *identity_ids* have ``moving == fixed`` (their true shift is zero).
    """
    fixed = rng.normal(size=(len(ids), _CELL, _CELL)).astype(np.float32)
    moving = np.empty_like(fixed)
    for i, key in enumerate(ids):
        if key in identity_ids:
            moving[i] = fixed[i]
        else:
            moving[i] = np.roll(np.roll(fixed[i], -int(_TRUE_PEAK[0]), axis=0), -int(_TRUE_PEAK[1]), axis=1)
    return ldc.MeasuredCellRois(
        ids=list(ids), fixed=fixed, moving=moving, valid=None, roi_kinds=['rigid'] * len(ids))


def _records(ids: list[tuple[int, int]], peaks: dict[tuple[int, int], np.ndarray] | None = None) -> list:
    peaks = peaks or {}
    return [
        SimpleNamespace(
            ID=key,
            SourcePoint=np.array([10.0 * key[0], 10.0 * key[1]]),
            TargetPoint=np.array([10.0 * key[0], 10.0 * key[1]]),
            peak=np.asarray(peaks.get(key, _TRUE_PEAK), dtype=np.float64),
            roi_candidate='rigid')
        for key in ids
    ]


def _settings() -> SimpleNamespace:
    # zncc_decoy_radius_px scales the base radius by cell/128, so 20 -> 5 at a 32 px cell.
    return SimpleNamespace(
        cell_size=(_CELL, _CELL),
        peak_ratio_exclusion_radius=_RADIUS * 128.0 / _CELL,
        ring_scale_fraction_max=0.05,
        ring_angle_max_degrees=5.0,
        ring_allow_flip_change=False,
        target_image=None,
        source_image=None,
        target_image_stats=None,
        source_image_stats=None)


def _no_rebuild(*_args, **_kwargs):
    raise AssertionError('ZNCC must score the measured ROIs, not re-extract them')


_GRID_3X3 = [(r, c) for r in range(3) for c in range(3)]
# Two cells with exactly one neighbor each and one with none: all below the neighbor minimum.
_SPARSE = [(5, 5), (5, 6), (9, 9)]


class TestZnccOnMeasuredRois(unittest.TestCase):
    """The lock gate scores the pass's own ROI stacks with a neighbor-ROI null."""

    def test_scores_without_re_extracting_rois(self) -> None:
        rng = np.random.default_rng(20)
        ids = _GRID_3X3 + _SPARSE
        measured = _measured_grid(rng, ids)
        with patch.object(ldc, 'ApproximateRigidTransformBySourcePoints', side_effect=_no_rebuild), \
             patch.object(ldc, 'BuildAlignmentROIsBatched', side_effect=_no_rebuild), \
             patch.object(ldc, 'BuildExactMovingROIsBatched', side_effect=_no_rebuild), \
             patch.object(ldc, 'BuildAlignmentROIs', side_effect=_no_rebuild):
            scores = ldc._compute_zncc_for_candidates(
                _records(ids), set(ids), object(), _settings(),
                travel_eps=0.5, measured_rois=measured)
        self.assertEqual(set(scores), set(ids))
        for key, score in scores.items():
            self.assertGreater(score.peak, 0.9, msg=str(key))
            self.assertGreater(score.prominence, 4.0, msg=str(key))
            self.assertTrue(zncc_score_passes(score, prominence_min=4.0), msg=str(key))

    def test_neighbor_null_for_connected_cells_cardinal_fallback_for_sparse(self) -> None:
        rng = np.random.default_rng(21)
        ids = _GRID_3X3 + _SPARSE
        measured = _measured_grid(rng, ids)
        neighbor_calls: list[np.ndarray] = []
        cardinal_calls: list[int] = []
        real_neighbor = ldc._zncc_neighbor_null_scores
        real_cardinal = ldc._zncc_cardinal_null_scores

        def spy_neighbor(fixed_cells, valid_cells, neighbor_rows, all_moving, all_valid):
            neighbor_calls.append(np.array(neighbor_rows))
            return real_neighbor(fixed_cells, valid_cells, neighbor_rows, all_moving, all_valid)

        def spy_cardinal(fixed_stack, moving_stack, peaks, **kwargs):
            cardinal_calls.append(int(fixed_stack.shape[0]))
            return real_cardinal(fixed_stack, moving_stack, peaks, **kwargs)

        with patch.object(ldc, '_zncc_neighbor_null_scores', side_effect=spy_neighbor), \
             patch.object(ldc, '_zncc_cardinal_null_scores', side_effect=spy_cardinal):
            scores = ldc._compute_zncc_for_candidates(
                _records(ids), set(ids), object(), _settings(),
                travel_eps=0.5, measured_rois=measured)

        self.assertEqual(len(neighbor_calls), 1)
        rows = neighbor_calls[0]
        self.assertEqual(rows.shape, (len(_GRID_3X3), 4))
        counts = np.count_nonzero(rows >= 0, axis=1)
        # Corners of the 3x3 have 2 neighbors, edges 3, the center 4.
        self.assertEqual(sorted(counts.tolist()), [2, 2, 2, 2, 3, 3, 3, 3, 4])
        self.assertEqual(cardinal_calls, [len(_SPARSE)])
        self.assertEqual(set(scores), set(ids))

    def test_neighbor_rows_map_to_grid_neighbors(self) -> None:
        row_of = {key: i for i, key in enumerate(_GRID_3X3)}
        rows = ldc._four_connected_neighbor_rows([(1, 1), (0, 0), (7, 7)], row_of)
        np.testing.assert_array_equal(rows[0], [row_of[(0, 1)], row_of[(2, 1)], row_of[(1, 0)], row_of[(1, 2)]])
        np.testing.assert_array_equal(rows[1], [-1, row_of[(1, 0)], -1, row_of[(0, 1)]])
        np.testing.assert_array_equal(rows[2], [-1, -1, -1, -1])

    def test_identity_rival_blocks_far_claim_on_unshifted_cell(self) -> None:
        """A cell whose ROIs already match at zero shift cannot lock a far-away claimed peak."""
        rng = np.random.default_rng(22)
        ids = _GRID_3X3
        measured = _measured_grid(rng, ids, identity_ids={(1, 1)})
        far_peak = np.array([_RADIUS + 3.0, 0.0])
        scores = ldc._compute_zncc_for_candidates(
            _records(ids, peaks={(1, 1): far_peak}), {(1, 1)}, object(), _settings(),
            travel_eps=0.5, measured_rois=measured)
        score = scores[(1, 1)]
        self.assertGreater(score.decoy_max, 0.9, 'identity ZNCC must be the max rival')
        self.assertLess(score.peak, score.decoy_max)
        self.assertFalse(zncc_score_passes(score, prominence_min=4.0))

    def test_identity_not_scored_for_near_identity_peak(self) -> None:
        """Within the decoy radius the zero shift is a near-duplicate of the claim, not a rival."""
        rng = np.random.default_rng(23)
        ids = _GRID_3X3
        measured = _measured_grid(rng, ids, identity_ids={(1, 1)})
        near_peak = np.array([0.6, 0.0])  # rounds to a 1 px shift of an unshifted cell
        scores = ldc._compute_zncc_for_candidates(
            _records(ids, peaks={(1, 1): near_peak}), {(1, 1)}, object(), _settings(),
            travel_eps=0.5, measured_rois=measured)
        # Neighbor textures are independent, so without the identity rival the max stays small.
        self.assertLess(scores[(1, 1)].decoy_max, 0.5)

    def test_candidates_missing_from_stack_fall_back_to_re_extract(self) -> None:
        rng = np.random.default_rng(24)
        measured = _measured_grid(rng, _GRID_3X3)
        ids = _GRID_3X3 + [(7, 7)]
        calls: list[int] = []

        def fake_rigid(**kwargs):
            calls.append(int(np.asarray(kwargs['source_points']).shape[0]))
            return [object()] * calls[-1]

        def fake_batched(**kwargs):
            n = int(np.asarray(kwargs['target_points']).shape[0])
            fixed = rng.normal(size=(n, _CELL, _CELL)).astype(np.float32)
            moving = np.roll(np.roll(fixed, -2, axis=1), 1, axis=2)
            return fixed, moving, np.zeros((n, _CELL, _CELL), dtype=bool)

        with patch.object(ldc, 'ApproximateRigidTransformBySourcePoints', side_effect=fake_rigid), \
             patch.object(ldc, '_stos_settings_images', return_value=(None, None)), \
             patch.object(ldc, 'BuildAlignmentROIsBatched', side_effect=fake_batched):
            scores = ldc._compute_zncc_for_candidates(
                _records(ids), set(ids), object(), _settings(),
                travel_eps=0.5, measured_rois=measured)
        self.assertEqual(calls, [1], 'only the unmeasured candidate is re-extracted')
        self.assertEqual(set(scores), set(ids))
        self.assertGreater(scores[(7, 7)].peak, 0.9)


class TestZnccNullCache(unittest.TestCase):
    """Null statistics are reused across passes; peak and identity are not."""

    def test_second_pass_skips_null_zncc_and_reuses_stats(self) -> None:
        rng = np.random.default_rng(30)
        ids = _GRID_3X3 + _SPARSE
        cache = ldc.ZnccNullCache()
        first = ldc._compute_zncc_for_candidates(
            _records(ids), set(ids), object(), _settings(),
            travel_eps=0.5, measured_rois=_measured_grid(rng, ids), null_cache=cache)
        self.assertEqual(len(cache), len(ids))
        self.assertEqual(cache.misses, len(ids))

        # A fresh stack: peaks are re-scored on it, the null comes from the cache.
        second_stack = _measured_grid(rng, ids)
        with patch.object(ldc, '_zncc_neighbor_null_scores', side_effect=_no_rebuild), \
             patch.object(ldc, '_zncc_cardinal_null_scores', side_effect=_no_rebuild):
            second = ldc._compute_zncc_for_candidates(
                _records(ids), set(ids), object(), _settings(),
                travel_eps=0.5, measured_rois=second_stack, null_cache=cache)
        self.assertEqual(cache.hits, len(ids))
        for key in ids:
            self.assertEqual(second[key].decoy_med, first[key].decoy_med)
            self.assertGreater(second[key].peak, 0.9)

    def test_identity_rival_is_fresh_even_on_cache_hit(self) -> None:
        rng = np.random.default_rng(31)
        ids = _GRID_3X3
        cache = ldc.ZnccNullCache()
        ldc._compute_zncc_for_candidates(
            _records(ids), set(ids), object(), _settings(),
            travel_eps=0.5, measured_rois=_measured_grid(rng, ids), null_cache=cache)
        far_peak = np.array([_RADIUS + 3.0, 0.0])
        scores = ldc._compute_zncc_for_candidates(
            _records(ids, peaks={(1, 1): far_peak}), {(1, 1)}, object(), _settings(),
            travel_eps=0.5, measured_rois=_measured_grid(rng, ids, identity_ids={(1, 1)}),
            null_cache=cache)
        self.assertGreater(scores[(1, 1)].decoy_max, 0.9)
        self.assertFalse(zncc_score_passes(scores[(1, 1)], prominence_min=4.0))

    def test_cleared_cache_recomputes(self) -> None:
        rng = np.random.default_rng(32)
        ids = _GRID_3X3
        cache = ldc.ZnccNullCache()
        ldc._compute_zncc_for_candidates(
            _records(ids), set(ids), object(), _settings(),
            travel_eps=0.5, measured_rois=_measured_grid(rng, ids), null_cache=cache)
        cache.clear()
        self.assertEqual(len(cache), 0)
        ldc._compute_zncc_for_candidates(
            _records(ids), set(ids), object(), _settings(),
            travel_eps=0.5, measured_rois=_measured_grid(rng, ids), null_cache=cache)
        self.assertEqual(len(cache), len(ids))
        self.assertEqual(cache.misses, 2 * len(ids))

    def _cell_size_settings(self, cell: int) -> SimpleNamespace:
        return SimpleNamespace(
            cell_size=np.array([cell, cell], dtype=np.int64),
            num_iterations=5,
            source_image=np.zeros((2048, 2048), dtype=np.float32),
            target_image=np.zeros((2048, 2048), dtype=np.float32))

    def test_cell_size_grow_clears_cache(self) -> None:
        from nornir_imageregistration.refine_shared import SourceContentCache
        cache = ldc.ZnccNullCache()
        cache.remember((0, 0), 0.0, 0.1, 0.02)
        grew = ldc._grow_refine_cell_size_after_failure(
            self._cell_size_settings(128), SourceContentCache(),
            pass_index=1, final_pass=False, zncc_null_cache=cache)
        self.assertTrue(grew)
        self.assertEqual(len(cache), 0)

    def test_cell_size_restore_clears_cache(self) -> None:
        from nornir_imageregistration.refine_shared import SourceContentCache
        cache = ldc.ZnccNullCache()
        cache.remember((0, 0), 0.0, 0.1, 0.02)
        restored = ldc._restore_refine_cell_size_after_success(
            self._cell_size_settings(256), SourceContentCache(), (128, 128),
            pass_index=1, final_pass=False, zncc_null_cache=cache)
        self.assertTrue(restored)
        self.assertEqual(len(cache), 0)


if __name__ == '__main__':
    unittest.main()
