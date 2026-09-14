"""Dual-candidate STOS refine cells: rigid-fit ROI plus exact-transform ROI.

A single similarity fitted to a ring of points cannot follow a mesh that is locally
sheared or anisotropically scaled, so refine also measures cells whose rigid result is
soft from a ROI warped through the real transform and keeps the better candidate.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import local_distortion_correction as ldc
from nornir_imageregistration.refine_shared.peak_ratio_gates import PEAK_RATIO_EARLY
from nornir_imageregistration.transforms.meshwithrbffallback import MeshWithRBFFallback


def _sheared_mesh(size: int) -> MeshWithRBFFallback:
    """A control-point mesh whose x-scale varies across the image, so no one similarity fits it."""
    ys, xs = np.meshgrid(np.linspace(0, size - 1, 5), np.linspace(0, size - 1, 5), indexing='ij')
    source = np.stack((ys.ravel(), xs.ravel()), axis=1)
    target = source.copy()
    target[:, 1] = source[:, 1] * (0.9 + 0.2 * source[:, 0] / (size - 1))
    return MeshWithRBFFallback(np.hstack((target, source)))


class TestChooseTranslationCandidates(unittest.TestCase):

    def _pick(self, rigid, exact):
        rigid_t = (np.array([[0.0, 1.0]]), np.array([rigid[0]]), np.array([rigid[1]]))
        exact_t = (np.array([[2.0, 3.0]]), np.array([exact[0]]), np.array([exact[1]]))
        peaks, weights, ratios, used_exact = ldc._choose_translation_candidates(rigid_t, exact_t)
        return bool(used_exact[0]), peaks[0], weights[0], ratios[0]

    def test_higher_ratio_wins(self) -> None:
        used_exact, peak, _, ratio = self._pick(rigid=(2.0, 1.1), exact=(1.0, 1.6))
        self.assertTrue(used_exact)
        np.testing.assert_array_equal(peak, [2.0, 3.0])
        self.assertEqual(ratio, 1.6)

        used_exact, peak, _, _ = self._pick(rigid=(1.0, 1.6), exact=(2.0, 1.1))
        self.assertFalse(used_exact)
        np.testing.assert_array_equal(peak, [0.0, 1.0])

    def test_tie_keeps_rigid_unless_exact_weight_is_higher(self) -> None:
        self.assertFalse(self._pick(rigid=(1.0, 1.4), exact=(1.0, 1.4))[0])
        self.assertTrue(self._pick(rigid=(1.0, 1.4), exact=(1.5, 1.4))[0])

    def test_unusable_candidate_loses(self) -> None:
        self.assertFalse(self._pick(rigid=(1.0, 1.1), exact=(0.0, 9.0))[0])
        self.assertTrue(self._pick(rigid=(0.0, 9.0), exact=(1.0, 1.1))[0])


class TestExactMovingROIs(unittest.TestCase):

    def test_batched_exact_rois_match_per_cell_warp(self) -> None:
        size = 256
        cell = np.array((48, 48))
        rng = np.random.default_rng(3)
        source_image = rng.random((size, size)).astype(np.float32)
        transform = _sheared_mesh(size)
        target_points = np.array([[80.0, 90.0], [128.0, 128.0], [170.0, 150.0]])

        stack, nan_mask = ldc.BuildExactMovingROIsBatched(
            transform=transform,
            source_image=source_image,
            source_image_stats=None,
            target_points=target_points,
            alignment_area=cell,
            xp=np)
        self.assertIsNone(nan_mask)
        self.assertEqual(stack.shape, (3, 48, 48))

        for i, point in enumerate(target_points):
            _, expected = ldc.BuildAlignmentROIs(
                transform=transform,
                targetImage_param=source_image,
                sourceImage_param=source_image,
                target_image_stats=None,
                source_image_stats=None,
                target_controlpoint=point,
                alignmentArea=cell)[:2]
            expected = nornir_imageregistration.EnsureNumpyArray(expected)
            self.assertEqual(expected.shape, stack[i].shape)
            corr = np.corrcoef(expected.ravel(), stack[i].ravel())[0, 1]
            self.assertGreater(corr, 0.999, f'cell {i}: exact batched ROI diverges from per-cell warp (corr={corr:.4f})')

    def test_exact_rois_differ_from_rigid_fit_on_sheared_mesh(self) -> None:
        """The reason the second candidate exists: on a sheared mesh the rigid ROI is not the exact ROI."""
        size = 256
        cell = np.array((64, 64))
        rng = np.random.default_rng(5)
        source_image = rng.random((size, size)).astype(np.float32)
        transform = _sheared_mesh(size)
        target_points = np.array([[128.0, 128.0]])
        source_points = nornir_imageregistration.EnsureNumpyArray(transform.InverseTransform(target_points))
        rigid = ldc.ApproximateRigidTransformBySourcePoints(
            input_transform=transform, source_points=source_points, cell_size=cell)

        rigid_stack = ldc.BuildAlignmentROIsBatched(
            rigid_transforms=rigid,
            target_image=source_image,
            source_image=source_image,
            target_image_stats=None,
            source_image_stats=None,
            target_points=target_points,
            alignment_area=cell)
        self.assertIsNotNone(rigid_stack)
        exact_stack, _ = ldc.BuildExactMovingROIsBatched(
            transform=transform,
            source_image=source_image,
            source_image_stats=None,
            target_points=target_points,
            alignment_area=cell,
            xp=np)
        corr = np.corrcoef(rigid_stack[1][0].ravel(), exact_stack[0].ravel())[0, 1]
        self.assertLess(corr, 0.9, f'rigid and exact ROIs should differ on a sheared mesh (corr={corr:.4f})')


class TestSelectiveExactCandidate(unittest.TestCase):
    """The exact ROI is warped and measured only for cells whose rigid result is soft."""

    def test_soft_rigid_cells(self) -> None:
        weights = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        peaks = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [np.nan, np.nan], [2.0, 2.0], [0.0, 1.0]])
        ratios = np.array([2.0, 1.2, 3.0, 2.0, PEAK_RATIO_EARLY, np.nan])
        soft = ldc._soft_rigid_cells(weights, peaks, ratios)
        np.testing.assert_array_equal(soft, [False, True, True, True, False, True])

    def test_fft_working_dtype_floors_at_float32(self) -> None:
        self.assertEqual(ldc._fft_working_dtype(np.zeros(1, np.float32), np.zeros(1, np.float32)), np.float32)
        self.assertEqual(ldc._fft_working_dtype(np.zeros(1, np.float64)), np.float64)
        self.assertEqual(ldc._fft_working_dtype(np.zeros(1, np.float16)), np.float32)
        self.assertEqual(ldc._fft_working_dtype(np.zeros(1, np.uint8)), np.float32)

    def test_exact_rois_built_only_for_soft_cells_and_sink_keeps_winners(self) -> None:
        cell = 32
        n = 5
        rng = np.random.default_rng(11)
        fixed = rng.normal(size=(n, cell, cell)).astype(np.float32)
        shift = (2, -1)
        shifted = np.roll(np.roll(fixed, -shift[0], axis=1), -shift[1], axis=2)
        # Rows 0, 1, 4 register strongly from the rigid ROI; rows 2, 3 are uncorrelated
        # noise there and only the exact ROI carries the shifted copy.
        rigid_moving = shifted.copy()
        rigid_moving[2] = rng.normal(size=(cell, cell)).astype(np.float32)
        rigid_moving[3] = rng.normal(size=(cell, cell)).astype(np.float32)
        exact_all = shifted.copy()
        target_points = np.array([[100.0 + 40.0 * i, 100.0] for i in range(n)])
        keys = [(i, 0) for i in range(n)]
        exact_requests: list[np.ndarray] = []

        def fake_rigid_rois(**_kwargs):
            return fixed.copy(), rigid_moving.copy(), None

        def fake_exact_rois(**kwargs):
            pts = np.asarray(kwargs['target_points'])
            exact_requests.append(pts)
            rows = [int(np.flatnonzero(np.all(target_points == p, axis=1))[0]) for p in pts]
            return exact_all[rows].copy(), None

        settings = SimpleNamespace(
            cell_size=(cell, cell),
            target_image=fixed[0], source_image=fixed[0],
            target_image_stats=None, source_image_stats=None,
            min_alignment_overlap=0.5,
            peak_ratio_exclusion_radius=3)
        sink = ldc.MeasuredRoiSink()
        with patch.object(ldc, '_stos_settings_images', return_value=(fixed[0], fixed[0])), \
             patch.object(ldc, 'BuildAlignmentROIsBatched', side_effect=fake_rigid_rois), \
             patch.object(ldc, 'BuildExactMovingROIsBatched', side_effect=fake_exact_rois):
            records = ldc._attempt_align_points_translation_batched(
                keys=keys,
                source_points=target_points.copy(),
                target_points=target_points,
                rigid_transforms=[object()] * n,
                settings=settings,
                exact_transform=_sheared_mesh(256),
                roi_sink=sink)

        self.assertIsNotNone(records)
        by_id = {rec.ID: rec for rec in records}
        self.assertEqual(set(by_id), set(keys))
        self.assertEqual(len(exact_requests), 1, 'one batched exact warp for the soft subset')
        soft_rows = sorted(int(np.flatnonzero(np.all(target_points == p, axis=1))[0]) for p in exact_requests[0])
        self.assertEqual(soft_rows, [2, 3], 'confident rigid cells must not pay for the exact ROI')
        for i in (0, 1, 4):
            self.assertEqual(by_id[(i, 0)].roi_candidate, 'rigid')
        for i in (2, 3):
            self.assertEqual(by_id[(i, 0)].roi_candidate, 'exact')
            np.testing.assert_allclose(by_id[(i, 0)].peak, shift, atol=0.5)

        measured = sink.current
        self.assertIsNotNone(measured)
        self.assertEqual(measured.ids, keys)
        self.assertEqual(measured.roi_kinds, ['rigid', 'rigid', 'exact', 'exact', 'rigid'])
        np.testing.assert_array_equal(measured.fixed, fixed)
        for i in (0, 1, 4):
            np.testing.assert_array_equal(measured.moving[i], rigid_moving[i])
        for i in (2, 3):
            np.testing.assert_array_equal(measured.moving[i], exact_all[i])
        self.assertEqual(measured.fixed.dtype, np.float32, 'no float64 upcast before the batched FFT')

    def test_no_exact_warp_when_every_rigid_cell_is_confident(self) -> None:
        cell = 32
        n = 4
        rng = np.random.default_rng(12)
        fixed = rng.normal(size=(n, cell, cell)).astype(np.float32)
        moving = np.roll(fixed, -2, axis=1)
        target_points = np.array([[100.0 + 40.0 * i, 100.0] for i in range(n)])
        settings = SimpleNamespace(
            cell_size=(cell, cell),
            target_image=fixed[0], source_image=fixed[0],
            target_image_stats=None, source_image_stats=None,
            min_alignment_overlap=0.5,
            peak_ratio_exclusion_radius=3)

        def never(**_kwargs):
            raise AssertionError('exact ROI must not be built when rigid is confident everywhere')

        with patch.object(ldc, '_stos_settings_images', return_value=(fixed[0], fixed[0])), \
             patch.object(ldc, 'BuildAlignmentROIsBatched', return_value=(fixed, moving, None)), \
             patch.object(ldc, 'BuildExactMovingROIsBatched', side_effect=never):
            records = ldc._attempt_align_points_translation_batched(
                keys=[(i, 0) for i in range(n)],
                source_points=target_points.copy(),
                target_points=target_points,
                rigid_transforms=[object()] * n,
                settings=settings,
                exact_transform=_sheared_mesh(256))
        self.assertEqual(len(records), n)
        self.assertTrue(all(rec.roi_candidate == 'rigid' for rec in records))


if __name__ == '__main__':
    unittest.main()
