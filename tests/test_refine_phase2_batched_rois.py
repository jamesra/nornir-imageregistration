"""Parity tests for Phase 2 batched rigid ROI extraction.

``BuildAlignmentROIsBatched`` must agree with per-cell ``BuildAlignmentROIs``
(defer_oob_check=True) on geometry, OOB masks, and in-bounds pixel values for the
translation-only STOS grid-refine path.
"""

from __future__ import annotations

import unittest

import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import nornir_imageregistration


class TestBuildAlignmentROIsBatchedParity(unittest.TestCase):
    """Batched extract must match the legacy per-cell path."""

    _IMAGE_SHAPE = (64, 64)
    _CELL_SIZE = np.array((8, 8))

    def _make_images_and_stats(self):
        image = np.zeros(self._IMAGE_SHAPE, dtype=nornir_imageregistration.default_image_dtype())
        for y in range(self._IMAGE_SHAPE[0]):
            for x in range(self._IMAGE_SHAPE[1]):
                image[y, x] = (((x % 8) + (y % 8)) / 8.0) * 0.6 + 0.2
        stats = nornir_imageregistration.ImageStats.Create(image)
        return image, stats

    def setUp(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.target_image, self.target_stats = self._make_images_and_stats()
        self.source_image, self.source_stats = self._make_images_and_stats()

    def _per_cell(self, transforms, target_points):
        fixed = []
        moving = []
        masks = []
        for i, transform in enumerate(transforms):
            target_roi, source_roi, nan_mask = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIs(
                transform=transform,
                targetImage_param=self.target_image,
                sourceImage_param=self.source_image,
                target_image_stats=self.target_stats,
                source_image_stats=self.source_stats,
                target_controlpoint=target_points[i],
                alignmentArea=self._CELL_SIZE,
                defer_oob_check=True)
            fixed.append(np.asarray(target_roi, dtype=np.float64))
            moving.append(np.asarray(source_roi, dtype=np.float64))
            masks.append(np.asarray(nan_mask))
        return np.stack(fixed), np.stack(moving), np.stack(masks)

    def test_identity_translation_parity(self):
        identity = nornir_imageregistration.transforms.RigidTranslation(
            target_offset=np.array((0.0, 0.0)))
        target_points = np.array([
            (16.0, 16.0),
            (32.0, 40.0),
            (48.0, 24.0),
            (20.0, 50.0),
        ])
        transforms = [identity] * len(target_points)

        batched = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIsBatched(
            rigid_transforms=transforms,
            target_image=self.target_image,
            source_image=self.source_image,
            target_image_stats=self.target_stats,
            source_image_stats=self.source_stats,
            target_points=target_points,
            alignment_area=self._CELL_SIZE)
        self.assertIsNotNone(batched)
        fixed_b, moving_b, mask_b = batched
        fixed_p, moving_p, mask_p = self._per_cell(transforms, target_points)

        np.testing.assert_array_equal(np.asarray(mask_b), mask_p)
        # Target crops are integer copies — exact match.
        np.testing.assert_allclose(np.asarray(fixed_b), fixed_p, rtol=0, atol=0)
        # Fully in-bounds identity cells: noise fill never runs; cubic sample should match.
        in_bounds = ~mask_p.any(axis=(1, 2))
        self.assertTrue(bool(np.any(in_bounds)))
        np.testing.assert_allclose(
            np.asarray(moving_b)[in_bounds],
            moving_p[in_bounds],
            rtol=1e-5,
            atol=1e-4)

    def test_rotated_rigid_single_cell_exact_parity(self):
        """One cell → AABB crop matches per-cell SourceImageToTargetSpace exactly."""
        rigid = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=np.array((2.0, -1.5)),
            source_rotation_center=np.array((32.0, 32.0)),
            angle=0.15,
            scalar=1.0)
        target_points = np.array([(24.0, 24.0)])
        transforms = [rigid]

        batched = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIsBatched(
            rigid_transforms=transforms,
            target_image=self.target_image,
            source_image=self.source_image,
            target_image_stats=self.target_stats,
            source_image_stats=self.source_stats,
            target_points=target_points,
            alignment_area=self._CELL_SIZE)
        self.assertIsNotNone(batched)
        fixed_b, moving_b, mask_b = batched
        fixed_p, moving_p, mask_p = self._per_cell(transforms, target_points)

        np.testing.assert_array_equal(np.asarray(mask_b), mask_p)
        np.testing.assert_allclose(np.asarray(fixed_b), fixed_p, rtol=0, atol=0)
        valid = ~mask_p
        np.testing.assert_allclose(
            np.asarray(moving_b)[valid],
            moving_p[valid],
            rtol=1e-5,
            atol=1e-4)

    def test_rotated_rigid_multi_cell_mask_and_close_pixels(self):
        """Multi-cell union crop can change cubic prefilter slightly vs per-cell crops."""
        rigid = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
            target_offset=np.array((2.0, -1.5)),
            source_rotation_center=np.array((32.0, 32.0)),
            angle=0.15,
            scalar=1.0)
        target_points = np.array([
            (24.0, 24.0),
            (40.0, 40.0),
            (28.0, 44.0),
        ])
        transforms = [rigid] * len(target_points)

        batched = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIsBatched(
            rigid_transforms=transforms,
            target_image=self.target_image,
            source_image=self.source_image,
            target_image_stats=self.target_stats,
            source_image_stats=self.source_stats,
            target_points=target_points,
            alignment_area=self._CELL_SIZE)
        self.assertIsNotNone(batched)
        fixed_b, moving_b, mask_b = batched
        fixed_p, moving_p, mask_p = self._per_cell(transforms, target_points)

        np.testing.assert_array_equal(np.asarray(mask_b), mask_p)
        np.testing.assert_allclose(np.asarray(fixed_b), fixed_p, rtol=0, atol=0)
        valid = ~mask_p
        # Union-AABB prefilter vs per-cell crops: allow small intensity drift; OOB mask
        # and geometry must still match exactly (asserted above).
        abs_err = np.abs(np.asarray(moving_b)[valid] - moving_p[valid])
        self.assertLess(float(np.mean(abs_err)), 0.02)
        self.assertLess(float(np.max(abs_err)), 0.08)

    def test_fully_oob_mask_agrees(self):
        identity = nornir_imageregistration.transforms.RigidTranslation(
            target_offset=np.array((0.0, 0.0)))
        target_points = np.array([
            (16.0, 16.0),
            (500.0, 500.0),
            (32.0, 32.0),
        ])
        transforms = [identity] * len(target_points)

        batched = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIsBatched(
            rigid_transforms=transforms,
            target_image=self.target_image,
            source_image=self.source_image,
            target_image_stats=self.target_stats,
            source_image_stats=self.source_stats,
            target_points=target_points,
            alignment_area=self._CELL_SIZE)
        self.assertIsNotNone(batched)
        _fixed_b, _moving_b, mask_b = batched
        _fixed_p, _moving_p, mask_p = self._per_cell(transforms, target_points)

        np.testing.assert_array_equal(np.asarray(mask_b), mask_p)
        self.assertTrue(bool(mask_p[1].all()))
        self.assertFalse(bool(mask_p[0].any()))

    def test_attempt_align_still_drops_fully_oob(self):
        image = self.target_image
        settings = nornir_imageregistration.settings.GridRefinement.CreateWithUnproccessedImages(
            target_image=image.copy(),
            source_image=image.copy(),
            cell_size=self._CELL_SIZE,
            single_thread_processing=True)
        identity = nornir_imageregistration.transforms.RigidTranslation(
            target_offset=np.array((0.0, 0.0)))
        target_points = np.array([
            (16.0, 16.0),
            (32.0, 32.0),
            (48.0, 20.0),
            (20.0, 48.0),
            (500.0, 500.0),
        ])
        records = nornir_imageregistration.local_distortion_correction._attempt_align_points_translation_batched(
            keys=[(i, 0) for i in range(len(target_points))],
            source_points=target_points.copy(),
            target_points=target_points,
            rigid_transforms=[identity] * len(target_points),
            settings=settings)
        self.assertIsNotNone(records)
        aligned_keys = {record.ID for record in records}
        self.assertNotIn((4, 0), aligned_keys)
        self.assertGreaterEqual(len(aligned_keys), 3)

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "requires CuPy")
    def test_batched_records_copy_cupy_lattice_to_host_once(self) -> None:
        import cupy as cupy_mod

        previous = nornir_imageregistration.GetActiveComputationLib()
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.cupy)
        try:
            image = self.target_image
            settings = nornir_imageregistration.settings.GridRefinement.CreateWithUnproccessedImages(
                target_image=image.copy(),
                source_image=image.copy(),
                cell_size=self._CELL_SIZE,
                single_thread_processing=True)
            identity = nornir_imageregistration.transforms.RigidTranslation(
                target_offset=np.array((0.0, 0.0)))
            target_points = np.array([
                (16.0, 16.0),
                (32.0, 32.0),
                (48.0, 20.0),
            ])
            records = nornir_imageregistration.local_distortion_correction._attempt_align_points_translation_batched(
                keys=[(i, 0) for i in range(len(target_points))],
                source_points=cupy_mod.asarray(target_points),
                target_points=cupy_mod.asarray(target_points),
                rigid_transforms=[identity] * len(target_points),
                settings=settings)
        finally:
            nornir_imageregistration.SetActiveComputationLib(previous)
        self.assertIsNotNone(records)
        for record in records:
            self.assertIs(
                nornir_imageregistration.cp.get_array_module(record.TargetPoint), np)
            self.assertIs(
                nornir_imageregistration.cp.get_array_module(record.SourcePoint), np)

    def test_alignment_roi_botlefts_numpy_lattice(self) -> None:
        from nornir_imageregistration.local_distortion_correction import _alignment_roi_botlefts

        points = np.array([[16.0, 16.0], [32.0, 40.0]], dtype=np.float64)
        botlefts = _alignment_roi_botlefts(points, np.array([8.0, 8.0]))
        self.assertIs(nornir_imageregistration.cp.get_array_module(botlefts), np)
        np.testing.assert_array_equal(botlefts, np.array([[12.0, 12.0], [28.0, 36.0]]))

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "requires CuPy")
    def test_alignment_roi_botlefts_and_batched_keep_cupy_target_points(self) -> None:
        import cupy as cupy_mod
        from nornir_imageregistration.local_distortion_correction import _alignment_roi_botlefts

        points_np = np.array([[16.0, 16.0], [32.0, 40.0]], dtype=np.float64)
        area = np.array([8.0, 8.0])
        host = nornir_imageregistration.EnsureNumpyArray(_alignment_roi_botlefts(points_np, area))
        gpu_pts = cupy_mod.asarray(points_np)
        gpu = _alignment_roi_botlefts(gpu_pts, area)
        self.assertTrue(isinstance(gpu, cupy_mod.ndarray))
        np.testing.assert_array_equal(host, nornir_imageregistration.EnsureNumpyArray(gpu))

        identity = nornir_imageregistration.transforms.RigidTranslation(
            target_offset=np.array((0.0, 0.0)))
        batched = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIsBatched(
            rigid_transforms=[identity, identity],
            target_image=self.target_image,
            source_image=self.source_image,
            target_image_stats=self.target_stats,
            source_image_stats=self.source_stats,
            target_points=gpu_pts,
            alignment_area=self._CELL_SIZE)
        self.assertIsNotNone(batched)
        fixed_b, _moving_b, _mask_b = batched
        self.assertEqual(int(fixed_b.shape[0]), 2)

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "requires CuPy")
    def test_populate_then_batched_roi_on_cupy_target_points(self) -> None:
        import cupy as cupy_mod
        from nornir_imageregistration.grid_subdivision import ITKGridDivision
        from nornir_imageregistration.transforms.meshwithrbffallback import (
            MeshWithRBFFallback_GPUComponent,
        )

        previous = nornir_imageregistration.GetActiveComputationLib()
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.cupy)
        try:
            pairs = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 64.0, 0.0, 64.0],
                    [64.0, 0.0, 64.0, 0.0],
                    [64.0, 64.0, 64.0, 64.0],
                ],
                dtype=np.float64,
            )
            mesh = MeshWithRBFFallback_GPUComponent(pairs)
            mesh.InitializeDataStructures()
            grid = ITKGridDivision(
                source_shape=(64, 64),
                cell_size=(16, 16),
                grid_spacing=(16, 16),
            )
            mapped = grid.PopulateTargetPoints(mesh)
            self.assertTrue(isinstance(mapped, cupy_mod.ndarray))
            identity = nornir_imageregistration.transforms.RigidTranslation(
                target_offset=np.array((0.0, 0.0)))
            n_pts = int(grid.TargetPoints.shape[0])
            batched = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIsBatched(
                rigid_transforms=[identity] * n_pts,
                target_image=self.target_image,
                source_image=self.source_image,
                target_image_stats=self.target_stats,
                source_image_stats=self.source_stats,
                target_points=grid.TargetPoints,
                alignment_area=self._CELL_SIZE)
            self.assertIsNotNone(batched)
        finally:
            nornir_imageregistration.SetActiveComputationLib(previous)

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "requires CuPy")
    def test_batched_roi_numpy_images_stay_numpy_when_process_is_cupy(self) -> None:
        """Host STOS images must not be upgraded just because UsingCupy() is true."""
        import cupy as cupy_mod

        previous = nornir_imageregistration.GetActiveComputationLib()
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.cupy)
        try:
            identity = nornir_imageregistration.transforms.RigidTranslation(
                target_offset=np.array((0.0, 0.0)))
            target_points = np.array([(16.0, 16.0), (32.0, 40.0)], dtype=np.float64)
            batched = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIsBatched(
                rigid_transforms=[identity, identity],
                target_image=self.target_image,
                source_image=self.source_image,
                target_image_stats=self.target_stats,
                source_image_stats=self.source_stats,
                target_points=target_points,
                alignment_area=self._CELL_SIZE)
            self.assertIsNotNone(batched)
            fixed_b, moving_b, mask_b = batched
            self.assertFalse(isinstance(fixed_b, cupy_mod.ndarray))
            self.assertFalse(isinstance(moving_b, cupy_mod.ndarray))
            self.assertIsInstance(fixed_b, np.ndarray)
            self.assertIsInstance(moving_b, np.ndarray)
            self.assertIsNotNone(mask_b)
            assert mask_b is not None
            self.assertFalse(isinstance(mask_b, cupy_mod.ndarray))
        finally:
            nornir_imageregistration.SetActiveComputationLib(previous)

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "requires CuPy")
    def test_batched_roi_cupy_images_stay_cupy_when_process_is_cupy(self) -> None:
        """Device STOS images stay on CuPy through batched ROI extract."""
        import cupy as cupy_mod

        previous = nornir_imageregistration.GetActiveComputationLib()
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.cupy)
        try:
            identity = nornir_imageregistration.transforms.RigidTranslation(
                target_offset=np.array((0.0, 0.0)))
            target_points = np.array([(16.0, 16.0), (32.0, 40.0)], dtype=np.float64)
            batched = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIsBatched(
                rigid_transforms=[identity, identity],
                target_image=cupy_mod.asarray(self.target_image),
                source_image=cupy_mod.asarray(self.source_image),
                target_image_stats=self.target_stats,
                source_image_stats=self.source_stats,
                target_points=target_points,
                alignment_area=self._CELL_SIZE)
            self.assertIsNotNone(batched)
            fixed_b, moving_b, mask_b = batched
            self.assertIsInstance(fixed_b, cupy_mod.ndarray)
            self.assertIsInstance(moving_b, cupy_mod.ndarray)
            self.assertIsNotNone(mask_b)
            assert mask_b is not None
            self.assertIsInstance(mask_b, cupy_mod.ndarray)
        finally:
            nornir_imageregistration.SetActiveComputationLib(previous)

    @given(
        points=arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(1, 6), st.just(2)),
            elements=st.floats(min_value=0.0, max_value=128.0, allow_nan=False, allow_infinity=False),
            unique=False,
        ),
        area_y=st.floats(min_value=2.0, max_value=32.0, allow_nan=False, allow_infinity=False),
        area_x=st.floats(min_value=2.0, max_value=32.0, allow_nan=False, allow_infinity=False),
    )
    @example(
        points=np.array([[16.0, 16.0], [32.0, 40.0]], dtype=np.float64),
        area_y=8.0,
        area_x=8.0,
    )
    @settings(max_examples=20, deadline=None)
    def test_alignment_roi_botlefts_numpy_matches_cupy_property(
            self, points, area_y, area_x) -> None:
        from nornir_imageregistration.local_distortion_correction import _alignment_roi_botlefts

        area = np.array([area_y, area_x], dtype=np.float64)
        host = nornir_imageregistration.EnsureNumpyArray(
            _alignment_roi_botlefts(points, area))
        self.assertEqual(host.shape, (points.shape[0], 2))
        if not nornir_imageregistration.HasCupy():
            return
        import cupy as cupy_mod
        gpu = _alignment_roi_botlefts(cupy_mod.asarray(points), area)
        self.assertTrue(isinstance(gpu, cupy_mod.ndarray))
        np.testing.assert_allclose(
            host, nornir_imageregistration.EnsureNumpyArray(gpu), rtol=0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
