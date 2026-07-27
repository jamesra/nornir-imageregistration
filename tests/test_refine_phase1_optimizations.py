"""Unit tests for the Phase 1 STOS grid-refine per-cell sync removal.

Covers the two low-level primitives touched by the fix:

- ``assemble.write_to_target_roi_coords``'s ``IRigidTransform`` fast path, which must
  produce results identical to the general (``InvalidIndices``-based) path when the
  transform cannot produce NaNs.
- ``local_distortion_correction.BuildAlignmentROIs``'s ``defer_oob_check`` parameter,
  which must agree with the eager (default) behavior about which cells are
  entirely out of bounds, just without the immediate ``ValueError``/sync.

And an end-to-end check of ``_attempt_align_points_translation_batched``, which now
relies on both of the above to batch the accept/reject decision once per pass.
"""

from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.transforms.base import ITransform


class _NonRigidWrapper(ITransform):
    """Delegates to a real rigid transform but is intentionally *not* an IRigidTransform.

    Used to force write_to_target_roi_coords onto its general (InvalidIndices) path so
    it can be compared against the IRigidTransform fast path for parity.
    """

    def __init__(self, inner: nornir_imageregistration.ITransform):
        self._inner = inner

    def Transform(self, points, **kwargs):
        return self._inner.Transform(points, **kwargs)

    def InverseTransform(self, points, **kwargs):
        return self._inner.InverseTransform(points, **kwargs)

    def Load(self, *args, **kwargs):
        return self._inner.Load(*args, **kwargs)

    def ToITKString(self, *args, **kwargs):
        return self._inner.ToITKString(*args, **kwargs)

    @property
    def type(self):
        return self._inner.type


class TestWriteToTargetROICoordsFastPath(unittest.TestCase):
    """IRigidTransform fast path in write_to_target_roi_coords must match the general path."""

    def test_fast_path_matches_general_path_for_rigid_transform(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

        rigid_transform = nornir_imageregistration.transforms.Rigid(
            target_offset=np.array((3.5, -2.25)),
            source_rotation_center=np.array((10.0, 10.0)),
            angle=0.3)
        wrapped_transform = _NonRigidWrapper(rigid_transform)

        self.assertTrue(isinstance(rigid_transform, nornir_imageregistration.transforms.IRigidTransform))
        self.assertFalse(isinstance(wrapped_transform, nornir_imageregistration.transforms.IRigidTransform))

        botleft = (5.0, 5.0)
        area = (16, 16)

        fast_read, fast_write = nornir_imageregistration.assemble.write_to_target_roi_coords(
            rigid_transform, botleft=botleft, area=area, extrapolate=True)
        general_read, general_write = nornir_imageregistration.assemble.write_to_target_roi_coords(
            wrapped_transform, botleft=botleft, area=area, extrapolate=True)

        np.testing.assert_array_equal(fast_read, general_read)
        np.testing.assert_array_equal(fast_write, general_write)
        # A rigid transform on finite input never produces NaN, so no rows should have
        # been dropped by either path.
        expected_num_coords = int(area[0]) * int(area[1])
        self.assertEqual(fast_read.shape[0], expected_num_coords)
        self.assertEqual(general_read.shape[0], expected_num_coords)


class TestBuildAlignmentROIsDeferOOBCheck(unittest.TestCase):
    """defer_oob_check=True must agree with the eager default about OOB cells."""

    _IMAGE_SHAPE = (40, 40)
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
        self.identity_transform = nornir_imageregistration.transforms.RigidTranslation(
            target_offset=np.array((0.0, 0.0)))

    def test_fully_in_bounds_cell_agrees_and_matches(self):
        target_controlpoint = np.array((20.0, 20.0))

        target_roi_eager, source_roi_eager = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIs(
            transform=self.identity_transform,
            targetImage_param=self.target_image,
            sourceImage_param=self.source_image,
            target_image_stats=self.target_stats,
            source_image_stats=self.source_stats,
            target_controlpoint=target_controlpoint,
            alignmentArea=self._CELL_SIZE)

        target_roi_deferred, source_roi_deferred, nan_mask = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIs(
            transform=self.identity_transform,
            targetImage_param=self.target_image,
            sourceImage_param=self.source_image,
            target_image_stats=self.target_stats,
            source_image_stats=self.source_stats,
            target_controlpoint=target_controlpoint,
            alignmentArea=self._CELL_SIZE,
            defer_oob_check=True)

        self.assertIsNotNone(nan_mask)
        self.assertFalse(bool(np.asarray(nan_mask).any()), "Fully in-bounds cell should have no NaNs")
        np.testing.assert_array_equal(target_roi_eager, target_roi_deferred)
        # No pixels are NaN, so RandomNoiseMask has nothing to fill and both calls are
        # deterministic and must match exactly.
        np.testing.assert_array_equal(source_roi_eager, source_roi_deferred)

    def test_fully_out_of_bounds_cell_agrees(self):
        target_controlpoint = np.array((500.0, 500.0))

        with self.assertRaises(ValueError):
            nornir_imageregistration.local_distortion_correction.BuildAlignmentROIs(
                transform=self.identity_transform,
                targetImage_param=self.target_image,
                sourceImage_param=self.source_image,
                target_image_stats=self.target_stats,
                source_image_stats=self.source_stats,
                target_controlpoint=target_controlpoint,
                alignmentArea=self._CELL_SIZE)

        target_roi_deferred, source_roi_deferred, nan_mask = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIs(
            transform=self.identity_transform,
            targetImage_param=self.target_image,
            sourceImage_param=self.source_image,
            target_image_stats=self.target_stats,
            source_image_stats=self.source_stats,
            target_controlpoint=target_controlpoint,
            alignmentArea=self._CELL_SIZE,
            defer_oob_check=True)

        self.assertIsNotNone(nan_mask)
        self.assertTrue(bool(np.asarray(nan_mask).all()),
                        "Cell entirely outside the source image must be reported as entirely NaN")
        self.assertEqual(tuple(source_roi_deferred.shape), tuple(int(v) for v in self._CELL_SIZE))

    def test_partially_out_of_bounds_cell_is_not_fully_oob(self):
        target_controlpoint = np.array((2.0, 2.0))

        # Should not raise: this cell straddles the source image edge, so it is not
        # entirely out of bounds.
        target_roi_eager, source_roi_eager = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIs(
            transform=self.identity_transform,
            targetImage_param=self.target_image,
            sourceImage_param=self.source_image,
            target_image_stats=self.target_stats,
            source_image_stats=self.source_stats,
            target_controlpoint=target_controlpoint,
            alignmentArea=self._CELL_SIZE)

        target_roi_deferred, source_roi_deferred, nan_mask = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIs(
            transform=self.identity_transform,
            targetImage_param=self.target_image,
            sourceImage_param=self.source_image,
            target_image_stats=self.target_stats,
            source_image_stats=self.source_stats,
            target_controlpoint=target_controlpoint,
            alignmentArea=self._CELL_SIZE,
            defer_oob_check=True)

        self.assertIsNotNone(nan_mask)
        nan_mask_host = np.asarray(nan_mask)
        self.assertTrue(bool(nan_mask_host.any()), "Cell should have some out-of-bounds pixels")
        self.assertFalse(bool(nan_mask_host.all()), "Cell should not be entirely out of bounds")
        self.assertEqual(tuple(target_roi_eager.shape), tuple(target_roi_deferred.shape))
        self.assertEqual(tuple(source_roi_eager.shape), tuple(source_roi_deferred.shape))


class TestAttemptAlignPointsTranslationBatched(unittest.TestCase):
    """The batched accept/reject rewrite must still reject fully-OOB cells and align the rest."""

    def test_fully_oob_point_is_dropped_others_still_align(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

        image_shape = (40, 40)
        image = np.zeros(image_shape, dtype=nornir_imageregistration.default_image_dtype())
        for y in range(image_shape[0]):
            for x in range(image_shape[1]):
                image[y, x] = (((x % 8) + (y % 8)) / 8.0) * 0.6 + 0.2

        settings = nornir_imageregistration.settings.GridRefinement.CreateWithUnproccessedImages(
            target_image=image.copy(),
            source_image=image.copy(),
            cell_size=np.array((8, 8)),
            single_thread_processing=True)

        identity_transform = nornir_imageregistration.transforms.RigidTranslation(
            target_offset=np.array((0.0, 0.0)))

        # Four in-bounds points plus one point whose cell is entirely outside both images.
        target_points = np.array([
            (10.0, 10.0),
            (30.0, 30.0),
            (10.0, 30.0),
            (30.0, 10.0),
            (500.0, 500.0),
        ])
        source_points = target_points.copy()
        keys = [(i, 0) for i in range(len(target_points))]
        rigid_transforms = [identity_transform] * len(target_points)

        records = nornir_imageregistration.local_distortion_correction._attempt_align_points_translation_batched(
            keys=keys,
            source_points=source_points,
            target_points=target_points,
            rigid_transforms=rigid_transforms,
            settings=settings)

        self.assertIsNotNone(records, "Expected at least the four in-bounds points to align")
        aligned_keys = {record.ID for record in records}
        self.assertNotIn((4, 0), aligned_keys, "The fully out-of-bounds point must be rejected")
        self.assertLessEqual(len(aligned_keys), 4)
        self.assertGreaterEqual(len(aligned_keys), 3)


if __name__ == "__main__":
    unittest.main()
