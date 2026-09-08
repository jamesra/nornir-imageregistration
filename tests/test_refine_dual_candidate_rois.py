"""Dual-candidate STOS refine cells: rigid-fit ROI plus exact-transform ROI.

A single similarity fitted to a ring of points cannot follow a mesh that is locally
sheared or anisotropically scaled, so refine also measures each cell from a ROI warped
through the real transform and keeps the better candidate.
"""
from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import local_distortion_correction as ldc
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


if __name__ == '__main__':
    unittest.main()
