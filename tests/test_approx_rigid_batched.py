"""Smoke tests for batched ApproximateRigidTransformBySourcePoints."""

from __future__ import annotations

import math
import unittest

import numpy as np

from nornir_imageregistration.local_distortion_correction import (
    ApproximateRigidTransformBySourcePoints,
    calculate_offset,
)
from nornir_imageregistration.refine_shared.ring_pose_limits import (
    RING_SCALE_FRACTION_MAX,
    reference_pose_from_transform,
)
from nornir_imageregistration.transforms.converters import (
    EstimateRigidComponentsFromControlPointsBatched,
)
from nornir_imageregistration.transforms.meshwithrbffallback import MeshWithRBFFallback


def _center_plus_ring(center_yx: np.ndarray, cell_size: np.ndarray) -> np.ndarray:
    """Reproduce the 9-point ring used by ApproximateRigidTransformBySourcePoints."""
    center = np.asarray(center_yx, dtype=np.float64).reshape(1, 2)
    offset = calculate_offset(center, cell_size)
    radius = float(np.linalg.norm(offset))
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring = np.vstack((np.cos(angles), np.sin(angles))).T * radius
    return np.vstack((center, center + ring))


def _sparse_180_mesh() -> tuple[MeshWithRBFFallback, np.ndarray]:
    """20-point ~180° scale-1 mesh with high-frequency warp.

    Global Kabsch stays near scale 1 / 180°, but a hull-edge ring samples
    mixed in-hull / RBF points and can invent scale ~4–5 (the 784-782 shape).
    """
    rng = np.random.default_rng(1)
    src = rng.uniform(low=2200.0, high=5000.0, size=(20, 2))
    center = np.array([3606.0, 3496.0], dtype=np.float64)
    tgt = (2.0 * center) - src
    tgt = tgt + np.stack(
        (500.0 * np.sin(src[:, 0] / 25.0), 500.0 * np.cos(src[:, 1] / 25.0)),
        axis=1)
    return MeshWithRBFFallback(np.hstack([tgt, src])), src


class TestApproximateRigidBatched(unittest.TestCase):
    """Batched ring Transform should match per-point rigid fit at query centers."""

    def test_batch_maps_centers_near_parent_transform(self) -> None:
        src = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float64)
        tgt = src + np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]], dtype=np.float64)
        control = np.hstack([tgt, src])
        transform = MeshWithRBFFallback(control)
        query = np.array([[25.0, 25.0], [50.0, 50.0], [75.0, 40.0]], dtype=np.float64)
        rigids = ApproximateRigidTransformBySourcePoints(
            transform, query, cell_size=np.array([32.0, 32.0], dtype=np.float64))
        self.assertEqual(len(rigids), 3)
        for i, point in enumerate(query):
            predicted = np.asarray(rigids[i].Transform(point.reshape(1, 2))).reshape(2)
            expected = np.asarray(transform.Transform(point.reshape(1, 2))).reshape(2)
            self.assertLess(float(np.linalg.norm(predicted - expected)), 2.0)

    def test_clamped_ring_pins_center_to_parent_transform(self) -> None:
        mesh, src = _sparse_180_mesh()
        query = src[0:1].copy()
        cell_size = np.array([256.0, 256.0], dtype=np.float64)
        rigid = ApproximateRigidTransformBySourcePoints(mesh, query, cell_size=cell_size)[0]
        predicted = np.asarray(rigid.Transform(query)).reshape(2)
        expected = np.asarray(mesh.Transform(query)).reshape(2)
        self.assertLess(float(np.linalg.norm(predicted - expected)), 0.25)

    def test_oob_ring_on_sparse_180_mesh_does_not_invent_scale(self) -> None:
        """Regression shape of RC2 784-782: OOB ring must not fit scale ~2.6."""
        mesh, src = _sparse_180_mesh()
        query = src[0:1].copy()
        cell_size = np.array([256.0, 256.0], dtype=np.float64)
        ring_src = _center_plus_ring(query[0], cell_size)
        ring_tgt = np.asarray(mesh.Transform(ring_src), dtype=np.float64)
        unclamped = EstimateRigidComponentsFromControlPointsBatched(
            ring_src.reshape(1, -1, 2), ring_tgt.reshape(1, -1, 2))[0]
        self.assertGreater(abs(unclamped.scale - 1.0), 1.0)

        ref = reference_pose_from_transform(mesh)
        self.assertLess(abs(ref.scale - 1.0), 0.05)
        self.assertGreater(abs(ref.angle), math.pi * 0.75)

        rigid = ApproximateRigidTransformBySourcePoints(mesh, query, cell_size=cell_size)[0]
        self.assertLess(
            abs(float(rigid.scalar) / ref.scale - 1.0),
            RING_SCALE_FRACTION_MAX + 1e-6)


if __name__ == '__main__':
    unittest.main()
