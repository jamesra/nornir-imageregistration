"""Smoke tests for batched ApproximateRigidTransformBySourcePoints."""

from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.local_distortion_correction import (
    ApproximateRigidTransformBySourcePoints,
)
from nornir_imageregistration.transforms.meshwithrbffallback import MeshWithRBFFallback


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


if __name__ == '__main__':
    unittest.main()
