"""Regression for #190: host Triangulation.Scale must rebind, not mutate in place."""
from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.transforms.triangulation import Triangulation


def _square_points() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 10.0, 0.0],
            [0.0, 10.0, 0.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
        ],
        dtype=np.float64,
    )


class TestTriangulationScaleRebind(unittest.TestCase):
    def test_scale_does_not_mutate_points_alias(self) -> None:
        """#190: ``*=`` mutated aliases of ``.points``; GPU/Landmark rebind."""
        tri = Triangulation(_square_points())
        alias = tri.points
        original = alias.copy()
        tri.Scale(2.0)
        np.testing.assert_array_equal(alias, original)
        np.testing.assert_allclose(tri.points, original * 2.0)
        self.assertFalse(np.shares_memory(alias, tri.points))


if __name__ == '__main__':
    unittest.main()
