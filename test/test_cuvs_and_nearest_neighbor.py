"""Tests for HasCuVS() and the nearest-neighbor index abstraction (scipy/CuVS)."""
import unittest
import numpy as np
import nornir_imageregistration
from nornir_imageregistration.nearest_neighbor import build_nearest_neighbor_index


def _to_numpy(x):
    """Convert array to numpy for assertions (handles CuPy)."""
    return np.asarray(x) if not hasattr(x, "get") else x.get()


class TestHasCuVS(unittest.TestCase):
    """HasCuVS() is detected and cached at startup; no import error when CuVS is missing."""

    def test_has_cuvs_returns_bool(self):
        self.assertIsInstance(nornir_imageregistration.HasCuVS(), bool)

    def test_has_cuvs_false_when_cupy_missing(self):
        # When CuPy is not available, HasCuVS must be False. When CuPy is available
        # but CuVS is not (e.g. Windows), HasCuVS is also False.
        if not nornir_imageregistration.HasCupy():
            self.assertFalse(nornir_imageregistration.HasCuVS())


class TestNearestNeighborIndex(unittest.TestCase):
    """build_nearest_neighbor_index and .query(points, k=1) match cKDTree-style behavior."""

    def test_build_and_query_single_point(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        idx = build_nearest_neighbor_index(pts)
        d, i = idx.query(pts[:1], k=1)
        self.assertEqual(d, 0.0)
        self.assertEqual(i, 0)

    def test_build_and_query_batch(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        idx = build_nearest_neighbor_index(pts)
        d, i = idx.query(pts, k=1)
        np.testing.assert_array_almost_equal(_to_numpy(d), [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(_to_numpy(i), [0, 1, 2])

    def test_query_nearest_other_point(self):
        pts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        idx = build_nearest_neighbor_index(pts)
        # query [1.5, 0] -> unique nearest is index 1 (distance 0.5)
        d, i = idx.query(np.array([[1.5, 0.0]], dtype=np.float32), k=1)
        self.assertEqual(int(_to_numpy(i)), 1)
        self.assertAlmostEqual(float(_to_numpy(d)), 0.5)


if __name__ == "__main__":
    unittest.main()
