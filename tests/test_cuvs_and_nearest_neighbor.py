"""Tests for HasCuVS() and the nearest-neighbor index abstraction (scipy/CuVS)."""
import unittest
import numpy as np
import nornir_imageregistration
from nornir_imageregistration.nearest_neighbor import (
    CUVS_NN_MIN_POINTS_DEFAULT,
    _CuVSNNIndex,
    _ScipyNNIndex,
    build_nearest_neighbor_index,
)
from nornir_imageregistration.spatial_distance import cdist as pairwise_cdist


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

    def test_below_gate_uses_scipy_ckdtree(self):
        rng = np.random.RandomState(0)
        pts = rng.randn(256, 2).astype(np.float32)
        idx = build_nearest_neighbor_index(pts)
        self.assertIsInstance(idx, _ScipyNNIndex)

    def test_at_gate_uses_cuvs_when_available(self):
        rng = np.random.RandomState(0)
        pts = rng.randn(CUVS_NN_MIN_POINTS_DEFAULT, 2).astype(np.float32)
        idx = build_nearest_neighbor_index(pts)
        if nornir_imageregistration.UsingCupy() and nornir_imageregistration.HasCuVS():
            self.assertIsInstance(idx, _CuVSNNIndex)
            d, i = idx.query(pts[:1], k=1)
            self.assertEqual(int(_to_numpy(i)), 0)
            self.assertAlmostEqual(float(_to_numpy(d)), 0.0, places=5)
        else:
            self.assertIsInstance(idx, _ScipyNNIndex)


class TestGpuCdistUsesCuVS(unittest.TestCase):
    """GPU pairwise cdist stays on device when CuVS is available (no size gate)."""

    @unittest.skipUnless(
        nornir_imageregistration.HasCupy() and nornir_imageregistration.HasCuVS(),
        "CuPy and CuVS required",
    )
    def test_cupy_cdist_stays_on_device(self):
        import cupy as cp

        rng = np.random.RandomState(0)
        host = rng.randn(32, 2).astype(np.float32)
        xa = cp.asarray(host)
        dist = pairwise_cdist(xa, xa)
        self.assertIs(cp.get_array_module(dist), cp)
        np.testing.assert_allclose(
            _to_numpy(dist.diagonal()),
            np.zeros(host.shape[0]),
            atol=1e-5,
        )

    @unittest.skipUnless(
        nornir_imageregistration.HasCupy() and nornir_imageregistration.HasCuVS(),
        "CuPy and CuVS required",
    )
    def test_cupy_cdist_mixed_float32_float64(self):
        import cupy as cp

        xa = cp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=cp.float32)
        xb = cp.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=cp.float64)
        dist = pairwise_cdist(xa, xb)
        self.assertIs(cp.get_array_module(dist), cp)
        np.testing.assert_allclose(_to_numpy(dist[0, 0]), 0.0, atol=1e-5)
        np.testing.assert_allclose(_to_numpy(dist[1, 1]), np.sqrt(2.0), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
