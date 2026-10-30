"""Regression for #119 / #113: vectorized GetPointPairsInRect."""
from __future__ import annotations

import time
import unittest

import numpy as np

import nornir_imageregistration as nir
from nornir_imageregistration.transforms import MeshWithRBFFallback
from nornir_imageregistration.transforms.controlpointbase import _select_point_pairs_in_rect


def _legacy_loop(point_pairs: np.ndarray, query_yx: np.ndarray, bounds) -> np.ndarray | None:
    output = None
    bounds_arr = nir.Rectangle.PrimitiveToRectangle(bounds).ToArray()
    for i in range(query_yx.shape[0]):
        y, x = query_yx[i, :]
        if nir.Rectangle.contains(bounds_arr, (y, x)):
            row = point_pairs[i, :]
            output = row if output is None else np.vstack((output, row))
    if output is not None and output.ndim == 1:
        output = np.reshape(output, (1, output.shape[0]))
    return output


class TestSelectPointPairsInRect(unittest.TestCase):
    def test_matches_legacy_loop(self):
        rng = np.random.default_rng(0)
        pairs = rng.random((200, 4)) * 100.0
        bounds = nir.Rectangle.CreateFromBounds((20.0, 20.0, 80.0, 80.0))
        legacy = _legacy_loop(pairs, pairs[:, 0:2], bounds)
        vectorized = _select_point_pairs_in_rect(pairs, pairs[:, 0:2], bounds)
        self.assertIsNotNone(legacy)
        self.assertIsNotNone(vectorized)
        np.testing.assert_allclose(vectorized, legacy)

    def test_empty_rect_returns_none(self):
        pairs = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
        bounds = nir.Rectangle.CreateFromBounds((50.0, 50.0, 60.0, 60.0))
        self.assertIsNone(_select_point_pairs_in_rect(pairs, pairs[:, 0:2], bounds))

    def test_mesh_get_fixed_points_in_rect(self):
        pairs = np.array(
            [[0.0, 0.0, 0.0, 0.0],
             [10.0, 10.0, 10.0, 10.0],
             [5.0, 5.0, 5.0, 5.0],
             [100.0, 100.0, 100.0, 100.0]],
            dtype=np.float64,
        )
        mesh = MeshWithRBFFallback(pairs)
        bounds = nir.Rectangle.CreateFromBounds((0.0, 0.0, 20.0, 20.0))
        got = mesh.GetFixedPointsInRect(bounds)
        self.assertEqual(got.shape[0], 3)

    def test_vectorized_faster_than_legacy_loop(self):
        rng = np.random.default_rng(1)
        n = 4000
        pairs = rng.random((n, 4)) * 100.0
        bounds = nir.Rectangle.CreateFromBounds((25.0, 25.0, 75.0, 75.0))
        t0 = time.perf_counter()
        for _ in range(3):
            _legacy_loop(pairs, pairs[:, 0:2], bounds)
        legacy_ms = (time.perf_counter() - t0) / 3 * 1000
        t0 = time.perf_counter()
        for _ in range(30):
            _select_point_pairs_in_rect(pairs, pairs[:, 0:2], bounds)
        vec_ms = (time.perf_counter() - t0) / 30 * 1000
        self.assertLess(vec_ms * 20, legacy_ms, msg=f'legacy={legacy_ms:.2f}ms vec={vec_ms:.2f}ms')

    @unittest.skipUnless(nir.HasCupy(), 'CuPy required')
    def test_cupy_pairs_stay_on_device(self):
        import cupy as cp

        pairs = cp.asarray(
            [[0.0, 0.0, 0.0, 0.0],
             [10.0, 10.0, 10.0, 10.0],
             [100.0, 100.0, 100.0, 100.0]],
            dtype=cp.float64,
        )
        bounds = nir.Rectangle.CreateFromBounds((0.0, 0.0, 20.0, 20.0))
        got = _select_point_pairs_in_rect(pairs, pairs[:, 0:2], bounds)
        self.assertIsNotNone(got)
        self.assertIs(cp.get_array_module(got), cp)
        self.assertEqual(int(got.shape[0]), 2)


if __name__ == '__main__':
    unittest.main()
