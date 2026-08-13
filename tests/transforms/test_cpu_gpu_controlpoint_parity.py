"""Parity checks for control-point mutate/dedupe/flip on NumPy vs CuPy."""

from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.transforms.controlpointbase import ControlPointBase
from nornir_imageregistration.transforms.landmark import Landmark_CPU, Landmark_GPU
from nornir_imageregistration.transforms.triangulation import Triangulation, Triangulation_GPUComponent

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None


def _cupy_available() -> bool:
    if cp is None:
        return False
    if getattr(cp, "__name__", "") == "nornir_imageregistration.cupy_thunk":
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def _sample_point_pairs() -> np.ndarray:
    """Four control points with one fixed-space duplicate (rows 0 and 2)."""
    return np.array(
        [
            [10.0, 20.0, 100.0, 200.0],
            [30.0, 40.0, 110.0, 210.0],
            [10.0001, 20.0001, 120.0, 220.0],  # rounds to same fixed as row 0
            [50.0, 60.0, 130.0, 230.0],
        ],
        dtype=np.float32,
    )


def _as_host(arr) -> np.ndarray:
    getter = getattr(arr, "get", None)
    if callable(getter):
        return np.asarray(getter(), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


class TestCpuGpuControlPointParity(unittest.TestCase):
    """Same inputs → NumPy vs CuPy for dedupe, Flip, and FlipWarped."""

    def setUp(self) -> None:
        if not _cupy_available():
            self.skipTest("CuPy not available")

    def test_remove_duplicate_control_points_numpy_cupy_match(self) -> None:
        host = _sample_point_pairs()
        device = cp.asarray(host)
        host_out = ControlPointBase.RemoveDuplicateControlPoints(host)
        device_out = ControlPointBase.RemoveDuplicateControlPoints(device)
        self.assertEqual(host_out.shape[0], 3)
        np.testing.assert_allclose(_as_host(device_out), _as_host(host_out), rtol=0, atol=1e-5)

    def test_flip_numpy_cupy_match(self) -> None:
        host_pts = _sample_point_pairs()
        # Need three unique fixed points for Landmark; drop the duplicate row for Flip.
        host_pts = host_pts[[0, 1, 3], :]
        cpu = Landmark_CPU(host_pts.copy())
        gpu = Landmark_GPU(cp.asarray(host_pts.copy()))
        before = _as_host(cpu.points).copy()
        cpu.Flip()
        gpu.Flip()
        np.testing.assert_allclose(_as_host(gpu.points), _as_host(cpu.points), rtol=0, atol=1e-4)
        # X columns (indices 1 and 3) must change; Y columns stay put.
        self.assertFalse(np.allclose(before[:, 1], _as_host(cpu.points)[:, 1]))
        np.testing.assert_allclose(before[:, 0], _as_host(cpu.points)[:, 0], rtol=0, atol=1e-4)

    def test_flip_warped_numpy_cupy_match(self) -> None:
        host_pts = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0, 10.0],
                [10.0, 0.0, 10.0, 0.0],
                [10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float32,
        )
        cpu = Triangulation(host_pts.copy())
        gpu = Triangulation_GPUComponent(cp.asarray(host_pts.copy()))
        center = np.array([5.0, 5.0], dtype=np.float64)
        cpu.FlipWarped(flip_center=center)
        gpu.FlipWarped(flip_center=cp.asarray(center))
        np.testing.assert_allclose(_as_host(gpu.points), _as_host(cpu.points), rtol=0, atol=1e-4)
        # Source X flipped about center; source Y restored (no permanent Y shift).
        expected_source = host_pts[:, 2:4].copy()
        expected_source[:, 1] = -(expected_source[:, 1] - 5.0) + 5.0
        np.testing.assert_allclose(_as_host(cpu.points)[:, 2:4], expected_source, rtol=0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
