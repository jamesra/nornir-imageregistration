"""Tests for CuPy LinearNDInterpolator fallback to SciPy Qhull."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pytest
import scipy.spatial
from scipy.interpolate import LinearNDInterpolator

import nornir_imageregistration
from nornir_imageregistration.grid_subdivision import ITKGridDivision
from nornir_imageregistration.transforms.gridtransform import GridTransform_GPUComponent
from nornir_imageregistration.transforms.triangulation import Triangulation_GPUComponent

try:
    import cupy as cp
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp

try:
    from tests.transforms.data import IdentityTransformPoints
except ImportError:
    from transforms.data import IdentityTransformPoints

_CUPY_DEGENERATE_ERROR = ValueError(
    'The input is degenerate, the extreme points are close to coplanar',
)


def _sample_grid_transform() -> GridTransform_GPUComponent:
    """Return a GPU grid transform with enough finite control points for Qhull."""
    transform = nornir_imageregistration.transforms.Rigid(
        target_offset=(500, 1000),
        source_rotation_center=(150, 300),
        angle=0,
    )
    grid_data = ITKGridDivision(
        source_shape=(1000, 1000),
        cell_size=(256, 256),
        grid_spacing=(192, 192),
        transform=transform,
    )
    return GridTransform_GPUComponent(grid_data)


def _identity_triangulation_gpu() -> Triangulation_GPUComponent:
    """Return a GPU triangulation with identity control points."""
    return Triangulation_GPUComponent(IdentityTransformPoints)


@pytest.mark.skipif(not nornir_imageregistration.HasCupy(), reason='CuPy required')
class TestLinearNDCuPyFallback(unittest.TestCase):
    """Verify SciPy Qhull fallback when cupyx triangulation fails."""

    def setUp(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)

    def test_grid_inverse_falls_back_when_cupy_build_fails(self) -> None:
        """When cuLinearND fails on a degenerate grid, CPU analytic inverse is selected."""
        transform = _sample_grid_transform()
        target_np = nornir_imageregistration.EnsureNumpyArray(transform.TargetPoints)
        query_points = cp.asarray(target_np[:2], dtype=np.float64)

        with mock.patch(
                'nornir_imageregistration.transforms.gridtransform.cuLinearNDInterpolator',
                side_effect=_CUPY_DEGENERATE_ERROR,
        ):
            _ = transform.InverseInterpolator

        self.assertTrue(transform._scipy_inverse_interp)
        self.assertIsNotNone(transform._InverseInterpolator)

        with mock.patch(
                'nornir_imageregistration.transforms.gridtransform.cuLinearNDInterpolator',
                side_effect=_CUPY_DEGENERATE_ERROR,
        ):
            reference = _sample_grid_transform()
            reference._InverseInterpolator = None
            reference._scipy_inverse_interp = True
            _ = reference.InverseInterpolator

        expected = nornir_imageregistration.EnsureNumpyArray(reference.InverseTransform(query_points))
        actual = nornir_imageregistration.EnsureNumpyArray(transform.InverseTransform(query_points))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)

    def test_triangulation_inverse_falls_back_when_cupy_build_fails(self) -> None:
        """Triangulation GPU inverse matches SciPy reference after CuPy build failure."""
        transform = _identity_triangulation_gpu()
        query_points = cp.asarray([[0.25, 0.25], [0.75, 0.75]], dtype=np.float32)

        with mock.patch(
                'nornir_imageregistration.transforms.gridtransform.cuLinearNDInterpolator',
                side_effect=_CUPY_DEGENERATE_ERROR,
        ):
            _ = transform.InverseInterpolator

        self.assertTrue(transform._scipy_inverse_interp)
        self.assertIsNotNone(transform._InverseInterpolator)

        reference = _identity_triangulation_gpu()
        reference._InverseInterpolator = None
        reference._scipy_inverse_interp = True
        _ = reference.InverseInterpolator

        expected = nornir_imageregistration.EnsureNumpyArray(reference.InverseTransform(query_points))
        actual = nornir_imageregistration.EnsureNumpyArray(transform.InverseTransform(query_points))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)

    def test_grid_inverse_returns_nan_when_all_build_paths_fail(self) -> None:
        """When cuLinearND and CPU analytic both fail, inverse coordinates are NaN."""
        transform = _sample_grid_transform()
        target_np = nornir_imageregistration.EnsureNumpyArray(transform.TargetPoints)
        query_points = cp.asarray(target_np[:1], dtype=np.float64)

        with mock.patch(
                'nornir_imageregistration.transforms.gridtransform.cuLinearNDInterpolator',
                side_effect=_CUPY_DEGENERATE_ERROR,
        ), mock.patch(
                'nornir_imageregistration.transforms.gridtransform._build_scipy_linear_nd_interpolator',
                return_value=None,
        ):
            result = nornir_imageregistration.EnsureNumpyArray(transform.InverseTransform(query_points))

        self.assertTrue(transform._scipy_inverse_interp)
        self.assertTrue(np.all(np.isnan(result)))

    def test_grid_inverse_query_retry_falls_back_after_cupy_query_failure(self) -> None:
        """CuPy query degeneracy invalidates GPU interpolator and retries via SciPy."""
        transform = _sample_grid_transform()
        target_np = nornir_imageregistration.EnsureNumpyArray(transform.TargetPoints)
        query_points = cp.asarray(target_np[:1], dtype=np.float64)
        call_count = {'n': 0}

        class _FlakyGpuInterpolator:
            def __call__(self, points: cp.ndarray) -> cp.ndarray:
                call_count['n'] += 1
                if call_count['n'] == 1:
                    raise _CUPY_DEGENERATE_ERROR
                raise AssertionError('GPU interpolator should not be called after fallback retry')

        transform._InverseInterpolator = _FlakyGpuInterpolator()
        transform._scipy_inverse_interp = False

        expected = nornir_imageregistration.EnsureNumpyArray(
            _sample_grid_transform().InverseTransform(query_points),
        )
        actual = nornir_imageregistration.EnsureNumpyArray(transform.InverseTransform(query_points))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)
        self.assertTrue(transform._scipy_inverse_interp)

    def test_grid_inverse_uses_scipy_qhull_when_culinear_degenerate(self) -> None:
        """Grid inverse falls back to SciPy Qhull when cuLinearND rejects the mesh as degenerate."""
        transform = _sample_grid_transform()
        target_np = nornir_imageregistration.EnsureNumpyArray(transform.TargetPoints)
        source_np = nornir_imageregistration.EnsureNumpyArray(transform.SourcePoints)
        query_points = cp.asarray(target_np[:1], dtype=np.float64)

        with mock.patch(
                'nornir_imageregistration.transforms.gridtransform.cuLinearNDInterpolator',
                side_effect=_CUPY_DEGENERATE_ERROR,
        ):
            result = nornir_imageregistration.EnsureNumpyArray(transform.InverseTransform(query_points))

        self.assertTrue(transform._scipy_inverse_interp)
        self.assertIsInstance(transform._InverseInterpolator, LinearNDInterpolator)
        self.assertFalse(np.any(np.isnan(result)))
        np.testing.assert_allclose(result, source_np[:1], rtol=0, atol=1e-5)

    def test_grid_inverse_uses_gpu_path_by_default(self) -> None:
        """GridTransform_GPUComponent uses cuLinearNDInterpolator for inverse (not CPU analytic).

        Before the grid_dims gate removal, grid_dims is not None forced the analytic CPU path
        unconditionally. After the fix, cuLinearNDInterpolator is attempted first for inverse
        because TargetPoints (deformed fixed-space) are not collinear.
        """
        transform = _sample_grid_transform()
        # Force rebuild of the inverse interpolator.
        transform._InverseInterpolator = None
        transform._scipy_inverse_interp = False

        _ = transform.InverseInterpolator

        # The GPU path should have been used: scipy flag must be False.
        self.assertFalse(
            transform._scipy_inverse_interp,
            'InverseInterpolator should be cuLinearNDInterpolator (GPU), not CPU analytic path',
        )

    def test_grid_inverse_gpu_matches_scipy_qhull(self) -> None:
        """GPU cuLinearNDInterpolator inverse matches the SciPy Qhull host reference.

        Queries a dense interior grid so the interpolated case is covered on both
        backends. This is the host/device parity guard for the grid inverse.
        """
        from nornir_imageregistration.transforms.gridtransform import (
            _build_scipy_linear_nd_interpolator,
        )

        gpu_transform = _sample_grid_transform()
        target_np = nornir_imageregistration.EnsureNumpyArray(gpu_transform.TargetPoints)
        source_np = nornir_imageregistration.EnsureNumpyArray(gpu_transform.SourcePoints)

        cpu_interp = _build_scipy_linear_nd_interpolator(target_np, source_np)
        assert cpu_interp is not None, 'SciPy Qhull reference interpolator must build'

        # Dense interior query grid (avoids extrapolation which both paths handle as NaN).
        y_min, x_min = target_np.min(axis=0)
        y_max, x_max = target_np.max(axis=0)
        ys = np.linspace(y_min + 1, y_max - 1, 20)
        xs = np.linspace(x_min + 1, x_max - 1, 20)
        gy, gx = np.meshgrid(ys, xs, indexing='ij')
        query_np = np.column_stack([gy.ravel(), gx.ravel()]).astype(np.float64)

        cpu_result = cpu_interp(query_np)
        gpu_result = nornir_imageregistration.EnsureNumpyArray(
            gpu_transform.InverseTransform(cp.asarray(query_np, dtype=np.float64))
        )

        # Mask NaN where CPU analytic also returns NaN (boundary extrapolation).
        valid = ~np.isnan(cpu_result).any(axis=1) & ~np.isnan(gpu_result).any(axis=1)
        self.assertGreater(valid.sum(), 50, 'Expected at least 50 valid interior query points')
        np.testing.assert_allclose(
            gpu_result[valid], cpu_result[valid],
            rtol=0, atol=1e-3,
            err_msg='GPU inverse does not match SciPy Qhull inverse within tolerance',
        )


class TestScipyQhullInverseFallback(unittest.TestCase):
    """Tests for the SciPy Qhull host interpolator used when cupyx Delaunay is degenerate.

    The GPU analytic grid-topology interpolator these tests originally covered was
    removed in 696479f8 because it regressed section assemble from ~9s to ~60s;
    SciPy Qhull is now the only fallback, so the parity checks target it instead.
    """

    def setUp(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)

    def _build_interpolators(self):
        from nornir_imageregistration.transforms.gridtransform import (
            _build_scipy_linear_nd_interpolator,
        )
        transform = _sample_grid_transform()
        target_np = nornir_imageregistration.EnsureNumpyArray(transform.TargetPoints)
        source_np = nornir_imageregistration.EnsureNumpyArray(transform.SourcePoints)
        cpu = _build_scipy_linear_nd_interpolator(target_np, source_np)
        assert cpu is not None, 'SciPy Qhull reference interpolator must build'
        return cpu, target_np, source_np

    def _interior_query(self, target_np: np.ndarray, n: int = 30) -> np.ndarray:
        y_min, x_min = target_np.min(axis=0)
        y_max, x_max = target_np.max(axis=0)
        ys = np.linspace(y_min + 1, y_max - 1, n)
        xs = np.linspace(x_min + 1, x_max - 1, n)
        gy, gx = np.meshgrid(ys, xs, indexing='ij')
        return np.column_stack([gy.ravel(), gx.ravel()]).astype(np.float64)

    @pytest.mark.skipif(not nornir_imageregistration.HasCupy(), reason='CuPy required')
    def test_gpu_inverse_returns_cupy_array(self) -> None:
        """The default (non-degenerate) GPU inverse keeps results on the device."""
        transform = _sample_grid_transform()
        target_np = nornir_imageregistration.EnsureNumpyArray(transform.TargetPoints)
        result = transform.InverseTransform(cp.asarray(target_np[:3], dtype=np.float64))
        self.assertIsInstance(result, cp.ndarray)

    @pytest.mark.skipif(not nornir_imageregistration.HasCupy(), reason='CuPy required')
    def test_control_points_exact(self) -> None:
        """SciPy Qhull fallback returns source values at grid control points (on-vertex case)."""
        cpu, target_np, source_np = self._build_interpolators()
        cpu_result = np.asarray(cpu(target_np))
        valid = ~np.isnan(cpu_result).any(axis=1)
        self.assertGreater(valid.sum(), len(target_np) // 2)
        np.testing.assert_allclose(
            cpu_result[valid], source_np[valid],
            rtol=0, atol=1e-5,
            err_msg='Fallback interpolator does not return source values at control points',
        )

    @pytest.mark.skipif(not nornir_imageregistration.HasCupy(), reason='CuPy required')
    def test_end_to_end_inverse_transform_uses_scipy_qhull(self) -> None:
        """Full InverseTransform matches the SciPy reference when cuLinearND is degenerate."""
        cpu, target_np, _ = self._build_interpolators()
        transform = _sample_grid_transform()
        transform._InverseInterpolator = None
        transform._scipy_inverse_interp = False

        query_np = self._interior_query(target_np)
        cpu_result = np.asarray(cpu(query_np))

        with mock.patch(
                'nornir_imageregistration.transforms.gridtransform.cuLinearNDInterpolator',
                side_effect=_CUPY_DEGENERATE_ERROR,
        ):
            gpu_result = nornir_imageregistration.EnsureNumpyArray(
                transform.InverseTransform(cp.asarray(query_np, dtype=cp.float64))
            )

        valid = ~np.isnan(cpu_result).any(axis=1) & ~np.isnan(gpu_result).any(axis=1)
        self.assertGreater(valid.sum(), 200)
        np.testing.assert_allclose(
            gpu_result[valid], cpu_result[valid],
            rtol=0, atol=1e-4,
            err_msg='End-to-end inverse does not match the SciPy Qhull reference',
        )


if __name__ == '__main__':
    unittest.main()
