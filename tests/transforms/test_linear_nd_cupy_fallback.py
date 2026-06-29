"""Tests for CuPy LinearNDInterpolator fallback to SciPy Qhull."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pytest
import scipy.spatial

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

    def test_grid_inverse_uses_cpu_analytic_when_culinear_degenerate(self) -> None:
        """Grid inverse uses CPU analytic when cuLinearND fails (degenerate triangulation)."""
        from nornir_imageregistration.transforms.gridtransform import _GridTopologyLinearInterpolator

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
        self.assertIsInstance(transform._InverseInterpolator, _GridTopologyLinearInterpolator)
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

    def test_grid_inverse_gpu_matches_cpu_analytic(self) -> None:
        """GPU cuLinearNDInterpolator inverse matches CPU _GridTopologyLinearInterpolator.

        Queries the full set of TargetPoints (known exact positions) plus a dense
        interior grid so both the on-control-point and interpolated cases are covered.
        """
        from nornir_imageregistration.transforms.gridtransform import (
            _GridTopologyLinearInterpolator,
            _build_scipy_linear_nd_interpolator,
        )

        gpu_transform = _sample_grid_transform()
        target_np = nornir_imageregistration.EnsureNumpyArray(gpu_transform.TargetPoints)
        source_np = nornir_imageregistration.EnsureNumpyArray(gpu_transform.SourcePoints)

        # Build CPU reference directly.
        cpu_interp = _GridTopologyLinearInterpolator(target_np, source_np, gpu_transform.grid_dims)

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
            err_msg='GPU inverse does not match CPU analytic inverse within tolerance',
        )


class TestCuGridTopologyInterpolator(unittest.TestCase):
    """Tests for _CuGridTopologyInterpolator — GPU analytic barycentric grid lookup."""

    def setUp(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)

    def _build_interpolators(self):
        from nornir_imageregistration.transforms.gridtransform import (
            _CuGridTopologyInterpolator,
            _GridTopologyLinearInterpolator,
        )
        transform = _sample_grid_transform()
        target_np = nornir_imageregistration.EnsureNumpyArray(transform.TargetPoints)
        source_np = nornir_imageregistration.EnsureNumpyArray(transform.SourcePoints)
        dims = transform.grid_dims
        cpu = _GridTopologyLinearInterpolator(target_np, source_np, dims)
        gpu = _CuGridTopologyInterpolator(target_np, source_np, dims)
        return cpu, gpu, target_np, source_np

    def _interior_query(self, target_np: np.ndarray, n: int = 30) -> np.ndarray:
        y_min, x_min = target_np.min(axis=0)
        y_max, x_max = target_np.max(axis=0)
        ys = np.linspace(y_min + 1, y_max - 1, n)
        xs = np.linspace(x_min + 1, x_max - 1, n)
        gy, gx = np.meshgrid(ys, xs, indexing='ij')
        return np.column_stack([gy.ravel(), gx.ravel()]).astype(np.float64)

    @pytest.mark.skipif(not nornir_imageregistration.HasCupy(), reason='CuPy required')
    def test_returns_cupy_array(self) -> None:
        """_CuGridTopologyInterpolator returns a CuPy array."""
        _, gpu, target_np, _ = self._build_interpolators()
        query = cp.asarray(target_np[:3], dtype=cp.float64)
        result = gpu(query)
        self.assertIsInstance(result, cp.ndarray)

    @pytest.mark.skipif(not nornir_imageregistration.HasCupy(), reason='CuPy required')
    def test_control_points_exact(self) -> None:
        """GPU interpolator returns source values at grid control points (on-vertex case)."""
        cpu, gpu, target_np, source_np = self._build_interpolators()
        query = cp.asarray(target_np, dtype=cp.float64)
        gpu_result = nornir_imageregistration.EnsureNumpyArray(gpu(query))
        valid = ~np.isnan(gpu_result).any(axis=1)
        self.assertGreater(valid.sum(), len(target_np) // 2)
        np.testing.assert_allclose(
            gpu_result[valid], source_np[valid],
            rtol=0, atol=1e-5,
            err_msg='GPU interpolator does not return source values at control points',
        )

    @pytest.mark.skipif(not nornir_imageregistration.HasCupy(), reason='CuPy required')
    def test_matches_cpu_interior(self) -> None:
        """GPU interpolator matches CPU analytic on a dense interior query grid."""
        cpu, gpu, target_np, _ = self._build_interpolators()
        query_np = self._interior_query(target_np)
        cpu_result = cpu(query_np)
        gpu_result = nornir_imageregistration.EnsureNumpyArray(
            gpu(cp.asarray(query_np, dtype=cp.float64))
        )
        valid = ~np.isnan(cpu_result).any(axis=1) & ~np.isnan(gpu_result).any(axis=1)
        self.assertGreater(valid.sum(), 200, 'Expected at least 200 valid interior points')
        np.testing.assert_allclose(
            gpu_result[valid], cpu_result[valid],
            rtol=0, atol=1e-4,
            err_msg='GPU analytic does not match CPU analytic within tolerance',
        )

    @pytest.mark.skipif(not nornir_imageregistration.HasCupy(), reason='CuPy required')
    def test_used_as_inverse_interpolator_after_culinear_fails(self) -> None:
        """After cuLinearNDInterpolator degeneracy, CPU analytic inverse is selected."""
        from nornir_imageregistration.transforms.gridtransform import _GridTopologyLinearInterpolator

        transform = _sample_grid_transform()
        transform._InverseInterpolator = None
        transform._scipy_inverse_interp = False

        with mock.patch(
                'nornir_imageregistration.transforms.gridtransform.cuLinearNDInterpolator',
                side_effect=_CUPY_DEGENERATE_ERROR,
        ):
            _ = transform.InverseInterpolator

        self.assertIsInstance(
            transform._InverseInterpolator,
            _GridTopologyLinearInterpolator,
            'Expected CPU analytic inverse after cuLinearNDInterpolator failure',
        )
        self.assertTrue(
            transform._scipy_inverse_interp,
            '_scipy_inverse_interp should be True for CPU analytic path',
        )

    @pytest.mark.skipif(not nornir_imageregistration.HasCupy(), reason='CuPy required')
    def test_end_to_end_inverse_transform_uses_cpu_analytic(self) -> None:
        """Full InverseTransform via CPU analytic matches reference when cuLinearND fails."""
        cpu, _, target_np, _ = self._build_interpolators()
        transform = _sample_grid_transform()
        transform._InverseInterpolator = None
        transform._scipy_inverse_interp = False

        query_np = self._interior_query(target_np)
        cpu_result = cpu(query_np)

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
            err_msg='End-to-end GPU analytic inverse does not match CPU reference',
        )


if __name__ == '__main__':
    unittest.main()
