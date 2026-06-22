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
        """CuPy degeneracy during build selects SciPy inverse interpolator."""
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

    def test_grid_inverse_returns_nan_when_both_build_paths_fail(self) -> None:
        """Dual CuPy and SciPy triangulation failure yields NaN inverse coordinates."""
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

    def test_scipy_triangulation_failure_returns_nan(self) -> None:
        """SciPy Qhull failure during fallback build returns NaN without raising."""
        transform = _sample_grid_transform()
        target_np = nornir_imageregistration.EnsureNumpyArray(transform.TargetPoints)
        query_points = cp.asarray(target_np[:1], dtype=np.float64)

        with mock.patch(
                'nornir_imageregistration.transforms.gridtransform.cuLinearNDInterpolator',
                side_effect=_CUPY_DEGENERATE_ERROR,
        ), mock.patch(
                'scipy.spatial.Delaunay',
                side_effect=scipy.spatial.QhullError('mock qhull failure', '', ''),
        ):
            result = nornir_imageregistration.EnsureNumpyArray(transform.InverseTransform(query_points))

        self.assertTrue(np.all(np.isnan(result)))


if __name__ == '__main__':
    unittest.main()
