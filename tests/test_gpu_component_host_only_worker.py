"""A host-only pool worker must not move points to the GPU in GridWithRBFFallback_GPUComponent (RC3 516, 2026-11-10).

e57b480 switched workers to SciPy interpolators, but the transform still pushed its points to CuPy on entry and exit,
so each of ~128 prewarp workers still allocated GPU memory and the card filled up.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.transforms import gridtransform
from nornir_imageregistration.transforms.gridwithrbffallback import GridWithRBFFallback_GPUComponent

EXTENT = 1000.0
GRID_N = 5


def _grid():
    from nornir_imageregistration.grid_subdivision import ITKGridDivision
    grid = ITKGridDivision(
        source_shape=np.asarray((EXTENT, EXTENT), dtype=np.float64),
        cell_size=np.asarray((EXTENT / GRID_N, EXTENT / GRID_N), dtype=np.float64),
        grid_dims=np.asarray((GRID_N, GRID_N), dtype=np.int64))
    src = np.asarray(grid.SourcePoints, dtype=np.float64)
    tgt = src.copy()
    tgt[:, 0] += 7.25
    tgt[:, 1] += -3.75
    grid.TargetPoints = tgt
    return grid


class TestHostOnlyWorker(unittest.TestCase):

    def setUp(self) -> None:
        self.addCleanup(setattr, gridtransform, '_scipy_interpolators_only', gridtransform._scipy_interpolators_only)

    def test_inverse_transform_never_touches_cupy_in_a_host_only_worker(self) -> None:
        transform = GridWithRBFFallback_GPUComponent(_grid())
        gridtransform.use_scipy_interpolators_in_this_process()
        points = np.array([[100.0, 100.0], [500.0, 500.0], [900.0, 250.0]])
        boom = mock.Mock(side_effect=AssertionError('points moved to the GPU in a host-only worker'))
        with mock.patch.object(nornir_imageregistration, 'EnsurePointsAre2DCuPyArray', boom):
            result = transform.InverseTransform(points, extrapolate=False)
            empty = transform.InverseTransform(np.empty((0, 2)), extrapolate=False)
        boom.assert_not_called()
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual((3, 2), result.shape)
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(result, points - np.array([7.25, -3.75]), atol=1e-6)
        self.assertEqual((0, 2), empty.shape)


if __name__ == '__main__':
    unittest.main()
