"""Process-pool workers must not build CuPy triangulations (RC3 516: grid_prewarp ran out of GPU memory)."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from nornir_imageregistration.transforms import gridtransform


class TestScipyInterpolatorsOnly(unittest.TestCase):

    def setUp(self) -> None:
        self.addCleanup(setattr, gridtransform, '_scipy_interpolators_only', gridtransform._scipy_interpolators_only)

    def _points(self):
        grid = np.array([[0, 0], [0, 10], [10, 0], [10, 10], [5, 5]], dtype=np.float64)
        return grid, grid + 1.0

    def test_a_worker_never_calls_the_cupy_interpolator(self) -> None:
        cupy_interpolator = mock.Mock(side_effect=AssertionError('CuPy interpolator built in a worker'))
        with mock.patch.object(gridtransform, 'cuLinearNDInterpolator', cupy_interpolator):
            gridtransform.use_scipy_interpolators_in_this_process()
            target, source = self._points()
            interpolator, uses_scipy = gridtransform._build_linear_nd_interpolator(target, source)
        self.assertTrue(uses_scipy)
        self.assertIsNotNone(interpolator)
        cupy_interpolator.assert_not_called()

    def test_the_main_process_still_uses_cupy_when_it_is_available(self) -> None:
        cupy_interpolator = mock.Mock(return_value='device interpolator')
        with mock.patch.object(gridtransform, 'cuLinearNDInterpolator', cupy_interpolator), \
                mock.patch.object(gridtransform, '_scipy_interpolators_only', False), \
                mock.patch.object(gridtransform.cp, 'asarray', side_effect=lambda a, dtype=None: a):
            target, source = self._points()
            interpolator, uses_scipy = gridtransform._build_linear_nd_interpolator(target, source)
        self.assertFalse(uses_scipy)
        self.assertEqual('device interpolator', interpolator)


if __name__ == '__main__':
    unittest.main()
