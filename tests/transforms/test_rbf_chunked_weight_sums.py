"""Regression for #121 / C05-P004: chunked weight sums must not need per-chunk asserts."""
from __future__ import annotations

import inspect
import unittest

import numpy as np

from nornir_imageregistration.transforms.one_way_rbftransform import (
    OneWayRBFWithLinearCorrection,
    OneWayRBFWithLinearCorrection_GPUComponent,
)


class TestChunkedWeightSums(unittest.TestCase):
    def test_chunked_matches_unchunked_host(self):
        rng = np.random.default_rng(0)
        ctrl = rng.uniform(0, 100, size=(12, 2))
        target = ctrl + rng.normal(0, 0.5, size=ctrl.shape)
        rbf = OneWayRBFWithLinearCorrection(ctrl, target)
        _ = rbf.Weights
        points = rng.uniform(0, 100, size=(20_000, 2))
        unchunked = rbf._GetMatrixWeightSums(points, rbf.SourcePoints, MaxChunkSize=points.shape[0] + 1)
        chunked = rbf._GetMatrixWeightSums(points, rbf.SourcePoints, MaxChunkSize=1024)
        np.testing.assert_allclose(chunked[0], unchunked[0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(chunked[1], unchunked[1], rtol=1e-5, atol=1e-5)

    def test_no_device_scalar_assert_in_chunk_loop(self):
        """Indexing MatrixWeightSumX[iStart] in assert forces a GPU→host sync per chunk."""
        for cls in (OneWayRBFWithLinearCorrection, OneWayRBFWithLinearCorrection_GPUComponent):
            src = inspect.getsource(cls._GetMatrixWeightSums)
            self.assertNotIn('MatrixWeightSumX[iStart]', src)
            self.assertNotIn('MatrixWeightSumY[iStart]', src)


if __name__ == '__main__':
    unittest.main()
