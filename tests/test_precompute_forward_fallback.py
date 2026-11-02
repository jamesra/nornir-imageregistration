"""A transform pickled into pool tasks carries its solved RBF fallback weights.

MosaicToVolume pickles one slice-to-volume transform into every tile task. Its RBF
fallback is solved lazily, so each of ~900 tasks per section re-solved the same system
(srv, RC3, 2026-11-02: scipy.linalg.solve at 98% of every worker, ~40 s per section).
"""
from __future__ import annotations

import pickle
import unittest

import nornir_imageregistration
import numpy as np
from nornir_imageregistration import assemble
from nornir_imageregistration.transforms import one_way_rbftransform as owr

_OUTSIDE = np.array([[-50.0, -40.0], [460.0, 30.0], [30.0, 470.0]])


def _grid_transform() -> nornir_imageregistration.transforms.GridWithRBFFallback:
    grid = nornir_imageregistration.ITKGridDivision((400, 400), cell_size=(128, 128), grid_dims=(4, 4))
    grid.TargetPoints = grid.SourcePoints + np.random.default_rng(1).normal(0, 2, grid.SourcePoints.shape)
    return nornir_imageregistration.transforms.GridWithRBFFallback(grid)


def _mesh_transform() -> nornir_imageregistration.transforms.MeshWithRBFFallback:
    rng = np.random.default_rng(2)
    source = rng.uniform(0, 400, (30, 2))
    return nornir_imageregistration.transforms.MeshWithRBFFallback(
        np.hstack((source + rng.normal(0, 2, source.shape), source)))


class TestPrecomputeForwardFallback(unittest.TestCase):

    def setUp(self) -> None:
        self._lib = nornir_imageregistration.GetActiveComputationLib()
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.calls = 0
        real = owr.OneWayRBFWithLinearCorrection.CalculateRBFWeights

        def counting(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        self._real = real
        owr.OneWayRBFWithLinearCorrection.CalculateRBFWeights = staticmethod(counting)

    def tearDown(self) -> None:
        owr.OneWayRBFWithLinearCorrection.CalculateRBFWeights = staticmethod(self._real)
        nornir_imageregistration.SetActiveComputationLib(self._lib)

    def _solves_in_pickled_copy(self, transform) -> int:
        copy = pickle.loads(pickle.dumps(transform))
        before = self.calls
        copy.Transform(_OUTSIDE)
        return self.calls - before

    def test_without_precompute_each_copy_solves(self):
        """Precondition: the lazy solve really happens in each pickled copy."""
        self.assertGreater(self._solves_in_pickled_copy(_grid_transform()), 0)

    def test_precomputed_grid_copy_does_not_solve(self):
        transform = _grid_transform()
        assemble.precompute_forward_fallback(transform)

        self.assertEqual(self._solves_in_pickled_copy(transform), 0)

    def test_precomputed_mesh_copy_does_not_solve(self):
        transform = _mesh_transform()
        assemble.precompute_forward_fallback(transform)

        self.assertEqual(self._solves_in_pickled_copy(transform), 0)

    def test_results_are_unchanged(self):
        lazy, eager = _grid_transform(), _grid_transform()
        assemble.precompute_forward_fallback(eager)

        np.testing.assert_array_equal(pickle.loads(pickle.dumps(lazy)).Transform(_OUTSIDE),
                                      pickle.loads(pickle.dumps(eager)).Transform(_OUTSIDE))

    def test_other_transforms_are_ignored(self):
        assemble.precompute_forward_fallback(nornir_imageregistration.transforms.RigidTranslation((1.0, 2.0)))


if __name__ == '__main__':
    unittest.main()
