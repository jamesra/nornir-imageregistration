"""Image warping extrapolates mesh/grid transforms with the nearest hull triangle's affine.

The RBF fallback outside the control-point hull dominated RC2 Grid32 overlays
(2026-11-02); James chose the nearest triangle's affine for warping. Inside the hull
nothing changes; outside, the warp continues linearly from the nearest hull triangle.
"""
from __future__ import annotations

import unittest

import nornir_imageregistration
import numpy as np
from nornir_imageregistration import assemble
from nornir_imageregistration.transforms import hull_extrapolation
from nornir_imageregistration.transforms import one_way_rbftransform as owr

_AFFINE = np.array([[1.02, 0.03], [-0.04, 0.97]])
_OFFSET = np.array([12.0, -7.0])


def _affine_mesh() -> nornir_imageregistration.transforms.MeshWithRBFFallback:
    """Control points from one global affine: target = source @ A + b."""
    source = np.random.default_rng(3).uniform(100, 500, (40, 2))
    target = source @ _AFFINE + _OFFSET
    return nornir_imageregistration.transforms.MeshWithRBFFallback(np.hstack((target, source)))


def _warped_grid() -> nornir_imageregistration.transforms.GridWithRBFFallback:
    grid = nornir_imageregistration.ITKGridDivision((600, 600), cell_size=(128, 128), grid_dims=(5, 5))
    grid.TargetPoints = grid.SourcePoints + np.random.default_rng(4).normal(0, 4, grid.SourcePoints.shape)
    return nornir_imageregistration.transforms.GridWithRBFFallback(grid)


class _NumpyCase(unittest.TestCase):

    def setUp(self) -> None:
        self._lib = nornir_imageregistration.GetActiveComputationLib()
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

    def tearDown(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(self._lib)


class TestHullAffineExtrapolator(_NumpyCase):

    def test_a_global_affine_is_reproduced_far_outside_the_hull(self):
        mesh = _affine_mesh()
        extrapolate = hull_extrapolation.inverse_extrapolator(mesh)
        far = np.array([[-300.0, -200.0], [900.0, 40.0], [50.0, 1200.0], [1000.0, 1000.0]])

        expected_source = (far - _OFFSET) @ np.linalg.inv(_AFFINE)
        # The transform keeps its points in float32, so exact means to ~1e-3 px here.
        np.testing.assert_allclose(extrapolate(far), expected_source, atol=1e-2)

    def test_continuous_at_the_hull_boundary(self):
        """0.01 px either side of each hull edge's midpoint maps to nearly the same place."""
        mesh = _warped_grid()
        hull = scipy_hull_points(mesh)
        starts, ends = hull, np.roll(hull, -1, axis=0)
        mid = (starts + ends) / 2.0
        edge = ends - starts
        inward = np.stack((-edge[:, 1], edge[:, 0]), axis=1)
        inward /= np.linalg.norm(inward, axis=1, keepdims=True)
        if np.mean(np.einsum('ij,ij->i', hull.mean(axis=0) - mid, inward)) < 0:
            inward = -inward
        inner = nornir_imageregistration.EnsureNumpyArray(
            mesh.InverseTransform(mid + 0.01 * inward, extrapolate=False))
        outer = hull_extrapolation.inverse_extrapolator(mesh)(mid - 0.01 * inward)

        self.assertFalse(np.isnan(inner).any())
        # A warped grid's border is not exactly convex, so between it and the Delaunay hull
        # the seam is sub-pixel rather than exact (0.06 px here); a mesh's is exact.
        np.testing.assert_allclose(outer, inner, atol=0.1)

    def test_built_once_per_transform(self):
        mesh = _affine_mesh()
        self.assertIs(hull_extrapolation.inverse_extrapolator(mesh), hull_extrapolation.inverse_extrapolator(mesh))
        self.assertIs(hull_extrapolation.forward_extrapolator(mesh), hull_extrapolation.forward_extrapolator(mesh))
        self.assertIsNot(hull_extrapolation.forward_extrapolator(mesh), hull_extrapolation.inverse_extrapolator(mesh))

    def test_forward_reproduces_a_global_affine_far_outside_the_hull(self):
        mesh = _affine_mesh()
        far = np.array([[-300.0, -200.0], [900.0, 40.0], [50.0, 1200.0], [1000.0, 1000.0]])
        np.testing.assert_allclose(
            hull_extrapolation.forward_extrapolator(mesh)(far), far @ _AFFINE + _OFFSET, atol=1e-2)

    def test_forward_rebuilds_after_points_move(self):
        mesh = _affine_mesh()
        before = hull_extrapolation.forward_extrapolator(mesh)
        mesh.TranslateFixed(np.array([10.0, 0.0]))
        after = hull_extrapolation.forward_extrapolator(mesh)
        self.assertIsNot(before, after)
        far = np.array([[-300.0, -200.0]])
        np.testing.assert_allclose(after(far), before(far) + np.array([10.0, 0.0]), atol=1e-2)


def scipy_hull_points(transform) -> np.ndarray:
    import scipy.spatial
    target = nornir_imageregistration.EnsureNumpyArray(transform.TargetPoints)
    return target[scipy.spatial.ConvexHull(target).vertices]


class TestWarpCoordinates(_NumpyCase):
    """``write_to_target_roi_coords`` is the warping path (assembly, STOS overlays)."""

    def _rbf_solves(self, fn):
        calls = [0]
        real = owr.OneWayRBFWithLinearCorrection.CalculateRBFWeights

        def counting(*args, **kwargs):
            calls[0] += 1
            return real(*args, **kwargs)

        owr.OneWayRBFWithLinearCorrection.CalculateRBFWeights = staticmethod(counting)
        try:
            result = fn()
        finally:
            owr.OneWayRBFWithLinearCorrection.CalculateRBFWeights = staticmethod(real)
        return result, calls[0]

    def _check(self, transform):
        botleft, area = (-100, -100), (800, 800)  # extends well past the hull
        inside_read, inside_write = assemble.write_to_target_roi_coords(transform, botleft, area, extrapolate=False)
        (read, write), solves = self._rbf_solves(
            lambda: assemble.write_to_target_roi_coords(transform, botleft, area, extrapolate=True))
        read, write = nornir_imageregistration.EnsureNumpyArray(read), nornir_imageregistration.EnsureNumpyArray(write)

        self.assertEqual(solves, 0, 'warping must not build the RBF fallback')
        self.assertEqual(write.shape[0], 800 * 800, 'every pixel is mapped')
        self.assertFalse(np.isnan(read).any())
        inside_write = nornir_imageregistration.EnsureNumpyArray(inside_write)
        lookup = {tuple(p): i for i, p in enumerate(write.astype(np.int64).tolist())}
        rows = [lookup[tuple(p)] for p in inside_write.astype(np.int64).tolist()]
        np.testing.assert_array_equal(read[rows], nornir_imageregistration.EnsureNumpyArray(inside_read),
                                      'inside the hull nothing changes')

    def test_mesh(self):
        self._check(_affine_mesh())

    def test_grid(self):
        self._check(_warped_grid())


if __name__ == '__main__':
    unittest.main()
