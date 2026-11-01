"""Topology-preserving grid inverse after TargetPoints edits (no full retriangulation)."""

from __future__ import annotations

import unittest

import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

import nornir_imageregistration
from nornir_imageregistration.grid_subdivision import ITKGridDivision
from nornir_imageregistration.transforms.gridtransform import (
    GridTransform_GPUComponent,
    _FixedSimplicesLinearND,
    delaunay_topology_still_valid,
)


def _sample_grid_transform() -> GridTransform_GPUComponent:
    """Small GPU grid with room for a topology-preserving nudge."""
    transform = nornir_imageregistration.transforms.Rigid(
        target_offset=(0, 0),
        source_rotation_center=(50, 50),
        angle=0,
    )
    return GridTransform_GPUComponent(ITKGridDivision(
        source_shape=(200, 200),
        cell_size=(50, 50),
        grid_spacing=(50, 50),
        transform=transform,
    ))


class TestDelaunayTopologyCertificate(unittest.TestCase):
    """Lawson / orientation certificate used to skip retriangulation."""

    def test_uniform_translate_keeps_topology(self) -> None:
        pts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 1.0]], dtype=np.float64)
        tri = Delaunay(pts)
        moved = pts + 0.05
        self.assertTrue(delaunay_topology_still_valid(moved, tri.simplices, tri.neighbors))

    def test_crossing_move_rejects_topology(self) -> None:
        pts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 1.0]], dtype=np.float64)
        tri = Delaunay(pts)
        moved = pts.copy()
        moved[4] = [3.5, 1.0]
        self.assertFalse(delaunay_topology_still_valid(moved, tri.simplices, tri.neighbors))


@pytest.mark.skipif(not nornir_imageregistration.HasCupy(), reason='CuPy required')
class TestGridInverseTopologyPreserve(unittest.TestCase):
    """Spacebar-sized TargetPoints edits should not re-enter cupyx Delaunay."""

    def setUp(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.cupy)

    def test_small_edit_reuses_simplices_for_inverse(self) -> None:
        grid = _sample_grid_transform()
        _ = grid.InverseInterpolator
        self.assertIsNotNone(grid._fixedtri)
        cached = grid._fixedtri
        simplices_before = np.array(cached.simplices, copy=True)

        targets = nornir_imageregistration.EnsureNumpyArray(grid.TargetPoints).copy()
        sources = nornir_imageregistration.EnsureNumpyArray(grid.SourcePoints).copy()
        targets = targets + 0.01
        grid.UpdateTargetPointsByIndex(np.arange(targets.shape[0]), targets)

        self.assertIs(grid._fixedtri, cached)
        self.assertTrue(np.array_equal(grid._fixedtri.simplices, simplices_before))

        _ = grid.InverseInterpolator
        self.assertIsInstance(grid._InverseInterpolator, _FixedSimplicesLinearND)
        self.assertTrue(grid._scipy_inverse_interp)

        actual = nornir_imageregistration.EnsureNumpyArray(grid.InverseTransform(targets))
        expected = LinearNDInterpolator(Delaunay(targets), sources)(targets)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)
        np.testing.assert_allclose(actual, sources, rtol=0, atol=1e-5)

    def test_topology_breaking_edit_rebuilds_scipy_cache(self) -> None:
        grid = _sample_grid_transform()
        _ = grid.InverseInterpolator
        cached = grid._fixedtri
        targets = nornir_imageregistration.EnsureNumpyArray(grid.TargetPoints).copy()
        # Pull one interior vertex far outside the mesh so edges must flip.
        mid = targets.shape[0] // 2
        targets[mid] = targets.max(axis=0) + np.array([500.0, 500.0], dtype=np.float64)
        grid.UpdateTargetPointsByIndex(np.arange(targets.shape[0]), targets)

        self.assertIsNotNone(grid._fixedtri)
        self.assertIsNot(grid._fixedtri, cached)


if __name__ == '__main__':
    unittest.main()
