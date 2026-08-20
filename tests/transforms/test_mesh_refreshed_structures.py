"""Snapshot/install of mesh Delaunay interpolators and RBF must not mutate until apply."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from nornir_imageregistration import interactive_edit
from nornir_imageregistration.transforms.meshwithrbffallback import MeshWithRBFFallback


def _identity_mesh() -> MeshWithRBFFallback:
    points = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 32.0, 0.0, 32.0],
            [32.0, 0.0, 32.0, 0.0],
            [32.0, 32.0, 32.0, 32.0],
        ],
        dtype=np.float64,
    )
    return MeshWithRBFFallback(points)


class TestMeshRefreshedStructures(unittest.TestCase):
    """Off-UI bundle construction must leave the live mesh untouched until apply."""

    def test_build_does_not_mutate_live_structures(self) -> None:
        mesh = _identity_mesh()
        mesh.InitializeDataStructures()
        live_rbf = mesh._ForwardRBFInstance
        live_fixed = mesh._fixedtri
        self.assertIsNotNone(live_rbf)
        self.assertIsNotNone(live_fixed)

        new_target = np.asarray(mesh.TargetPoints[0], dtype=np.float64) + np.array((5.0, -3.0))
        mesh.UpdateTargetPointsByIndex(0, new_target)
        self.assertIsNone(mesh._ForwardRBFInstance)
        self.assertIsNone(mesh._fixedtri)
        self.assertTrue(mesh._continuous_stale)

        bundle = mesh.build_refreshed_continuous()
        self.assertIsNone(mesh._ForwardRBFInstance)
        self.assertIsNone(mesh._fixedtri)
        self.assertIsNot(bundle.forward_rbf, live_rbf)
        self.assertIsNotNone(bundle.fixedtri)

        mesh.apply_refreshed_continuous(bundle)
        self.assertIs(mesh._ForwardRBFInstance, bundle.forward_rbf)
        self.assertIs(mesh._fixedtri, bundle.fixedtri)
        self.assertFalse(mesh._continuous_stale)

    def test_stale_forward_uses_source_delaunay_without_qhull(self) -> None:
        mesh = _identity_mesh()
        _ = mesh.source_space_trianglulation
        new_target = np.asarray(mesh.TargetPoints[0], dtype=np.float64) + np.array((4.0, 2.0))
        interactive_edit.begin()
        try:
            mesh.UpdateTargetPointsByIndex(0, new_target)
            with patch("scipy.spatial.Delaunay", side_effect=AssertionError("Qhull")):
                mapped = mesh.Transform(np.asarray(mesh.SourcePoints), extrapolate=False)
        finally:
            interactive_edit.end()
        np.testing.assert_allclose(mapped[0], new_target, rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(
            mapped[1:], np.asarray(mesh.TargetPoints[1:]), rtol=1e-6, atol=1e-5)
        self.assertIsNone(mesh._ForwardInterpolator)
        self.assertIsNone(mesh._ForwardRBFInstance)
