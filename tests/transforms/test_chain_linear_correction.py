"""Tests for chain-consistent rigid linear targets in transform composition."""

import unittest

import numpy as np

import nornir_imageregistration.transforms
from nornir_imageregistration.transforms import MeshWithRBFFallback
from nornir_imageregistration.transforms.addition import AddTransforms, AddTransformsWithLinearCorrection
from nornir_imageregistration.transforms.converters import ConvertTransformToRigidTransform


def _mesh_from_points(source_points: np.ndarray, target_points: np.ndarray) -> MeshWithRBFFallback:
    point_pairs = np.hstack((target_points, source_points))
    return MeshWithRBFFallback(point_pairs)


class TestChainConsistentRigidLinear(unittest.TestCase):
    def test_composed_rigid_chain_differs_from_mesh_rigid_fit(self) -> None:
        """Composed per-hop rigids should differ from a single rigid fit to the composed mesh."""
        source = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float64)
        # Mild per-link similarity plus extra local warp so the composed mesh is not globally rigid.
        target_ab = source + np.array([[4.0, 1.0], [1.0, 4.0], [-4.0, 1.0], [1.0, -4.0]])
        target_bc = target_ab + np.array([[6.0, -2.0], [-2.0, 6.0], [6.0, 2.0], [2.0, 6.0]])
        local_warp = np.array([[0.0, 0.0], [12.0, -8.0], [-10.0, 9.0], [7.0, 11.0]])
        target_bc = target_bc + local_warp

        mesh_ab = _mesh_from_points(source, target_ab)
        mesh_bc = _mesh_from_points(target_ab, target_bc)

        rigid_ab = ConvertTransformToRigidTransform(mesh_ab)
        rigid_bc = ConvertTransformToRigidTransform(mesh_bc)
        composed_rigid = AddTransforms(rigid_bc, rigid_ab)  # type: ignore[arg-type]

        composed_mesh = AddTransforms(mesh_bc, mesh_ab)  # type: ignore[arg-type]
        mesh_rigid_fit = ConvertTransformToRigidTransform(composed_mesh)  # type: ignore[arg-type]

        composed_corners = composed_rigid.Transform(source)
        mesh_fit_corners = mesh_rigid_fit.Transform(source)
        self.assertGreater(float(np.max(np.abs(composed_corners - mesh_fit_corners))), 0.001)

    def test_explicit_b_to_c_linear_changes_blend_result(self) -> None:
        """An explicit B_To_C_Linear override should change the blended output."""
        source = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float64)
        target_ab = source + np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0], [0.0, -5.0]])
        target_bc = source + np.array([[20.0, 10.0], [10.0, 20.0], [-20.0, -10.0], [-10.0, -20.0]])

        mesh_ab = _mesh_from_points(source, target_ab)
        mesh_bc = _mesh_from_points(target_ab, target_bc)

        blended_default = AddTransformsWithLinearCorrection(
            mesh_bc,
            mesh_ab,  # type: ignore[arg-type]
            min_blend=0.2,
        )
        alternate_rigid_bc = nornir_imageregistration.transforms.RigidTranslation(
            target_offset=np.array([15.0, 10.0], dtype=np.float32))
        blended_override = AddTransformsWithLinearCorrection(
            mesh_bc,
            mesh_ab,  # type: ignore[arg-type]
            min_blend=0.2,
            B_To_C_Linear=alternate_rigid_bc,
        )

        self.assertGreater(
            float(np.max(np.abs(blended_default.TargetPoints - blended_override.TargetPoints))),  # type: ignore[attr-defined]
            0.01,
        )


if __name__ == '__main__':
    unittest.main()
