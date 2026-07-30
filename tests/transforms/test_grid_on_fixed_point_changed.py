"""Regression tests for grid transform fixed-point edit notifications."""

from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import interactive_edit
from nornir_imageregistration.grid_subdivision import ITKGridDivision
from nornir_imageregistration.transforms.gridtransform import GridTransform_GPUComponent
from nornir_imageregistration.transforms.gridwithrbffallback import GridWithRBFFallback


def _sample_grid_data() -> ITKGridDivision:
    """Small identity-like grid subdivision."""
    transform = nornir_imageregistration.transforms.Rigid(
        target_offset=(0, 0),
        source_rotation_center=(50, 50),
        angle=0,
    )
    return ITKGridDivision(
        source_shape=(200, 200),
        cell_size=(100, 100),
        grid_spacing=(100, 100),
        transform=transform,
    )


def _sample_grid_transform() -> GridTransform_GPUComponent:
    """GPU grid component with a small identity-like grid."""
    return GridTransform_GPUComponent(_sample_grid_data())


class TestGridOnFixedPointChanged(unittest.TestCase):
    """UpdateTargetPointsByIndex during interactive edit must not raise."""

    def test_gpu_component_update_target_during_interactive_edit(self) -> None:
        grid = _sample_grid_transform()
        interactive_edit.begin()
        try:
            point = grid.TargetPoints[0].copy()
            point[0] += 1.0
            grid.UpdateTargetPointsByIndex(0, point)
        finally:
            interactive_edit.end()

    def test_grid_with_rbf_accepts_numpy_target_point(self) -> None:
        """Pyre MovePoint passes NumPy points into CuPy continuous RBF buffers."""
        model = GridWithRBFFallback(_sample_grid_data())
        before = np.asarray(model.TargetPoints[0], dtype=np.float64)
        # Host ndarray (not CuPy) — the failure mode from composite CP drag.
        point = np.asarray(before + np.array([3.0, 5.0], dtype=np.float64), dtype=np.float64)
        model.UpdateTargetPointsByIndex(0, point)
        after = np.asarray(model.TargetPoints[0], dtype=np.float64)
        np.testing.assert_allclose(after, point, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
