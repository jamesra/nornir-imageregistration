"""Regression tests for grid transform fixed-point edit notifications."""

from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import interactive_edit
from nornir_imageregistration.grid_subdivision import ITKGridDivision
from nornir_imageregistration.transforms.gridtransform import GridTransform_GPUComponent


def _sample_grid_transform() -> GridTransform_GPUComponent:
    """GPU grid component with a small identity-like grid."""
    transform = nornir_imageregistration.transforms.Rigid(
        target_offset=(0, 0),
        source_rotation_center=(50, 50),
        angle=0,
    )
    grid_data = ITKGridDivision(
        source_shape=(200, 200),
        cell_size=(100, 100),
        grid_spacing=(100, 100),
        transform=transform,
    )
    return GridTransform_GPUComponent(grid_data)


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


if __name__ == "__main__":
    unittest.main()
