"""Regression for #114 / C05-B008: grid compose keeps A-space source + C-space target."""
from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration as nir
from nornir_imageregistration.transforms import addition


class TestAddGridTransforms(unittest.TestCase):
    def test_grid_compose_preserves_source_and_composes_target(self):
        shape = (64, 64)
        grid_ab = nir.ITKGridDivision(source_shape=shape, cell_size=(32, 32), grid_dims=(3, 3))
        grid_ab.TargetPoints = grid_ab.SourcePoints + np.array([2.0, 3.0])
        a_to_b = nir.transforms.GridWithRBFFallback(grid_ab)

        grid_bc = nir.ITKGridDivision(source_shape=shape, cell_size=(32, 32), grid_dims=(3, 3))
        grid_bc.TargetPoints = grid_bc.SourcePoints + np.array([10.0, -5.0])
        b_to_c = nir.transforms.GridWithRBFFallback(grid_bc)

        a_to_c = addition._AddGridTransforms(b_to_c, a_to_b)
        expected_target = b_to_c.Transform(a_to_b.TargetPoints)

        self.assertTrue(np.allclose(a_to_c.SourcePoints, a_to_b.SourcePoints))
        self.assertTrue(np.allclose(a_to_c.TargetPoints, expected_target))
        self.assertTrue(np.allclose(a_to_c.points[:, :2], expected_target))
        self.assertTrue(np.allclose(a_to_c.points[:, 2:], a_to_b.SourcePoints))


if __name__ == '__main__':
    unittest.main()
