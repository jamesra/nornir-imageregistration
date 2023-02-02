import unittest
import hypothesis
from hypothesis import given, strategies
import numpy as np
import nornir_imageregistration

from nornir_imageregistration.grid_subdivision import ITKGridDivision, CenteredGridDivision

class testITKGridDivision(unittest.TestCase):

    grid_spacing_strategy=strategies.tuples(strategies.integers(min_value=1, max_value=128), strategies.integers(min_value=1, max_value=128))
    cell_size_strategy=strategies.tuples(strategies.integers(min_value=128, max_value=256), strategies.integers(min_value=128, max_value=256))
    source_shape_strategy=strategies.tuples(strategies.integers(min_value=1000, max_value=2048), strategies.integers(min_value=1000, max_value=2048))

    def test_source_target_points_transform(self):

        grid_spacing = np.array((192, 192), dtype=int)
        cell_size = np.array((256, 256), dtype=int)
        transform = nornir_imageregistration.transforms.Rigid(target_offset=(500,1000), source_rotation_center=(150,300), angle=0)
        grid = ITKGridDivision(source_shape=(1000, 1000), cell_size=cell_size,
                               grid_spacing=grid_spacing, transform=transform)

        self.assertTrue(np.alltrue(grid.grid_spacing <= grid_spacing))

        inverted_target_points = transform.InverseTransform(grid.TargetPoints)

        self.assertTrue(np.allclose(inverted_target_points, grid.SourcePoints))

    @given(grid_spacing=grid_spacing_strategy, cell_size=cell_size_strategy, source_shape=source_shape_strategy)
    def test_grid_spacing(self, grid_spacing: tuple[int, int], cell_size: tuple[int, int], source_shape: tuple[int, int]):
        self.check_grid_spacing(grid_spacing=grid_spacing, cell_size=cell_size, source_shape=source_shape)

    def check_grid_spacing(self, grid_spacing: tuple[int, int], cell_size: tuple[int, int], source_shape: tuple[int, int]):
        """Ensure the grid_spacing is always equal or less than the passed parameter"""
        grid = ITKGridDivision(source_shape=source_shape, cell_size=cell_size,
                               grid_spacing=grid_spacing)
        self.assertTrue(np.alltrue(grid.grid_spacing <= grid_spacing))
        self.assertTrue(np.alltrue(grid.cell_size == cell_size))

        return grid

    def test_grid_spacing_even(self):
        grid_spacing = (100, 66)
        grid = self.check_grid_spacing(grid_spacing=grid_spacing, cell_size=(100, 66), source_shape=(500, 500))

        self.assertTrue(np.alltrue(grid.grid_spacing[0] == grid_spacing[0]), "Grid spacing should not be changed for a perfect fit")

    def test_axis_points(self):
        source_shape = (1000, 1000)
        grid_spacing = (100, 200)
        transform = nornir_imageregistration.transforms.Rigid(target_offset=(500, 1000),
                                                              source_rotation_center=(150, 300), angle=0)
        grid = ITKGridDivision(source_shape=(1000, 1000), cell_size=(200, 200),
                               grid_spacing=(100, 200), transform=transform)

        self.assertEqual(len(grid.axis_points), 2, "There should be two axes")

        self.assertTrue(np.allclose(grid.axis_points[0], np.arange(0, source_shape[0]+grid_spacing[0], grid_spacing[0])))
        self.assertTrue(np.allclose(grid.axis_points[1], np.arange(0, source_shape[1]+grid_spacing[1], grid_spacing[1])))



if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()