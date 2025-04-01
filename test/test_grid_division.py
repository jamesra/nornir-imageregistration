import unittest

from hypothesis import given, strategies
import numpy as np

import nornir_imageregistration
from nornir_imageregistration.grid_subdivision import ITKGridDivision, IGrid, CenteredGridDivision
from nornir_imageregistration.views import build_grid_rois


class testITKGridDivision(unittest.TestCase):
    grid_spacing_strategy = strategies.tuples(strategies.integers(min_value=1, max_value=128),
                                              strategies.integers(min_value=1, max_value=128))
    cell_size_strategy = strategies.tuples(strategies.integers(min_value=128, max_value=256),
                                           strategies.integers(min_value=128, max_value=256))
    source_shape_strategy = strategies.tuples(strategies.integers(min_value=1000, max_value=2048),
                                              strategies.integers(min_value=1000, max_value=2048))

    def test_source_target_points_transform(self):
        grid_spacing = np.array((192, 192), dtype=int)
        cell_size = np.array((256, 256), dtype=int)
        transform = nornir_imageregistration.transforms.Rigid(target_offset=(500, 1000),
                                                              source_rotation_center=(150, 300), angle=0)
        grid = ITKGridDivision(source_shape=(1000, 1000), cell_size=cell_size,
                               grid_spacing=grid_spacing, transform=transform)

        self.assertTrue(np.all(grid.grid_spacing <= grid_spacing))

        inverted_target_points = transform.InverseTransform(grid.TargetPoints)

        self.assertTrue(np.allclose(inverted_target_points, grid.SourcePoints))

    @given(grid_spacing=grid_spacing_strategy, cell_size=cell_size_strategy, source_shape=source_shape_strategy)
    def test_grid_spacing(self, grid_spacing: tuple[int, int], cell_size: tuple[int, int],
                          source_shape: tuple[int, int]):
        self.check_grid_spacing(grid_spacing=grid_spacing, cell_size=cell_size, source_shape=source_shape)

    def check_grid_spacing(self, grid_spacing: tuple[int, int], cell_size: tuple[int, int],
                           source_shape: tuple[int, int]):
        """Ensure the grid_spacing is always equal or less than the passed parameter"""
        grid = ITKGridDivision(source_shape=source_shape, cell_size=cell_size,
                               grid_spacing=grid_spacing)
        self.assertTrue(np.all(grid.grid_spacing <= grid_spacing))
        self.assertTrue(np.all(grid.cell_size == cell_size))

        return grid

    def test_grid_spacing_even(self):
        grid_spacing = (100, 66)
        grid = self.check_grid_spacing(grid_spacing=grid_spacing, cell_size=(100, 66), source_shape=(500, 500))

        self.assertTrue(np.all(grid.grid_spacing[0] == grid_spacing[0]),
                        "Grid spacing should not be changed for a perfect fit")

    def test_axis_points(self):
        source_shape = (1000, 1000)
        grid_spacing = (100, 200)
        transform = nornir_imageregistration.transforms.Rigid(target_offset=(500, 1000),
                                                              source_rotation_center=(150, 300), angle=0)
        grid = ITKGridDivision(source_shape=(1000, 1000), cell_size=(200, 200),
                               grid_spacing=(100, 200), transform=transform)

        self.assertEqual(len(grid.axis_points), 2, "There should be two axes")

        self.assertTrue(
            np.allclose(grid.axis_points[0], np.arange(0, source_shape[0] + grid_spacing[0], grid_spacing[0])))
        self.assertTrue(
            np.allclose(grid.axis_points[1], np.arange(0, source_shape[1] + grid_spacing[1], grid_spacing[1])))


class testCenteredGridDivision(unittest.TestCase):
    grid_spacing_strategy = strategies.tuples(strategies.integers(min_value=1, max_value=128),
                                              strategies.integers(min_value=1, max_value=128))
    cell_size_strategy = strategies.tuples(strategies.integers(min_value=128, max_value=256),
                                           strategies.integers(min_value=128, max_value=256))
    source_shape_strategy = strategies.tuples(strategies.integers(min_value=1000, max_value=2048),
                                              strategies.integers(min_value=1000, max_value=2048))

    def test_source_target_points_transform(self):
        grid_spacing = np.array((192, 192), dtype=int)
        cell_size = np.array((256, 256), dtype=int)
        transform = nornir_imageregistration.transforms.Rigid(target_offset=(500, 1000),
                                                              source_rotation_center=(150, 300), angle=0)
        grid = nornir_imageregistration.CenteredGridDivision(source_shape=(1000, 1000), cell_size=cell_size,
                                                             grid_spacing=grid_spacing, transform=transform)

        self.assertTrue(np.all(grid.grid_spacing <= grid_spacing))

        inverted_target_points = transform.InverseTransform(grid.TargetPoints)

        self.assertTrue(np.allclose(inverted_target_points, grid.SourcePoints))

    @given(grid_spacing=grid_spacing_strategy, cell_size=cell_size_strategy, source_shape=source_shape_strategy)
    def test_grid_spacing(self, grid_spacing: tuple[int, int], cell_size: tuple[int, int],
                          source_shape: tuple[int, int]):
        self.check_grid_spacing(grid_spacing=grid_spacing, cell_size=cell_size, source_shape=source_shape)

    def check_grid_spacing(self, grid_spacing: tuple[int, int], cell_size: tuple[int, int],
                           source_shape: tuple[int, int]):
        """Ensure the grid_spacing is always equal or less than the passed parameter"""
        grid = nornir_imageregistration.CenteredGridDivision(source_shape=source_shape, cell_size=cell_size,
                                                             grid_spacing=grid_spacing)
        self.assertTrue(np.all(grid.grid_spacing <= grid_spacing))
        self.assertTrue(np.all(grid.cell_size == cell_size))

        return grid

    def test_grid_spacing_even(self):
        grid_spacing = (128, 128)
        image_size = (512, 512)

        grid = self.check_grid_spacing(grid_spacing=grid_spacing, cell_size=(100, 66), source_shape=(500, 500))


class TestGridDivisionAndMasking(unittest.TestCase):

    def test_masking_simple(self):
        cell_size = np.asarray((128, 128), dtype=int)
        grid_spacing = np.asarray((96, 96), dtype=int)
        image_size = np.asarray((512, 512), dtype=int)

        source_image = np.random.standard_normal(image_size)
        target_image = np.random.standard_normal(image_size)

        source_mask = nornir_imageregistration.overlapmasking.GetOverlapMask(image_size, image_size, image_size,
                                                                             0.5, 0.75)
        target_mask = nornir_imageregistration.overlapmasking.GetOverlapMask(image_size, image_size, image_size,
                                                                             0.25, 0.75)

        transform = nornir_imageregistration.transforms.Rigid(target_offset=(128, 0),
                                                              source_rotation_center=(256, 256), angle=np.pi / 3)

        grid = nornir_imageregistration.CenteredGridDivision(source_shape=image_size,
                                                             cell_size=cell_size,
                                                             grid_spacing=grid_spacing)

        grid.PopulateTargetPoints(transform=transform)

        grid.RemoveCellsUsingSourceImageMask(source_mask, min_unmasked_area=0.25)
        grid.RemoveCellsUsingTargetImageMask(target_mask, min_unmasked_area=0.25)

        max_cells = np.prod(image_size / grid_spacing)

        self.assertTrue(grid.coords.shape[0] > 0, "Not all cells should be removed")
        self.assertTrue(grid.coords.shape[0] < max_cells, "Some cells should be removed")

        source_rois, target_rois = build_grid_rois(grid)

        nornir_imageregistration.ShowGrayscale(input_params=[source_image, source_mask, target_image, target_mask],
                                               title="Masked Grid, Source and Target Images, Rectangles with more than 25% in masked areas should be removed.",
                                               image_titles=["Source", "Source Mask", "Target", "Target Mask"],
                                               rois=[source_rois, source_rois, target_rois, target_rois],
                                               PassFail=True)
        #
        # target_image_data = nornir_imageregistration.ImagePermutationHelper(img=target_image,
        #                                                                     mask=target_mask,
        #                                                                     extrema_mask_size_cuttoff=None,
        #                                                                     dtype=nornir_imageregistration.default_image_dtype())
        #
        # source_image_data = nornir_imageregistration.ImagePermutationHelper(img=source_mask,
        #                                                                     mask=target_mask,
        #                                                                     extrema_mask_size_cuttoff=None,
        #                                                                     dtype=nornir_imageregistration.default_image_dtype())

    #
    # with nornir_imageregistration.settings.GridRefinement.CreateWithPreprocessedImages(
    #         target_img_data=target_image_data,
    #         source_img_data=source_image_data,
    #         num_iterations=10,
    #         cell_size=128, grid_spacing=96,
    #         angles_to_search=None, final_pass_angles=[0],
    #         max_travel_for_finalization=None,
    #         min_alignment_overlap=0.5,
    #         min_unmasked_area=0.49) as settings:


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
