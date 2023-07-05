import numpy as np

import nornir_imageregistration
from nornir_imageregistration import assemble as assemble
import setup_imagetest


class TestAssembleImageRegion(setup_imagetest.ImageTestBase):

    def test_write_to_target_image_coord_generation(self):
        """
        Define an ROI on an image in target space we want to fill with interpolated values from coordinates in source space
        :return:
        """
        target_bottom_left = np.array((0, 0))
        target_area = np.array((3, 3))
        # source_image = self.create_gradient_image((9, 9))
        target_coords = assemble.GetROICoords(botleft=target_bottom_left, area=target_area)
        source_to_target_offset = np.array((3.5, 1))
        transform = nornir_imageregistration.transforms.RigidNoRotation(target_offset=source_to_target_offset)

        roi_read_coords, roi_write_coords = assemble.write_to_target_roi_coords(transform, target_bottom_left,
                                                                                target_area)
        self.assertTrue(np.array_equal(target_coords, roi_write_coords))
        read_bottom_left = roi_read_coords.min(0)
        read_area = (roi_read_coords.max(0) - read_bottom_left) + 1
        self.assertTrue(np.array_equal(target_area, read_area))
        self.assertTrue(np.array_equal(read_bottom_left, target_bottom_left - source_to_target_offset))

    def test_write_to_source_image_coord_generation(self):
        """
        Transform uniform coordinates from target_space
        :return:
        """
        source_bottom_left = np.array((0, 0))
        source_area = np.array((3, 3))

        source_coords = assemble.GetROICoords(botleft=source_bottom_left, area=source_area)
        source_to_target_offset = np.array((3.5, 1))
        transform = nornir_imageregistration.transforms.RigidNoRotation(target_offset=source_to_target_offset)

        roi_read_coords, roi_write_coords = assemble.write_to_source_roi_coords(transform, source_bottom_left,
                                                                                source_area)
        self.assertTrue(np.array_equal(source_coords, roi_write_coords))
        read_bottom_left = roi_read_coords.min(0)
        read_area = (roi_read_coords.max(0) - read_bottom_left) + 1
        self.assertTrue(np.array_equal(source_area, read_area))
        self.assertTrue(np.array_equal(read_bottom_left, source_bottom_left + source_to_target_offset))

    def test_grid_division_identity(self):
        """
        Divide an image into 3x3 tile regions and return each cell
        :return:
        """

        source_image = self.create_gradient_image((9, 10))
        target_image = self.create_gradient_image((9, 10))
        source_to_target_offset = np.array((0, 0))
        cell_size = np.array((3, 3))
        grid_dims = np.array((3, 3))
        self.run_grid_division(source_image=source_image,
                               target_image=target_image,
                               source_to_target_offset=source_to_target_offset,
                               cell_size=cell_size,
                               grid_dims=grid_dims)

    def test_grid_division_offset(self):
        """
        Divide an image into 3x3 tile regions and return each cell
        :return:
        """

        source_image = self.create_nested_squares_image((9, 10))
        target_image = self.create_nested_squares_image((9, 10))

        source_to_target_offset = np.array((1, -2))

        cell_size = np.array((3, 3))
        grid_dims = np.array((3, 3))
        self.run_grid_division(source_image=source_image,
                               target_image=target_image,
                               source_to_target_offset=source_to_target_offset,
                               cell_size=cell_size,
                               grid_dims=grid_dims)

    def run_grid_division(self, source_image, target_image, source_to_target_offset, cell_size, grid_dims):
        transform = nornir_imageregistration.transforms.RigidNoRotation(target_offset=source_to_target_offset)

        source_stats = nornir_imageregistration.ImageStats.Create(source_image)
        target_stats = nornir_imageregistration.ImageStats.Create(target_image)

        grid_division = nornir_imageregistration.ITKGridDivision(source_shape=source_image.shape,
                                                                 cell_size=cell_size,
                                                                 grid_dims=grid_dims,
                                                                 transform=transform)

        source_image = nornir_imageregistration.CropImage(source_image, Xo=source_to_target_offset[1],
                                                          Yo=source_to_target_offset[0],
                                                          Width=target_image.shape[1] + np.abs(
                                                              source_to_target_offset[1]),
                                                          Height=target_image.shape[0] + np.abs(
                                                              source_to_target_offset[0]),
                                                          cval=0)

        for source_point in grid_division.SourcePoints:
            # target_image_roi = nornir_imageregistration.assemble.TargetImageToSourceSpace(transform,
            #                                                                               DataToTransform=target_image,
            #                                                                               output_botleft=source_rectangle.BottomLeft,
            #                                                                               output_area=source_rectangle.Size,
            #                                                                               extrapolate=True, cval=np.nan)
            target_roi, source_roi = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIs(
                transform=transform,
                targetImage_param=target_image,
                sourceImage_param=source_image,
                target_image_stats=target_stats,
                source_image_stats=source_stats,
                target_controlpoint=source_point,
                alignmentArea=cell_size)

            target_point = transform.Transform(source_point)
            roi_diff = np.abs(target_roi - source_roi)
            target_roi_rect = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(source_point, cell_size)
            source_roi_rect = nornir_imageregistration.Rectangle.translate(target_roi_rect, -source_to_target_offset)
            self.assertTrue(nornir_imageregistration.ShowGrayscale(
                ((source_image, target_image), (source_roi, roi_diff, target_roi)),
                title=f"Pair of cells extracted at point {target_point} (y, X)\nSource -> target offset {source_to_target_offset}\nROI should match a {cell_size} cell removed from top image.\nArea outside the image should have random values.\nDelta image should be black for non-random pixels",
                image_titles=(("Source", "Target"), ("Source ROI", "ROI Delta", "Target ROI")),
                rois=((source_roi_rect, target_roi_rect), None),
                PassFail=True))
