import numpy as np
from numpy._typing import NDArray
import scipy

import nornir_imageregistration
from nornir_imageregistration import ShapeLike, VectorLike

import setup_imagetest

try:
    from . import create_gradient_image, create_nested_squares_image
except ImportError:
    from test import create_gradient_image, create_nested_squares_image


class TestROIExtractionFromGrid(setup_imagetest.ImageTestBase):

    def test_grid_division_identity(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.grid_division_identity()

    def test_grid_division_identity_gpu(self):
        if not nornir_imageregistration.HasCupy():
            return
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        self.grid_division_identity()

    def grid_division_identity(self):
        """
        Divide an image into 3x3 tile regions and return each cell
        :return:
        """

        source_image = create_gradient_image((9, 10))
        target_image = create_gradient_image((9, 10))
        source_to_target_offset = np.array((0, 0))
        cell_size = np.array((3, 3))
        grid_dims = np.array((3, 3))
        transform = nornir_imageregistration.transforms.RigidNoRotation(target_offset=source_to_target_offset)
        self.run_grid_division(source_image=source_image,
                               target_image=target_image,
                               transform=transform,
                               cell_size=cell_size,
                               grid_dims=grid_dims)

    def test_grid_division_offset(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.grid_division_offset()

    def test_grid_division_offset_gpu(self):
        if not nornir_imageregistration.HasCupy():
            return
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        self.grid_division_offset()

    def grid_division_offset(self):
        """
        Divide an image into 3x3 tile regions and return each cell
        :return:
        """

        source_image = create_nested_squares_image((9, 10))
        target_image = create_nested_squares_image((9, 10))

        source_to_target_offset = np.array((1, -2))

        cell_size = np.array((3, 3))
        grid_dims = np.array((3, 3))
        transform = nornir_imageregistration.transforms.RigidNoRotation(target_offset=source_to_target_offset)
        self.run_grid_division(source_image=source_image,
                               target_image=target_image,
                               transform=transform,
                               cell_size=cell_size,
                               grid_dims=grid_dims)

    def test_grid_division_rotated(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.grid_division_rotated()

    def test_grid_division_rotated_gpu(self):
        if not nornir_imageregistration.HasCupy():
            return
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        self.grid_division_rotated()

    def grid_division_rotated(self):
        """
        Divide an image into 3x3 tile regions, rotate the source image 90 degrees, and return each cell
        :return:
        """

        image_shape = (9, 10)
        source_image = create_gradient_image(image_shape)

        angle = np.pi / 2
        source_to_target_offset = (0, 0)
        center_of_rotation = (np.array(image_shape) - 1) / 2
        transform = nornir_imageregistration.transforms.Rigid(target_offset=source_to_target_offset,
                                                              source_rotation_center=center_of_rotation, angle=angle)
        target_image = nornir_imageregistration.assemble.SourceImageToTargetSpace(transform=transform,
                                                                                  DataToTransform=source_image)

        approved = nornir_imageregistration.ShowGrayscale((source_image, target_image),
                                                          title="Bottom image is rotated 90 degrees",
                                                          image_titles=('Source Image', 'Target Image'),
                                                          PassFail=True)
        self.assertTrue(approved)

        source_to_target_offset = np.array((0, 0))

        cell_size = np.array((3, 3))
        grid_dims = np.array((3, 3))
        self.run_grid_division(source_image=source_image,
                               target_image=target_image,
                               transform=transform,
                               cell_size=cell_size,
                               grid_dims=grid_dims)

    def run_grid_division(self,
                          source_image: NDArray[np.floating],
                          target_image: NDArray[np.floating],
                          transform: nornir_imageregistration.IRigidTransform,
                          cell_size: ShapeLike,
                          grid_dims: ShapeLike):

        source_to_target_offset = transform.target_offset
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

        adjusted_transforms = nornir_imageregistration.local_distortion_correction.ApproximateRigidTransformBySourcePoints(
            transform,
            source_points=grid_division.SourcePoints,
            cell_size=cell_size)

        for i, source_point in enumerate(grid_division.SourcePoints):
            # target_image_roi = nornir_imageregistration.assemble.TargetImageToSourceSpace(transform,
            #                                                                               DataToTransform=target_image,
            #                                                                               output_botleft=source_rectangle.BottomLeft,
            #                                                                               output_area=source_rectangle.Size,
            #                                                                               extrapolate=True, cval=np.nan)

            adjusted_transform = adjusted_transforms[i]
            target_roi, source_roi = nornir_imageregistration.local_distortion_correction.BuildAlignmentROIs(
                transform=adjusted_transform,
                targetImage_param=target_image,
                sourceImage_param=source_image,
                target_image_stats=target_stats,
                source_image_stats=source_stats,
                target_controlpoint=source_point,
                alignmentArea=cell_size)

            target_point = adjusted_transform.Transform(source_point)
            roi_diff = np.abs(target_roi - source_roi)
            target_roi_rect = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(target_point, cell_size)
            source_roi_rect = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(source_point, cell_size)
            self.assertTrue(nornir_imageregistration.ShowGrayscale(
                ((source_image, target_image), (source_roi, roi_diff, target_roi)),
                title=f"Pair of cells extracted at point {target_point} (y, X)\nSource -> target offset {source_to_target_offset}\nROI should match a {cell_size} cell removed from top image.\nArea outside the image should have random values.\nDelta image should be black for non-random pixels",
                image_titles=(("Source", "Target"), ("Source ROI", "ROI Delta", "Target ROI")),
                rois=((source_roi_rect, target_roi_rect), None),
                PassFail=True))
