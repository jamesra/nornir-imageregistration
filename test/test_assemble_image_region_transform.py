import numpy as np

try:
    from . import create_gradient_image, create_nested_squares_image
except ImportError:
    from test import create_gradient_image, create_nested_squares_image

import nornir_imageregistration
from nornir_imageregistration import assemble as assemble
import setup_imagetest

# Check if cupy is available, and if it is not import thunks that refer to scipy/numpy
try:
    import cupy as cp
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp


class TestAssembleImageRegionTranslateOnly(setup_imagetest.ImageTestBase):

    def test_write_to_target_image_coord_generation(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.write_to_target_image_coord_generation_translate_only()

    def test_write_to_target_image_coord_generation_gpu(self):
        if not nornir_imageregistration.HasCupy():
            return

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        self.write_to_target_image_coord_generation_translate_only()

    def write_to_target_image_coord_generation_translate_only(self):
        """
        Define an ROI on an image in target space we want to fill with interpolated values from coordinates in source space
        :return:
        """
        xp = nornir_imageregistration.GetComputationModule()
        target_bottom_left = np.array((0, 0))
        target_area = np.array((3, 3))
        # source_image = self.create_gradient_image((9, 9))
        target_coords = assemble.GetROICoords(botleft=target_bottom_left, area=target_area)
        source_to_target_offset = np.array((3.5, 1))
        transform = nornir_imageregistration.transforms.RigidNoRotation(target_offset=source_to_target_offset)

        roi_read_coords, roi_write_coords = assemble.write_to_target_roi_coords(transform,
                                                                                target_bottom_left,
                                                                                target_area)
        self.assertTrue(np.array_equal(target_coords, roi_write_coords))
        read_bottom_left = roi_read_coords.min(0)
        read_area = (roi_read_coords.max(0) - read_bottom_left) + 1

        read_area = read_area.get() if xp == cp else read_area
        read_bottom_left = read_bottom_left.get() if xp == cp else read_bottom_left

        self.assertTrue(np.array_equal(target_area, read_area))
        self.assertTrue(np.array_equal(read_bottom_left, target_bottom_left - source_to_target_offset))

    def test_write_to_source_image_coord_generation(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.write_to_source_image_coord_generation()

    def test_write_to_source_image_coord_generation_gpu(self):
        if not nornir_imageregistration.HasCupy():
            return

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        self.write_to_source_image_coord_generation()

    def write_to_source_image_coord_generation(self):
        """
        Transform uniform coordinates from target_space
        :return:
        """
        xp = nornir_imageregistration.GetComputationModule()
        source_bottom_left = np.array((0, 0))
        source_area = np.array((3, 3))

        source_coords = assemble.GetROICoords(botleft=source_bottom_left, area=source_area)
        source_to_target_offset = np.array((3.5, 1))
        transform = nornir_imageregistration.transforms.RigidNoRotation(target_offset=source_to_target_offset)

        roi_read_coords, roi_write_coords = assemble.write_to_source_roi_coords(transform, source_bottom_left,
                                                                                source_area)
        self.assertTrue(xp.array_equal(source_coords, roi_write_coords))
        read_bottom_left = roi_read_coords.min(0)
        read_area = (roi_read_coords.max(0) - read_bottom_left) + 1

        read_area = read_area.get() if xp is cp else read_area
        read_bottom_left = read_bottom_left.get() if xp is cp else read_bottom_left

        self.assertTrue(xp.array_equal(source_area, read_area))
        self.assertTrue(xp.array_equal(read_bottom_left, source_bottom_left + source_to_target_offset))
