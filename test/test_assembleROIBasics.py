import numpy

import nornir_imageregistration
from nornir_imageregistration import assemble as assemble
import setup_imagetest

#Check if cupy is available, and if it is not import thunks that refer to scipy/numpy
try:
    import cupy as cp 
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp 
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp 


class TestROIBasics(setup_imagetest.ImageTestBase):

    def test_crop_to_fit_coords_all_coords_contained(self):
        image = TestROIBasics.create_gradient_image((8, 8))
        coords = assemble.GetROICoords(botleft=(0, 0), area=(4, 4))
          
        (cropped_image, adjusted_coords, adjusted_coords_mask) = assemble._CropImageToFitCoords(image, coords,
                                                                                                padding=1)
        correct_image = image[0:4, 0:4]

        self.assertTrue(nornir_imageregistration.ShowGrayscale((image, (correct_image, cropped_image)),
                                                               title=f"Cropped image showing bottom left of original image with origin at {numpy.min(coords, 0)}\nPadded by one pixel around edges.",
                                                               PassFail=True))
        
        if self.use_cp():
            coords = coords.get() 
        
        self.assertTrue(numpy.allclose(coords + numpy.array((1, 1)), adjusted_coords),
                        "Coordinates should not change if all coordinates are inside the image")

    def test_crop_to_fit_coords_some_coords_external(self):
        image = TestROIBasics.create_gradient_image((8, 8))
        bottom_left = (-1, -1)
        area = (4, 4)
        coords, cropped_image, adjusted_coords = self.show_cropped_image(image, bottom_left=bottom_left, area=area)

    def test_crop_to_fit_coords_some_coords_external_two(self):
        image = TestROIBasics.create_gradient_image((8, 8))
        bottom_left = (6, 6)
        area = (4, 4)
        coords, cropped_image, adjusted_coords = self.show_cropped_image(image, bottom_left=bottom_left, area=area)

    def test_crop_to_fit_coords_some_coords_external_three(self):
        image = TestROIBasics.create_gradient_image((8, 8))
        bottom_left = (-4, -4)
        area = (8, 8)
        coords, cropped_image, adjusted_coords = self.show_cropped_image(image, bottom_left=bottom_left, area=area)

    def test_crop_to_fit_coords_some_coords_external_four(self):
        image = TestROIBasics.create_gradient_image((8, 8))
        bottom_left = (4, 4)
        area = (8, 8)
        coords, cropped_image, adjusted_coords = self.show_cropped_image(image, bottom_left=bottom_left, area=area)

    def test_crop_to_fit_coords_some_coords_external_no_padding(self):
        image = TestROIBasics.create_gradient_image((8, 8))
        bottom_left = (4, 4)
        area = (8, 8)
        coords, cropped_image, adjusted_coords = self.show_cropped_image(image, bottom_left=bottom_left, area=area,
                                                                         padding=0)

    def test_crop_to_fit_coords_some_coords_external_no_padding_two(self):
        image = TestROIBasics.create_gradient_image((8, 8))
        bottom_left = (2, 2)
        area = (5, 5)
        coords, cropped_image, adjusted_coords = self.show_cropped_image(image, bottom_left=bottom_left, area=area,
                                                                         padding=0)

    def show_cropped_image(self, image, bottom_left, area, padding=1):
        xp = cp.get_array_module(image)
        area = numpy.array(area)
        bottom_left = numpy.array(bottom_left)

        roi = nornir_imageregistration.Rectangle.CreateFromPointAndArea(bottom_left, area)
        coords = assemble.GetROICoords(botleft=bottom_left, area=area)
        self.assertTrue(numpy.allclose(numpy.min(coords, 0), bottom_left))
        self.assertTrue(numpy.allclose((numpy.max(coords, 0) - numpy.min(coords, 0)) + 1, area))

        (cropped_image, adjusted_coords, adjusted_coords_mask) = assemble._CropImageToFitCoords(image, coords,
                                                                                                padding=padding)
        valid_adjusted_coords, valid_coords_mask = nornir_imageregistration.assemble.get_valid_coords(coords,
                                                                                                      image.shape,
                                                                                                      origin=(0, 0),
                                                                                                      area=area)

        self.assertTrue(xp.allclose(adjusted_coords_mask, valid_coords_mask))
        flat_valid_coords = nornir_imageregistration.ravel_index(valid_adjusted_coords, image.shape)
        expected_image = xp.zeros(image.shape, dtype=numpy.float64)
        
        if self.use_cp():
            expected_image = expected_image.ravel()
            image1d = image.ravel()
            expected_image[flat_valid_coords] = image1d[flat_valid_coords]
            expected_image = expected_image.reshape(image.shape)
        else:
            expected_image.flat[flat_valid_coords] = image.flat[flat_valid_coords]

        self.assertTrue(nornir_imageregistration.ShowGrayscale((image, (expected_image, cropped_image)),
                                                               title=f"Cropped version of image padded by {padding} pixel if outside boundaries.\nOrigin at {bottom_left} and area {area}",
                                                               rois=(roi, (None, None)), PassFail=True))
        self.assertTrue(numpy.sum(adjusted_coords_mask), adjusted_coords.shape[0])
        return coords, cropped_image, adjusted_coords
