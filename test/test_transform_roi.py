import numpy
import scipy

import nornir_imageregistration
import setup_imagetest
from nornir_imageregistration import AlignmentRecord, assemble as assemble, spatial as spatial


class TestTransformROI(setup_imagetest.ImageTestBase):

    @classmethod
    def create_tiny_image(cls, shape):
        shape = nornir_imageregistration.EnsurePointsAre1DNumpyArray(shape, numpy.int32)
        image = numpy.zeros(shape,
                            dtype=numpy.float32)  # Needs to be float32 for numpy functions used in the test to support it
        for x in range(0, shape[1]):
            for y in range(0, shape[0]):
                color_index = (((x % 4) + (y % 4)) % 4) / 4
                image[y, x] = (color_index * 0.8) + 0.2

        # Make origin bright white
        image[0, 0] = 1.0

        return image

    def test_identity(self):

        arecord = AlignmentRecord(peak=(0, 0), weight=100, angle=0.0)

        # Shape in numpy is (height, width)
        sourceShape = (2, 6)
        targetShape = sourceShape
        transform = arecord.ToTransform(targetShape, sourceShape)

        (fixedpoints, points) = assemble.write_to_target_roi_coords(transform, (0, 0), targetShape)

        self.show_test_image(transform, sourceShape, (numpy.max(points, 0) - numpy.min(points, 0)) + 1,
                             "Identity transform, should be identical")
        # Transform ROI should return coordinates as
        # ([Y1,X1],
        # ([Y2,X2], ...

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), 5, delta=0.01)

    def test_translate(self):

        offset = numpy.array((1, 2), numpy.int32)
        arecord = AlignmentRecord(peak=offset, weight=100, angle=0.0)

        sourceShape = (2, 6)
        targetShape = sourceShape
        transform = arecord.ToTransform(targetShape, sourceShape)

        (read_coords, target_coords) = assemble.write_to_target_roi_coords(transform, offset, targetShape)

        self.show_test_image(transform, sourceShape, targetShape * numpy.array((2)),
                             f"Translate by x:{offset[1]} y:{offset[0]}")

        self.assertAlmostEqual(min(target_coords[:, spatial.iPoint.Y]), 0 + offset[spatial.iPoint.Y], delta=0.01)
        self.assertAlmostEqual(max(target_coords[:, spatial.iPoint.Y]), (sourceShape[0] - 1) + offset[spatial.iPoint.Y],
                               delta=0.01)
        self.assertAlmostEqual(min(target_coords[:, spatial.iPoint.X]), 0 + offset[spatial.iPoint.X], delta=0.01)
        self.assertAlmostEqual(max(target_coords[:, spatial.iPoint.X]), (sourceShape[1] - 1) + offset[spatial.iPoint.X],
                               delta=0.01)

    def test_Rotate180(self):
        sourceShape = (2, 6)
        targetShape = sourceShape
        offset = (0, 0)  # numpy.array(canvasShape) / 2.0
        arecord = AlignmentRecord(peak=offset, weight=100, angle=180.0)
        transform = arecord.ToTransform(targetShape, sourceShape)

        (fixedpoints, points) = assemble.write_to_target_roi_coords(transform, offset, targetShape, extrapolate=True)

        self.show_test_image(transform, sourceShape, targetShape, "Rotate 180 degrees")

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), 5, delta=0.01)

    def test_Rotate180_odd_offset(self):
        sourceShape = (2, 6)
        targetShape = (3, 7)
        offset = (1, 1)  # numpy.array(canvasShape) / 2.0
        arecord = AlignmentRecord(peak=offset, weight=100, angle=180.0)
        transform = arecord.ToTransform(targetShape, sourceShape)

        (fixedpoints, points) = assemble.write_to_target_roi_coords(transform, offset, targetShape, extrapolate=True)

        self.show_test_image(transform, sourceShape, targetShape, "Rotate 180 degrees, offset by 1,1")

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), 5, delta=0.01)

    def test_Rotate90(self):

        sourceShape = (3, 6)
        targetShape = (6, 3)
        offset = (0, 0)  # numpy.array(targetShape) / 2.0
        arecord = AlignmentRecord(peak=offset, weight=100, angle=90.0)
        transform = arecord.ToTransform(targetShape, sourceShape)

        (fixedpoints, points) = assemble.write_to_target_roi_coords(transform, offset, targetShape, extrapolate=True)

        self.show_test_image(transform, sourceShape, targetShape,
                             "Rotate 90 degrees.  An increase in angle should rotate counter-clockwise (RHS)")

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), targetShape[0] - 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), targetShape[1] - 1, delta=0.01)

    def test_Rotate90_square(self):

        sourceShape = (5, 5)
        targetShape = (5, 5)
        offset = (0, 0)  # numpy.array(targetShape) / 2.0
        arecord = AlignmentRecord(peak=offset, weight=100, angle=90.0)
        transform = arecord.ToTransform(targetShape, sourceShape)

        (fixedpoints, points) = assemble.write_to_target_roi_coords(transform, offset, targetShape, extrapolate=True)

        self.show_test_image(transform, sourceShape, targetShape,
                             "Rotate 90 degrees.  An increase in angle should rotate counter-clockwise (RHS)")

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), targetShape[0] - 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), targetShape[1] - 1, delta=0.01)

    def test_Rotate90_expandedCanvas_even(self):
        sourceShape = numpy.array((3, 6))
        sourceCenter = (sourceShape / 2.0)
        targetShapeEven = numpy.array((8,
                                       10))  # Weirdly I've had cases where the test passes or fails based on whether target shape is an even or odd number
        targetCenter = (targetShapeEven / 2.0)  # Subtract 0.5 so we rotate at the center of the image
        targetShape = targetShapeEven
        offset = (0, 0)  # numpy.array(targetShape) / 2.0

        arecord = AlignmentRecord(peak=offset, weight=100, angle=90.0)
        transform = arecord.ToTransform(targetShape, sourceShape)

        (fixedpoints, points) = assemble.write_to_target_roi_coords(transform, offset, targetShape, extrapolate=True)

        self.show_test_image(transform, sourceShape, targetShape,
                             f"Rotate 90 degrees\nEven canvas dimensions, offset: {offset}")

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), sourceShape[1] - 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), 1, delta=0.01)

    def test_Rotate90_expandedCanvas_odd(self):
        sourceShape = numpy.array((3, 6))
        sourceCenter = (sourceShape) / 2.0
        targetShapeOdd = numpy.array((7,
                                      9))  # Weirdly I've had cases where the test passes or fails based on whether target shape is an even or odd number
        targetCenter = (targetShapeOdd / 2.0)  # Subtract 0.5 so we rotate at the center of the image
        targetShape = targetShapeOdd
        offset = (0, 0)  # numpy.array(targetShape) / 2.0

        arecord = AlignmentRecord(peak=offset, weight=100, angle=90.0)
        transform = arecord.ToTransform(targetShape, sourceShape)

        (fixedpoints, points) = assemble.write_to_target_roi_coords(transform, offset, targetShape, extrapolate=True)

        self.show_test_image(transform, sourceShape, targetShape,
                             f"Rotate 90 degrees\nOdd canvas dimensions, offset: {offset}")

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), -3, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), 5, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), -0.5, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), 5.5, delta=0.01)

    def show_test_image(self, transform, image_shape, target_space_shape, title):
        image = TestTransformROI.create_tiny_image(image_shape)
        transformedImage = assemble.SourceImageToTargetSpace(transform, image, output_area=target_space_shape)
        if transform.angle != 0:
            scipyImage = scipy.ndimage.rotate(image, -(transform.angle / (2 * numpy.pi)) * 360)

            self.assertTrue(nornir_imageregistration.ShowGrayscale((image, transformedImage, scipyImage), title=title,
                                                                   image_titles=(
                                                                   'input', 'output', 'scipy.ndimage.rotate'),
                                                                   PassFail=True))
        else:
            self.assertTrue(nornir_imageregistration.ShowGrayscale((image, transformedImage), title=title,
                                                                   image_titles=('input', 'output'), PassFail=True))
