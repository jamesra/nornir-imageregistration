import numpy
from numpy.typing import NDArray
import scipy
from imageutilities import create_tiny_image

import nornir_imageregistration
from nornir_imageregistration import AlignmentRecord, assemble as assemble, spatial as spatial
import setup_imagetest


class TestTransformROI(setup_imagetest.ImageTestBase):

    def test_identity(self):
        arecord = AlignmentRecord(peak=(0, 0), weight=100, angle=0.0)

        # Shape in numpy is (height, width)
        sourceShape = (2, 6)
        targetShape = sourceShape
        transform = arecord.ToImageTransform(targetShape, sourceShape)

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

    def test_identity_flipped(self):
        """Flip the image upside down"""
        arecord = AlignmentRecord(peak=(0, 0), weight=100, angle=0.0, flipped_ud=True)

        # Shape in numpy is (height, width)
        sourceShape = (2, 6)
        targetShape = sourceShape
        transform = arecord.ToImageTransform(targetShape, sourceShape)

        (fixedpoints, points) = assemble.write_to_target_roi_coords(transform, (0, 0), targetShape)

        self.show_test_image(transform, sourceShape, (numpy.max(points, 0) - numpy.min(points, 0)) + 1,
                             "Flipped up/down transform, top left should be white")
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
        transform = arecord.ToImageTransform(targetShape, sourceShape)

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
        transform = arecord.ToImageTransform(targetShape, sourceShape)

        (fixedpoints, points) = assemble.write_to_target_roi_coords(transform, offset, targetShape, extrapolate=True)

        self.show_test_image(transform, sourceShape, targetShape, "Rotate 180 degrees")

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), 5, delta=0.01)

    def test_Rotate180_odd_offset(self):
        sourceShape = (2, 6)
        targetShape = (2, 9)
        offset = (0, -.5)  # numpy.array(canvasShape) / 2.0
        arecord = AlignmentRecord(peak=offset, weight=100, angle=180.0)
        transform = arecord.ToImageTransform(targetShape, sourceShape)

        (source_coords, target_coords) = assemble.write_to_target_roi_coords(transform, offset, targetShape,
                                                                            extrapolate=True)

        self.show_test_image(transform, sourceShape, targetShape, f"Rotate 180 degrees, offset by {offset}")

        # The target grid spans the full 9-pixel canvas. GetROICoords is an integer
        # meshgrid, so the -0.5 X component of botleft truncates to 0; the half-pixel
        # shift is carried by the transform (the AlignmentRecord peak), not by the
        # write coordinates, which must stay on whole output pixels.
        self.assertAlmostEqual(min(target_coords[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(target_coords[:, spatial.iPoint.Y]), targetShape[0] - 1, delta=0.01)
        self.assertAlmostEqual(min(target_coords[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(target_coords[:, spatial.iPoint.X]), targetShape[1] - 1, delta=0.01)

    def test_Rotate90(self):

        sourceShape = (3, 6)
        targetShape = (6, 3)
        offset = (0, 0)  # numpy.array(targetShape) / 2.0
        arecord = AlignmentRecord(peak=offset, weight=100, angle=90.0)
        transform = arecord.ToImageTransform(targetShape, sourceShape)

        (fixedpoints, points) = assemble.write_to_target_roi_coords(transform, offset, targetShape, extrapolate=True)

        self.show_test_image(transform, sourceShape, targetShape,
                             "Rotate 90 degrees.  An increase in angle should rotate counter-clockwise (RHS)")

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), targetShape[0] - 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), targetShape[1] - 1, delta=0.01)

    def test_Rotate90_square_odd(self):

        sourceShape = (5, 5)
        targetShape = (5, 5)
        offset = (0, 0)  # numpy.array(targetShape) / 2.0
        arecord = AlignmentRecord(peak=offset, weight=100, angle=90.0)
        transform = arecord.ToImageTransform(targetShape, sourceShape)

        (fixedpoints, points) = assemble.write_to_target_roi_coords(transform, offset, targetShape, extrapolate=True)

        self.show_test_image(transform, sourceShape, targetShape,
                             "Rotate 90 degrees.  An increase in angle should rotate counter-clockwise (RHS)")

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), targetShape[0] - 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), targetShape[1] - 1, delta=0.01)

    def test_Rotate90_square_even(self):

        sourceShape = (5, 5)
        targetShape = (5, 5)
        offset = (0, 0)  # numpy.array(targetShape) / 2.0
        arecord = AlignmentRecord(peak=offset, weight=100, angle=90.0)
        transform = arecord.ToImageTransform(targetShape, sourceShape)

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
        transform = arecord.ToImageTransform(targetShape, sourceShape)

        (source_coords, target_coords) = assemble.write_to_target_roi_coords(transform, offset, targetShape,
                                                                            extrapolate=True)

        self.show_test_image(transform, sourceShape, targetShape,
                             f"Rotate 90 degrees\nEven canvas dimensions, offset: {offset}")

        # Unlike the same-canvas tests above, the expanded-canvas cases assert the
        # source-space extent the target grid reads from: that is what exposes the
        # even/odd half-pixel sensitivity this pair was written to pin down. The
        # extent must stay centred on the source pixel centre (1.0, 2.5).
        self.assertAlmostEqual(min(source_coords[:, spatial.iPoint.Y]), -3.5, delta=0.01)
        self.assertAlmostEqual(max(source_coords[:, spatial.iPoint.Y]), 5.5, delta=0.01)
        self.assertAlmostEqual(min(source_coords[:, spatial.iPoint.X]), -1.0, delta=0.01)
        self.assertAlmostEqual(max(source_coords[:, spatial.iPoint.X]), 6.0, delta=0.01)

    def test_Rotate90_expandedCanvas_odd(self):
        sourceShape = numpy.array((3, 6))
        sourceCenter = sourceShape / 2.0
        targetShapeOdd = numpy.array((7,
                                      9))  # Weirdly I've had cases where the test passes or fails based on whether target shape is an even or odd number
        targetCenter = (targetShapeOdd / 2.0)  # Subtract 0.5 so we rotate at the center of the image
        targetShape = targetShapeOdd
        offset = (0, 0)  # numpy.array(targetShape) / 2.0

        arecord = AlignmentRecord(peak=offset, weight=100, angle=90.0)
        transform = arecord.ToImageTransform(targetShape, sourceShape)

        (source_coords, target_coords) = assemble.write_to_target_roi_coords(transform, offset, targetShape,
                                                                            extrapolate=True)

        self.show_test_image(transform, sourceShape, targetShape,
                             f"Rotate 90 degrees\nOdd canvas dimensions, offset: {offset}")

        # See the even-canvas variant above: these bounds are the source-space extent,
        # centred on the source pixel centre (1.0, 2.5).
        self.assertAlmostEqual(min(source_coords[:, spatial.iPoint.Y]), -3, delta=0.01)
        self.assertAlmostEqual(max(source_coords[:, spatial.iPoint.Y]), 5, delta=0.01)
        self.assertAlmostEqual(min(source_coords[:, spatial.iPoint.X]), -0.5, delta=0.01)
        self.assertAlmostEqual(max(source_coords[:, spatial.iPoint.X]), 5.5, delta=0.01)

    def show_test_image(self,
                        transform: nornir_imageregistration.ITransform,
                        image_shape: NDArray,
                        target_space_shape: NDArray,
                        title: str):
        image = create_tiny_image(image_shape)
        transformedImage = assemble.SourceImageToTargetSpace(transform, image, output_area=target_space_shape)
        if transform.angle != 0:
            scipyImage = scipy.ndimage.rotate(image.astype(numpy.float32), -(transform.angle / (2 * numpy.pi)) * 360)

            self.assertTrue(nornir_imageregistration.ShowGrayscale((image, transformedImage, scipyImage), title=title,
                                                                   image_titles=(
                                                                       'input', 'output', 'scipy.ndimage.rotate'),
                                                                   PassFail=True))
        else:
            self.assertTrue(nornir_imageregistration.ShowGrayscale((image, transformedImage), title=title,
                                                                   image_titles=('input', 'output'), PassFail=True))
