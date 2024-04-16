'''
Created on Mar 21, 2013

@author: u0490822

Rotation tests should have a positive angle result in a counter-clockwise rotation.
That is the behavior consistent with numpy and a right handed system.
Corner points should be (BotLeft, BotRight, TopLeft, TopRight)

Raw Data::

  2---3
  |   |
  |   |
  0---1
  
Rotated 90 Degrees::

  3---1
  |   |
  |   |
  2---0


'''
import os
import unittest

import numpy as np
from scipy import pi

import nornir_imageregistration
from nornir_imageregistration.files.stosfile import StosFile
import setup_imagetest

try:
    from transforms.data import TranslateRotateTransformPoints, TranslateRotateFlippedTransformPoints, \
        RotateTransformPoints
    from transforms.checks import TransformCheck
except ImportError:
    from test.transforms.data import TranslateRotateTransformPoints, TranslateRotateFlippedTransformPoints, \
        RotateTransformPoints
    from test.transforms.checks import TransformCheck


# ##An alignment record records how a warped image should be translated and rotated to be
# ##positioned over a fixed image.  For this reason if we map 0,0 from the warped image it
# ##should return the -peak in the alignment record
class TestAlignmentRecord(unittest.TestCase):

    def testIdentity(self):
        xp = nornir_imageregistration.GetComputationModule()

        record = nornir_imageregistration.AlignmentRecord((0, 0), 100, 0)
        self.assertEqual(round(record.rangle, 3), 0.0, "Degrees angle not converting to radians")

        # Get the corners for a 10,10  image rotated 90 degrees
        predictedArray = np.array([[0, 0],
                                   [0, 10],
                                   [10, 0],
                                   [10, 10]])
        # predictedArray[:, [0, 1]] = predictedArray[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points

        Corners = record.GetTransformedCornerPoints(np.array((10, 10), int))
        np.testing.assert_allclose(Corners.get() if nornir_imageregistration.UsingCupy() else Corners,
                                   predictedArray)

        transform = record.ToImageTransform([10, 10], [10, 10])
        TransformCheck(self, transform, [[4.5, 4.5]], [[4.5, 4.5]])
        TransformCheck(self, transform, [[0, 0]], [[0, 0]])

    def testIdentityFlipped(self):
        xp = nornir_imageregistration.GetComputationModule()
        record = nornir_imageregistration.AlignmentRecord((0, 0), 100, 0, True)
        self.assertEqual(round(record.rangle, 3), 0.0, "Degrees angle not converting to radians")

        # Get the corners for a 10,10  image flipped over the X axis (up/down flip)
        predictedArray = np.array([[10, 0],
                                   [10, 10],
                                   [0, 0],
                                   [0, 10]])
        # predictedArray[:, [0, 1]] = predictedArray[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points

        Corners = record.GetTransformedCornerPoints(np.array((10, 10), int))
        np.testing.assert_allclose(Corners.get() if nornir_imageregistration.UsingCupy() else Corners,
                                   predictedArray)

        transform = record.ToImageTransform([10, 10], [10, 10])
        TransformCheck(self, transform, [[4.5, 4.5]], [[4.5, 4.5]])  # Transform the very center of the image
        TransformCheck(self, transform, [[5, 4.5]], [[4, 4.5]])  # Transform one half of a pixel up from center
        TransformCheck(self, transform, [[0, 0]], [[9, 0]])

    def testRotation(self):
        xp = nornir_imageregistration.GetComputationModule()
        record = nornir_imageregistration.AlignmentRecord((0, 0), 100, 90)
        self.assertEqual(round(record.rangle, 3), round(pi / 2.0, 3), "Degrees angle not converting to radians")

        # Get the corners for a 10,10  image rotated 90 degrees.
        # Angles increase in clockwise order if the origin is at the bottom left of a display and values increase going up and to the right.
        ydim, xdim = 10, 10

        # Input array from GetTransformedCornerPoints, verify before trusting if you come back to this years later:
        # corners = np.array([[0, 0],
        #                 [0, xmax],
        #                 [ymax, 0],
        #                 [ymax, xmax]])

        predictedArray = np.array([[0, xdim],
                                   [ydim, xdim],
                                   [0, 0],
                                   [ydim, 0]])
        # predictedArray[:, [0, 1]] = predictedArray[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points

        Corners = record.GetTransformedCornerPoints(np.array((ydim, xdim), int))
        self.assertTrue(np.isclose(Corners.get() if nornir_imageregistration.UsingCupy() else Corners,
                                   predictedArray).all())

        transform = record.ToImageTransform([ydim, xdim], [ydim,
                                                           xdim])  # The center of this image would be 4.5, 4.5 since indexing starts at 0 and ends at 9
        center = (ydim - 1) / 2.0, (xdim - 1) / 2.0  # The center, assuming this is an image, is 4.5, 4.5
        TransformCheck(self, transform, [center], [center])
        TransformCheck(self, transform, [[0, 0]], [[0, 9]])

        transform = record.ToSpatialTransform([0, 0], [0, 0])
        sourcePoints = RotateTransformPoints[:, 2:]
        targetPoints = RotateTransformPoints[:, 0:2]

        TransformCheck(self, transform, sourcePoints, targetPoints)

    def testRotationTranslate(self):
        xp = nornir_imageregistration.GetComputationModule()
        translate = np.array((1, 2))
        record = nornir_imageregistration.AlignmentRecord(translate, 100, 90)
        self.assertEqual(round(record.rangle, 3), round(pi / 2.0, 3), "Degrees angle not converting to radians")

        # Get the corners for a 10,10  image rotated 90 degrees.
        # Angles increase in clockwise order if the origin is at the bottom left of a display and values increase going up and to the right.
        ydim, xdim = 10, 10

        # Input array from GetTransformedCornerPoints, verify before trusting if you come back to this years later:
        # corners = np.array([[0, 0],
        #                 [0, xmax],
        #                 [ymax, 0],
        #                 [ymax, xmax]])

        predictedArray = np.array([[0, xdim],
                                   [ydim, xdim],
                                   [0, 0],
                                   [ydim, 0]])
        predictedArray += translate
        # predictedArray[:, [0, 1]] = predictedArray[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points

        Corners = record.GetTransformedCornerPoints(np.array((ydim, xdim), int))
        np.testing.assert_allclose(Corners.get() if nornir_imageregistration.UsingCupy() else Corners,
                                   predictedArray)

        transform = record.ToImageTransform([ydim, xdim], [ydim,
                                                           xdim])  # The center of this image would be 4.5, 4.5 since indexing starts at 0 and ends at 9
        center = (ydim - 1) / 2.0, (xdim - 1) / 2.0  # The center, assuming this is an image, is 4.5, 4.5
        TransformCheck(self, transform, [center], [np.array(center) + translate])
        TransformCheck(self, transform, [(0, 0)], np.array([[0, 9]]) + translate)

        transform = record.ToSpatialTransform([0, 0], [0, 0])
        sourcePoints = TranslateRotateTransformPoints[:, 2:]
        targetPoints = TranslateRotateTransformPoints[:, 0:2]

        TransformCheck(self, transform, sourcePoints, targetPoints)

    def testRotationFlipped(self):
        xp = nornir_imageregistration.GetComputationModule()
        record = nornir_imageregistration.AlignmentRecord((0, 0), 100, 90, True)

        # Get the corners for a 10,10  image rotated 90 degrees
        ydim, xdim = 10, 10

        # Input array from GetTransformedCornerPoints, verify before trusting if you come back to this years later:
        # corners = np.array([[0, 0],
        #                 [0, xmax],
        #                 [ymax, 0],
        #                 [ymax, xmax]])

        predictedArray = np.array([[ydim, 0],
                                   [0, 0],
                                   [ydim, xdim],
                                   [0, xdim]])

        predictedArray = np.array([[10, 10],
                                   [0, 10],
                                   [10, 0],
                                   [0, 0]])
        # predictedArray[:, [0, 1]] = predictedArray[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points

        Corners = record.GetTransformedCornerPoints(np.array((10, 10), int))
        np.testing.assert_allclose(Corners.get() if nornir_imageregistration.UsingCupy() else Corners,
                                   predictedArray)

        transform = record.ToSpatialTransform([10, 10], [10, 10])
        TransformCheck(self, transform, [[5, 5]], [[5, 5]])
        TransformCheck(self, transform, [[0, 0]], [[10, 10]])

    def test_TranslateRotateFlippedTransformPoints(self):
        xp = nornir_imageregistration.GetComputationModule()
        record = nornir_imageregistration.AlignmentRecord((1, 2), 100, 90, True)
        transform = record.ToSpatialTransform([0, 0], [0, 0])
        sourcePoints = TranslateRotateFlippedTransformPoints[:, 2:]
        targetPoints = TranslateRotateFlippedTransformPoints[:, 0:2]

        TransformCheck(self, transform, sourcePoints, targetPoints)

    def testTranslate(self):
        xp = nornir_imageregistration.GetComputationModule()
        peak = [3, 1]
        record = nornir_imageregistration.AlignmentRecord(peak, 100, 0)

        # Get the corners for a 10,10  translated 3x, 1y
        predictedArray = np.array([[3, 1],
                                   [3, 11],
                                   [13, 1],
                                   [13, 11]])
        # predictedArray[:, [0, 1]] = predictedArray[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points

        Corners = record.GetTransformedCornerPoints(np.array((10, 10), int))
        np.testing.assert_allclose(Corners.get() if nornir_imageregistration.UsingCupy() else Corners,
                                   predictedArray)

        transform = record.ToImageTransform([10, 10], [10, 10])
        TransformCheck(self, transform, [[0, 0]], [peak])
        TransformCheck(self, transform, [[5, 5]], [[8, 6]])

    def testAlignmentRecord(self):
        xp = nornir_imageregistration.GetComputationModule()
        record = nornir_imageregistration.AlignmentRecord((2.5, 0), 100, 90)
        self.assertEqual(round(record.rangle, 3), round(pi / 2.0, 3), "Degrees angle not converting to radians")

        # Get the corners for a 10,10  image rotated 90 degrees
        predictedArray = np.array([[2.5, 10],
                                   [12.5, 10],
                                   [2.5, 0],
                                   [12.5, 0]
                                   ])
        # predictedArray[:, [0, 1]] = predictedArray[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points

        Corners = record.GetTransformedCornerPoints(np.array((10, 10), int))
        np.testing.assert_allclose(Corners.get() if nornir_imageregistration.UsingCupy() else Corners,
                                   predictedArray)

        record = nornir_imageregistration.AlignmentRecord((-2.5, 2.5), 100, -90)
        self.assertEqual(round(record.rangle, 3), -round(pi / 2.0, 3), "Degrees angle not converting to radians")

        # Get the corners for a 10,10  image rotated -90 degrees
        predictedArray = np.array([[7.5, 2.5],
                                   [-2.5, 2.5],
                                   [7.5, 12.5],
                                   [-2.5, 12.5]])

        # predictedArray[:, [0, 1]] = predictedArray[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points

        Corners = record.GetTransformedCornerPoints(np.array((10, 10), int))
        self.assertTrue(np.isclose(Corners, predictedArray).all())

    def testAlignmentTransformSizeMismatch(self):
        '''An alignment record where the fixed and warped images are differenct sizes.  No scaling occurs'''

        record = nornir_imageregistration.AlignmentRecord((0, 0), 100, 0)

        transform = record.ToImageTransform([100, 1000], [10, 10])

        # OK, we should be able to map points
        TransformCheck(self, transform, [[2.5, 2.5]], [[47.5, 497.5]])
        TransformCheck(self, transform, [[7.5, 7.5]], [[52.5, 502.5]])

        transform = record.ToImageTransform([100, 1000], [10, 50])

        # OK, we should be able to map points
        TransformCheck(self, transform, [[2.5, 2.5]], [[47.5, 477.5]])
        TransformCheck(self, transform, [[7.5, 7.5]], [[52.5, 482.5]])

    def testAlignmentTransformSizeMismatchWithRotation(self):
        record = nornir_imageregistration.AlignmentRecord((0, 0), 100, 90)
        self.assertEqual(round(record.rangle, 3), round(pi / 2.0, 3), "Degrees angle not converting to radians")

        transform = record.ToSpatialTransform([100, 100], [10, 10])

        # OK, we should be able to map points
        TransformCheck(self, transform, [[2.5, 2.5]], [[47.5, 52.5]])
        TransformCheck(self, transform, [[7.5, 7.5]], [[52.5, 47.5]])

    def testAlignmentTransformTranslate(self):
        record = nornir_imageregistration.AlignmentRecord((1, 1), 100, 0)

        transform = record.ToImageTransform([10, 10], [10, 10])

        # OK, we should be able to map points
        TransformCheck(self, transform, [[4.5, 4.5]], [[5.5, 5.5]])


class TestIO(setup_imagetest.ImageTestBase):

    def testReadWriteTransformSimple(self):
        '''A simple test of a transform which maps points from a 10,10 image to a 100,100 without translation or rotation'''
        WarpedImagePath = os.path.join(self.ImportedDataPath, "10x10.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImagePath = os.path.join(self.ImportedDataPath, "10x10.png")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        FixedSize = (10, 10)
        WarpedSize = (10, 10)

        arecord = nornir_imageregistration.AlignmentRecord(peak=(0, 0), weight=100, angle=0)
        alignmentTransform = arecord.ToImageTransform(FixedSize, WarpedSize)

        self.assertEqual(FixedSize[0], nornir_imageregistration.GetImageSize(FixedImagePath)[0])
        self.assertEqual(FixedSize[1], nornir_imageregistration.GetImageSize(FixedImagePath)[1])
        self.assertEqual(WarpedSize[0], nornir_imageregistration.GetImageSize(WarpedImagePath)[0])
        self.assertEqual(WarpedSize[1], nornir_imageregistration.GetImageSize(WarpedImagePath)[1])

        TransformCheck(self, alignmentTransform, [[0, 0]], [[0, 0]])
        TransformCheck(self, alignmentTransform, [[9, 9]], [[9, 9]])
        TransformCheck(self, alignmentTransform, [[0, 9]], [[0, 9]])
        TransformCheck(self, alignmentTransform, [[9, 0]], [[9, 0]])

        # OK, try to save the stos file and reload it.  Make sure the transforms match
        savedstosObj = arecord.ToStos(FixedImagePath, WarpedImagePath, PixelSpacing=1)
        self.assertIsNotNone(savedstosObj)
        stosfilepath = os.path.join(self.VolumeDir, 'TestRWScaleOnly.stos')
        savedstosObj.Save(stosfilepath)

        loadedStosObj = StosFile.Load(stosfilepath)
        self.assertIsNotNone(loadedStosObj)

        loadedTransform = nornir_imageregistration.transforms.LoadTransform(loadedStosObj.Transform)
        self.assertIsNotNone(loadedTransform)

        if hasattr(alignmentTransform, 'points'):
            self.assertTrue((alignmentTransform.points == loadedTransform.points).all(),
                            "Transform different after save/load")

        TransformCheck(self, loadedTransform, [[0, 0]], [[0, 0]])
        TransformCheck(self, loadedTransform, [[9, 9]], [[9, 9]])
        TransformCheck(self, loadedTransform, [[0, 9]], [[0, 9]])
        TransformCheck(self, loadedTransform, [[9, 0]], [[9, 0]])

    def testReadWriteTransformSizeMismatch(self):
        '''A simple test of a transform which maps points from a 10,10 image to a 100,100 without translation or rotation'''
        WarpedImagePath = os.path.join(self.ImportedDataPath, "10x10.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImagePath = os.path.join(self.ImportedDataPath, "1000x100.png")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        FixedSize = (100, 1000)
        WarpedSize = (10, 10)

        arecord = nornir_imageregistration.AlignmentRecord(peak=(0, 0), weight=100, angle=0)
        alignmentTransform = arecord.ToImageTransform(FixedSize, WarpedSize)

        self.assertEqual(FixedSize[0], nornir_imageregistration.GetImageSize(FixedImagePath)[0])
        self.assertEqual(FixedSize[1], nornir_imageregistration.GetImageSize(FixedImagePath)[1])
        self.assertEqual(WarpedSize[0], nornir_imageregistration.GetImageSize(WarpedImagePath)[0])
        self.assertEqual(WarpedSize[1], nornir_imageregistration.GetImageSize(WarpedImagePath)[1])

        TransformCheck(self, alignmentTransform, [[0, 0]], [[45, 495]])
        TransformCheck(self, alignmentTransform, [[9, 9]], [[54, 504]])

        # OK, try to save the stos file and reload it.  Make sure the transforms match
        savedstosObj = arecord.ToStos(FixedImagePath, WarpedImagePath, PixelSpacing=1)
        self.assertIsNotNone(savedstosObj)
        stosfilepath = os.path.join(self.VolumeDir, 'TestRWScaleOnly.stos')
        savedstosObj.Save(stosfilepath)

        loadedStosObj = StosFile.Load(stosfilepath)
        self.assertIsNotNone(loadedStosObj)

        loadedTransform = nornir_imageregistration.transforms.LoadTransform(loadedStosObj.Transform)
        self.assertIsNotNone(loadedTransform)

        if hasattr(alignmentTransform, 'points'):
            self.assertTrue((alignmentTransform.points == loadedTransform.points).all(),
                            "Transform different after save/load")

        TransformCheck(self, loadedTransform, [[0, 0]], [[45, 495]])
        TransformCheck(self, alignmentTransform, [[9, 9]], [[54, 504]])

    def testTranslateReadWriteAlignment(self):

        WarpedImagePath = os.path.join(self.ImportedDataPath,
                                       "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImagePath = os.path.join(self.ImportedDataPath,
                                      "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        peak = (20, 5)
        arecord = nornir_imageregistration.AlignmentRecord(peak, weight=100, angle=0)

        FixedSize = nornir_imageregistration.GetImageSize(FixedImagePath)
        WarpedSize = nornir_imageregistration.GetImageSize(WarpedImagePath)

        alignmentTransform = arecord.ToImageTransform(FixedSize, WarpedSize)

        TransformCheck(self, alignmentTransform, [[(WarpedSize[0] / 2.0), (WarpedSize[1] / 2.0)]],
                       [[(FixedSize[0] / 2.0) + peak[0], (FixedSize[1] / 2.0) + peak[1]]])

        # OK, try to save the stos file and reload it.  Make sure the transforms match
        savedstosObj = arecord.ToStos(FixedImagePath, WarpedImagePath, PixelSpacing=1)
        self.assertIsNotNone(savedstosObj)
        stosfilepath = os.path.join(self.VolumeDir, '17-18_brute.stos')
        savedstosObj.Save(stosfilepath)

        loadedStosObj = StosFile.Load(stosfilepath)
        self.assertIsNotNone(loadedStosObj)

        loadedTransform = nornir_imageregistration.transforms.LoadTransform(loadedStosObj.Transform)
        self.assertIsNotNone(loadedTransform)

        if hasattr(alignmentTransform, 'points'):
            self.assertTrue((alignmentTransform.points == loadedTransform.points).all(),
                            "Transform different after save/load")


#    def TestReadWriteAlignment(self):
#
#        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
#        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
#        FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
#        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")
#
#        peak = (-4,22)
#        arecord = AlignmentRecord(peak, weight = 100, angle = 132.0)
#
#        # OK, try to save the stos file and reload it.  Make sure the transforms match
#        savedstosObj = arecord.ToStos(FixedImagePath, WarpedImagePath, PixelSpacing = 1)
#        self.assertIsNotNone(savedstosObj)
#
#        FixedSize = Utils.Images.GetImageSize(FixedImagePath)
#        WarpedSize = Utils.Images.GetImageSize(WarpedImagePath)
#
#        alignmentTransform = arecord.ToImageTransform(FixedSize, WarpedSize)
#
#        TransformCheck(self, alignmentTransform, [[(WarpedSize[0] / 2.0), (WarpedSize[1] / 2.0)]], [[(FixedSize[0] / 2.0) + peak[0], (FixedSize[1] / 2.0) + peak[1]]])
#
#        stosfilepath = os.path.join(self.VolumeDir, '17-18_brute.stos')
#
#        savedstosObj.Save(stosfilepath)
#
#        loadedStosObj = IrTools.IO.stosfile.StosFile.Load(stosfilepath)
#        self.assertIsNotNone(loadedStosObj)
#
#        loadedTransform = IrTools.Transforms.factory.LoadTransform(loadedStosObj.Transform)
#        self.assertIsNotNone(loadedTransform)
#
#        self.assertTrue((alignmentTransform.points == loadedTransform.points).all(), "Transform different after save/load")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
