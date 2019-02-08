'''
Created on Mar 21, 2013

@author: u0490822

Rotation tests should have a positive angle result in a clockwise rotation
Corner points should be (BotLeft, BotRight, TopLeft, TopRight)

Raw Data::

  2---3
  |   |
  |   |
  0---1
  
Rotated 90 Degrees::

  0---2
  |   |
  |   |
  1---3
  



'''
import unittest
import os

import nornir_imageregistration 
from nornir_imageregistration.files.stosfile import StosFile
from scipy import pi
import test.setup_imagetest
import numpy as np


from . import TransformCheck, ForwardTransformCheck, NearestFixedCheck, NearestWarpedCheck
 


# ##An alignment record records how a warped image should be translated and rotated to be
# ##positioned over a fixed image.  For this reason if we map 0,0 from the warped image it
# ##should return the -peak in the alignment record
class TestAlignmentRecord(unittest.TestCase):

    def testIdentity(self):
        record = nornir_imageregistration.AlignmentRecord((0, 0), 100, 0)
        self.assertEqual(round(record.rangle, 3), 0.0, "Degrees angle not converting to radians")

        # Get the corners for a 10,10  image rotated 90 degrees
        predictedArray = np.array([[0, 0],
                                   [0, 9],
                                   [9, 0],
                                   [9, 9]])
        # predictedArray[:, [0, 1]] = predictedArray[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points

        Corners = record.GetTransformedCornerPoints([10, 10])
        self.assertTrue((Corners == predictedArray).all())

        transform = record.ToTransform([10, 10], [10, 10])
        TransformCheck(self, transform, [[4.5, 4.5]], [[4.5, 4.5]])
        TransformCheck(self, transform, [[0, 0]], [[0, 0]])

    def testIdentityFlipped(self):
        record = nornir_imageregistration.AlignmentRecord((0, 0), 100, 0, True)
        self.assertEqual(round(record.rangle, 3), 0.0, "Degrees angle not converting to radians")

        # Get the corners for a 10,10  image rotated 90 degrees
        predictedArray = np.array([[9, 0],
                                   [9, 9],
                                   [0, 0],
                                   [0, 9]])
        # predictedArray[:, [0, 1]] = predictedArray[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points

        Corners = record.GetTransformedCornerPoints([10, 10])
        self.assertTrue((Corners == predictedArray).all())

        transform = record.ToTransform([10, 10], [10, 10])
        TransformCheck(self, transform, [[4.5, 4.5]], [[4.5, 4.5]])
        TransformCheck(self, transform, [[0, 0]], [[9, 0]])

    def testRotation(self):
        record = nornir_imageregistration.AlignmentRecord((0, 0), 100, 90)
        self.assertEqual(round(record.rangle, 3), round(pi / 2.0, 3), "Degrees angle not converting to radians")

        # Get the corners for a 10,10  image rotated 90 degrees
        predictedArray = np.array([[9, 0],
                                   [0, 0],
                                   [9, 9],
                                   [0, 9]])
        # predictedArray[:, [0, 1]] = predictedArray[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points

        Corners = record.GetTransformedCornerPoints([10, 10])
        self.assertTrue((Corners == predictedArray).all())

        transform = record.ToTransform([10, 10], [10, 10])
        TransformCheck(self, transform, [[4.5, 4.5]], [[4.5, 4.5]])
        TransformCheck(self, transform, [[0, 0]], [[9, 0]])

    def testRotationFlipped(self):
        record = nornir_imageregistration.AlignmentRecord((0, 0), 100, 90, True)
        self.assertEqual(round(record.rangle, 3), round(pi / 2.0, 3), "Degrees angle not converting to radians")

        # Get the corners for a 10,10  image rotated 90 degrees
        predictedArray = np.array([[9, 9],
                                   [0, 9],
                                   [9, 0],
                                   [0, 0]])
        # predictedArray[:, [0, 1]] = predictedArray[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points

        Corners = record.GetTransformedCornerPoints([10, 10])
        self.assertTrue((Corners == predictedArray).all())

        transform = record.ToTransform([10, 10], [10, 10])
        TransformCheck(self, transform, [[4.5, 4.5]], [[4.5, 4.5]])
        TransformCheck(self, transform, [[0, 0]], [[9, 9]])

    def testTranslate(self):
        peak = [3, 1]
        record = nornir_imageregistration.AlignmentRecord(peak, 100, 0)

        # Get the corners for a 10,10  translated 3x, 1y
        predictedArray = np.array([[3, 1],
                                   [3, 10],
                                   [12, 1],
                                   [12, 10]])
        # predictedArray[:, [0, 1]] = predictedArray[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points


        Corners = record.GetTransformedCornerPoints([10, 10])

        self.assertTrue((Corners == predictedArray).all())

        transform = record.ToTransform([10, 10], [10, 10])
        TransformCheck(self, transform, [[0, 0]], [peak])
        TransformCheck(self, transform, [[4.5, 4.5]], [[7.5, 5.5]])

    def testAlignmentRecord(self):
        record = nornir_imageregistration.AlignmentRecord((2.5, 0), 100, 90)
        self.assertEqual(round(record.rangle, 3), round(pi / 2.0, 3), "Degrees angle not converting to radians")

        # Get the corners for a 10,10  image rotated 90 degrees
        predictedArray = np.array([[11.5, 0],
                                   [2.5, 0],
                                   [11.5, 9],
                                   [2.5, 9]])
        # predictedArray[:, [0, 1]] = predictedArray[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points

        Corners = record.GetTransformedCornerPoints([10, 10])
        self.assertTrue((Corners == predictedArray).all())

        record = nornir_imageregistration.AlignmentRecord((-2.5, 2.5), 100, 90)
        self.assertEqual(round(record.rangle, 3), round(pi / 2.0, 3), "Degrees angle not converting to radians")

        # Get the corners for a 10,10  image rotated 90 degrees
        predictedArray = np.array([[6.5, 2.5],
                                   [-2.5, 2.5],
                                   [6.5, 11.5],
                                   [-2.5, 11.5]])

        # predictedArray[:, [0, 1]] = predictedArray[:, [1, 0]]  # Swapped when GetTransformedCornerPoints switched to Y,X points

        Corners = record.GetTransformedCornerPoints([10, 10])
        self.assertTrue((Corners == predictedArray).all())

    def testAlignmentTransformSizeMismatch(self):
        '''An alignment record where the fixed and warped images are differenct sizes'''

        record = nornir_imageregistration.AlignmentRecord((0, 0), 100, 0)

        transform = record.ToTransform([100, 100], [10, 10])

        # OK, we should be able to map points
        TransformCheck(self, transform, [[2.5, 2.5]], [[47.5, 47.5]])
        TransformCheck(self, transform, [[7.5, 7.5]], [[52.5, 52.5]])

        transform = record.ToTransform([100, 100], [10, 50])

        # OK, we should be able to map points
        TransformCheck(self, transform, [[2.5, 2.5]], [[47.5, 27.5]])
        TransformCheck(self, transform, [[7.5, 7.5]], [[52.5, 32.5]])

    def testAlignmentTransformSizeMismatchWithRotation(self):
        record = nornir_imageregistration.AlignmentRecord((0, 0), 100, 90)
        self.assertEqual(round(record.rangle, 3), round(pi / 2.0, 3), "Degrees angle not converting to radians")

        transform = record.ToTransform([100, 100], [10, 10])

        # OK, we should be able to map points
        TransformCheck(self, transform, [[2.5, 2.5]], [[51.5, 47.5]])
        TransformCheck(self, transform, [[7.5, 7.5]], [[46.5, 52.5]])

    def testAlignmentTransformTranslate(self):
        record = nornir_imageregistration.AlignmentRecord((1, 1), 100, 0)

        transform = record.ToTransform([10, 10], [10, 10])

        # OK, we should be able to map points
        TransformCheck(self, transform, [[4.5, 4.5]], [[5.5, 5.5]])


class TestIO(test.setup_imagetest.ImageTestBase):

    def TestReadWriteTransform(self):
        '''A simple test of a transform which maps points from a 10,10 image to a 100,100 without translation or rotation'''
        WarpedImagePath = os.path.join(self.ImportedDataPath, "10x10.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImagePath = os.path.join(self.ImportedDataPath, "1000x100.png")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        FixedSize = (100, 1000)
        WarpedSize = (10, 10)

        arecord = nornir_imageregistration.AlignmentRecord(peak=(0, 0), weight=100, angle=0)
        alignmentTransform = arecord.ToTransform(FixedSize, WarpedSize)

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

        self.assertTrue((alignmentTransform.points == loadedTransform.points).all(), "Transform different after save/load")

        TransformCheck(self, loadedTransform, [[0, 0]], [[45, 495]])
        TransformCheck(self, alignmentTransform, [[9, 9]], [[54, 504]])


    def TestTranslateReadWriteAlignment(self):

        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        peak = (20, 5)
        arecord = nornir_imageregistration.AlignmentRecord(peak, weight=100, angle=0)

        FixedSize = nornir_imageregistration.GetImageSize(FixedImagePath)
        WarpedSize = nornir_imageregistration.GetImageSize(WarpedImagePath)

        alignmentTransform = arecord.ToTransform(FixedSize, WarpedSize)

        TransformCheck(self, alignmentTransform, [[(WarpedSize[0] / 2.0), (WarpedSize[1] / 2.0)]], [[(FixedSize[0] / 2.0) + peak[0], (FixedSize[1] / 2.0) + peak[1]]])

        # OK, try to save the stos file and reload it.  Make sure the transforms match
        savedstosObj = arecord.ToStos(FixedImagePath, WarpedImagePath, PixelSpacing=1)
        self.assertIsNotNone(savedstosObj)
        stosfilepath = os.path.join(self.VolumeDir, '17-18_brute.stos')
        savedstosObj.Save(stosfilepath)

        loadedStosObj = StosFile.Load(stosfilepath)
        self.assertIsNotNone(loadedStosObj)

        loadedTransform = nornir_imageregistration.transforms.LoadTransform(loadedStosObj.Transform)
        self.assertIsNotNone(loadedTransform)

        self.assertTrue((alignmentTransform.points == loadedTransform.points).all(), "Transform different after save/load")

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
#        alignmentTransform = arecord.ToTransform(FixedSize, WarpedSize)
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
