'''
Created on Apr 3, 2013

@author: u0490822
'''
import os
import unittest

import nornir_imageregistration
from nornir_imageregistration import AlignmentRecord
import numpy
from scipy.misc import imsave
from scipy.ndimage import interpolation

import nornir_imageregistration.assemble as assemble
import nornir_imageregistration.core as core
import nornir_imageregistration.spatial as spatial
import nornir_shared.images as images

from . import setup_imagetest


def ShowComparison(*args, **kwargs):
    return core.ShowGrayscale(*args, **kwargs)

class TestTransformROI(setup_imagetest.ImageTestBase):


    def test_identity(self):

        arecord = AlignmentRecord(peak=(0, 0), weight=100, angle=0.0)

        # Shape in numpy is (height, width)
        canvasShape = (2, 6)
        flipCanvasShape = (6, 2)
        transform = arecord.ToTransform(canvasShape, canvasShape)

        (fixedpoints, points) = assemble.DestinationROI_to_SourceROI(transform, (0, 0), canvasShape)

        # Transform ROI should return coordinates as
        # ([Y1,X1],
        # ([Y2,X2], ...

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), 5, delta=0.01)

    def test_translate(self):

        arecord = AlignmentRecord(peak=(1, 2), weight=100, angle=0.0)

        canvasShape = (2, 6)
        flipCanvasShape = (6, 2)
        transform = arecord.ToTransform(canvasShape, canvasShape)

        (fixedpoints, points) = assemble.DestinationROI_to_SourceROI(transform, (1, 2), canvasShape)

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), 5, delta=0.01)

    def test_Rotate180(self):

        arecord = AlignmentRecord(peak=(0, 0), weight=100, angle=180.0)
        canvasShape = (2, 6)
        flipCanvasShape = (6, 2)
        transform = arecord.ToTransform(canvasShape, canvasShape)

        (fixedpoints, points) = assemble.DestinationROI_to_SourceROI(transform, (0, 0), canvasShape)

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), 5, delta=0.01)

    def test_Rotate90(self):

        arecord = AlignmentRecord(peak=(0, 0), weight=100, angle=90.0)
        canvasShape = (2, 6)
        flipCanvasShape = (6, 2)
        transform = arecord.ToTransform(canvasShape, flipCanvasShape)

        (fixedpoints, points) = assemble.DestinationROI_to_SourceROI(transform, (transform.FixedBoundingBox[spatial.iRect.MinY], transform.FixedBoundingBox[spatial.iRect.MinX]), canvasShape)

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), 5, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), 1, delta=0.01)

class TestAssemble(setup_imagetest.ImageTestBase):
    
    def test_TransformImageIdentity(self):
        # Too small to require more than one tile
        self.CallTransformImage(imageDim=10)
        
        # Large enough to require more than one tile
        self.CallTransformImage(imageDim=4097)
        
    def CallTransformImage(self, imageDim):
        
        Height = int(imageDim)
        Width = int(numpy.round(imageDim / 2))
        
        identity_transform = nornir_imageregistration.transforms.triangulation.Triangulation(numpy.array([[0, 0, 0, 0],
                                                                              [Height, 0, Height, 0],
                                                                              [0, Width, 0, Width],
                                                                              [Height, Width, Height, Width]]))
        
        warpedImage = numpy.ones([Height, Width])

        outputImage = assemble.TransformImage(identity_transform, numpy.array([Height, Width]), warpedImage, CropUndefined=False)
        self.assertIsNotNone(outputImage, msg="No image produced by TransformImage")
        self.assertEqual(outputImage.shape[0], Height, msg="Output image height should match")
        self.assertEqual(outputImage.shape[1], Width, msg="Output image width should match")
        
    def test_warpedImageToFixedSpaceTranslate(self):
        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")

        angle = 0
        arecord = AlignmentRecord(peak=(50, 100), weight=100, angle=angle)

        fixedImage = core.LoadImage(WarpedImagePath)
        warpedImage = core.LoadImage(WarpedImagePath)

        transform = arecord.ToTransform(fixedImage.shape, warpedImage.shape)

        transformedImage = assemble.WarpedImageToFixedSpace(transform, fixedImage.shape, warpedImage)
        imsave("C:\\Temp\\17Translate.png", transformedImage)

        #rotatedWarped = interpolation.rotate(warpedImage.astype(numpy.float32), angle=angle)
#
        self.assertTrue(ShowComparison([fixedImage, transformedImage], title="Image should be translated +100x,+50y but not rotated.", PassFail=True))
        return

        # delta = fixedImage[1:64, 1:64] - transformedImage
        # self.assertTrue((delta < 0.01).all())


    def test_warpedImageToFixedSpaceRotateTransform(self):
        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")

        angle = 30
        arecord = AlignmentRecord(peak=(0, 0), weight=100, angle=angle)

        fixedImage = core.LoadImage(WarpedImagePath)
        warpedImage = core.LoadImage(WarpedImagePath)

        transform = arecord.ToTransform(fixedImage.shape, warpedImage.shape)

        transformedImage = assemble.WarpedImageToFixedSpace(transform, fixedImage.shape, warpedImage)
        imsave("C:\\Temp\\17Rotate.png", transformedImage)

        rotatedWarped = interpolation.rotate(warpedImage.astype(numpy.float32), angle=angle)
#
        self.assertTrue(ShowComparison([fixedImage, rotatedWarped, transformedImage],title="Rotate transform should match scipy.interpolate.rotate result", PassFail=True))

        # delta = fixedImage[512:544, 512:544] - rotatedWarped
        # self.assertTrue((delta < 0.01).all())

    def test_warpedImageToFixedSpaceIdentityTransform(self):
        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")


        arecord = AlignmentRecord(peak=(0, 0), weight=100, angle=0.0)

        fixedImage = core.LoadImage(WarpedImagePath)
        warpedImage = core.LoadImage(WarpedImagePath)

        transform = arecord.ToTransform(fixedImage.shape, warpedImage.shape)

        transformedImage = assemble.WarpedImageToFixedSpace(transform, fixedImage.shape, warpedImage, (0, 0), (64, 64))
        # imsave("C:\\Temp\\17.png", transformedImage)

        delta = fixedImage[0:64, 0:64] - transformedImage

        # core.ShowGrayscale([fixedImage[0:64, 0:64], transformedImage, delta])
        self.assertTrue((delta < 0.01).all())


    def test_warpedImageToFixedSpace(self):
        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        arecord = AlignmentRecord(peak=(22, -4), weight=100, angle=-132.0)

        fixedImage = core.LoadImage(FixedImagePath)
        warpedImage = core.LoadImage(WarpedImagePath)

        transform = arecord.ToTransform(fixedImage.shape, warpedImage.shape)

        transformedImage = assemble.WarpedImageToFixedSpace(transform, fixedImage.shape, warpedImage)
        imsave(os.path.join(self.VolumeDir, "test_warpedImageToFixedSpace.png"), transformedImage)

class TestStosFixedMovingAssemble(setup_imagetest.ImageTestBase):
    '''Runs assemble on the same fixed.png, moving.png images using different transform files'''

    def setUp(self):
        super(TestStosFixedMovingAssemble, self).setUp()

        self.WarpedImagePath = os.path.join(self.ImportedDataPath, "Moving.png")
        self.assertTrue(os.path.exists(self.WarpedImagePath), "Missing test input")

        self.FixedImagePath = os.path.join(self.ImportedDataPath, "Fixed.png")
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")

    def RunStosAssemble(self, stosFullPath):
        OutputPath = os.path.join(self.VolumeDir, "test_StosAssemble.png");

        warpedImage = assemble.TransformStos(stosFullPath, OutputPath, self.FixedImagePath, self.WarpedImagePath)
        self.assertIsNotNone(warpedImage)

        self.assertTrue(os.path.exists(OutputPath), "RegisteredImage does not exist")

        self.assertEquals(core.GetImageSize(self.FixedImagePath)[0], core.GetImageSize(OutputPath)[0])
        self.assertEquals(core.GetImageSize(self.FixedImagePath)[1], core.GetImageSize(OutputPath)[1])

    def test_GridStosAssemble(self):
        stosFullPath = os.path.join(self.ImportedDataPath, "..", "Transforms", "FixedMoving_Grid.stos")
        self.RunStosAssemble(stosFullPath)

    def test_MeshStosAssemble(self):
        stosFullPath = os.path.join(self.ImportedDataPath, "..", "Transforms", "FixedMoving_Mesh.stos")
        self.RunStosAssemble(stosFullPath)
         


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_warpedImageToFixedSpace']
    unittest.main()
