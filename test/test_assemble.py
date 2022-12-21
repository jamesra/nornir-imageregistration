'''
Created on Apr 3, 2013

@author: u0490822
'''
import os
import unittest
import numpy
import scipy

import nornir_imageregistration
from nornir_imageregistration import AlignmentRecord

import nornir_imageregistration.assemble as assemble 
import nornir_imageregistration.spatial as spatial

import setup_imagetest


def ShowComparison(*args, **kwargs):
    return nornir_imageregistration.ShowGrayscale(*args, **kwargs)


class TestTransformROI(setup_imagetest.ImageTestBase):

    @classmethod
    def create_tiny_image(cls, shape):
        shape = nornir_imageregistration.EnsurePointsAre1DNumpyArray(shape, numpy.int32)
        image = numpy.zeros(shape, dtype=numpy.float32)
        for x in range(0,shape[1]):
            for y in range(0,shape[0]):
                color_index = (((x % 4) + (y % 4)) % 4) / 4
                image[y,x] = (color_index * 0.8) + 0.2
        
        return image

    def test_identity(self):

        arecord = AlignmentRecord(peak=(0, 0), weight=100, angle=0.0)

        # Shape in numpy is (height, width)
        sourceShape = (2, 6)
        targetShape = sourceShape 
        transform = arecord.ToTransform(targetShape, sourceShape)

        (fixedpoints, points) = assemble.write_to_source_roi_coords(transform, (0, 0), targetShape)

        self.show_test_image(transform, sourceShape, (numpy.max(points,0) - numpy.min(points,0)) + 1, "Identity transform, should be identical")
        # Transform ROI should return coordinates as
        # ([Y1,X1],
        # ([Y2,X2], ...

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), 5, delta=0.01)

    def test_translate(self):

        offset = numpy.array((1,2),numpy.int32)
        arecord = AlignmentRecord(peak=offset, weight=100, angle=0.0)

        sourceShape = (2, 6)
        targetShape = sourceShape
        transform = arecord.ToTransform(targetShape, sourceShape)

        (fixedpoints, points) = assemble.write_to_source_roi_coords(transform, offset, targetShape)
        
        self.show_test_image(transform, sourceShape, targetShape * numpy.array((2)), f"Translate by x:{offset[1]} y:{offset[0]}")

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0 + offset[spatial.iPoint.Y], delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), (sourceShape[0] - 1) + offset[spatial.iPoint.Y], delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0 + offset[spatial.iPoint.X], delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), (sourceShape[1] - 1) + offset[spatial.iPoint.X], delta=0.01)

    def test_Rotate180(self): 
        sourceShape = (2, 6)
        targetShape = sourceShape
        offset = (0,0)#numpy.array(canvasShape) / 2.0
        arecord = AlignmentRecord(peak=offset, weight=100, angle=180.0)
        transform = arecord.ToTransform(targetShape, sourceShape)

        (fixedpoints, points) = assemble.write_to_source_roi_coords(transform, offset, targetShape, extrapolate=True)
        
        self.show_test_image(transform, sourceShape, targetShape, "Rotate 180 degrees")

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), 5, delta=0.01)
        
    def test_Rotate180_odd_offset(self): 
        sourceShape = (2, 6)
        targetShape = (3, 7)
        offset = (1,1)#numpy.array(canvasShape) / 2.0
        arecord = AlignmentRecord(peak=offset, weight=100, angle=180.0)
        transform = arecord.ToTransform(targetShape, sourceShape)

        (fixedpoints, points) = assemble.write_to_source_roi_coords(transform, offset, targetShape, extrapolate=True)
        
        self.show_test_image(transform, sourceShape, targetShape, "Rotate 180 degrees, offset by 1,1")

        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), 5, delta=0.01)

    def test_Rotate90(self): 
        
        sourceShape = (3, 6)
        targetShape = (6, 3)
        offset =  (0,0)#numpy.array(targetShape) / 2.0
        arecord = AlignmentRecord(peak=offset, weight=100, angle=90.0)
        transform = arecord.ToTransform(targetShape, sourceShape)
        
        (fixedpoints, points) = assemble.write_to_source_roi_coords(transform, offset, targetShape, extrapolate=True)
        
        self.show_test_image(transform, sourceShape,  targetShape,"Rotate 90 degrees")
          
        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), targetShape[0] - 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), targetShape[1] - 1, delta=0.01)
        
    def test_Rotate90_expandedCanvas_even(self):
        sourceShape = numpy.array((3, 6))
        sourceCenter = (sourceShape / 2.0)
        targetShapeEven = numpy.array((8, 10)) #Weirdly I've had cases where the test passes or fails based on whether target shape is an even or odd number
        targetCenter = (targetShapeEven / 2.0)  # Subtract 0.5 so we rotate at the center of the image
        targetShape = targetShapeEven
        offset =  (0,0)#numpy.array(targetShape) / 2.0 
        
        arecord = AlignmentRecord(peak=offset, weight=100, angle=90.0)
        transform = arecord.ToTransform(targetShape, sourceShape)
        
        (fixedpoints, points) = assemble.write_to_source_roi_coords(transform, offset, targetShape, extrapolate=True)
        
        self.show_test_image(transform, sourceShape,  targetShape, f"Rotate 90 degrees\nEven canvas dimensions, offset: {offset}")
          
        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), sourceShape[1] - 1, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), 0, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), 1, delta=0.01)
    
    def test_Rotate90_expandedCanvas_odd(self):
        sourceShape = numpy.array((3, 6))
        sourceCenter = (sourceShape) / 2.0
        targetShapeOdd = numpy.array((7, 9)) #Weirdly I've had cases where the test passes or fails based on whether target shape is an even or odd number
        targetCenter = (targetShapeOdd / 2.0)  # Subtract 0.5 so we rotate at the center of the image
        targetShape=targetShapeOdd
        offset =  (0,0)#numpy.array(targetShape) / 2.0 
        
        arecord = AlignmentRecord(peak=offset, weight=100, angle=90.0)
        transform = arecord.ToTransform(targetShape, sourceShape)
        
        (fixedpoints, points) = assemble.write_to_source_roi_coords(transform, offset, targetShape, extrapolate=True)
        
        self.show_test_image(transform, sourceShape,  targetShape, f"Rotate 90 degrees\nOdd canvas dimensions, offset: {offset}")
          
        self.assertAlmostEqual(min(points[:, spatial.iPoint.Y]), -3, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.Y]), 5, delta=0.01)
        self.assertAlmostEqual(min(points[:, spatial.iPoint.X]), -0.5, delta=0.01)
        self.assertAlmostEqual(max(points[:, spatial.iPoint.X]), 5.5, delta=0.01)
         
    def show_test_image(self, transform, image_shape, target_space_shape, title):
        image = TestTransformROI.create_tiny_image(image_shape)
        transformedImage = assemble.SourceImageToTargetSpace(transform, image)
        
        self.assertTrue(nornir_imageregistration.ShowGrayscale((image, transformedImage), title=title, image_titles=('input', 'output'), PassFail=True))
        

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

        fixedImage = nornir_imageregistration.LoadImage(WarpedImagePath)
        warpedImage = nornir_imageregistration.LoadImage(WarpedImagePath)

        transform = arecord.ToTransform(fixedImage.shape, warpedImage.shape)

        transformedImage = assemble.WarpedImageToFixedSpace(transform, warpedImage)
        nornir_imageregistration.SaveImage("C:\\Temp\\17Translate.png", transformedImage, bpp=8)

        #rotatedWarped = interpolation.rotate(warpedImage.astype(numpy.float32), angle=angle)
#
        self.assertTrue(ShowComparison([fixedImage, transformedImage], title="Image should be translated +100x,+50y but not rotated.", PassFail=True, image_titles=('Original', 'Translated')))
        return

        # delta = fixedImage[1:64, 1:64] - transformedImage
        # self.assertTrue((delta < 0.01).all())


    def test_warpedImageToFixedSpaceRotateTransform(self):
        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")

        angle = 30
        arecord = AlignmentRecord(peak=(0, 0), weight=100, angle=angle)

        fixedImage = nornir_imageregistration.LoadImage(WarpedImagePath)
        warpedImage = nornir_imageregistration.LoadImage(WarpedImagePath)

        transform = arecord.ToTransform(fixedImage.shape, warpedImage.shape)

        transformedImage = assemble.WarpedImageToFixedSpace(transform, warpedImage)
        nornir_imageregistration.SaveImage("C:\\Temp\\17Rotate.png", transformedImage, bpp=8)

        rotatedWarped = scipy.ndimage.rotate(warpedImage.astype(numpy.float32), angle=angle, reshape=False)
#
        self.assertTrue(ShowComparison([fixedImage, rotatedWarped, transformedImage], title="Rotate transform should match scipy.interpolate.rotate result", PassFail=True, image_titles=('Target', 'Scipy', 'My Transform')))

        # delta = fixedImage[512:544, 512:544] - rotatedWarped
        # self.assertTrue((delta < 0.01).all())

    def test_warpedImageToFixedSpaceIdentityTransform(self):
        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")


        arecord = AlignmentRecord(peak=(0, 0), weight=100, angle=0.0)

        fixedImage = nornir_imageregistration.LoadImage(WarpedImagePath)
        warpedImage = nornir_imageregistration.LoadImage(WarpedImagePath)

        transform = arecord.ToTransform(fixedImage.shape, warpedImage.shape)

        transformedImage = assemble.SourceImageToTargetSpace(transform, warpedImage, (0, 0), (64, 64))
        # nornir_imageregistration.SaveImage("C:\\Temp\\17.png", transformedImage)

        delta = fixedImage[0:64, 0:64] - transformedImage

        # nornir_imageregistration.ShowGrayscale([fixedImage[0:64, 0:64], transformedImage, delta])
        self.assertTrue((delta < 0.01).all())


    def test_warpedImageToFixedSpace(self):
        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        arecord = AlignmentRecord(peak=(22, -4), weight=100, angle=-132.0)

        fixedImage = nornir_imageregistration.LoadImage(FixedImagePath)
        warpedImage = nornir_imageregistration.LoadImage(WarpedImagePath)

        transform = arecord.ToTransform(fixedImage.shape, warpedImage.shape)

        transformedImage = assemble.WarpedImageToFixedSpace(transform, warpedImage)
        nornir_imageregistration.SaveImage(os.path.join(self.VolumeDir, "test_warpedImageToFixedSpace.png"), transformedImage, bpp=8)

class TestStosFixedMovingAssemble(setup_imagetest.ImageTestBase):
    '''Runs assemble on the same fixed.png, moving.png images using different transform files'''

    def setUp(self):
        super(TestStosFixedMovingAssemble, self).setUp()

        self.WarpedImagePath = os.path.join(self.ImportedDataPath, "Moving.png")
        self.assertTrue(os.path.exists(self.WarpedImagePath), "Missing test input")

        self.FixedImagePath = os.path.join(self.ImportedDataPath, "Fixed.png")
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")

    def RunStosAssemble(self, stosFullPath):
        OutputPath = os.path.join(self.VolumeDir, "test_StosAssemble.png")

        warpedImage = assemble.TransformStos(stosFullPath, OutputPath, self.FixedImagePath, self.WarpedImagePath)
        self.assertIsNotNone(warpedImage)

        self.assertTrue(os.path.exists(OutputPath), "RegisteredImage does not exist")

        self.assertEquals(nornir_imageregistration.GetImageSize(self.FixedImagePath)[0], nornir_imageregistration.GetImageSize(OutputPath)[0])
        self.assertEquals(nornir_imageregistration.GetImageSize(self.FixedImagePath)[1], nornir_imageregistration.GetImageSize(OutputPath)[1])

    def test_GridStosAssemble(self):
        stosFullPath = os.path.join(self.ImportedDataPath, "..", "Transforms", "FixedMoving_Grid.stos")
        self.RunStosAssemble(stosFullPath)

    def test_MeshStosAssemble(self):
        stosFullPath = os.path.join(self.ImportedDataPath, "..", "Transforms", "FixedMoving_Mesh.stos")
        self.RunStosAssemble(stosFullPath)
         


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_warpedImageToFixedSpace']
    unittest.main()
