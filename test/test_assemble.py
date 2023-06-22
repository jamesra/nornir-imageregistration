'''
Created on Apr 3, 2013

@author: u0490822
'''
import os
import unittest
import numpy as np
import scipy

import nornir_imageregistration
from nornir_imageregistration import AlignmentRecord

import nornir_imageregistration.assemble as assemble

import setup_imagetest


def ShowComparison(*args, **kwargs):
    return nornir_imageregistration.ShowGrayscale(*args, **kwargs)


class TestAssemble(setup_imagetest.ImageTestBase):

    def setUp(self):
        super(TestAssemble, self).setUp()
        self.WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")

    def test_TransformImageIdentity(self):
        # Too small to require more than one tile
        self.CallTransformImage(imageDim=10)
        
        # Large enough to require more than one tile
        self.CallTransformImage(imageDim=4097)
        
    def CallTransformImage(self, imageDim):
        
        Height = int(imageDim)
        Width = int(np.round(imageDim / 2))
        
        identity_transform = nornir_imageregistration.transforms.triangulation.Triangulation(np.array([[0, 0, 0, 0],
                                                                              [Height, 0, Height, 0],
                                                                              [0, Width, 0, Width],
                                                                              [Height, Width, Height, Width]]))
        
        warpedImage = np.ones([Height, Width])

        outputImage = assemble.TransformImage(identity_transform, np.array([Height, Width]), warpedImage, CropUndefined=False)
        self.assertIsNotNone(outputImage, msg="No image produced by TransformImage")
        self.assertEqual(outputImage.shape[0], Height, msg="Output image height should match")
        self.assertEqual(outputImage.shape[1], Width, msg="Output image width should match")

    def test_SourceImageToTargetSpace_Translate(self):
        angle = 0
        arecord = AlignmentRecord(peak=(50, 100), weight=100, angle=angle)

        (fixedImage, warpedImage, transformedImage) = self.Run_SourceImageToTargetSpaceTransform(self.WarpedImagePath,
                                                                     self.WarpedImagePath,
                                                                     arecord)

        nornir_imageregistration.SaveImage("C:\\Temp\\17Translate.png", transformedImage, bpp=8)
        # self.assertTrue(ShowComparison([fixedImage, transformedImage], title="Image should be translated +100x,+50y but not rotated.", PassFail=True, image_titles=('Original', 'Translated')))

        # delta = fixedImage[1:64, 1:64] - transformedImage
        # self.assertTrue((delta < 0.01).all())

    def test_SourceImageToTargetSpace_Translate_GPU(self):
        angle = 0
        arecord = AlignmentRecord(peak=(50, 100), weight=100, angle=angle)

        (fixedImage, warpedImage, transformedImage) = self.Run_SourceImageToTargetSpaceTransform(self.WarpedImagePath,
                                                                     self.WarpedImagePath,
                                                                     arecord,
                                                                     use_cp=True)

        nornir_imageregistration.SaveImage("C:\\Temp\\17Translate.png", transformedImage, bpp=8)
        # self.assertTrue(ShowComparison([fixedImage, transformedImage],
        #                                title="Image should be translated +100x,+50y but not rotated.",
        #                                PassFail=True, image_titles=('Original', 'Translated')))

        # delta = fixedImage[1:64, 1:64] - transformedImage
        # self.assertTrue((delta < 0.01).all())

    def test_SourceImageToTargetSpace_RotateTransform(self):
        angle = 30
        arecord = AlignmentRecord(peak=(0, 0), weight=100, angle=angle)

        (fixedImage, warpedImage, transformedImage) = self.Run_SourceImageToTargetSpaceTransform(self.WarpedImagePath,
                                                                     self.WarpedImagePath,
                                                                     arecord)

        nornir_imageregistration.SaveImage("C:\\Temp\\17Rotate.png", transformedImage, bpp=8)

        rotatedWarpedA = scipy.ndimage.rotate(warpedImage.astype(np.float32, copy=False), angle=angle, reshape=False)
        rotatedWarpedB = scipy.ndimage.rotate(warpedImage.astype(np.float32, copy=False), angle=-angle, reshape=False)

        # self.assertTrue(ShowComparison(((fixedImage, transformedImage), (rotatedWarpedA, rotatedWarpedB)),
        #                                title="Rotate transform should match scipy.interpolate.rotate result", PassFail=True,
        #                                image_titles=(('Target', 'My Transform'),('SciPy ', 'SciPy negated angle'))))
        #
        # delta = fixedImage[512:544, 512:544] - rotatedWarped
        # self.assertTrue((delta < 0.01).all())

    def test_SourceImageToTargetSpace_RotateTransform_GPU(self):
        angle = 30
        arecord = AlignmentRecord(peak=(0, 0), weight=100, angle=angle)

        (fixedImage, warpedImage, transformedImage) = self.Run_SourceImageToTargetSpaceTransform(self.WarpedImagePath,
                                                                     self.WarpedImagePath,
                                                                     arecord,
                                                                     use_cp=True)

        nornir_imageregistration.SaveImage("C:\\Temp\\17Rotate.png", transformedImage, bpp=8)

        rotatedWarpedA = scipy.ndimage.rotate(warpedImage.astype(np.float32, copy=False), angle=angle,
                                              reshape=False)
        rotatedWarpedB = scipy.ndimage.rotate(warpedImage.astype(np.float32, copy=False), angle=-angle,
                                              reshape=False)

        # self.assertTrue(ShowComparison(((fixedImage, transformedImage), (rotatedWarpedA, rotatedWarpedB)),
        #                                title="Rotate transform should match scipy.interpolate.rotate result",
        #                                PassFail=True,
        #                                image_titles=(
        #                                ('Target', 'My Transform'), ('SciPy ', 'SciPy negated angle'))))

        # delta = fixedImage[512:544, 512:544] - rotatedWarped
        # self.assertTrue((delta < 0.01).all())

    def test_SourceImageToTargetSpace_IdentityTransform(self):
        arecord = AlignmentRecord(peak=(0, 0), weight=100, angle=0.0)

        (fixedImage, warpedImage, transformedImage) = self.Run_SourceImageToTargetSpaceTransform(self.WarpedImagePath,
                                                                     self.WarpedImagePath,
                                                                     arecord,
                                                                     (0, 0),
                                                                     (64, 64))

        # nornir_imageregistration.SaveImage("C:\\Temp\\17.png", transformedImage)

        delta = fixedImage[0:64, 0:64] - transformedImage

        #nornir_imageregistration.ShowGrayscale([fixedImage[0:64, 0:64], transformedImage, delta])
        self.assertTrue((delta < 0.01).all())

    def test_SourceImageToTargetSpace_IdentityTransform_GPU(self):
        arecord = AlignmentRecord(peak=(0, 0), weight=100, angle=0.0)

        (fixedImage, warpedImage, transformedImage) = self.Run_SourceImageToTargetSpaceTransform(self.WarpedImagePath,
                                                                     self.WarpedImagePath,
                                                                     arecord,
                                                                     (0, 0),
                                                                     (64, 64),
                                                                     use_cp=True)

        # nornir_imageregistration.SaveImage("C:\\Temp\\17.png", transformedImage)

        delta = fixedImage[0:64, 0:64] - transformedImage

        #nornir_imageregistration.ShowGrayscale([fixedImage[0:64, 0:64], transformedImage, delta])
        self.assertTrue((delta < 0.01).all())


    def test_SourceImageToTargetSpace(self):
        arecord = AlignmentRecord(peak=(22, -4), weight=100, angle=-132.0)

        (fixedImage, warpedImage, transformedImage) = self.Run_SourceImageToTargetSpaceTransform(self.FixedImagePath,
                                                                     self.WarpedImagePath,
                                                                     arecord)

        nornir_imageregistration.SaveImage(os.path.join(self.VolumeDir, "test_warpedImageToFixedSpace.png"), transformedImage, bpp=8)

    def test_SourceImageToTargetSpace_GPU(self):
        arecord = AlignmentRecord(peak=(22, -4), weight=100, angle=-132.0)

        (fixedImage, warpedImage, transformedImage) = self.Run_SourceImageToTargetSpaceTransform(self.FixedImagePath,
                                                                     self.WarpedImagePath,
                                                                     arecord,
                                                                     use_cp=True)
        nornir_imageregistration.SaveImage(os.path.join(self.VolumeDir, "test_warpedImageToFixedSpace.png"), transformedImage, bpp=8)

    def Run_SourceImageToTargetSpaceTransform(self, FixedImagePath: str,
                                             WarpedImagePath: str,
                                             alignment_record: AlignmentRecord,
                                             output_botleft = None,
                                             output_area = None,
                                             use_cp: bool=False):
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        fixedImage = nornir_imageregistration.LoadImage(FixedImagePath)
        warpedImage = nornir_imageregistration.LoadImage(WarpedImagePath)

        transform = alignment_record.ToTransform(fixedImage.shape, warpedImage.shape)
        transformedImage = assemble.SourceImageToTargetSpace(transform,
                                                             warpedImage,
                                                             output_botleft = output_botleft,
                                                             output_area = output_area,
                                                             use_cp = use_cp)
        return fixedImage, warpedImage, transformedImage


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
