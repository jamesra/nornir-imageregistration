'''
Created on Mar 25, 2013

@author: u0490822
'''
import logging
import os
import unittest

from pylab import *

import nornir_imageregistration.core as core
import nornir_imageregistration.stos_brute as stos_brute

from . import setup_imagetest


class Test(setup_imagetest.ImageTestBase):


#    def testSciPyRavel(self):
#        '''I was having a problem with SciPy's ravel operator not returning the same image after converting to 1D array and back.
#            I never got ravel working so I only use the .flat or .flatiter from now on'''
#        self.FixedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
#        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")
#
#        Image = imread(self.FixedImagePath)
#        Image1D = ravel(Image, 1);
#
#        ReshapeImage = Image1D.reshape(Image.shape)
#
#        self.assertEqual(Image.shape, ReshapeImage.shape)
#        self.assertTrue((Image == ReshapeImage).all(), "Images should match exactly after converting to 1D and back to 2D")

    def __CheckRangeForPowerOfTwo(self, overlap):

        for v in range(2, 129):
            newDim = core.NearestPowerOfTwoWithOverlap(v, overlap=overlap)

            logOriginalDim = math.log(v, 2)
            logNewDim = math.log(newDim, 2)

            self.assertTrue(newDim % 2 == 0, "%d is not a power of two" % newDim)
            self.assertTrue(logOriginalDim <= logNewDim, "%d padded to %d instead of nearest power of two" % (v, newDim))

            print("%d -> %d" % (v, newDim))

    def testNearestPowerOfTwo(self):

        self.__CheckRangeForPowerOfTwo(1.0)
        self.__CheckRangeForPowerOfTwo(0.5)

    def testROIRange(self):

        r = core.ROIRange(0, 16, 32)
        self.assertEqual(r[0], 0)
        self.assertEqual(r[-1], 15)
        self.assertEqual(len(r), 16)

        r = core.ROIRange(-3, 16, 32)
        self.assertEqual(r[0], 0)
        self.assertEqual(r[-1], 15)
        self.assertEqual(len(r), 16)

        r = core.ROIRange(24, 16, 32)
        self.assertEqual(r[0], 16)
        self.assertEqual(r[-1], 31)
        self.assertEqual(len(r), 16)

        r = core.ROIRange(24, 16, 5)
        self.assertIsNone(r)

    def testShowGrayscale(self):

        imageA = np.random.rand(64, 64)
        imageB = np.random.rand(64, 64)

        self.assertFalse(core.ShowGrayscale(imageA, title="Ensure the FAIL button works by clicking it now, select Pass button for all other tests", PassFail=True))

        self.assertTrue(core.ShowGrayscale(imageA, title="A single image with a title followed by a single image with no title", PassFail=True))
        self.assertTrue(core.ShowGrayscale(imageA, title=None, PassFail=True))

        self.assertTrue(core.ShowGrayscale([imageA], title="A single image in a list with a title followed by a single image with no title", PassFail=True))
        self.assertTrue(core.ShowGrayscale([imageA], title=None, PassFail=True))

        self.assertTrue(core.ShowGrayscale([imageA, imageB], title="Two images in a list with a title, followed by two images with no title", PassFail=True))
        self.assertTrue(core.ShowGrayscale([imageA, imageB], title=None, PassFail=True))


#    def test_SaveImageJPeg2000(self):
#        
#         image_full_path = self.GetImagePath('0162_ds16.png')
#         image = core.LoadImage(image_full_path)
#         
#         jpeg2000_full_path = os.path.join(self.TestOutputPath, '0162_ds16.jp2')
# 
#         tile_dims = core.TileGridShape(image, (512,512))        
#         image_tiles = core.ImageToTiles(image, (512, 512))
#         
#         for iY in range(0,tile_dims[0]):
#             for iX in range(0,tile_dims[1]):
#                 core.SaveImage_JPeg2000_Tile(jpeg2000_full_path, image_tiles[iY,iX], tile_coord=(iX,iY), tile_dim=None)
#         
#         self.assertTrue(os.path.exists(jpeg2000_full_path), "Jpeg 2000 file does not exist")


    def testCrop(self):

        image_dim = (16, 16)
        image = np.zeros(image_dim)

        row_fill = np.array((range(1, image_dim[1] + 1)))
        for iy in range(0, 16):
            image[iy, :] = row_fill

        cropsize = 4
        cropped = core.CropImage(image, 0, 0, cropsize, cropsize)
        for i in range(0, cropsize):
            self.assertEqual(cropped[i, i], i + 1, "cropped image not correct")

        cropped = core.CropImage(image, -1, -1, cropsize, cropsize)
        for i in range(0, cropsize):
            testVal = i
            if testVal < 0:
                testVal = 0

            self.assertEqual(cropped[i, i], testVal, "cropped image not correct")

        cropsize = 20
        cropped = core.CropImage(image, 0, 0, cropsize, cropsize)
        for i in range(0, cropsize):
            testVal = i + 1
            if testVal < 0:
                testVal = 0

            if testVal > 16:
                testVal = 0

            self.assertEqual(cropped[i, i], testVal, "cropped image not correct")

        cropsize = 20
        cropped = core.CropImage(image, -1, -1, cropsize, cropsize)
        for i in range(0, cropsize):
            testVal = i
            if testVal < 0:
                testVal = 0

            if testVal > 16:
                testVal = 0

            self.assertEqual(cropped[i, i], testVal, "cropped image not correct")


        cropsize = 17
        cropped = core.CropImage(image, 1, 1, cropsize, cropsize)
        for i in range(0, cropsize):
            testVal = i + 2
            if testVal < 0:
                testVal = 0

            if testVal > 16:
                testVal = 0

            self.assertEqual(cropped[i, i], testVal, "cropped image not correct")

        cropsize = 8
        cropped = core.CropImage(image, 1, 0, cropsize, 16)
        for i in range(0, cropsize):
            testVal = i + 2
            if testVal < 0:
                testVal = 0

            if testVal > 16:
                testVal = 0

            self.assertEqual(cropped[i + 1, i], testVal, "cropped image not correct")
            
        cropsize = image_dim[0]
        constant_value = 3
        cropped = core.CropImage(image, -cropsize, image_dim[0], cropsize, cropsize, cval=constant_value)
        for i in range(0, cropsize):
            self.assertEqual(cropped[i, i], constant_value, "Cropped region outside original image should use cval")
            
        cropped = core.CropImage(image, -(cropsize / 2.0), -(cropsize / 2.0), cropsize, cropsize, cval='random')
        for i in range(0, cropsize):
            self.assertGreaterEqual(cropped[i, i], 0, "Cropped region outside original image should use random value")
        
        self.assertTrue(core.ShowGrayscale(cropped, title="The bottom left quadrant is a gradient.  The remainder is random noise.", PassFail=True))
            
    
    def testImageToTiles(self):
        self.FixedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")

        image = imread(self.FixedImagePath)
        tiles = core.ImageToTiles(image, tile_size=(256, 512))
        
        self.assertTrue(core.ShowGrayscale(list(tiles.values()), "Expecting 512 wide x 256 tall tiles", PassFail=True))
        

    def testReplaceImageExtramaWithNoise(self):

        self.FixedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")

        image = imread(self.FixedImagePath)
        updatedImage = core.ReplaceImageExtramaWithNoise(image)
        self.assertIsNotNone(updatedImage)
        self.assertFalse((image == updatedImage).all())

    def testRandomNoiseMask(self):

        self.FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")

        self.FixedImageMaskPath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(self.FixedImageMaskPath), "Missing test input")

        image = imread(self.FixedImagePath)
        mask = imread(self.FixedImageMaskPath)

        self.assertTrue(mask[0][0] == 0, "The mask pixel we are going to test is not masked, test is broken")
        self.assertFalse(mask[256][256] == 0, "The unmasked pixel we are going to test is  masked, test is broken")

        updatedImage = core.RandomNoiseMask(image, mask, Copy=True)
        # core.ShowGrayscale(updatedImage)

        self.assertIsNotNone(updatedImage)
        self.assertNotEqual(updatedImage[0][0], image[0][0], "Masked off pixel should not be the same as input image value after the test")
        self.assertEqual(updatedImage[256][256], image[256][256], "Unmasked pixel should equal input image")



    def testRandomNoiseMask2(self):
        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")

        WarpedImageMaskPath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")

        image = imread(WarpedImagePath)
        mask = imread(WarpedImageMaskPath)

        self.assertTrue(mask[0][0] == 0, "The mask pixel we are going to test is not masked, test is broken")
        self.assertFalse(mask[32][32] == 0, "The unmasked pixel we are going to test is  masked, test is broken")

        updatedImage = core.RandomNoiseMask(image, mask, Copy=True)

        # core.ShowGrayscale(updatedImage)
        self.assertIsNotNone(updatedImage)
        self.assertNotEqual(updatedImage[0][0], image[0][0], "Masked off pixel should not be the same as input image value after the test")
        self.assertEqual(updatedImage[32][32], image[32][32], "Unmasked pixel should equal input image")

#    def testPadImage(self):
#
#        self.FixedImagePath = os.path.join(self.ImportedDataPath, "PadImageTestPattern.png")
#        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")
#
#        image = imread(self.FixedImagePath)
#
#        paddedimage = core.PadImageForPhaseCorrelation(image)
#
#        core.ShowGrayscale(paddedimage)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testReplaceImageExtramaWithNoise']
    unittest.main()
