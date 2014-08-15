'''
Created on Mar 25, 2013

@author: u0490822
'''
import unittest
import os
from pylab import *
import nornir_imageregistration.core as core
import logging
from . import setup_imagetest
import nornir_imageregistration.stos_brute as stos_brute

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


    def testCrop(self):

        image = np.zeros((16, 16))

        for i in range(0, 16):
            image[i, i] = i + 1

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
            
    
    def testImageToTiles(self):
        self.FixedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")

        image = imread(self.FixedImagePath)
        tiles = core.ImageToTiles(image, tile_size=(256,512))
        
        core.ShowGrayscale(tiles.values(), "Expecting 512 wide x 256 tall tiles")
        

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

        updatedImage = core.RandomNoiseMask(image, mask)
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

        updatedImage = core.RandomNoiseMask(image, mask)

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