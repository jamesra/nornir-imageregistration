'''
Created on Mar 25, 2013

@author: u0490822
'''
import unittest
import os

import hypothesis
from pylab import *

import nornir_imageregistration
import setup_imagetest


class TestCore(setup_imagetest.ImageTestBase):

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
            newDim = nornir_imageregistration.NearestPowerOfTwoWithOverlap(v, overlap=overlap)

            logOriginalDim = math.log(v, 2)
            logNewDim = math.log(newDim, 2)

            self.assertTrue(newDim % 2 == 0, "%d is not a power of two" % newDim)
            self.assertTrue(logOriginalDim <= logNewDim,
                            "%d padded to %d instead of nearest power of two" % (v, newDim))

            print("%d -> %d" % (v, newDim))

    def testNearestPowerOfTwo(self):

        self.__CheckRangeForPowerOfTwo(1.0)
        self.__CheckRangeForPowerOfTwo(0.5)

    def testSafeROIRange(self):

        r = self.check_SafeROIResult(0, 16, 32)
        self.assertEqual(r[0], 0)
        self.assertEqual(r[-1], 15)
        self.assertEqual(len(r), 16)

        r = self.check_SafeROIResult(0, 16, 32)
        self.assertEqual(r[0], 0)
        self.assertEqual(r[-1], 15)
        self.assertEqual(len(r), 16)

        r = self.check_SafeROIResult(-3, 16, 32)
        self.assertEqual(r[0], 0)
        self.assertEqual(r[-1], 15)
        self.assertEqual(len(r), 16)

        r = self.check_SafeROIResult(24, 16, 32)
        self.assertEqual(r[0], 16)
        self.assertEqual(r[-1], 31)
        self.assertEqual(len(r), 16)

        r = self.check_SafeROIResult(24, 16, 5)
        self.assertIsNone(r)

        r = self.check_SafeROIResult(0, 1, 0, 0)
        self.assertIsNone(r)

    @hypothesis.given(start=hypothesis.strategies.integers(-5, 256), num=hypothesis.strategies.integers(-5, 256),
                      maxVal=hypothesis.strategies.integers(-5, 256), minVal=hypothesis.strategies.integers(-10, 64))
    def test_ROIRange_with_hypothesis(self, start, num, maxVal, minVal):
        r = self.check_SafeROIResult(start, num, maxVal, minVal)
        if r is None:
            if (maxVal < minVal):
                hypothesis.event('maxVal < minVal')
                return

            if (maxVal - minVal < num):
                hypothesis.event('maxVal - minVal < num')
                return

            self.fail("None is not an expected result")

        if num <= 0:
            self.assertTrue(len(r) == 0)
            hypothesis.event('trivial - count equals 0')

    def check_SafeROIResult(self, start, num, maxVal, minVal=0):
        r = None
        try:
            r = nornir_imageregistration.SafeROIRange(start, num, maxVal, minVal)
        except ValueError:
            if (maxVal < minVal):
                return

            if (maxVal - minVal < num):
                return

        if num <= 0:
            self.assertTrue(len(r) == 0)
            return r

        self.assertEqual(len(r), num, "SafeROI needs to return the requested size of ROI")
        self.assertGreaterEqual(r[0], minVal, "min list value must be greater than or equal minVal")
        self.assertLess(r[-1], maxVal, "max list value must be less than maxVal")

        if r[0] < start:
            self.assertEqual(r[0], (maxVal - num))
            self.assertEqual(r[-1], maxVal - 1, "Last entry must be maxVal - 1 if start was shifted to fit ROI")
        elif r[0] > start:
            self.assertEqual(r[0], minVal,
                             "If start value was increased then first entry should be minVal, which was greater than requested start value")
            self.assertEqual(r[-1], minVal + num - 1, "Last entry must be maxVal - 1 if start was shifted to fit ROI")

        return r

    #    def test_SaveImageJPeg2000(self):
    #
    #         image_full_path = self.GetImagePath('0162_ds16.png')
    #         image = nornir_imageregistration.LoadImage(image_full_path)
    #
    #         jpeg2000_full_path = os.path.join(self.TestOutputPath, '0162_ds16.jp2')
    #
    #         tile_dims = nornir_imageregistration.TileGridShape(image, (512,512))
    #         image_tiles = nornir_imageregistration.ImageToTiles(image, (512, 512))
    #
    #         for iY in range(0,tile_dims[0]):
    #             for iX in range(0,tile_dims[1]):
    #                 nornir_imageregistration.SaveImage_JPeg2000_Tile(jpeg2000_full_path, image_tiles[iY,iX], tile_coord=(iX,iY), tile_dim=None)
    #
    #         self.assertTrue(os.path.exists(jpeg2000_full_path), "Jpeg 2000 file does not exist")

    def testCrop(self):

        image_dim = (16, 16)
        image = np.zeros(image_dim)

        row_fill = np.array((range(1, image_dim[1] + 1)))
        for iy in range(0, 16):
            image[iy, :] = row_fill

        cropsize = 4
        cropped = nornir_imageregistration.CropImage(image, 0, 0, cropsize, cropsize)
        for i in range(0, cropsize):
            self.assertEqual(cropped[i, i], i + 1, "cropped image not correct")

        cropped = nornir_imageregistration.CropImage(image, -1, -1, cropsize, cropsize)
        for i in range(0, cropsize):
            testVal = i
            if testVal < 0:
                testVal = 0

            self.assertEqual(cropped[i, i], testVal, "cropped image not correct")

        cropsize = 20
        cropped = nornir_imageregistration.CropImage(image, 0, 0, cropsize, cropsize)
        for i in range(0, cropsize):
            testVal = i + 1
            if testVal < 0:
                testVal = 0

            if testVal > 16:
                testVal = 0

            self.assertEqual(cropped[i, i], testVal, "cropped image not correct")

        cropsize = 20
        cropped = nornir_imageregistration.CropImage(image, -1, -1, cropsize, cropsize, cval=0)
        for i in range(0, cropsize):
            testVal = i
            if testVal < 0:
                testVal = 0

            if testVal > 16:
                testVal = 0

            self.assertEqual(cropped[i, i], testVal, "cropped image not correct")

        cropsize = 17
        cropped = nornir_imageregistration.CropImage(image, 1, 1, cropsize, cropsize)
        for i in range(0, cropsize):
            testVal = i + 2
            if testVal < 0:
                testVal = 0

            if testVal > 16:
                testVal = 0

            self.assertEqual(cropped[i, i], testVal, "cropped image not correct")

        cropsize = 8
        cropped = nornir_imageregistration.CropImage(image, 1, 0, cropsize, 16)
        for i in range(0, cropsize):
            testVal = i + 2
            if testVal < 0:
                testVal = 0

            if testVal > 16:
                testVal = 0

            self.assertEqual(cropped[i + 1, i], testVal, "cropped image not correct")

        cropsize = image_dim[0]
        constant_value = 3
        cropped = nornir_imageregistration.CropImage(image, -cropsize, image_dim[0], cropsize, cropsize,
                                                     cval=constant_value)
        for i in range(0, cropsize):
            self.assertEqual(cropped[i, i], constant_value, "Cropped region outside original image should use cval")

        cropped = nornir_imageregistration.CropImage(image, -(cropsize / 2.0), -(cropsize / 2.0), cropsize, cropsize,
                                                     cval='random')
        for i in range(0, cropsize):
            self.assertGreaterEqual(cropped[i, i], 0, "Cropped region outside original image should use random value")

        cropped = cropped / cropped.max()
        self.assertTrue(nornir_imageregistration.ShowGrayscale(cropped,
                                                               title="The bottom left quadrant is a gradient.  The remainder is random noise.",
                                                               PassFail=True))

    def testImageToTiles(self):
        self.FixedImagePath = os.path.join(self.ImportedDataPath,
                                           "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")

        tiles = nornir_imageregistration.ImageToTiles(self.FixedImagePath, tile_size=(256, 512))
        self.assertTrue(
            nornir_imageregistration.ShowGrayscale(list(tiles.values()), "Expecting 512 wide x 256 tall tiles",
                                                   PassFail=True))

    def testReplaceImageExtramaWithNoise(self):

        self.FixedImagePath = os.path.join(self.ImportedDataPath, "0226_mosaic_32.png")
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")

        image = imread(self.FixedImagePath)
        updatedImage = nornir_imageregistration.ReplaceImageExtremaWithNoise(image)

        minima = 0
        maxima = 1.0
        extrema_mask = np.logical_not(np.logical_or(image == maxima, image == minima))
        mask = nornir_imageregistration.CreateExtremaMask(image, size_cutoff=0.001)
        self.assertIsNotNone(updatedImage)
        self.assertFalse((image == updatedImage).all())

        self.assertTrue(nornir_imageregistration.ShowGrayscale((image, extrema_mask, updatedImage, mask),
                                                               title="Image should have extrema values removed, but not small regions saturated cells.\nTop mask should mark all extrema.  Bottom mask should remove small extrema regions.",
                                                               PassFail=True))

    def testReplaceImageExtramaWithNoiseSmall(self):

        self.FixedImagePath = os.path.join(self.ImportedDataPath,
                                           "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")

        image = imread(self.FixedImagePath)
        updatedImage = nornir_imageregistration.ReplaceImageExtremaWithNoise(image)

        minima = 0
        maxima = 1.0
        extrema_mask = np.logical_not(np.logical_or(image == maxima, image == minima))
        mask = nornir_imageregistration.CreateExtremaMask(image, size_cutoff=0.001)
        self.assertIsNotNone(updatedImage)
        self.assertFalse((image == updatedImage).all())

        self.assertTrue(nornir_imageregistration.ShowGrayscale((image, extrema_mask, updatedImage, mask),
                                                               title="Image should have extrema values removed, but not small regions saturated cells.\nTop mask should mark all extrema.  Bottom mask should remove small extrema regions.",
                                                               PassFail=True))

    def testRandomNoiseMask(self):

        self.FixedImagePath = os.path.join(self.ImportedDataPath,
                                           "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")

        self.FixedImageMaskPath = os.path.join(self.ImportedDataPath,
                                               "mini_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(self.FixedImageMaskPath), "Missing test input")

        image = imread(self.FixedImagePath)
        mask = imread(self.FixedImageMaskPath)

        self.assertTrue(mask[0][0] == 0, "The mask pixel we are going to test is not masked, test is broken")
        self.assertFalse(mask[256][256] == 0, "The unmasked pixel we are going to test is  masked, test is broken")

        updatedImage = nornir_imageregistration.RandomNoiseMask(image, mask, Copy=True)
        # nornir_imageregistration.ShowGrayscale(updatedImage)

        self.assertIsNotNone(updatedImage)
        self.assertNotEqual(updatedImage[0][0], image[0][0],
                            "Masked off pixel should not be the same as input image value after the test")
        self.assertEqual(updatedImage[256][256], image[256][256], "Unmasked pixel should equal input image")

    def testRandomNoiseMask2(self):
        WarpedImagePath = os.path.join(self.ImportedDataPath,
                                       "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")

        WarpedImageMaskPath = os.path.join(self.ImportedDataPath,
                                           "0017_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")

        image = imread(WarpedImagePath)
        mask = imread(WarpedImageMaskPath)

        self.assertTrue(mask[0][0] == 0, "The mask pixel we are going to test is not masked, test is broken")
        self.assertFalse(mask[32][32] == 0, "The unmasked pixel we are going to test is  masked, test is broken")

        updatedImage = nornir_imageregistration.RandomNoiseMask(image, mask, Copy=True)

        # nornir_imageregistration.ShowGrayscale(updatedImage)
        self.assertIsNotNone(updatedImage)
        self.assertNotEqual(updatedImage[0][0], image[0][0],
                            "Masked off pixel should not be the same as input image value after the test")
        self.assertEqual(updatedImage[32][32], image[32][32], "Unmasked pixel should equal input image")


#    def testPadImage(self):
#
#        self.FixedImagePath = os.path.join(self.ImportedDataPath, "PadImageTestPattern.png")
#        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")
#
#        image = imread(self.FixedImagePath)
#
#        paddedimage = nornir_imageregistration.PadImageForPhaseCorrelation(image)
#
#        nornir_imageregistration.ShowGrayscale(paddedimage)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testReplaceImageExtramaWithNoise']
    unittest.main()
