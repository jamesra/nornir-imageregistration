'''
Created on Feb 11, 2014

@author: u0490822
'''
import os
import unittest

import nornir_imageregistration as nir
import nornir_imageregistration.tileset as tiles
from . import setup_imagetest


class TestTiles(setup_imagetest.ImageTestBase):

    def testBrightfieldShading(self):

        ReferenceImagePath = self.GetImagePath("400.png")
        ShadedImagePath = self.GetImagePath("400_Shaded.png")
        ShadingReferencePath = self.GetImagePath("BrightfieldShading.png")

        self.assertTrue(os.path.exists(ReferenceImagePath))
        self.assertTrue(os.path.exists(ShadedImagePath))
        self.assertTrue(os.path.exists(ShadingReferencePath))

        originalImage = nir.LoadImage(ReferenceImagePath)
        shadedImage = nir.LoadImage(ShadedImagePath)
        shadingMask = nir.LoadImage(ShadingReferencePath)

        shadedImageV2 = originalImage * shadingMask
        shadedImageV2Path = os.path.join(self.TestOutputPath, "TestGeneratedShadedImage.png")

        nir.SaveImage(shadedImageV2Path, shadedImageV2)

        OutputPaths = tiles.ShadeCorrect([ShadedImagePath, shadedImageV2Path], shadingMask, self.TestOutputPath,
                                         correction_type=tiles.ShadeCorrectionTypes.BRIGHTFIELD)

        shownImages = [originalImage, shadedImage, shadingMask]

        for path in OutputPaths:
            correctedImage = nir.LoadImage(path)
            shownImages.append(correctedImage)

        # nir.ShowGrayscale(shownImages)

        pass

    def testBrightfieldShadingA(self):
        ShadedImagePath = self.GetImagePath("CorrectionA_Tile.png")
        ShadingReferencePath = self.GetImagePath("CorrectionA_Shading.png")

        self.ExamineBrightfieldShading(ShadedImagePath, ShadingReferencePath)

    def testBrightfieldShadingB(self):
        ShadedImagePath = self.GetImagePath("CorrectionB_Tile.png")
        ShadingReferencePath = self.GetImagePath("CorrectionB_Shading.png")

        self.ExamineBrightfieldShading(ShadedImagePath, ShadingReferencePath)

    def ExamineBrightfieldShading(self, ShadedImagePath, ShadingReferencePath):

        self.assertTrue(os.path.exists(ShadedImagePath))
        self.assertTrue(os.path.exists(ShadingReferencePath))

        shadedImage = nir.LoadImage(ShadedImagePath)
        shadingMask = nir.LoadImage(ShadingReferencePath)

        OutputPaths = tiles.ShadeCorrect([ShadedImagePath], shadingMask, self.TestOutputPath,
                                         correction_type=tiles.ShadeCorrectionTypes.BRIGHTFIELD)

        shownImages = [shadedImage, shadingMask]

        for path in OutputPaths:
            correctedImage = nir.LoadImage(path)
            shownImages.append(correctedImage)

        # nir.ShowGrayscale(shownImages)

        pass


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
