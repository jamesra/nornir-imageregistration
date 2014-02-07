'''
Created on Oct 28, 2013

@author: u0490822
'''
import unittest
import setup_imagetest
import glob
import nornir_imageregistration.assemble_tiles as at
import nornir_imageregistration.tiles as tiles
from nornir_imageregistration.files.mosaicfile import MosaicFile
import os
import nornir_imageregistration.transforms.factory as tfactory
# from pylab import *
from scipy.misc import imsave
import numpy as np
from scipy import stats

from nornir_shared.tasktimer import TaskTimer

from nornir_imageregistration.mosaic  import Mosaic

class TestMosaicAssemble(setup_imagetest.MosaicTestBase):

    @property
    def MosaicFiles(self, testName=None):
        if testName is None:
            testName = "Test1"

        return glob.glob(os.path.join(self.ImportedDataPath, testName, "*.mosaic"))


    def test_CreateDistanceBuffer(self):

        firstShape = (10, 10)
        dMatrix = at.CreateDistanceImage(firstShape)
        self.assertAlmostEqual(dMatrix[0, 0], 7.07, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[9, 9], 7.07, 2, "Distance matrix incorrect")

        secondShape = (11, 11)
        dMatrix = at.CreateDistanceImage(secondShape)

        self.assertAlmostEqual(dMatrix[0, 0], 7.78, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[10, 10], 7.78, 2, "Distance matrix incorrect")

        thirdShape = (10, 11)
        dMatrix = at.CreateDistanceImage(thirdShape)

        self.assertAlmostEqual(dMatrix[0, 0], 7.43, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[9, 0], 7.43, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[9, 10], 7.43, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[0, 10], 7.43, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[0, 5], 5, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[4, 0], 5.53, 2, "Distance matrix incorrect")


    def test_MosaicBoundsEachMosaicType(self):

        for m in self.MosaicFiles:

            mosaic = Mosaic.LoadFromMosaicFile(m)

            self.assertIsNotNone(mosaic.MappedBoundingBox, "No bounding box returned for mosiac")

            self.Logger.info(m + " mapped bounding box: " + str(mosaic.MappedBoundingBox))

            self.assertIsNotNone(mosaic.FixedBoundingBox, "No bounding box returned for mosiac")

            self.Logger.info(m + " fixed bounding box: " + str(mosaic.FixedBoundingBox))


    def AssembleMosaic(self, mosaicFilePath, outputMosaicPath, parallel=False, downsamplePath=None):

        if downsamplePath is None:
            downsamplePath = '001'

        mosaic = Mosaic.LoadFromMosaicFile(mosaicFilePath)
        mosaicBaseName = os.path.basename(mosaicFilePath)

        (mosaicBaseName, ext) = os.path.splitext(mosaicBaseName)

        TilesDir = os.path.join(self.ImportedDataPath, 'Test1', 'Leveled', 'TilePyramid', downsamplePath)

        mosaic.TranslateToZeroOrigin()

        assembleScale = tiles.MostCommonScalar(mosaic.ImageToTransform.values(), mosaic.TileFullPaths(TilesDir))

        expectedScale = 1.0 / float(downsamplePath)

        self.assertEqual(assembleScale, expectedScale, "Scale for assemble does not match the expected scale")

        timer = TaskTimer()

        timer.Start("AssembleTiles " + TilesDir)

        (mosaicImage, mask) = mosaic.AssembleTiles(TilesDir, usecluster=parallel)

        timer.End("AssembleTiles " + TilesDir, True)

        OutputDir = os.path.join(self.TestOutputPath, outputMosaicPath)

        outputImagePath = os.path.join(OutputDir, mosaicBaseName + '.png')
        outputImageMaskPath = os.path.join(OutputDir, mosaicBaseName + '_mask.png')

        self.assertEqual(mosaicImage.shape[0], np.ceil(mosaic.FixedBoundingBoxHeight * expectedScale), "Output mosaic height does not match .mosaic height %g vs %g" % (mosaicImage.shape[0], mosaic.FixedBoundingBoxHeight * expectedScale))
        self.assertEqual(mosaicImage.shape[1], np.ceil(mosaic.FixedBoundingBoxWidth * expectedScale), "Output mosaic width does not match .mosaic height %g vs %g" % (mosaicImage.shape[1], mosaic.FixedBoundingBoxWidth * expectedScale))
        # img = im.fromarray(mosaicImage, mode="I")
        # img.save(outputImagePath, mosaicImage, bits=8)

        if not os.path.exists(OutputDir):
            os.makedirs(OutputDir)

        imsave(outputImagePath, mosaicImage)
        imsave(outputImageMaskPath, mask)

        self.assertTrue(os.path.exists(outputImagePath), "OutputImage not found")

        del mosaicImage
        del mask


    def test_CreateAssembleEachMosaicType(self):

        # Make sure we save images before starting a long test
        z = np.zeros((16, 16))
        outputImagePath = os.path.join(self.TestOutputPath, 'z.png')
        imsave(outputImagePath, z)
        self.assertTrue(os.path.exists(outputImagePath), "OutputImage not found")

        for m in self.MosaicFiles:

            self. AssembleMosaic(m, 'CreateAssembleEachMosaicTypeDS4', parallel=False, downsamplePath='004')
            self. AssembleMosaic(m, 'CreateAssembleEachMosaicType', parallel=False)

        print("All done")


    def test_ParallelAssembleEachMosaicType(self):
        # Make sure we save images before starting a long test
        z = np.zeros((16, 16))
        outputImagePath = os.path.join(self.TestOutputPath, 'z.png')
        imsave(outputImagePath, z)
        self.assertTrue(os.path.exists(outputImagePath), "OutputImage not found")

        for m in self.MosaicFiles:

            self. AssembleMosaic(m, 'ParallelAssembleEachMosaicTypeDS4', parallel=True, downsamplePath='004')
            self. AssembleMosaic(m, 'ParallelAssembleEachMosaicType', parallel=True)

        print("All done")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']

    import nornir_shared.misc
    nornir_shared.misc.RunWithProfiler("unittest.main()")