'''
Created on Oct 28, 2013

@author: u0490822
'''
import unittest
from . import setup_imagetest
import glob
import nornir_imageregistration.assemble_tiles as at
import nornir_imageregistration.tileset as tiles
from nornir_imageregistration.files.mosaicfile import MosaicFile
import os
import nornir_imageregistration.core as core
import nornir_imageregistration.transforms.factory as tfactory

# from pylab import *
from scipy.misc import imsave
import numpy as np
from scipy import stats

from nornir_shared.tasktimer import TaskTimer

from nornir_imageregistration.mosaic  import Mosaic


class TestMosaicAssemble(setup_imagetest.MosaicTestBase):

    def DownsampleFromTilePath(self, tilePath):
        downsamplePath = os.path.basename(tilePath)
        return float(downsamplePath)

    def setUp(self):
        setup_imagetest.MosaicTestBase.setUp(self)

        # Make sure we save images before starting a long test
        z = np.zeros((16, 16))
        outputImagePath = os.path.join(self.TestOutputPath, 'z.png')
        imsave(outputImagePath, z)

        self.assertTrue(os.path.exists(outputImagePath), "OutputImage not found")

        os.remove(outputImagePath)


    def AssembleMosaic(self, mosaicFilePath, tilesDir, outputMosaicPath=None, parallel=False, downsamplePath=None):

        SaveFiles = not outputMosaicPath is None

        mosaic = Mosaic.LoadFromMosaicFile(mosaicFilePath)
        mosaicBaseName = os.path.basename(mosaicFilePath)
        (mosaicBaseName, ext) = os.path.splitext(mosaicBaseName)

        mosaic.TranslateToZeroOrigin()

        assembleScale = tiles.MostCommonScalar(list(mosaic.ImageToTransform.values()), mosaic.TileFullPaths(tilesDir))

        expectedScale = 1.0 / self.DownsampleFromTilePath(tilesDir)

        self.assertEqual(assembleScale, expectedScale, "Scale for assemble does not match the expected scale")

        timer = TaskTimer()

        timer.Start("AssembleTiles " + tilesDir)

        (mosaicImage, mask) = mosaic.AssembleTiles(tilesDir, usecluster=parallel)

        timer.End("AssembleTiles " + tilesDir, True)

        self.assertEqual(mosaicImage.shape[0], np.ceil(mosaic.FixedBoundingBoxHeight * expectedScale), "Output mosaic height does not match .mosaic height %g vs %g" % (mosaicImage.shape[0], mosaic.FixedBoundingBoxHeight * expectedScale))
        self.assertEqual(mosaicImage.shape[1], np.ceil(mosaic.FixedBoundingBoxWidth * expectedScale), "Output mosaic width does not match .mosaic height %g vs %g" % (mosaicImage.shape[1], mosaic.FixedBoundingBoxWidth * expectedScale))
        # img = im.fromarray(mosaicImage, mode="I")
        # img.save(outputImagePath, mosaicImage, bits=8)

        if SaveFiles:
            OutputDir = os.path.join(self.TestOutputPath, outputMosaicPath)

            if not os.path.exists(OutputDir):
                os.makedirs(OutputDir)

            outputImagePath = os.path.join(OutputDir, mosaicBaseName + '.png')
            outputImageMaskPath = os.path.join(OutputDir, mosaicBaseName + '_mask.png')

            core.SaveImage(outputImagePath, mosaicImage)
            core.SaveImage(outputImageMaskPath, mask)
            self.assertTrue(os.path.exists(outputImagePath), "OutputImage not found")

            outputMask = core.LoadImage(outputImageMaskPath)
            self.assertTrue(outputMask[outputMask.shape[0] / 2.0, outputMask.shape[1] / 2.0] > 0, "Center of assembled image mask should be non-zero")

            del mosaicImage
            del mask
        else:
            return (mosaicImage, mask)



    def CompareMosaicAsssembleAndTransformTile(self, mosaicFilePath, tilesDir):
        '''
        1) Assemble the entire mosaic
        2) Assemble subregion of the mosaic
        3) Assemble subregion directly using _TransformTile
        4) Check the output of all match
        '''

        mosaic = Mosaic.LoadFromMosaicFile(mosaicFilePath)
        self.assertIsNotNone(mosaic, "Mosaic not loaded")

        mosaic.TranslateToZeroOrigin()

        (imageKey, transform) = list(mosaic.ImageToTransform.items())[0]

        (MinY, MinX, MaxY, MaxZ) = transform.FixedBoundingBox

        FixedRegion = np.array([MinY + 512, MinX + 1024, MinY + 1024, MinX + 2048])
        ScaledFixedRegion = FixedRegion / self.DownsampleFromTilePath(tilesDir)

        (tileImage, tileMask) = mosaic.AssembleTiles(tilesDir, usecluster=False, FixedRegion=FixedRegion)
        # self.assertEqual(tileImage.shape, (ScaledFixedRegion[3], ScaledFixedRegion[2]))

        (clustertileImage, clustertileMask) = mosaic.AssembleTiles(tilesDir, usecluster=True, FixedRegion=FixedRegion)
        # self.assertEqual(tileImage.shape, (ScaledFixedRegion[3], ScaledFixedRegion[2]))

        self.assertTrue(np.sum(np.abs(clustertileImage - tileImage).flat) < 0.65, "Tiles generated with cluster should be identical to single threaded implementation")
        self.assertTrue(np.all(clustertileMask == tileMask), "Tiles generated with cluster should be identical to single threaded implementation")

        result = at.TransformTile(transform, os.path.join(tilesDir, imageKey), distanceImage=None, requiredScale=None, FixedRegion=FixedRegion)
        self.assertEqual(result.image.shape, (ScaledFixedRegion[2] - ScaledFixedRegion[0], ScaledFixedRegion[3] - ScaledFixedRegion[1]))

        # core.ShowGrayscale([tileImage, result.image])
        (wholeimage, wholemask) = self.AssembleMosaic(mosaicFilePath, tilesDir, outputMosaicPath=None, parallel=False)
        self.assertIsNotNone(wholeimage, "Assemble did not produce an image")
        self.assertIsNotNone(wholemask, "Assemble did not produce a mask")

        croppedWholeImage = core.CropImage(wholeimage,
                                                    int(ScaledFixedRegion[1]),
                                                    int(ScaledFixedRegion[0]),
                                                    int(ScaledFixedRegion[3] - ScaledFixedRegion[1]),
                                                    int(ScaledFixedRegion[2] - ScaledFixedRegion[0]))

        core.ShowGrayscale([result.image, tileImage, croppedWholeImage, wholeimage])


    def CreateAssembleOptimizedTile(self, mosaicFilePath, TilesDir):
        mosaic = Mosaic.LoadFromMosaicFile(mosaicFilePath)
        mosaicBaseName = os.path.basename(mosaicFilePath)

        mosaicDir = os.path.dirname(mosaicFilePath)
        (mosaicBaseName, ext) = os.path.splitext(mosaicBaseName)

        imageKey = list(mosaic.ImageToTransform.keys())[0]
        transform = mosaic.ImageToTransform[imageKey]

        (MinY, MinX, MaxY, MaxX) = transform.FixedBoundingBox

        expectedScale = 1.0 / self.DownsampleFromTilePath(TilesDir)

        result = at.TransformTile(transform, os.path.join(TilesDir, imageKey), distanceImage=None, requiredScale=None, FixedRegion=(MinY, MinX, MaxY, MinX + 256))
        self.assertEqual(result.image.shape, (np.ceil(transform.FixedBoundingBox.Height * expectedScale), np.ceil(256 * expectedScale)))

        result = at.TransformTile(transform, os.path.join(TilesDir, imageKey), distanceImage=None, requiredScale=None, FixedRegion=(MinY, MinX, MinY + 256, MaxX))
        self.assertEqual(result.image.shape, (np.ceil(256 * expectedScale), np.ceil(transform.FixedBoundingBox.Width * expectedScale)))

        result = at.TransformTile(transform, os.path.join(TilesDir, imageKey), distanceImage=None, requiredScale=None, FixedRegion=(MinY + 2048, MinX + 2048, MinY + 2048 + 512, MinX + 2048 + 512))
        self.assertEqual(result.image.shape, (np.ceil(512 * expectedScale), np.ceil(512 * expectedScale)))


    def CreateAssembleEachMosaic(self, mosaicFiles, tilesDir):

        for m in mosaicFiles:
            self.AssembleMosaic(m, tilesDir, 'CreateAssembleEachMosaicTypeDS4', parallel=False)
            # self. AssembleMosaic(m, 'CreateAssembleEachMosaicType', parallel=False)

        print("All done")


    def ParallelAssembleEachMosaic(self, mosaicFiles, tilesDir):

        for m in mosaicFiles:
            self.AssembleMosaic(m, tilesDir , 'ParallelAssembleEachMosaicTypeDS4', parallel=True)
            # self. AssembleMosaic(m, 'ParallelAssembleEachMosaicType', parallel=True)

        print("All done")


class BasicTests(TestMosaicAssemble):

    @property
    def TestName(self):
        return "PMG1"

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

        for m in self.GetMosaicFiles():

            mosaic = Mosaic.LoadFromMosaicFile(m)

            self.assertIsNotNone(mosaic.MappedBoundingBox, "No bounding box returned for mosiac")

            self.Logger.info(m + " mapped bounding box: " + str(mosaic.MappedBoundingBox))

            self.assertIsNotNone(mosaic.FixedBoundingBox, "No bounding box returned for mosiac")

            self.Logger.info(m + " fixed bounding box: " + str(mosaic.FixedBoundingBox))


class PMGTests(TestMosaicAssemble):

    @property
    def TestName(self):
        return "PMG1"

    def test_AssemblePMG(self):
        testName = "PMG1"

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath()

        self.CreateAssembleEachMosaic(mosaicFiles, tilesDir)

    def test_AssemblePMGParallel(self):
        testName = "PMG1"

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath()

        self.ParallelAssembleEachMosaic(mosaicFiles, tilesDir)


class IDOCTests(TestMosaicAssemble):

    @property
    def TestName(self):
        return "IDOC1"

    def test_AssembleIDOC(self):
        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath='004')

        self.CreateAssembleEachMosaic(mosaicFiles, tilesDir)

    def test_AssembleIDOCParallel(self):

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath='001')

        self.ParallelAssembleEachMosaic(mosaicFiles, tilesDir)

    def test_AssembleTilesIDoc(self):
        '''Assemble small 256x265 tiles from a transform and image in a mosaic'''

        downsamplePath = '004'

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath)

        self.CompareMosaicAsssembleAndTransformTile(mosaicFiles[0], tilesDir)
        self.CreateAssembleOptimizedTile(mosaicFiles[0], tilesDir)
        
    def test_AssembleTilesIDoc_JPeg2000(self):
        '''Assemble small 256x265 tiles from a transform and image in a mosaic'''

        downsamplePath = '001'

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath)
        
        ImageFullPath = os.path.join(self.TestOutputPath, 'Image.jp2')
        
        (image, mask) = self.AssembleMosaic(mosaicFiles[0], tilesDir, parallel=True)
        
        core.SaveImage_JPeg2000(ImageFullPath, image)
        self.assertTrue(os.path.exists(ImageFullPath), "File should be written to disk for JPeg2000")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']

    import nornir_shared.misc
    nornir_shared.misc.RunWithProfiler("unittest.main()")