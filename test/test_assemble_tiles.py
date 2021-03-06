'''
Created on Oct 28, 2013

@author: u0490822
'''
import glob
import os
import unittest

from nornir_imageregistration.files.mosaicfile import MosaicFile
from nornir_imageregistration.mosaic  import Mosaic
from scipy import stats
from scipy.misc import imsave

import nornir_imageregistration
import nornir_imageregistration.assemble_tiles as at
import nornir_imageregistration.core as core
import nornir_imageregistration.tileset as tiles
import nornir_imageregistration.transforms.factory as tfactory
from nornir_shared.tasktimer import TaskTimer
import numpy as np

import setup_imagetest


# from pylab import *
class TestMosaicAssemble(setup_imagetest.TransformTestBase):

    def DownsampleFromTilePath(self, tilePath):
        downsamplePath = os.path.basename(tilePath)
        return float(downsamplePath)

    def setUp(self):
        super(TestMosaicAssemble, self).setUp()
        
        self.assertTrue(len(self.GetMosaicFiles()) > 0, "No mosaic files found to test!")

        # Make sure we can save images before starting a long test
        z = np.zeros((16, 16))
        outputImagePath = os.path.join(self.TestOutputPath, 'z.png')
        imsave(outputImagePath, z)

        self.assertTrue(os.path.exists(outputImagePath), "OutputImage not found, test cannot write to disk!")

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

        timer.Start("AssembleImage " + tilesDir)

        (mosaicImage, mask) = mosaic.AssembleImage(tilesDir, usecluster=parallel)

        timer.End("AssembleImage " + tilesDir, True)

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
            self.assertTrue(outputMask[int(outputMask.shape[0] / 2.0), int(outputMask.shape[1] / 2.0)] > 0, "Center of assembled image mask should be non-zero")

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

        (imageKey, transform) = sorted(list(mosaic.ImageToTransform.items()))[0]

        (MinY, MinX, MaxY, MaxZ) = transform.FixedBoundingBox
        print("Scaled fixed region %s" % str(transform.FixedBoundingBox))
        
        FixedRegion = np.array([MinY + 512, MinX + 1024, MinY + 1024, MinX + 2048])
        ScaledFixedRegion = FixedRegion / self.DownsampleFromTilePath(tilesDir)
        

        (tileImage, tileMask) = mosaic.AssembleImage(tilesDir, usecluster=False, FixedRegion=FixedRegion)
        # self.assertEqual(tileImage.shape, (ScaledFixedRegion[3], ScaledFixedRegion[2]))

        (clustertileImage, clustertileMask) = mosaic.AssembleImage(tilesDir, usecluster=True, FixedRegion=FixedRegion)
        # self.assertEqual(tileImage.shape, (ScaledFixedRegion[3], ScaledFixedRegion[2]))

        self.assertTrue(np.sum(np.abs(clustertileImage - tileImage).flat) < 0.65, "Tiles generated with cluster should be identical to single threaded implementation")
        self.assertTrue(np.all(clustertileMask == tileMask), "Tiles generated with cluster should be identical to single threaded implementation")

        result = at.TransformTile(transform, os.path.join(tilesDir, imageKey), distanceImage=None, target_space_scale=None, TargetRegion=FixedRegion)
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

        self.assertTrue(nornir_imageregistration.ShowGrayscale([result.image, tileImage, croppedWholeImage, wholeimage], title="image: %s\n%s" % (imageKey, str(transform.FixedBoundingBox)), PassFail=True))


    def CreateAssembleOptimizedTile(self, mosaicFilePath, TilesDir):
        mosaic = Mosaic.LoadFromMosaicFile(mosaicFilePath)
        mosaicBaseName = os.path.basename(mosaicFilePath)

        mosaicDir = os.path.dirname(mosaicFilePath)
        (mosaicBaseName, ext) = os.path.splitext(mosaicBaseName)

        imageKey = list(mosaic.ImageToTransform.keys())[0]
        transform = mosaic.ImageToTransform[imageKey]

        (MinY, MinX, MaxY, MaxX) = transform.FixedBoundingBox

        expectedScale = 1.0 / self.DownsampleFromTilePath(TilesDir)

        result = at.TransformTile(transform, os.path.join(TilesDir, imageKey), distanceImage=None, target_space_scale=None, TargetRegion=(MinY, MinX, MaxY, MinX + 256))
        self.assertEqual(result.image.shape, (np.ceil(transform.FixedBoundingBox.Height * expectedScale), np.ceil(256 * expectedScale)))

        result = at.TransformTile(transform, os.path.join(TilesDir, imageKey), distanceImage=None, target_space_scale=None, TargetRegion=(MinY, MinX, MinY + 256, MaxX))
        self.assertEqual(result.image.shape, (np.ceil(256 * expectedScale), np.ceil(transform.FixedBoundingBox.Width * expectedScale)))

        result = at.TransformTile(transform, os.path.join(TilesDir, imageKey), distanceImage=None, target_space_scale=None, TargetRegion=(MinY + 2048, MinX + 2048, MinY + 2048 + 512, MinX + 2048 + 512))
        self.assertEqual(result.image.shape, (np.ceil(512 * expectedScale), np.ceil(512 * expectedScale)))

    def CreateAssembleOptimizedTileTwo(self, mosaicFilePath, TilesDir, tile_dims=None, numColumnsPerPass=None):
        
        if tile_dims is None:
            tile_dims = (512,512)
            
        tile_dims = np.asarray(tile_dims, dtype=np.int64)
        
        mosaic = Mosaic.LoadFromMosaicFile(mosaicFilePath)
        mosaic.TranslateToZeroOrigin()
        mosaicBaseName = os.path.basename(mosaicFilePath)

        mosaicDir = os.path.dirname(mosaicFilePath)
        (mosaicBaseName, ext) = os.path.splitext(mosaicBaseName)
        
        expectedScale = 1.0 / self.DownsampleFromTilePath(TilesDir)
        
        scaled_fixed_bounding_box_shape = np.ceil(mosaic.FixedBoundingBox.shape / (1 / expectedScale)).astype(np.int64)
        
        expected_grid_dims = nornir_imageregistration.TileGridShape(scaled_fixed_bounding_box_shape,
                                                           tile_size=tile_dims)
        
        mosaic_expected_grid_dims = mosaic.CalculateGridDimensions(tile_dims, expectedScale)
        self.assertTrue(np.array_equal(expected_grid_dims, mosaic_expected_grid_dims), "Mosaic object grid dimensions should match manually calculated grid dimensions")
        
        max_temp_image_dims = expected_grid_dims * tile_dims
        if numColumnsPerPass is not None:
            max_temp_image_dims[1] = tile_dims[1] * numColumnsPerPass
            
        max_temp_image_area = np.prod(max_temp_image_dims)
        
        tiles = [[None for iCol in range(expected_grid_dims[1])] for iRow in range(expected_grid_dims[0])]#[[None] * expected_grid_dims[1]] * expected_grid_dims[0]
        tile_returned = np.zeros(expected_grid_dims, dtype=np.bool)
        #out = list(mosaic.GenerateOptimizedTiles(tilesPath=TilesDir, tile_dims=tile_dims, usecluster=False, target_space_scale=expectedScale))
        for t in mosaic.GenerateOptimizedTiles(tilesPath=TilesDir,
                                               tile_dims=tile_dims,
                                               usecluster=False,
                                               max_temp_image_area=max_temp_image_area,
                                               target_space_scale=expectedScale):
            (iRow, iCol, tile_image) = t
            assert(iCol < expected_grid_dims[1])
            assert(iRow < expected_grid_dims[0])
            
#            nornir_imageregistration.ShowGrayscale(tile_image)
            self.assertFalse(tile_returned[iRow, iCol], "Duplicate Row,Column value returned by enumerator")
            tile_returned[iRow, iCol] = True

            print("{0},{1} Tile completed".format(iRow, iCol))            
            # origin = np.asarray((iCol, iRow), dtype=np.int64) * tile_dims
            # region = nornir_imageregistration.Rectangle.CreateFromPointAndArea(origin, tile_dims*(1/expectedScale))
                        
#             (controlImage, mask) = mosaic.AssembleImage(tilesPath=TilesDir, FixedRegion=region,
#                                                 usecluster=True, target_space_scale=expectedScale)
#             self.assertTrue(np.array_equal(tile_image, controlImage))
            
            tiles[iRow][iCol] = tile_image
        
        self.assertTrue(np.all(tile_returned.flat))
        title = ""
        if numColumnsPerPass is not None:
            title = "Generated {0} columns in a pass.\n".format(numColumnsPerPass)
        
        title = title + "Tiles should not overlap and look reasonable" 
        self.assertTrue(nornir_imageregistration.ShowGrayscale(tiles, title=title, PassFail=True))
        

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
        
        self.CreateAssembleOptimizedTileTwo(mosaicFiles[0], tilesDir, numColumnsPerPass=2)
        self.CreateAssembleOptimizedTileTwo(mosaicFiles[0], tilesDir, numColumnsPerPass=3)
        self.CreateAssembleOptimizedTileTwo(mosaicFiles[0], tilesDir, numColumnsPerPass=1)
        self.CreateAssembleOptimizedTileTwo(mosaicFiles[0], tilesDir)
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
