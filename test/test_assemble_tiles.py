'''
Created on Oct 28, 2013

@author: u0490822
'''
import glob
import os
import time
import unittest
from typing import AnyStr

from scipy import stats 

import nornir_imageregistration
import nornir_imageregistration.assemble_tiles as at
import nornir_imageregistration.tileset as tiles
import nornir_imageregistration.transforms.factory as tfactory
import nornir_imageregistration.mosaic_tileset

from nornir_imageregistration.files.mosaicfile import MosaicFile
from nornir_imageregistration.mosaic  import Mosaic

from nornir_shared.tasktimer import TaskTimer
import numpy as np
import cupy as cp

import setup_imagetest 

init_context = cp.zeros((64,64))

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
        
        nornir_imageregistration.SaveImage(outputImagePath, z, bpp=8)

        self.assertTrue(os.path.exists(outputImagePath), "OutputImage not found, test cannot write to disk!")

        os.remove(outputImagePath)


    def AssembleMosaic(self, mosaicFilePath, tilesDir, outputMosaicPath=None, parallel=False, use_cp: bool=False, downsamplePath=None):

        SaveFiles = not outputMosaicPath is None

        mosaic = Mosaic.LoadFromMosaicFile(mosaicFilePath)
        mosaicBaseName = os.path.basename(mosaicFilePath)
        (mosaicBaseName, ext) = os.path.splitext(mosaicBaseName)
 
        downsample = self.DownsampleFromTilePath(tilesDir)
        expectedScale = 1.0 / downsample

        mosaic_set = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaic, tilesDir, downsample)
        mosaic_set.TranslateToZeroOrigin()
        timer = TaskTimer()

        first_tile = next(iter(mosaic_set.values())) # type: nornir_imageregistration.ITransform
        timing_key = f'AssembleImage: {os.path.basename(mosaicFilePath)} {tilesDir}' if first_tile is None else \
                     f'AssembleImage: {first_tile.Transform.type} {os.path.basename(mosaicFilePath)} {tilesDir}'
                     
        timer.Start(timing_key)

        (mosaicImage, mask) = mosaic_set.AssembleImage(usecluster=parallel, use_cp=use_cp)

        timer.End(timing_key, True)

        self.assertEqual(mosaicImage.shape[0], np.ceil(mosaic_set.FixedBoundingBoxHeight * expectedScale), "Output mosaic height does not match .mosaic height %g vs %g" % (mosaicImage.shape[0], mosaic_set.FixedBoundingBoxHeight * expectedScale))
        self.assertEqual(mosaicImage.shape[1], np.ceil(mosaic_set.FixedBoundingBoxWidth * expectedScale), "Output mosaic width does not match .mosaic height %g vs %g" % (mosaicImage.shape[1], mosaic_set.FixedBoundingBoxWidth * expectedScale))
        # img = im.fromarray(mosaicImage, mode="I")
        # img.save(outputImagePath, mosaicImage, bits=8)

        if SaveFiles:
            OutputDir = os.path.join(self.TestOutputPath, outputMosaicPath)

            if not os.path.exists(OutputDir):
                os.makedirs(OutputDir)

            outputImagePath = os.path.join(OutputDir, mosaicBaseName + '.png')
            outputImageMaskPath = os.path.join(OutputDir, mosaicBaseName + '_mask.png')

            nornir_imageregistration.SaveImage(outputImagePath, mosaicImage, bpp=8)
            nornir_imageregistration.SaveImage(outputImageMaskPath, mask)
            self.assertTrue(os.path.exists(outputImagePath), "OutputImage not found")

            outputMask = nornir_imageregistration.LoadImage(outputImageMaskPath)
            self.assertTrue(outputMask[int(outputMask.shape[0] / 2.0), int(outputMask.shape[1] / 2.0)] > 0, "Center of assembled image mask should be non-zero")

            del mosaicImage
            del mask
        else:
            return (mosaicImage, mask)



    def CompareMosaicAsssembleAndTransformTile(self, mosaicFilePath:str, tilesDir:str, downsample:float, use_cp: bool=False):
        """
        1) Assemble the entire mosaic
        2) Assemble subregion of the mosaic
        3) Assemble subregion directly using _TransformTile
        4) Check the output of all match
        """

        mosaicObj = Mosaic.LoadFromMosaicFile(mosaicFilePath)
        self.assertIsNotNone(mosaicObj, "Mosaic not loaded")
        
        mosaicTileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaicObj, tilesDir, downsample)
        mosaicTileset.TranslateToZeroOrigin()

        mosaicObj.TranslateToZeroOrigin()

        (MinY, MinX, MaxY, MaxZ) = mosaicTileset.TargetBoundingBox
        print("Unscaled fixed region %s" % str(mosaicTileset.TargetBoundingBox))
        
        ScaledFixedRegion = np.array([MinY + 512, MinX + 1024, MinY + 1024, MinX + 2048])
        FixedRegion = nornir_imageregistration.Rectangle(ScaledFixedRegion * downsample)
        
        ### Find a tile that intersects our region of interest.
        intersecting_tile = mosaicTileset.TargetSpaceIntersections(FixedRegion)[0]
        imageKey = intersecting_tile.ImagePath
        transform = intersecting_tile.Transform
  
        (tileImage, tileMask) = mosaicTileset.AssembleImage(FixedRegion=FixedRegion, usecluster=False, use_cp = use_cp, target_space_scale=1.0/downsample)
        # self.assertEqual(tileImage.shape, (ScaledFixedRegion[3], ScaledFixedRegion[2]))

        (clustertileImage, clustertileMask) = mosaicTileset.AssembleImage(FixedRegion=FixedRegion, usecluster=True, target_space_scale=1.0/downsample)
        # self.assertEqual(tileImage.shape, (ScaledFixedRegion[3], ScaledFixedRegion[2]))

        cluster_delta = np.abs(clustertileImage - tileImage)
        cluster_delta_sum = np.sum(cluster_delta.flat) 
        if cluster_delta_sum >= 0.65:
            nornir_imageregistration.ShowGrayscale([cluster_delta, cluster_delta > 0],
                        title=f"Unexpected high delta of image: {imageKey}\n{str(transform.FixedBoundingBox)}\nPlease double check they are identical (nearly all black).\nSecond image is a mask showing non-zero values.", PassFail=False)
            
        #10-13-2022: This test passes if the cluster composites the tiles in the same order as the single-threaded assembly.


        test_tile = nornir_imageregistration.Tile(transform, os.path.join(tilesDir, imageKey), image_to_source_space_scale=downsample, ID=0)
        result = at.TransformTile(test_tile, distanceImage=None, target_space_scale=None, TargetRegion=FixedRegion)
        self.assertEqual(result.image.shape, (ScaledFixedRegion[2] - ScaledFixedRegion[0], ScaledFixedRegion[3] - ScaledFixedRegion[1]))

        # nornir_imageregistration.ShowGrayscale([tileImage, result.image])
        (wholeimage, wholemask) = self.AssembleMosaic(mosaicFilePath, tilesDir, outputMosaicPath=None, parallel=False)
        self.assertIsNotNone(wholeimage, "Assemble did not produce an image")
        self.assertIsNotNone(wholemask, "Assemble did not produce a mask")

        croppedWholeImage = nornir_imageregistration.CropImage(wholeimage,
                                                    int(ScaledFixedRegion[1]),
                                                    int(ScaledFixedRegion[0]),
                                                    int(ScaledFixedRegion[3] - ScaledFixedRegion[1]),
                                                    int(ScaledFixedRegion[2] - ScaledFixedRegion[0]))

        self.assertTrue(nornir_imageregistration.ShowGrayscale([(result.image, cluster_delta), (tileImage, clustertileImage), (croppedWholeImage, wholeimage)],
                        title="image: %s\n%s" % (imageKey, str(transform.FixedBoundingBox)),
                        image_titles=(("Transform Tile", "Cluster vs Single Thread Delta"), ("Assemble Image", "Multi-threaded Assemble"), ("Cropped Mosaic", "Assemble Mosaic")), PassFail=True))

        # self.assertTrue(cluster_delta_sum < 0.65, "Tiles generated with cluster should be identical to single threaded implementation")
        # self.assertTrue(np.array_equal(clustertileMask, tileMask), "Tiles generated with cluster should be identical to single threaded implementation")


    def CreateAssembleOptimizedTile(self, mosaicFilePath, TilesDir, downsample, SingleThread: bool=False, use_cp: bool=False):
        mosaic = Mosaic.LoadFromMosaicFile(mosaicFilePath)
        mosaicBaseName = os.path.basename(mosaicFilePath)

        mosaicDir = os.path.dirname(mosaicFilePath)
        (mosaicBaseName, ext) = os.path.splitext(mosaicBaseName)

        imageKey = list(mosaic.ImageToTransform.keys())[0]
        transform = mosaic.ImageToTransform[imageKey]

        (MinY, MinX, MaxY, MaxX) = nornir_imageregistration.Rectangle.SafeRound(transform.FixedBoundingBox)
        expected_height = MaxY - MinY
        expected_width = MaxX - MinX

        expectedScale = 1.0 / self.DownsampleFromTilePath(TilesDir)
        
        tile = nornir_imageregistration.tile.Tile(transform, os.path.join(TilesDir, imageKey),
                                                  image_to_source_space_scale=downsample, ID=0)

        result = at.TransformTile(tile, distanceImage=None, target_space_scale=None, TargetRegion=(MinY, MinX, MaxY, MinX + 256), SingleThreadedInvoke=SingleThread, use_cp=use_cp)
        self.assertEqual(result.image.shape, (np.ceil(expected_height * expectedScale), np.ceil(256 * expectedScale)))

        result = at.TransformTile(tile, distanceImage=None, target_space_scale=None, TargetRegion=(MinY, MinX, MinY + 256, MaxX), SingleThreadedInvoke=SingleThread, use_cp=use_cp)
        self.assertEqual(result.image.shape, (np.ceil(256 * expectedScale), np.ceil(expected_width * expectedScale)))

        result = at.TransformTile(tile, distanceImage=None, target_space_scale=None, TargetRegion=(MinY + 2048, MinX + 2048, MinY + 2048 + 512, MinX + 2048 + 512), SingleThreadedInvoke=SingleThread, use_cp=use_cp)
        self.assertEqual(result.image.shape, (np.ceil(512 * expectedScale), np.ceil(512 * expectedScale)))

    def CreateAssembleOptimizedTileTwo(self, mosaicTileset, tile_dims=None, numColumnsPerPass=None, usecluster: bool=False, use_cp: bool=False):
        
        if tile_dims is None:
            tile_dims = (512,512)
            
        tile_dims = np.asarray(tile_dims, dtype=np.int64)

        #mosaicBaseName = os.path.basename(mosaicFilePath)

        #mosaicDir = os.path.dirname(mosaicFilePath)
        #(mosaicBaseName, ext) = os.path.splitext(mosaicBaseName)
        
        expectedScale = 1.0 / mosaicTileset.image_to_source_space_scale
        #mosaicTileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaic,
        #                                               image_folder=TilesDir,
        #                                               source_space_scale=expectedScale)
        mosaicTileset.TranslateToZeroOrigin()
        
        scaled_fixed_bounding_box_shape = np.ceil(mosaicTileset.TargetBoundingBox.shape / (1 / expectedScale)).astype(np.int64)
        
        expected_grid_dims = nornir_imageregistration.TileGridShape(scaled_fixed_bounding_box_shape,
                                                           tile_size=tile_dims)
        
        mosaic_expected_grid_dims = mosaicTileset.CalculateGridDimensions(tile_dims, expectedScale)
        self.assertTrue(np.array_equal(expected_grid_dims, mosaic_expected_grid_dims), "Mosaic object grid dimensions should match manually calculated grid dimensions")
        
        max_temp_image_dims = expected_grid_dims * tile_dims
        if numColumnsPerPass is not None:
            max_temp_image_dims[1] = tile_dims[1] * numColumnsPerPass
            
        max_temp_image_area = np.prod(max_temp_image_dims)
        
        nRows = expected_grid_dims[0]
        tiles = [[None for iCol in range(expected_grid_dims[1])] for iRow in range(expected_grid_dims[0])]#[[None] * expected_grid_dims[1]] * expected_grid_dims[0]
        tile_returned = np.zeros(expected_grid_dims, dtype=bool)
        #out = list(mosaic.GenerateOptimizedTiles(tilesPath=TilesDir, tile_dims=tile_dims, usecluster=False, target_space_scale=expectedScale))
        for t in mosaicTileset.GenerateOptimizedTiles(tile_dims=tile_dims,
                                                      usecluster=usecluster,
                                                      use_cp=use_cp,
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
            
            tiles[(nRows - 1) - iRow][iCol] = tile_image
        
        self.assertTrue(np.all(tile_returned.flat))
        title = ""
        if numColumnsPerPass is not None:
            title = "Generated {0} columns in a pass.\n".format(numColumnsPerPass)
        
        title = title + "Tiles should not overlap and look reasonable" 
        self.assertTrue(nornir_imageregistration.ShowGrayscale(tiles, title=title, PassFail=True))
        

    def CreateAssembleEachMosaic(self, mosaicFiles: list[AnyStr], tilesDir: str, use_cp: bool=False):

        for m in mosaicFiles:
            self.AssembleMosaic(m, tilesDir, 'CreateAssembleEachMosaic', parallel=False, use_cp=use_cp)
            # self. AssembleMosaic(m, 'CreateAssembleEachMosaicType', parallel=False)

        print("All done")


    def ParallelAssembleEachMosaic(self, mosaicFiles: list[AnyStr], tilesDir: str):

        for m in mosaicFiles:
            self.AssembleMosaic(m, tilesDir , 'ParallelAssembleEachMosaic', parallel=True)
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

    def test_AssemblePMG_GPU(self):
        testName = "PMG1"

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath()

        self.CreateAssembleEachMosaic(mosaicFiles, tilesDir, use_cp=True)

    def test_AssemblePMG_Parallel(self):
        testName = "PMG1"

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath()

        self.ParallelAssembleEachMosaic(mosaicFiles, tilesDir)


class IDOCTests(TestMosaicAssemble):

    @property
    def TestName(self):
        return "IDOC1"

    def test_AssembleIDOC_DS1(self):
        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath='001')

        self.CreateAssembleEachMosaic(mosaicFiles, tilesDir)

    def test_AssembleIDOC_DS4(self):
        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath='004')

        self.CreateAssembleEachMosaic(mosaicFiles, tilesDir)

    def test_AssembleIDOC_DS1_GPU(self):
        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath='001')

        self.CreateAssembleEachMosaic(mosaicFiles, tilesDir, use_cp=True)

    def test_AssembleIDOC_DS4_GPU(self):
        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath='004')

        self.CreateAssembleEachMosaic(mosaicFiles, tilesDir, use_cp=True)

    def test_AssembleIDOC_DS1_Parallel(self):
    
        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath='001')
    
        self.ParallelAssembleEachMosaic(mosaicFiles, tilesDir)

    def test_AssembleIDOC_DS4_Parallel(self):
        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath='004')

        self.ParallelAssembleEachMosaic(mosaicFiles, tilesDir)

    def test_AssembleOptimizedTilesIDoc(self):
        '''Assemble small 512x512 tiles from a transform and image in a mosaic'''

        downsamplePath = '004'

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath)
         
        mosaicObj = Mosaic.LoadFromMosaicFile(mosaicFiles[0])
        mosaicTileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaicObj, tilesDir, float(downsamplePath))
        mosaicTileset.TranslateToZeroOrigin()

        print("CreateAssembleOptimizedTileTwo - numColumnsPerPass 2")
        self.CreateAssembleOptimizedTileTwo(mosaicTileset, numColumnsPerPass=2)
        print("CreateAssembleOptimizedTileTwo - numColumnsPerPass 3")
        self.CreateAssembleOptimizedTileTwo(mosaicTileset, numColumnsPerPass=3)
        print("CreateAssembleOptimizedTileTwo - numColumnsPerPass 1")
        self.CreateAssembleOptimizedTileTwo(mosaicTileset, numColumnsPerPass=1)
        print("CreateAssembleOptimizedTileTwo - numColumnsPerPass None")
        self.CreateAssembleOptimizedTileTwo(mosaicTileset)

    def test_AssembleOptimizedTilesIDoc_GPU(self):
        '''Assemble small 512x512 tiles from a transform and image in a mosaic'''

        downsamplePath = '004'

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath)

        mosaicObj = Mosaic.LoadFromMosaicFile(mosaicFiles[0])
        mosaicTileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaicObj, tilesDir,
                                                                                 float(downsamplePath))
        mosaicTileset.TranslateToZeroOrigin()

        print("CreateAssembleOptimizedTileTwo - numColumnsPerPass 2")
        self.CreateAssembleOptimizedTileTwo(mosaicTileset, numColumnsPerPass=2, use_cp=True)
        print("CreateAssembleOptimizedTileTwo - numColumnsPerPass 3")
        self.CreateAssembleOptimizedTileTwo(mosaicTileset, numColumnsPerPass=3, use_cp=True)
        print("CreateAssembleOptimizedTileTwo - numColumnsPerPass 1")
        self.CreateAssembleOptimizedTileTwo(mosaicTileset, numColumnsPerPass=1, use_cp=True)
        print("CreateAssembleOptimizedTileTwo - ColumnsPerPass None")
        self.CreateAssembleOptimizedTileTwo(mosaicTileset)

    def test_AssembleOptimizedTilesIDoc_Cluster(self):
        '''Assemble small 512x512 tiles from a transform and image in a mosaic'''

        downsamplePath = '004'

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath)

        mosaicObj = Mosaic.LoadFromMosaicFile(mosaicFiles[0])
        mosaicTileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaicObj, tilesDir,
                                                                                 float(downsamplePath))
        mosaicTileset.TranslateToZeroOrigin()

        print("CreateAssembleOptimizedTileTwo - numColumnsPerPass 2")
        self.CreateAssembleOptimizedTileTwo(mosaicTileset, numColumnsPerPass=2, usecluster=True, use_cp=False)
        print("CreateAssembleOptimizedTileTwo - numColumnsPerPass 3")
        self.CreateAssembleOptimizedTileTwo(mosaicTileset, numColumnsPerPass=3, usecluster=True, use_cp=False)
        print("CreateAssembleOptimizedTileTwo - numColumnsPerPass 1")
        self.CreateAssembleOptimizedTileTwo(mosaicTileset, numColumnsPerPass=1, usecluster=True, use_cp=False)
        print("CreateAssembleOptimizedTileTwo - ColumnsPerPass None")
        self.CreateAssembleOptimizedTileTwo(mosaicTileset)

    def test_AssembleAndTransformTileIDoc(self):
        '''Assemble small 512x512 tiles from a transform and image in a mosaic'''

        downsamplePath = '004'

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath)
         
        mosaicObj = Mosaic.LoadFromMosaicFile(mosaicFiles[0])
        mosaicTileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaicObj, tilesDir, float(downsamplePath))
        mosaicTileset.TranslateToZeroOrigin()
        
        self.CompareMosaicAsssembleAndTransformTile(mosaicFiles[0],tilesDir, float(downsamplePath))  
        self.CreateAssembleOptimizedTile(mosaicFiles[0], tilesDir, float(downsamplePath))

    def test_AssembleAndTransformTileIDoc_GPU(self):
        '''Assemble small 512x512 tiles from a transform and image in a mosaic'''

        downsamplePath = '004'

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath)

        mosaicObj = Mosaic.LoadFromMosaicFile(mosaicFiles[0])
        mosaicTileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaicObj, tilesDir,
                                                                                 float(downsamplePath))
        mosaicTileset.TranslateToZeroOrigin()

        self.CompareMosaicAsssembleAndTransformTile(mosaicFiles[0], tilesDir, float(downsamplePath), use_cp=True)
        self.CreateAssembleOptimizedTile(mosaicFiles[0], tilesDir, float(downsamplePath), SingleThread=True, use_cp=True)

    def test_AssembleOptimizedTileIDoc(self):
        '''Assemble small 512x512 tiles from a transform and image in a mosaic'''

        downsamplePath = '004'

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath)
         
        mosaicObj = Mosaic.LoadFromMosaicFile(mosaicFiles[0])
        mosaicTileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaicObj, tilesDir, float(downsamplePath))
        mosaicTileset.TranslateToZeroOrigin()
         
        self.CreateAssembleOptimizedTile(mosaicFiles[0], tilesDir, float(downsamplePath))

    def test_AssembleOptimizedTileIDoc_DS1_MultiThread(self):
        '''Assemble small 512x512 tiles from a transform and image in a mosaic'''

        downsamplePath = '001'

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath)

        mosaicObj = Mosaic.LoadFromMosaicFile(mosaicFiles[0])
        mosaicTileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaicObj, tilesDir,
                                                                                 float(downsamplePath))
        mosaicTileset.TranslateToZeroOrigin()

        self.CreateAssembleOptimizedTile(mosaicFiles[0], tilesDir, float(downsamplePath))

    def test_AssembleOptimizedTileIDoc_DS1_SingleThread(self):
        '''Assemble small 512x512 tiles from a transform and image in a mosaic'''

        downsamplePath = '001'

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath)

        mosaicObj = Mosaic.LoadFromMosaicFile(mosaicFiles[0])
        mosaicTileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaicObj, tilesDir,
                                                                                 float(downsamplePath))
        mosaicTileset.TranslateToZeroOrigin()

        self.CreateAssembleOptimizedTile(mosaicFiles[0], tilesDir, float(downsamplePath), SingleThread=True)

    def test_AssembleOptimizedTileIDoc_DS1_GPU(self):
        '''Assemble small 512x512 tiles from a transform and image in a mosaic'''

        downsamplePath = '001'

        mosaicFiles = self.GetMosaicFiles()
        tilesDir = self.GetTileFullPath(downsamplePath)

        mosaicObj = Mosaic.LoadFromMosaicFile(mosaicFiles[0])
        mosaicTileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaicObj, tilesDir,
                                                                                 float(downsamplePath))
        mosaicTileset.TranslateToZeroOrigin()

        self.CreateAssembleOptimizedTile(mosaicFiles[0], tilesDir, float(downsamplePath), SingleThread=True, use_cp=True)

    # def test_AssembleTilesIDoc_JPeg2000(self):
    #     '''Assemble small 256x265 tiles from a transform and image in a mosaic'''
    #
    #     downsamplePath = '001'
    #
    #     mosaicFiles = self.GetMosaicFiles()
    #     tilesDir = self.GetTileFullPath(downsamplePath)
    #
    #     ImageFullPath = os.path.join(self.TestOutputPath, 'Image.jp2')
    #
    #     (image, mask) = self.AssembleMosaic(mosaicFiles[0], tilesDir, parallel=True)
    #
    #     nornir_imageregistration.SaveImage_JPeg2000(ImageFullPath, image)
    #     self.assertTrue(os.path.exists(ImageFullPath), "File should be written to disk for JPeg2000")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']

    import nornir_shared.misc
    nornir_shared.misc.RunWithProfiler("unittest.main()")
