'''
Created on Dec 13, 2013

@author: James Anderson
'''

import unittest
from . import setup_imagetest
import glob
import nornir_imageregistration.assemble_tiles as at
import nornir_imageregistration.tileset as tiles
import nornir_imageregistration.tile as tile
import nornir_imageregistration.core as core
from nornir_imageregistration.alignment_record import AlignmentRecord
from nornir_imageregistration.files.mosaicfile import MosaicFile
import os
import nornir_imageregistration.transforms.factory as tfactory
# from pylab import *
from scipy.misc import imsave
import numpy as np
from scipy import stats
import nornir_imageregistration.arrange_mosaic as arrange

from nornir_shared.tasktimer import TaskTimer

from nornir_imageregistration.mosaic  import Mosaic


def _GetFirstOffsetPair(tiles):
    '''Return the first offset from the tile list'''

    tileA = tiles[0]

    tileB_ID = list(tileA.OffsetToTile.keys())[0]

    tileB = tiles[tileB_ID]

    offset = tileA.OffsetToTile[tileB_ID]

    return (tileA, tileB, offset)


class TestBasicTileAlignment(setup_imagetest.MosaicTestBase):

    def test_Alignments(self):

        self.TilesPath = os.path.join(self.ImportedDataPath, "PMG1", "Leveled", "TilePyramid", "002")

        Tile1Filename = "Tile000001.png"
        Tile2Filename = "Tile000002.png"
        Tile5Filename = "Tile000005.png"
        Tile6Filename = "Tile000006.png"
        Tile7Filename = "Tile000007.png"
        Tile9Filename = "Tile000009.png"

        self.RunAlignment(Tile1Filename, Tile2Filename, (0, 630))
        self.RunAlignment(Tile5Filename, Tile6Filename, (1, 630))
        self.RunAlignment(Tile7Filename, Tile9Filename, (454, 0))


    def test_MismatchSizeAlignments(self):

        self.TilesPath = os.path.join(self.TestInputPath, "Images", "Alignment")

        Tile1Filename = "402.png"
        Tile2Filename = "401_Subset.png"

        # self.RunAlignment(Tile1Filename, Tile2Filename, (-529, -93))


    def RunAlignment(self, TileAFilename, TileBFilename, ExpectedOffset):
        '''ExpectedOffset is (Y,X)'''

        imFixed = core.LoadImage(os.path.join(self.TilesPath, TileAFilename))
        imMoving = core.LoadImage(os.path.join(self.TilesPath, TileBFilename))

        imFixedPadded = core.PadImageForPhaseCorrelation(imFixed)
        imMovingPadded = core.PadImageForPhaseCorrelation(imMoving)

        alignrecord = core.FindOffset(imFixedPadded, imMovingPadded)

        self.assertAlmostEqual(alignrecord.peak[1], ExpectedOffset[1], delta=2, msg="X dimension incorrect: " + str(alignrecord.peak) + " != " + str(ExpectedOffset))
        self.assertAlmostEqual(alignrecord.peak[0], ExpectedOffset[0], delta=2, msg="Y dimension incorrect: " + str(alignrecord.peak) + " != " + str(ExpectedOffset))


class TestMosaicArrange(setup_imagetest.MosaicTestBase):

    @property
    def MosaicFiles(self, testName=None):
        if testName is None:
            testName = "PMG1"

        return glob.glob(os.path.join(self.ImportedDataPath, testName, "Stage.mosaic"))


    def RigidTransformForTile(self, tile, arecord=None):
        if arecord is None:
            arecord = AlignmentRecord((0, 0), 0, 0)

        return tfactory.CreateRigidTransform(tile.OriginalImageSize, tile.OriginalImageSize, 0, arecord.peak)

    def ShowTilesWithOffset(self, tileA, tileB, offset):

        # transformA = self.RigidTransformForTile(tileA, AlignmentRecord((0, -624 * 2.0), 0, 0))
        transformA = self.RigidTransformForTile(tileA)
        transformB = self.RigidTransformForTile(tileB, offset)

        ImageToTransform = {}
        ImageToTransform[tileA.ImagePath] = transformA
        ImageToTransform[tileB.ImagePath] = transformB

        mosaic = Mosaic(ImageToTransform)

        self._ShowMosaic(mosaic)

    def _ShowMosaic(self, mosaic):

        (assembledImage, mask) = mosaic.AssembleTiles(tilesPath=None, usecluster=False)

        core.ShowGrayscale(assembledImage)

    def __CheckNoOffsetsToSelf(self, tiles):

        for i, t in enumerate(tiles):
            self.assertFalse(i in t.OffsetToTile, "Tiles should not be registered to themselves")


    def __RemoveExtraImages(self, mosaic):

        '''Remove all but the first two images'''
        keys = list(mosaic.ImageToTransform.keys())
        keys.sort()

        for i, k in enumerate(keys):
            if i >= 2:
                del mosaic.ImageToTransform[k]


    def ArrangeMosaicDirect(self, mosaicFilePath, outputMosaicPath, parallel=False, downsamplePath=None):

        if downsamplePath is None:
            downsamplePath = '001'

        scale = 1.0 / float(downsamplePath)

        mosaic = Mosaic.LoadFromMosaicFile(mosaicFilePath)
        mosaicBaseName = os.path.basename(mosaicFilePath)

        (mosaicBaseName, ext) = os.path.splitext(mosaicBaseName)

        TilesDir = os.path.join(self.ImportedDataPath, 'Test1', 'Leveled', 'TilePyramid', downsamplePath)

#        mosaic.TranslateToZeroOrigin()

        self.__RemoveExtraImages(mosaic)

       # assembleScale = tiles.MostCommonScalar(mosaic.ImageToTransform.values(), mosaic.TileFullPaths(TilesDir))

       # expectedScale = 1.0 / float(downsamplePath)

      #  self.assertEqual(assembleScale, expectedScale, "Scale for assemble does not match the expected scale")

        timer = TaskTimer()

        timer.Start("ArrangeTiles " + TilesDir)

        tilesPathList = mosaic.CreateTilesPathList(TilesDir)

        tiles = tile.Tile.CreateTiles(list(mosaic.ImageToTransform.values()), tilesPathList)

        arrange._FindTileOffsets(tiles, scale)

        self.__CheckNoOffsetsToSelf(tiles)

        # Each tile should contain a dictionary with the known offsets.  Show the overlapping images using the calculated offsets

        (tileA, tileB, offset) = _GetFirstOffsetPair(tiles)

        self.ShowTilesWithOffset(tileA, tileB, offset)

        # mosaic.ArrangeTilesWithTranslate(TilesDir, usecluster=parallel)

        timer.End("ArrangeTiles " + TilesDir, True)

        OutputDir = os.path.join(self.TestOutputPath, outputMosaicPath)


    def ArrangeMosaic(self, mosaicFilePath, outputMosaicPath, parallel=False, downsamplePath=None):

        if downsamplePath is None:
            downsamplePath = '001'

        mosaic = Mosaic.LoadFromMosaicFile(mosaicFilePath)
        mosaicBaseName = os.path.basename(mosaicFilePath)

        (mosaicBaseName, ext) = os.path.splitext(mosaicBaseName)

        TilesDir = os.path.join(self.ImportedDataPath, 'Test1', 'Leveled', 'TilePyramid', downsamplePath)

#        mosaic.TranslateToZeroOrigin()

        # self.__RemoveExtraImages(mosaic)

       # assembleScale = tiles.MostCommonScalar(mosaic.ImageToTransform.values(), mosaic.TileFullPaths(TilesDir))

       # expectedScale = 1.0 / float(downsamplePath)

      #  self.assertEqual(assembleScale, expectedScale, "Scale for assemble does not match the expected scale")

        timer = TaskTimer()

        timer.Start("ArrangeTiles " + TilesDir)

        newMosaic = mosaic.ArrangeTilesWithTranslate(TilesDir, usecluster=False)

        self._ShowMosaic(newMosaic)

        timer.End("ArrangeTiles " + TilesDir, True)

        OutputDir = os.path.join(self.TestOutputPath, outputMosaicPath)

    def test_ArrangeMosaic(self):

        for m in self.MosaicFiles:
            self.ArrangeMosaic(m, 'ArrangeWithTranslateEachMosaicTyepe', parallel=False, downsamplePath='001')

        print("All done")

    def test_ArrangeMosaicDirect(self):

        for m in self.MosaicFiles:
            self.ArrangeMosaicDirect(m, 'ArrangeWithTranslateEachMosaicTyepe', parallel=False, downsamplePath='001')

        print("All done")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']

    import nornir_shared.misc
    nornir_shared.misc.RunWithProfiler("unittest.main()")