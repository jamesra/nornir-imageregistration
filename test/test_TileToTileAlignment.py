'''
Created on Dec 13, 2013

@author: James Anderson
'''

import os
import pickle

import numpy as np

import nornir_imageregistration
import nornir_imageregistration.arrange_mosaic
import nornir_imageregistration.core as core
# from nornir_imageregistration.files.mosaicfile import MosaicFile
import nornir_imageregistration.layout
# from nornir_imageregistration.mosaic import Mosaic
# from nornir_imageregistration.mosaic_tileset import MosaicTileset
import nornir_imageregistration.mosaic
import nornir_imageregistration.mosaic_tileset
import nornir_imageregistration.tile_overlap
import setup_imagetest
import test_arrange


class TestMosaicTilesetTileOffsets(setup_imagetest.TransformTestBase):
    @property
    def TestName(self):
        return self.__class__.__name__

    def test_Alignment_RC3_0001(self):
        volume_dir = os.path.join("D:\\", "Data", "RC3", "TEM", "0001", "TEM")

        layout_files_dir = os.path.join(volume_dir, "Layouts", "Min_0.0")
        mosaic_tileset_filepath = os.path.join(layout_files_dir, "mosaic_tileset.pickle")
        with open(mosaic_tileset_filepath, "rb") as mosaic_tileset_file:
            mosaic_tileset = pickle.load(mosaic_tileset_file)

        self.align_two_tiles(mosaic_tileset, volume_dir, 0, 898, 899)
        self.align_two_tiles(mosaic_tileset, volume_dir, 0, 847, 848)
        self.align_two_tiles(mosaic_tileset, volume_dir, 0, 868, 869)
        self.align_two_tiles(mosaic_tileset, volume_dir, 0, 776, 802)
        self.align_two_tiles(mosaic_tileset, volume_dir, 2, 372, 394)

    def align_two_tiles(self, mosaic_tileset, volume_dir, iPass, TileA_ID, TileB_ID):
        layout_files_dir = os.path.join(volume_dir, "Layouts", "Min_0.0")
        layout_file = os.path.join(layout_files_dir, f"pass_{iPass}_tile_overlaps.pickle")

        self.assertTrue(os.path.exists(layout_file), f"Input layout file does not exist: {layout_file}")
        with open(layout_file, "rb") as input_layout_file:
            layout = pickle.load(input_layout_file)
            every_overlap = layout[0]
            del layout

        tile_to_overlaps_dict = nornir_imageregistration.arrange_mosaic.CreateTileToOverlapsDict(every_overlap)

        overlap_dict = tile_to_overlaps_dict[TileA_ID]
        tile_overlap = overlap_dict[(TileA_ID, TileB_ID)].tile_overlap

        Downsample = 4
        DownsampleString = "%03d" % Downsample
        self.TilesPath = os.path.join(volume_dir, "Leveled", "TilePyramid", DownsampleString)

        nornir_imageregistration.arrange_mosaic.ScoreTileOverlaps([tile_overlap])

        translated_layout = nornir_imageregistration.arrange_mosaic._FindTileOffsets(tile_overlaps=[tile_overlap],
                                                                                     excess_scalar=3.0,
                                                                                     image_to_source_space_scale=Downsample,
                                                                                     existing_layout=None)

        test_arrange.ShowTilesWithOffset(self, translated_layout, mosaic_tileset,
                                         TileA_ID, TileB_ID,
                                         self.TestOutputPath,
                                         "OutputImage-{TileA_ID}-{TileB_ID}",
                                         image_to_source_space_scale=Downsample,
                                         openwindow=True,
                                         weight=None)

        # self.RunAlignment(Tile127Filename, Tile148Filename, (-2, (4096 - 466) / Downsample), epsilon=Downsample*2, min_overlap=0.05)


class TestBasicTileAlignment(setup_imagetest.TransformTestBase):

    @property
    def TestName(self):
        return self.__class__.__name__

    def test_Alignments(self):
        Downsample = 1.0
        DownsampleString = "%03d" % Downsample
        self.TilesPath = os.path.join(self.ImportedDataPath, "PMG1", "Leveled", "TilePyramid", DownsampleString)

        Tile1Filename = "Tile000001.png"
        Tile2Filename = "Tile000002.png"
        Tile5Filename = "Tile000005.png"
        Tile6Filename = "Tile000006.png"
        Tile7Filename = "Tile000007.png"
        Tile9Filename = "Tile000009.png"

        self.RunAlignment(Tile7Filename, Tile9Filename, (908 / Downsample, 0), epsilon=1.5, min_overlap=0.05)
        self.RunAlignment(Tile5Filename, Tile6Filename, (2 / Downsample, 1260 / Downsample), epsilon=1.5,
                          min_overlap=0.05)
        self.RunAlignment(Tile1Filename, Tile2Filename, (2 / Downsample, 1260 / Downsample), epsilon=1.5,
                          min_overlap=0.05)

    def test_Alignment_Redmond(self):
        Downsample = 4
        DownsampleString = "%03d" % Downsample
        self.TilesPath = os.path.join(self.TestInputPath, "Images", "Alignment", "Redmond", DownsampleString)

        Tile127Filename = "127.png"
        Tile148Filename = "148.png"

        self.RunAlignment(Tile127Filename, Tile148Filename, (-2, (4096 - 466) / Downsample), epsilon=Downsample * 2,
                          min_overlap=0.05)

    def test_Alignment_RPC2_989_87_88(self):
        Downsample = 4
        DownsampleString = "%03d" % Downsample
        self.TilesPath = os.path.join(self.TestInputPath, "Images", "Alignment", "RPC2", DownsampleString)

        TileAFilename = "087.png"
        TileBFilename = "088.png"

        self.RunAlignment(TileAFilename, TileBFilename, ((4096 - 320) / Downsample, 4 / Downsample),
                          epsilon=Downsample * 2, min_overlap=0.05)

    def test_Alignment_RPC2_989_87_98(self):
        Downsample = 4
        DownsampleString = "%03d" % Downsample
        self.TilesPath = os.path.join(self.TestInputPath, "Images", "Alignment", "RPC2", DownsampleString)

        TileAFilename = "087.png"
        TileBFilename = "098.png"

        self.RunAlignment(TileAFilename, TileBFilename, (40 / Downsample, (4096 - 360) / Downsample),
                          epsilon=Downsample * 2, min_overlap=0.05)

    def test_Alignment_RC3_372_394(self):
        Downsample = 4
        DownsampleString = "%03d" % Downsample
        self.TilesPath = os.path.join(self.TestInputPath, "Images", "Alignment", "RPC2", DownsampleString)

        TileAFilename = "087.png"
        TileBFilename = "098.png"

        self.RunAlignment(TileAFilename, TileBFilename, (40 / Downsample, (4096 - 360) / Downsample),
                          epsilon=Downsample * 2, min_overlap=0.05)

    def test_MismatchSizeAlignments(self):
        self.TilesPath = os.path.join(self.TestInputPath, "Images", "Alignment")

        Tile1Filename = "402.png"
        Tile2Filename = "401_Subset.png"

        self.RunAlignment(Tile1Filename, Tile2Filename, (-529, -94), epsilon=1.0, min_overlap=0.05)

    def RunAlignment(self, TileAFilename, TileBFilename, ExpectedOffset, epsilon, min_overlap=0.5):
        '''ExpectedOffset is (Y,X)'''

        imFixed = core.LoadImage(os.path.join(self.TilesPath, TileAFilename),
                                 dtype=nornir_imageregistration.default_image_dtype())
        imMoving = core.LoadImage(os.path.join(self.TilesPath, TileBFilename),
                                  dtype=nornir_imageregistration.default_image_dtype())

        imFixedPadded = core.PadImageForPhaseCorrelation(imFixed, MinOverlap=min_overlap)
        imMovingPadded = core.PadImageForPhaseCorrelation(imMoving, MinOverlap=min_overlap)

        alignrecord = core.FindOffset(imFixedPadded, imMovingPadded, MinOverlap=0.05, MaxOverlap=0.5,
                                      FixedImageShape=imFixed.shape, MovingImageShape=imMoving.shape)

        print(str(alignrecord))

        tile = nornir_imageregistration.Tile(
            transform=alignrecord.ToTransform(fixedImageSize=imFixed.shape, warpedImageSize=imMoving.shape),
            imagepath=os.path.join(self.TilesPath, TileBFilename),
            image_to_source_space_scale=1,
            ID=0)

        targetRegion = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0, 0), imFixed.shape)
        alignedImageData = tile.Assemble(TargetRegion=targetRegion, SingleThreadedInvoke=True)

        deltaImage = alignedImageData.image - imFixed
        deltaImage = np.abs(deltaImage)

        user_passed = nornir_imageregistration.ShowGrayscale(
            (imFixed, imMoving, imFixedPadded, imMovingPadded, alignedImageData.image, deltaImage),
            title=f"First four are fixed/warped and padded versions\n5th is aligned, 6th is delta, should be nearly black in overlap region.\n{alignrecord}",
            PassFail=True)

        self.assertTrue(user_passed)

        self.assertAlmostEqual(alignrecord.peak[1], ExpectedOffset[1], delta=epsilon,
                               msg="X dimension incorrect: " + str(alignrecord.peak[1]) + " != " + str(
                                   ExpectedOffset[1]))
        self.assertAlmostEqual(alignrecord.peak[0], ExpectedOffset[0], delta=epsilon,
                               msg="Y dimension incorrect: " + str(alignrecord.peak[0]) + " != " + str(
                                   ExpectedOffset[0]))
