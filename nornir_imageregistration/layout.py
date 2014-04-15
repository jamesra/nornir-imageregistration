import os
import logging

from . import core
from . import alignment_record
import nornir_imageregistration.transforms.factory as tfactory

from rtree import index
import numpy as np

from . import spatial


def CreateTiles(transforms, imagepaths):
    '''Create tiles from pairs of transforms and image paths
    :param transform transforms: List of N transforms
    :param str imagepaths: List of N paths to image files
    :return: List of N tile objects
    '''

    tiles = []
    for i, t in enumerate(transforms):

        if not os.path.exists(imagepaths[i]):
            log = logging.getLogger("CreateTiles")
            log.error("Missing tile: " + imagepaths[i])
            continue

        tile = Tile(t, imagepaths[i], i)
        tiles.append(tile)

    return tiles


class Tile(object):
    '''
    A combination of a transform and a path to an image on disk.  Image will be loaded on demand
    '''

    __nextID = 0

    @property
    def MappedBoundingBox(self):
        return self._transform.MappedBoundingBox

    @property
    def ControlBoundingBox(self):
        return self._transform.FixedBoundingBox

    @property
    def OriginalImageSize(self):
        dims = self.MappedBoundingBox
        return (dims[spatial.iRect.MaxY] - dims[spatial.iRect.MinY], dims[spatial.iRect.MaxX] - dims[spatial.iRect.MinY])

    @property
    def WarpedImageSize(self):
        dims = self.ControlBoundingBox
        return (dims[spatial.iRect.MaxY] - dims[spatial.iRect.MinY], dims[spatial.iRect.MaxX] - dims[spatial.iRect.MinY])

    @property
    def Transform(self):
        return self._transform

    @property
    def Image(self):
        if self._image is None:
            img = core.LoadImage(self._imagepath)
            self._image = core.PadImageForPhaseCorrelation(img)
            del img

        return self._image

    @property
    def ImagePath(self):
        return self._imagepath

    @property
    def FFTImage(self):
        if self._fftimage is None:
            self._fftimage = np.fft.rfft2(self.Image)

        return self._fftimage

    @property
    def ID(self):
        return self._ID

    @classmethod
    def CreateTiles(cls, transforms, imagepaths):

        tiles = []
        for i, t in enumerate(transforms):

            if not os.path.exists(imagepaths[i]):
                log = logging.getLogger("CreateTiles")
                log.error("Missing tile: " + imagepaths[i])
                continue

            tile = Tile(t, imagepaths[i], i)
            tiles.append(tile)

        return tiles

    def __init__(self, transform, imagepath, ID=None):

        global __nextID

        self._transform = transform
        self._imagepath = imagepath
        self._image = None
        self._fftimage = None
        if ID is None:
            self._ID = Tile.__nextID
            Tile.__nextID += 1
        else:
            self._ID = ID

    def __str__(self):
        return "%d: %s" % (self._ID, self._imagepath)


class TileLayout(object):
    '''Arranges tiles in 2D space to form a mosaic'''

    @property
    def TileIDs(self):
        return list(self._TileToTransform.keys())

    def Contains(self, ID):
        return ID in self.TileToTransform

    def GetTileOrigin(self, ID):
        refTransform = self.TileToTransform[ID]
        (minY, minX, maxY, maxX) = refTransform.FixedBoundingBox

        return (minY, minX)

    @property
    def Tiles(self):
        '''Tiles contained within this layout'''
        return self._tiles

    def __init__(self, tiles):
        self._tiles = tiles
        self.TileToTransform = dict()


    def _AddIsolatedTile(self, tile):
        '''Add a tile with no overlap to other tiles'''

        if tile.ID in self.TileToTransform:
            return False

        self.TileToTransform[tile.ID] = TranslationTransformForAlignmentRecord(tile, arecord=None)

        return True

    def AddTileViaAlignment(self, arecord):

        if not arecord.FixedTileID in self.TileToTransform:
            refTile = self._tiles[arecord.FixedTileID]
            self._AddIsolatedTile(refTile)

        # Figure out the reference tiles position in the ctontrol (mosaic) space.  Add our offset to that
        (minY, minX) = self.GetTileOrigin(arecord.FixedTileID)

        movingTile = self._tiles[arecord.MovingTileID]
        arecord.translate((minY, minX))

        newTransform = TranslationTransformForAlignmentRecord(movingTile, arecord)
        self.TileToTransform[arecord.MovingTileID] = newTransform


    @classmethod
    def MergeLayouts(cls, layoutA, layoutB, offset):
        '''
        Merge B with A by translating all B transforms by offset.
        Then update the dictionary of A
        '''

        for t in list(layoutB.values()):
            t.translate(offset)

        layoutA.TileToTransform.update(layoutB.TileToTransform)


    @classmethod
    def MergeLayoutsWithAlignmentRecord(cls, layoutA, layoutB, arecord):
        '''
        Merge B with A by translating all B transforms by offset.
        Then update the dictionary of A
        '''

        FixedLayout = TileLayout.GetLayoutForID([layoutA, layoutB], arecord.FixedTileID)
        MovingLayout = TileLayout.GetLayoutForID([layoutA, layoutB], arecord.MovingTileID)

        assert(not FixedLayout is None and not MovingLayout is None)

        (fixedMinY, fixedMinX) = FixedLayout.GetTileOrigin(arecord.FixedTileID)
        (movingMinY, movingMinX) = MovingLayout.GetTileOrigin(arecord.MovingTileID)

        # arecord.translate((fixedMinY, fixedMinX))
        ExpectedMovingTilePosition = arecord.peak + np.array([fixedMinY, fixedMinX])

        MovingPositionDifference = np.array([ExpectedMovingTilePosition[0] - movingMinY, ExpectedMovingTilePosition[1] - movingMinX])

        for t in list(MovingLayout.TileToTransform.values()):
            t.TranslateFixed(MovingPositionDifference)

        FixedLayout.TileToTransform.update(MovingLayout.TileToTransform)


    @classmethod
    def GetLayoutForID(cls, listLayouts, ID):
        '''Given a list of tile layouts, returns the layout containing the given ID'''

        if listLayouts is None:
            return None

        for layout in listLayouts:
            if layout.Contains(ID):
                return layout

        return None

def TranslationTransformForAlignmentRecord(OriginalImageSize, arecord=None):
    '''
    :param Tile tile: A tile object
    :param AlignmentRecord arecord: An alignment record
    :return: A transform positioning the tile in 2D space
    '''
    if arecord is None:
        arecord = alignment_record.AlignmentRecord((0, 0), 0, 0)

    return tfactory.CreateRigidTransform(OriginalImageSize, OriginalImageSize, 0, arecord.peak)