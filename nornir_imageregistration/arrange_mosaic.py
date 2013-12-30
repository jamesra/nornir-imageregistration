'''
Created on Jul 10, 2012

@author: Jamesan
'''

import scipy.sparse
import nornir_imageregistration.core as core
from rtree import index
import numpy as np
import logging
import os
import nornir_imageregistration.tiles as tileModule
from alignment_record import AlignmentRecord
import nornir_imageregistration.transforms.factory as tfactory

from operator import attrgetter

class Tile(object):

    __nextID = 0

    @property
    def MappedBoundingBox(self):
        return self._transform.MappedPointBoundingBox

    @property
    def ControlBoundingBox(self):
        return self._transform.ControlPointBoundingBox

    @property
    def OriginalImageSize(self):
        dims = self.MappedBoundingBox
        return (dims[3] - dims[1], dims[2] - dims[0])

    @property
    def WarpedImageSize(self):
        dims = self.ControlBoundingBox
        return (dims[3] - dims[1], dims[2] - dims[0])

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
    def OffsetToTile(self):
        '''Maps a tile ID to an offset'''
        return self._offsetToTile

    @property
    def ID(self):
        return self._ID

    def __init__(self, transform, imagepath, ID=None):

        global __nextID

        self._transform = transform
        self._imagepath = imagepath
        self._image = None
        self._fftimage = None
        self._offsetToTile = dict()

        if ID is None:
            self._ID = Tile.__nextID
            Tile.__nextID += 1
        else:
            self._ID = ID


    def __str__(self):
        return "%d: %s" % (self._ID, self._imagepath)



class TileLayout(object):

    @property
    def TileIDs(self):
        return self._TileToTransform.keys()

    def Contains(self, ID):
        return ID in self.TileToTransform

    def GetTileOrigin(self, ID):
        refTransform = self.TileToTransform[ID]
        (minX, minY, maxX, maxY) = refTransform.ControlPointBoundingBox

        return (minY, minX)

    @property
    def Tiles(self):
        return self._tiles

    def __init__(self, tiles):
        self._tiles = tiles
        self.TileToTransform = dict()



    def _AddIsolatedTile(self, tile):
        '''Add a tile with no overlap to other tiles'''

        if tile.ID in self.TileToTransform:
            return False

        self.TileToTransform[tile.ID] = _RigidTransformForTile(tile, arecord=None)

        return True

    def AddTileViaAlignment(self, arecord):

        if not arecord.FixedTileID in self.TileToTransform:
            refTile = self._tiles[arecord.FixedTileID]
            self._AddIsolatedTile(refTile)

        # Figure out the reference tiles position in the ctontrol (mosaic) space.  Add our offset to that
        (minY, minX) = self.GetTileOrigin(arecord.FixedTileID)

        movingTile = self._tiles[arecord.MovingTileID]
        arecord.translate((minY, minX))

        newTransform = _RigidTransformForTile(movingTile, arecord)
        self.TileToTransform[arecord.MovingTileID] = newTransform


    @classmethod
    def MergeLayouts(cls, layoutA, layoutB, offset):
        '''
        Merge B with A by translating all B transforms by offset.
        Then update the dictionary of A
        '''

        for t in layoutB.values():
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

        for t in MovingLayout.TileToTransform.values():
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




def CreateSpatialMap(tiles):
    '''
    Given a set of transforms, create an rtree index
    
    rtree pickles inserted into the tree.  This means we cannot modify them and query them later and get the modified object
    for this reason we have to keep a list of objects and store the index of the objects in the list as a key
    '''

    idx = index.Index()

    for i, t in enumerate(tiles):

        idx.insert(i, t.ControlBoundingBox, obj=t)

    return idx


def CreateTiles(transforms, imagepaths):

    tiles = []
    for i, t in enumerate(transforms):

        if not os.path.exists(imagepaths[i]):
            log = logging.getLogger("CreateTiles")
            log.error("Missing tile: " + imagepaths[i])
            continue

        tile = Tile(t, imagepaths[i], i)
        tiles.append(tile)

    return tiles


def __RemoveTilesWithKnownOffset(reference, overlappingtiles):
    '''
    '''

    for i in range(len(overlappingtiles) - 1, -1, -1):
        overlapping = overlappingtiles[i]

        if reference.ID == overlapping.ID:
            del overlappingtiles[i]
            continue

        if overlapping.ID in reference.OffsetToTile:
            del overlappingtiles[i]
            continue

def TranslateTiles(transforms, imagepaths, imageScale=None):
    '''
    
    '''

    tiles = CreateTiles(transforms, imagepaths)

    if imageScale is None:
        imageScale = tileModule.MostCommonScalar(transforms, imagepaths)

    _FindTileOffsets(tiles, imageScale)

    layout = BuildBestTransformFirstMosaic(tiles)

    # Create a mosaic file using the tile paths and transforms
    return layout


def _FindTileOffsets(tiles, imageScale=None):
    '''Populates the OffsetToTile dictionary for tiles'''

    if imageScale is None:
        imageScale = 1.0

    idx = CreateSpatialMap(tiles)

    CalculationCount = 0

    for t in tiles:
        intersectingTilesIndex = list(idx.intersection(t.ControlBoundingBox, objects=False))
        intersectingTiles = map(lambda x: tiles[x], intersectingTilesIndex)

        __RemoveTilesWithKnownOffset(t, intersectingTiles)

        # intersectingTiles = list(idx.intersection(t.ControlBoundingBox, objects=True))
        offsets = __FindOffsets(t, intersectingTiles)
        CalculationCount += len(intersectingTiles)

        for i, offset in enumerate(offsets):
            overlappingTile = intersectingTiles[i]
            offset.scale(1.0 / imageScale)
            t.OffsetToTile[overlappingTile.ID] = offset
            overlappingTile.OffsetToTile[t.ID] = offset.Invert()

        print "Calculated %d offsets for %s" % (len(intersectingTiles), str(t))

    # TODO: Reposition the tiles based on the offsets
    print("Total offset calculations: " + str(CalculationCount))

    return tiles


def __FindOffsets(reference, overlapping):
    '''
    The adjacent transforms are known to overlap the reference transform.  Find the offset from the reference to each overlapping transform
    '''

    fixedImage = reference.Image

    offsets = []

    for overlappingTile in overlapping:
        warpedImage = overlappingTile.Image
        offset = core.FindOffset(fixedImage, MovingImage=warpedImage)
        offsets.append(offset)

    return offsets


def _BuildAlignmentRecordListWithTileIDs(tiles):

    offsets = []

    for t in tiles:
        for targetID, Offset in t.OffsetToTile.items():
            Offset.FixedTileID = t.ID
            Offset.MovingTileID = targetID

            offsets.append(Offset)

    return offsets


def _RigidTransformForTile(tile, arecord=None):
        if arecord is None:
            arecord = AlignmentRecord((0, 0), 0, 0)

        return tfactory.CreateRigidTransform(tile.OriginalImageSize, tile.OriginalImageSize, 0, arecord.peak)




def BuildBestTransformFirstMosaic(tiles):
    '''
    Constructs a mosaic by sorting all of the match results according to strength. 
    '''

    placedTiles = dict()

    arecords = _BuildAlignmentRecordListWithTileIDs(tiles)

    arecords.sort(key=attrgetter('weight'), reverse=True)

    LayoutList = []

    for record in arecords:
        print str(record.FixedTileID) + ' -> ' + str(record.MovingTileID) + " " + str(record) + " " + os.path.basename(tiles[record.FixedTileID].ImagePath) + " " + os.path.basename(tiles[record.MovingTileID].ImagePath)

        if np.isnan(record.weight):
            print "Skip: Invalid weight, not a number"
            continue

        fixedTileLayout = TileLayout.GetLayoutForID(LayoutList, record.FixedTileID)
        movingTileLayout = TileLayout.GetLayoutForID(LayoutList, record.MovingTileID)

        if fixedTileLayout is None and movingTileLayout is None:
            fixedTileLayout = TileLayout(tiles)
            fixedTileLayout.AddTileViaAlignment(record)
            LayoutList.append(fixedTileLayout)
            print "New layout"

        elif (not fixedTileLayout is None) and (not movingTileLayout is None):
            # Need to merge the layouts? See if they are the same
            if fixedTileLayout == movingTileLayout:
                # Already mapped
                # print "Skip: Already mapped"
                continue
            else:
                TileLayout.MergeLayoutsWithAlignmentRecord(fixedTileLayout, movingTileLayout, record)
                print "Merged"
                LayoutList.remove(movingTileLayout)
        else:
            if fixedTileLayout is None and not movingTileLayout is  None:
                # We'll pick it up on the next pass
                print "Skip: Getting it next time"
                continue

            fixedTileLayout.AddTileViaAlignment(record)

    # OK, we should have a single list of layouts
    LargestLayout = LayoutList[0]

    return LargestLayout


def TranslateFiles(fileDict):
    '''Translate Images expects a dictionary of images, their position and size in pixel space.  It moves the images to what it believes their optimal position is for alignment 
       and returns a dictionary of the same form.  
       Input: dict[ImageFileName] = [x y width height]
       Output: dict[ImageFileName] = [x y width height]'''

    # We do not want to load each image multiple time, and we do not know how many images we will get so we should not load them all at once.
    # Therefore our first action is building a matrix of each image and their overlapping counterparts



if __name__ == '__main__':
    pass