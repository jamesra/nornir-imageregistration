'''
Created on Jul 10, 2012

@author: Jamesan
'''

import scipy.sparse
import nornir_imageregistration.core as core
import numpy as np
import logging
import os
import nornir_imageregistration.tiles as tileModule
from alignment_record import AlignmentRecord
import nornir_imageregistration.transforms.factory as tfactory

import layout

from operator import attrgetter

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
    Finds the optimal translation of a set of tiles to construct a larger seemless mosaic.
    '''

    tiles = layout.CreateTiles(transforms, imagepaths)

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

    idx = tiles.CreateSpatialMap([t.ControlBoundingBox for t in tiles], tiles)

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

        fixedTileLayout = layout.TileLayout.GetLayoutForID(LayoutList, record.FixedTileID)
        movingTileLayout = layout.TileLayout.GetLayoutForID(LayoutList, record.MovingTileID)

        if fixedTileLayout is None and movingTileLayout is None:
            fixedTileLayout = layout.TileLayout(tiles)
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
                layout.TileLayout.MergeLayoutsWithAlignmentRecord(fixedTileLayout, movingTileLayout, record)
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