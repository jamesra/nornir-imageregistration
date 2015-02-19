'''
Created on Jul 10, 2012

@author: Jamesan
'''
 
from operator import attrgetter
import os
import numpy as np
import scipy.spatial
import itertools
import math
 
import nornir_imageregistration.spatial as spatial 
import nornir_imageregistration.core as core
import nornir_imageregistration.tileset as tileModule 
import nornir_imageregistration.tileset as tileset
import nornir_imageregistration.layout
from nornir_imageregistration.alignment_record import AlignmentRecord

import nornir_pools


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

    tiles = nornir_imageregistration.layout.CreateTiles(transforms, imagepaths)

    if imageScale is None:
        imageScale = tileModule.MostCommonScalar(transforms, imagepaths)

    _FindTileOffsets(tiles, imageScale)
    
    final_layout = BuildBestTransformFirstMosaic(tiles, imageScale)

    # Create a mosaic file using the tile paths and transforms
    return final_layout


def _FindTileOffsets(tiles, imageScale=None):
    '''Populates the OffsetToTile dictionary for tiles
    :param dict tiles: Dictionary mapping TileID to a tile
    :param dict imageScale: downsample level if known.  None causes it to be calculated.'''

    if imageScale is None:
        imageScale = 1.0

    #idx = tileset.CreateSpatialMap([t.ControlBoundingBox for t in tiles], tiles)

    CalculationCount = 0
 
    pool = nornir_pools.GetLocalMachinePool()
     
    fft_tasks = [] 
    
    for t in tiles.values(): 
        task = pool.add_task("Create padded image", t.PrecalculateImages)
        task.tile = t
        fft_tasks.append(task)
        
    print("Calculating FFTs\n")
    for task in fft_tasks:
        task.wait()
             
    tasks = list()
       
    for A,B in __iterateOverlappingTiles(tiles):
        t = pool.add_task("Align %d -> %d %s", __tile_offset, A, B)
        t.A = A
        t.B = B
        tasks.append(t) 
        CalculationCount += 1
        print("Start alignment %d -> %d" % (A.ID, B.ID))

    for t in tasks:
        offset = t.wait_return()
        __assign_offset(t.A, t.B, offset) 

    # TODO: Reposition the tiles based on the offsets
    print(("Total offset calculations: " + str(CalculationCount)))

    return tiles

def __predicted_offset(A,B):
    '''The expected offset to align B onto A
       :rtype: numpy.array
       :returns: A point''' 
    return B.ControlBoundingBox.BottomLeft - A.ControlBoundingBox.BottomLeft

def __offset_discrepancy(offsetA,offsetB):
    '''The expected offset to align B onto A
       :rtype: numpy.array
       :returns: A point''' 
    return offsetB - offsetA

def __assign_offset(A, B, offset):
    A.OffsetToTile[B.ID] = offset
    B.OffsetToTile[A.ID] = offset.Invert()
    print("Align %d -> %d %s" % (A.ID, B.ID, str(offset)))
    return

def __tile_offset(A,B):
    return  core.FindOffset( A.FFTImage, B.FFTImage, FFT_Required=False)

def __iterateOverlappingTiles(tiles, minOverlap = 0.05):
    
    for (A,B) in itertools.combinations(tiles.values(), 2):
        if spatial.rectangle.Rectangle.overlap(A.ControlBoundingBox, B.ControlBoundingBox) >= minOverlap:
            yield (A,B)
        #if spatial.rectangle.Rectangle.contains(A.ControlBoundingBox, B.ControlBoundingBox):
        #   yield (A,B)

# 
# def __FindOffsets(reference, overlapping):
#     '''
#     The adjacent transforms are known to overlap the reference transform.  Find the offset from the reference to each overlapping transform
#     '''
# 
#     fixedImage = reference.Image
# 
#     offsets = []
# 
#     for overlappingTile in overlapping:
#         warpedImage = overlappingTile.Image
#         offset = core.FindOffset(fixedImage, MovingImage=warpedImage)
#         offsets.append(offset)
# 
#     return offsets


def _BuildAlignmentRecordListWithTileIDs(tiles):

    offsets = []

    for t in tiles.values():
        for targetID, Offset in list(t.OffsetToTile.items()):
            Offset.FixedTileID = t.ID
            Offset.MovingTileID = targetID

            offsets.append(Offset)

    return offsets

def _ScaleOffsetWeights(tile_dict):
    '''
    Take the known offsets for a tile.  Adjust the weights based on how far the offset is from the expected position
    '''
    
    for tile in tile_dict.values():
        _ScaleOffsetWeightsForTile(tile_dict, tile)
         
    return 

def _ScaleOffsetWeightsForTile(tile_dict, tile):
    '''
    Take the known offsets for a tile.  Adjust the weights based on how far the offset is from the expected position
    '''
    
    distanceList = {}
    for targetTileID, offset in tile.OffsetToTile.items():
        target_tile = tile_dict[targetTileID]
        predicted = __predicted_offset(tile, target_tile)
        discrepancy = __offset_discrepancy(predicted, offset.peak)
        #print("Discrepancy %04d -> %04d : %gx %gy" % (tile.ID, target_tile.ID, discrepancy[1], discrepancy[0]))
        distanceList[targetTileID] = np.sqrt((discrepancy[0] * discrepancy[0]) + (discrepancy[1] * discrepancy[1]))
    
    totalDistance = sum(list(distanceList.values()))
    medianDistance = np.median(np.array(list(distanceList.values())))
    for targetTileID, distance in distanceList.items():
        weight = medianDistance / distance
        
        #We only penalize offsets above the median.  We don't want to place too much emphasis on a potentially flawed stage layout
        #if weight > 1.0:
        #    weight = 1.0
        #weight = distance / totalDistance
        #inv_weight = 1.0 - weight
         
        #Adjust the weight of the offset by the calculated weight.
        
        offset = tile.OffsetToTile[targetTileID]
        offset.weight *= weight
        
        print("Discrepancy weight %04d -> %04d\tdist: %3g scalar: %3g final: %3g" % (tile.ID, target_tile.ID, distance, weight, offset.weight)) 
        
    return 


def BuildBestTransformFirstMosaic(tiles, image_size_scalar):
    '''
    Constructs a mosaic by sorting all of the match results according to strength. 
    
    :param dict tiles: Dictionary of tile objects containing alignment records to other tiles
    :param float image_size_scalar: Scalar for transforms when downsampled images are used to create alignment records
    '''

    placedTiles = dict()
    
    _ScaleOffsetWeights(tiles)

    arecords = _BuildAlignmentRecordListWithTileIDs(tiles)

    arecords.sort(key=attrgetter('weight'), reverse=True)

    LayoutList = []
    
    downsample = 1.0 / image_size_scalar

    for record in arecords:
        
        record.peak[0] *= downsample
        record.peak[1] *= downsample
          
        print(str(record.FixedTileID) + ' -> ' + str(record.MovingTileID) + " " + str(record) + " " + os.path.basename(tiles[record.FixedTileID].ImagePath) + " " + os.path.basename(tiles[record.MovingTileID].ImagePath))

        if np.isnan(record.weight):
            print("Skip: Invalid weight, not a number")
            continue

        fixedTileLayout = nornir_imageregistration.layout.TileLayout.GetLayoutForID(LayoutList, record.FixedTileID)
        movingTileLayout = nornir_imageregistration.layout.TileLayout.GetLayoutForID(LayoutList, record.MovingTileID)

        if fixedTileLayout is None and movingTileLayout is None:
            fixedTileLayout = nornir_imageregistration.layout.TileLayout(tiles)
            fixedTileLayout.AddTileViaAlignment(record)
            LayoutList.append(fixedTileLayout)
            print("New layout")

        elif (not fixedTileLayout is None) and (not movingTileLayout is None):
            # Need to merge the layouts? See if they are the same
            if fixedTileLayout == movingTileLayout:
                # Already mapped
                # print "Skip: Already mapped"
                continue
            else:
                nornir_imageregistration.layout.TileLayout.MergeLayoutsWithAlignmentRecord(fixedTileLayout, movingTileLayout, record)
                print("Merged")
                LayoutList.remove(movingTileLayout)
        else:
            if fixedTileLayout is None and not movingTileLayout is  None:
                # We'll pick it up on the next pass
                print("Skip: Getting it next time")
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