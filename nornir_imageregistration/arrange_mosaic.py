'''
Created on Jul 10, 2012

@author: Jamesan
'''
 
import itertools
import math
from operator import attrgetter
import os

import nornir_imageregistration
import nornir_imageregistration.assemble 
import nornir_imageregistration.assemble_tiles
import nornir_imageregistration.layout
import nornir_imageregistration.tile
import scipy.spatial

import nornir_imageregistration.core as core
import nornir_imageregistration.spatial as spatial 
import nornir_imageregistration.tileset as tileset
import nornir_pools
import numpy as np


def _CalculateImageFFTs(tiles):
    '''
    Ensure all tiles have FFTs calculated and cached
    '''
    pool = nornir_pools.GetGlobalLocalMachinePool()
     
    fft_tasks = [] 
    for t in tiles.values(): 
        task = pool.add_task("Create padded image", t.PrecalculateImages)
        task.tile = t
        fft_tasks.append(task)
        
    print("Calculating FFTs\n")
    pool.wait_completion()
    
    
def TranslateTiles(transforms, imagepaths, excess_scalar, imageScale=None, max_relax_iterations=None, max_relax_tension_cutoff=None):
    '''
    Finds the optimal translation of a set of tiles to construct a larger seemless mosaic.
    :param list transforms: list of transforms for tiles
    :param list imagepaths: list of paths to tile images, must be same length as transforms list
    :param float excess_scalar: How much additional area should we pad the overlapping regions with.
    :param float imageScale: The downsampling of the images in imagepaths.  If None then this is calculated based on the difference in the transform and the image file dimensions
    :param int max_relax_iterations: Maximum number of iterations in the relax stage
    :param float max_relax_tension_cutoff: Stop relaxation stage if the maximum tension vector is below this value
    :return: (offsets_collection, tiles) tuple
    '''
    
    if max_relax_iterations is None:
        max_relax_iterations=150
    
    if max_relax_tension_cutoff is None:
        max_relax_tension_cutoff = 1.0
            

    tiles = nornir_imageregistration.tile.CreateTiles(transforms, imagepaths)

    if imageScale is None:
        imageScale = tileset.MostCommonScalar(transforms, imagepaths)

    tile_layout = _FindTileOffsets(tiles, excess_scalar, imageScale=imageScale)
    
    nornir_imageregistration.layout.ScaleOffsetWeightsByPopulationRank(tile_layout, min_allowed_weight=0.25, max_allowed_weight=1.0)
    nornir_imageregistration.layout.RelaxLayout(tile_layout, max_tension_cutoff=max_relax_tension_cutoff, max_iter=max_relax_iterations)
    
    # final_layout = nornir_imageregistration.layout.BuildLayoutWithHighestWeightsFirst(offsets_collection)

    # Create a mosaic file using the tile paths and transforms
    return (tile_layout, tiles)


def RefineTranslations(transforms, imagepaths, imageScale=None, subregion_shape=None):
    '''
    Refine the initial translate results by registering a number of smaller regions and taking the average offset.  Then update the offsets.
    This still produces a translation only offset
    '''
    if imageScale is None:
        imageScale = 1.0
        
    if subregion_shape is None:
        subregion_shape = np.array([128, 128])
        
    downsample = 1.0 / imageScale
    
    tiles = nornir_imageregistration.tile.CreateTiles(transforms, imagepaths)
    list_tiles = list(tiles.values())
    pool = nornir_pools.GetGlobalMultithreadingPool()
    tasks = list()
    
    if imageScale is None:
        imageScale = tileset.MostCommonScalar(transforms, imagepaths)
    
    layout = nornir_imageregistration.layout.Layout()    
    for t in list_tiles:
        layout.CreateNode(t.ID, t.ControlBoundingBox.Center)
        
    for A, B in nornir_imageregistration.tile.IterateOverlappingTiles(list_tiles, minOverlap=0.03):
        # OK... add some small neighborhoods and register those...
        (downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment) = nornir_imageregistration.tile.Tile.Calculate_Overlapping_Regions(A, B, imageScale)
#         
          
        task = pool.add_task("Align %d -> %d" % (A.ID, B.ID), __RefineTileAlignmentRemote, A, B, downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment, imageScale, subregion_shape)
        task.A = A
        task.B = B
        task.OffsetAdjustment = OffsetAdjustment
        tasks.append(task)
#          
#         (point_pairs, net_offset) = __RefineTileAlignmentRemote(A, B, downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment, imageScale)
#         offset = net_offset[0:2] + OffsetAdjustment
#         weight = net_offset[2]
#           
#         print("%d -> %d : %s" % (A.ID, B.ID, str(net_offset)))
#           
#         layout.SetOffset(A.ID, B.ID, offset, weight)
         
        # print(str(net_offset))
        
    for t in tasks:
        try:
            (point_pairs, net_offset) = t.wait_return()
        except Exception as e:
            print("Could not register %d -> %d" % (t.A.ID, t.B.ID))
            print("%s" % str(e))
            continue 
        
        offset = net_offset[0:2] + (t.OffsetAdjustment * downsample)
        weight = net_offset[2]
        layout.SetOffset(t.A.ID, t.B.ID, offset, weight) 
        
        # Figure out what offset we found vs. what offset we expected
        PredictedOffset = t.B.ControlBoundingBox.Center - t.A.ControlBoundingBox.Center
        
        diff = offset - PredictedOffset
        distance = np.sqrt(np.sum(diff ** 2))
        
        print("%d -> %d = %g" % (t.A.ID, t.B.ID, distance))
        
    pool.wait_completion()
    
    return (layout, tiles)
            
        
    
    
def _FindTileOffsets(tiles, excess_scalar, min_overlap=0.05, imageScale=None):
    '''Populates the OffsetToTile dictionary for tiles
    :param dict tiles: Dictionary mapping TileID to a tile
    :param dict imageScale: downsample level if known.  None causes it to be calculated.
    :param float excess_scalar: How much additional area should we pad the overlapping rectangles with.
    :return: A layout object describing the optimal adjustment for each tile to align with each neighboring tile
    '''
    

    if imageScale is None:
        imageScale = 1.0
        
    downsample = 1.0 / imageScale

    # idx = tileset.CreateSpatialMap([t.ControlBoundingBox for t in tiles], tiles)

    CalculationCount = 0
    
    # _CalculateImageFFTs(tiles)
  
    # pool = nornir_pools.GetGlobalSerialPool()
    pool = nornir_pools.GetGlobalMultithreadingPool()
    tasks = list()
    
    layout = nornir_imageregistration.layout.Layout()
    list_tiles = list(tiles.values())
    
    for t in list_tiles:
        layout.CreateNode(t.ID, t.ControlBoundingBox.Center)
        
    print("Starting tile alignment") 
    for A, B in nornir_imageregistration.tile.IterateOverlappingTiles(list_tiles, min_overlap):
        # Used for debugging: __tile_offset(A, B, imageScale)
        # t = pool.add_task("Align %d -> %d %s", __tile_offset, A, B, imageScale)
        (downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment) = nornir_imageregistration.tile.Tile.Calculate_Overlapping_Regions(A, B, imageScale)
        
        #__tile_offset_remote(A.ImagePath, B.ImagePath, downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment, excess_scalar)
        
        t = pool.add_task("Align %d -> %d" % (A.ID, B.ID), __tile_offset_remote, A.ImagePath, B.ImagePath, downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment, excess_scalar)
        
        t.A = A
        t.B = B
        tasks.append(t) 
        CalculationCount += 1
        # print("Start alignment %d -> %d" % (A.ID, B.ID))

    for t in tasks:
        try:
            offset = t.wait_return()
        except FloatingPointError as e:  # Very rarely the overlapping region is entirely one color and this error is thrown.
            print("FloatingPointError: %d -> %d = %s -> Using stage coordinates." % (t.A.ID, t.B.ID, str(e)))
            
            #Create an alignment record using only stage position and a weight of zero 
            offset = nornir_imageregistration.AlignmentRecord(peak=OffsetAdjustment, weight=0)
            
        
        # Figure out what offset we found vs. what offset we expected
        PredictedOffset = t.B.ControlBoundingBox.Center - t.A.ControlBoundingBox.Center
        ActualOffset = offset.peak * downsample
        
        diff = ActualOffset - PredictedOffset
        distance = np.sqrt(np.sum(diff ** 2))
        
        print("%d -> %d = %g" % (t.A.ID, t.B.ID, distance))
        
        layout.SetOffset(t.A.ID, t.B.ID, ActualOffset, offset.weight) 
        
    pool.wait_completion()
    
    print(("Total offset calculations: " + str(CalculationCount)))

    return layout



def __get_overlapping_image(image, overlapping_rect, excess_scalar):
    '''
    Crop the tile's image so it contains the specified rectangle
    '''
    
    scaled_rect = spatial.Rectangle.SafeRound(spatial.Rectangle.scale(overlapping_rect, excess_scalar))
    return core.CropImage(image, Xo=int(scaled_rect.BottomLeft[1]), Yo=int(scaled_rect.BottomLeft[0]), Width=int(scaled_rect.Width), Height=int(scaled_rect.Height), cval='random')
    
    # return core.PadImageForPhaseCorrelation(cropped, MinOverlap=1.0, PowerOfTwo=True)

    
 

def __tile_offset_remote(A_Filename, B_Filename, overlapping_rect_A, overlapping_rect_B, OffsetAdjustment, excess_scalar):
    '''
    :param A_Filename: Path to tile A
    :param B_Filename: Path to tile B
    :param overlapping_rect_A: Region of overlap on tile A with tile B
    :param overlapping_rect_B: Region of overlap on tile B with tile A
    :param OffsetAdjustment: Offset to account for the (center) position of tile B relative to tile A.  If the overlapping rectangles are perfectly aligned the reported offset would be (0,0).  OffsetAdjustment would be added to that (0,0) result to ensure Tile B remained in the same position. 
    :param float excess_scalar: How much additional area should we pad the overlapping rectangles with.
    Return the offset required to align to image files.
    This function exists to minimize the inter-process communication
    '''
    
    A = core.LoadImage(A_Filename).astype(dtype=np.float16)
    B = core.LoadImage(B_Filename).astype(dtype=np.float16)

# I had to add the .astype call above for DM4 support, but I recall it broke PMG input.  Leave this comment here until the tests are passing
#    A = core.LoadImage(A_Filename) #.astype(dtype=np.float16)
#    B = core.LoadImage(B_Filename) #.astype(dtype=np.float16)

    
    # I tried a 1.0 overlap.  It works better for light microscopy where the reported stage position is more precise
    # For TEM the stage position can be less reliable and the 1.5 scalar produces better results
    OverlappingRegionA = __get_overlapping_image(A, overlapping_rect_A, excess_scalar=excess_scalar)
    OverlappingRegionB = __get_overlapping_image(B, overlapping_rect_B, excess_scalar=excess_scalar)
    
    #It is fairly common to underflow when dividing float16 images, so just warn and move on. 
    #I spent a day debugging why a mosaic was not building correctly to find the underflow 
    #issue, so don't remove it.  The underflow error removes one of the ties between a tile
    #and its neighbors.
    
    #Note error levelshould now be set in nornir_imageregistration.__init__
    #old_float_err_settings = np.seterr(under='warn')
    
    #If the entire region is a solid color, then return an alignment record with no offset and a weight of zero
    if (OverlappingRegionA.min() == OverlappingRegionA.max()) or \
        (OverlappingRegionA.max() == 0) or \
        (OverlappingRegionB.min() == OverlappingRegionB.max()) or \
        (OverlappingRegionB.max() == 0):
        return nornir_imageregistration.AlignmentRecord(peak=OffsetAdjustment, weight=0)
        
    OverlappingRegionA -= OverlappingRegionA.min()
    OverlappingRegionA /= OverlappingRegionA.max()
    
    OverlappingRegionB -= OverlappingRegionB.min()
    OverlappingRegionB /= OverlappingRegionB.max()
    
    # core.ShowGrayscale([OverlappingRegionA, OverlappingRegionB])
    
    record = core.FindOffset(OverlappingRegionA, OverlappingRegionB, FFT_Required=True)
    
    # overlapping_rect_B_AdjustedToPeak = spatial.Rectangle.translate(overlapping_rect_B, -record.peak) 
    # overlapping_rect_B_AdjustedToPeak = spatial.Rectangle.change_area(overlapping_rect_B_AdjustedToPeak, overlapping_rect_A.Size)
    # median_diff = __AlignmentScoreRemote(A, B, overlapping_rect_A, overlapping_rect_B_AdjustedToPeak)
    # diff_weight = 1.0 - median_diff
    #np.seterr(**old_float_err_settings)
    
    adjusted_record = nornir_imageregistration.AlignmentRecord(np.array(record.peak) + OffsetAdjustment, record.weight)
    return adjusted_record


def __tile_offset(A, B, imageScale):
    '''
    First crop the images so we only send the half of the images which can overlap
    '''
    
#     overlapping_rect = spatial.Rectangle.overlap_rect(A.ControlBoundingBox,B.ControlBoundingBox)
#     
#     overlapping_rect_A = __get_overlapping_imagespace_rect_for_tile(A, overlapping_rect)
#     overlapping_rect_B = __get_overlapping_imagespace_rect_for_tile(B, overlapping_rect)
#     #If the predicted alignment is perfect and we use only the overlapping regions  we would have an alignment offset of 0,0.  Therefore we add the existing offset between tiles to the result
#     OffsetAdjustment = (B.ControlBoundingBox.Center - A.ControlBoundingBox.Center) * imageScale
#    downsampled_overlapping_rect_A = spatial.Rectangle.SafeRound(spatial.Rectangle.CreateFromBounds(overlapping_rect_A.ToArray() * imageScale))
#    downsampled_overlapping_rect_B = spatial.Rectangle.SafeRound(spatial.Rectangle.CreateFromBounds(overlapping_rect_B.ToArray() * imageScale))

    (downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment) = nornir_imageregistration.tile.Tile.Calculate_Overlapping_Regions(A, B, imageScale)
    
    ImageA = __get_overlapping_image(A.Image, downsampled_overlapping_rect_A)
    ImageB = __get_overlapping_image(B.Image, downsampled_overlapping_rect_B)
    
    # core.ShowGrayscale([ImageA, ImageB])
    
    record = core.FindOffset(ImageA, ImageB, FFT_Required=True)
    adjusted_record = nornir_imageregistration.AlignmentRecord(np.array(record.peak) + OffsetAdjustment, record.weight)
    return adjusted_record


def BuildOverlappingTileDict(list_tiles, minOverlap=0.05):
    ''':return: A map of tile ID to all overlapping tile IDs'''
    
    list_rects = []
    for tile in list_tiles:
        list_rects.append(tile.ControlBoundingBox)
        
    rset = spatial.RectangleSet.Create(list_rects)
    return rset.BuildTileOverlapDict()
    

def ScoreMosaicQuality(transforms, imagepaths, imageScale=None):
    '''
    Walk each overlapping region between tiles.  Subtract the 
    '''
    
    tiles = nornir_imageregistration.tile.CreateTiles(transforms, imagepaths)
    
    if imageScale is None:
        imageScale = tileset.MostCommonScalar(transforms, imagepaths)
        
    list_tiles = list(tiles.values())
    total_score = 0
    total_pixels = 0
    
    pool = nornir_pools.GetGlobalMultithreadingPool()
    tasks = list()
    
    for A, B in nornir_imageregistration.tile.IterateOverlappingTiles(list_tiles):
        (downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment) = nornir_imageregistration.tile.Tile.Calculate_Overlapping_Regions(A, B, imageScale)
        
        #__AlignmentScoreRemote(A.ImagePath, B.ImagePath, downsampled_overlapping_rect_A, downsampled_overlapping_rect_B)
        
        t = pool.add_task("Score %d -> %d" % (A.ID, B.ID), __AlignmentScoreRemote, A.ImagePath, B.ImagePath, downsampled_overlapping_rect_A, downsampled_overlapping_rect_B)
        tasks.append(t)
        
#         OverlappingRegionA = __get_overlapping_image(A.Image, downsampled_overlapping_rect_A, excess_scalar=1.0)
#         OverlappingRegionB = __get_overlapping_image(B.Image, downsampled_overlapping_rect_B, excess_scalar=1.0)
#         
#         OverlappingRegionA -= OverlappingRegionB
#         absoluteDiff = np.fabs(OverlappingRegionA)
#         score = np.sum(absoluteDiff.flat)

    pool.wait_completion()

    for t in tasks:
        # (score, num_pixels) = t.wait_return()
        score = t.wait_return()
        total_score += score
        # total_pixels += np.prod(num_pixels)
        
    # return total_score / total_pixels
    return total_score / len(tasks)

def __AlignmentScoreRemote(A_Filename, B_Filename, overlapping_rect_A, overlapping_rect_B):
    '''Returns the difference between the images'''
    
    try:
        OverlappingRegionA = __get_overlapping_image(A_Filename, overlapping_rect_A, excess_scalar=1.0)
        OverlappingRegionB = __get_overlapping_image(B_Filename, overlapping_rect_B, excess_scalar=1.0)
        
        #If the entire region is a solid color, then return the maximum score possible
        if (OverlappingRegionA.min() == OverlappingRegionA.max()) or \
            (OverlappingRegionA.max() == 0) or \
            (OverlappingRegionB.min() == OverlappingRegionB.max()) or \
            (OverlappingRegionB.max() == 0):
            return 1.0 
        
        OverlappingRegionA -= OverlappingRegionA.min()
        OverlappingRegionA /= OverlappingRegionA.max()
        
        OverlappingRegionB -= OverlappingRegionB.min()
        OverlappingRegionB /= OverlappingRegionB.max()
        
        ignoreIndicies = OverlappingRegionA == OverlappingRegionA.max()
        ignoreIndicies |= OverlappingRegionA == OverlappingRegionA.min()
        ignoreIndicies |= OverlappingRegionB == OverlappingRegionB.max()
        ignoreIndicies |= OverlappingRegionB == OverlappingRegionB.min()
        
        #There was data in the aligned images, but not overlapping.  So we return the maximum value
        if np.alltrue(ignoreIndicies):
            return 1.0
        
        validIndicies = np.invert(ignoreIndicies)
        
        
          
        OverlappingRegionA -= OverlappingRegionB
        absoluteDiff = np.fabs(OverlappingRegionA)
        
        # core.ShowGrayscale([OverlappingRegionA, OverlappingRegionB, absoluteDiff])
        return np.mean(absoluteDiff[validIndicies])
    except FloatingPointError as e:
        print("FloatingPointError: {0} for images\n\t{1}\n\t{2}".format(str(e), A_Filename, B_Filename))
        raise e
        
    
    
    # return (np.sum(absoluteDiff[validIndicies].flat), np.sum(validIndicies))

def TranslateFiles(fileDict):
    '''Translate Images expects a dictionary of images, their position and size in pixel space.  It moves the images to what it believes their optimal position is for alignment 
       and returns a dictionary of the same form.  
       Input: dict[ImageFileName] = [x y width height]
       Output: dict[ImageFileName] = [x y width height]'''

    # We do not want to load each image multiple time, and we do not know how many images we will get so we should not load them all at once.
    # Therefore our first action is building a matrix of each image and their overlapping counterparts
    raise NotImplemented()

if __name__ == '__main__':
    pass
