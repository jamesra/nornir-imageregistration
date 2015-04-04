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

import nornir_imageregistration.assemble 
import nornir_imageregistration.assemble_tiles
import nornir_imageregistration.spatial as spatial 
import nornir_imageregistration.core as core
import nornir_imageregistration.tileset as tileModule 
import nornir_imageregistration.tileset as tileset
import nornir_imageregistration.tile
import nornir_imageregistration.layout
from nornir_imageregistration.alignment_record import AlignmentRecord

import nornir_pools


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
    
    
def TranslateTiles(transforms, imagepaths, imageScale=None):
    '''
    Finds the optimal translation of a set of tiles to construct a larger seemless mosaic.
    '''

    tiles = nornir_imageregistration.layout.CreateTiles(transforms, imagepaths)

    if imageScale is None:
        imageScale = tileModule.MostCommonScalar(transforms, imagepaths)

    offsets_collection = _FindTileOffsets(tiles, imageScale)
    
    nornir_imageregistration.layout.ScaleOffsetWeightsByPopulationRank(offsets_collection, min_allowed_weight=0.25, max_allowed_weight=1.0)
    nornir_imageregistration.layout.RelaxLayout(offsets_collection, max_tension_cutoff=1.0, max_iter=100)
    
    #final_layout = nornir_imageregistration.layout.BuildLayoutWithHighestWeightsFirst(offsets_collection)

    # Create a mosaic file using the tile paths and transforms
    return (offsets_collection, tiles)


def RefineTranslations(transforms, imagepaths, imageScale=None, subregion_shape=None):
    '''
    Refine the initial translate results by registering a number of smaller regions and taking the average offset.  Then update the offsets.
    This still produces a translation only offset
    '''
    if imageScale is None:
        imageScale = 1.0
        
    if subregion_shape is None:
        subregion_shape = np.array([128,128])
        
    downsample = 1.0 / imageScale
    
    tiles = nornir_imageregistration.layout.CreateTiles(transforms, imagepaths)
    list_tiles = list(tiles.values())
    pool = nornir_pools.GetGlobalMultithreadingPool()
    tasks = list()
    
    if imageScale is None:
        imageScale = tileModule.MostCommonScalar(transforms, imagepaths)
    
    layout = nornir_imageregistration.layout.Layout()    
    for t in list_tiles:
        layout.CreateNode(t.ID, t.ControlBoundingBox.Center)
        
    for A,B in __iterateOverlappingTiles(list_tiles, minOverlap=0.03):
        #OK... add some small neighborhoods and register those...
        (downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment) = __Calculate_Overlapping_Regions(A,B, imageScale)
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
         
        #print(str(net_offset))
        
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
        
        #Figure out what offset we found vs. what offset we expected
        PredictedOffset = t.B.ControlBoundingBox.Center - t.A.ControlBoundingBox.Center
        
        diff = offset - PredictedOffset
        distance = np.sqrt(np.sum(diff ** 2))
        
        print("%d -> %d = %g" % (t.A.ID, t.B.ID, distance))
        
        
        
    pool.wait_completion()
    
    return layout
            
        
def __RefineTileAlignmentRemote(A, B, overlapping_rect_A, overlapping_rect_B, OffsetAdjustment, imageScale, subregion_shape=None):
    
    if subregion_shape is None:
        subregion_shape = np.array([128,128])
        
    downsample = 1.0 / imageScale
        
    grid_dim = core.TileGridShape(overlapping_rect_A.Size, subregion_shape)
    
    #overlapping_rect_A = spatial.Rectangle.change_area(overlapping_rect_A, grid_dim * subregion_shape)
    #overlapping_rect_B = spatial.Rectangle.change_area(overlapping_rect_B, grid_dim * subregion_shape)
        
    overlapping_rect = spatial.Rectangle.overlap_rect(A.ControlBoundingBox,B.ControlBoundingBox)
    overlapping_rect = spatial.Rectangle.change_area(overlapping_rect, grid_dim * subregion_shape * downsample)
        
    ATransformedImageData = nornir_imageregistration.assemble_tiles.TransformTile(transform=A.Transform, imagefullpath=A.ImagePath, distanceImage=None, requiredScale=imageScale, FixedRegion=overlapping_rect.ToArray())
    BTransformedImageData = nornir_imageregistration.assemble_tiles.TransformTile(transform=B.Transform, imagefullpath=B.ImagePath, distanceImage=None, requiredScale=imageScale, FixedRegion=overlapping_rect.ToArray())
      
    #I tried a 1.0 overlap.  It works better for light microscopy where the reported stage position is more precise
    #For TEM the stage position can be less reliable and the 1.5 scalar produces better results
    #OverlappingRegionA = __get_overlapping_image(A, overlapping_rect_A,excess_scalar=1.0)
    #OverlappingRegionB = __get_overlapping_image(B, overlapping_rect_B,excess_scalar=1.0)
    A_image = core.RandomNoiseMask(ATransformedImageData.image, ATransformedImageData.centerDistanceImage < np.finfo(ATransformedImageData.centerDistanceImage.dtype).max, Copy=True)
    B_image = core.RandomNoiseMask(BTransformedImageData.image, BTransformedImageData.centerDistanceImage < np.finfo(BTransformedImageData.centerDistanceImage.dtype).max, Copy=True)
    
    #OK, create tiles from the overlapping regions
    #A_image = core.ReplaceImageExtramaWithNoise(ATransformedImageData.image)
    #B_image = core.ReplaceImageExtramaWithNoise(BTransformedImageData.image)
    #core.ShowGrayscale([A_image,B_image])
    A_tiles = core.ImageToTiles(A_image, subregion_shape,cval='random')
    B_tiles = core.ImageToTiles(B_image, subregion_shape,cval='random')
    
    #grid_dim = core.TileGridShape(ATransformedImageData.image.shape, subregion_shape)
    
    refine_dtype=np.dtype([('SourceY','f4'),
                           ('SourceX','f4'),
                           ('TargetY','f4'),
                           ('TargetX','f4'),
                           ('Weight','f4'),
                           ('Angle','f4')])
    
    point_pairs = np.empty(grid_dim, dtype=refine_dtype)
    
    net_displacement =  np.empty((grid_dim.prod(),3), dtype=np.float32) #Amount the refine points are moved from the defaultf
    
    cell_center_offset = subregion_shape / 2.0
    
    for iRow in range(0,grid_dim[0]):
        for iCol in range(0,grid_dim[1]): 
            subregion_offset = (np.array([iRow,iCol]) * subregion_shape) + cell_center_offset #Position subregion coordinate space
            source_tile_offset = subregion_offset + overlapping_rect_A.BottomLeft #Position tile image coordinate space
            global_offset = OffsetAdjustment + subregion_offset #Position subregion in tile's mosaic space
            
            if not (iRow,iCol) in A_tiles or not (iRow,iCol) in B_tiles:
                net_displacement[(iRow*grid_dim[1])+iCol,:] = np.array([0, 0, 0])
                point_pairs[iRow,iCol] = np.array((source_tile_offset[0], source_tile_offset[1], 0 + global_offset[0], 0 + global_offset[1], 0, 0), dtype=refine_dtype)
                continue 
            
            try:
                record = core.FindOffset(A_tiles[iRow,iCol], B_tiles[iRow, iCol], FFT_Required=True)
            except:
                net_displacement[(iRow*grid_dim[1])+iCol,:] = np.array([0, 0, 0])
                point_pairs[iRow,iCol] = np.array((source_tile_offset[0], source_tile_offset[1], 0 + global_offset[0], 0 + global_offset[1], 0, 0), dtype=refine_dtype)
                continue
            
            adjusted_record = AlignmentRecord(np.array(record.peak) * downsample, record.weight)
            
            #print(str(record))
            
            if np.any(np.isnan(record.peak)):
                net_displacement[(iRow*grid_dim[1])+iCol,:] = np.array([0, 0, 0])
                point_pairs[iRow,iCol] = np.array((source_tile_offset[0], source_tile_offset[1], 0 + global_offset[0], 0 + global_offset[1], 0, record.angle), dtype=refine_dtype)
            else:
                net_displacement[(iRow*grid_dim[1])+iCol,:] = np.array([adjusted_record.peak[0], adjusted_record.peak[1], record.weight])
                point_pairs[iRow,iCol] = np.array((source_tile_offset[0], source_tile_offset[1], adjusted_record.peak[0] + global_offset[0], adjusted_record.peak[1] + global_offset[1], record.weight, record.angle), dtype=refine_dtype)
            
    #TODO: return two sets of point pairs, one for each tile, with the offsets divided by two so both tiles are warping to improve the fit equally?
    #TODO: Try a number of angles
    #TODO: The original refine-grid assembled the image before trying to improve the fit.  Is this important or an artifact of the implementation?
    weighted_net_offset = np.copy(net_displacement)
    weighted_net_offset[:,2] /= np.sum(net_displacement[:,2])
    weighted_net_offset[:,0] *= weighted_net_offset[:,2]
    weighted_net_offset[:,1] *= weighted_net_offset[:,2]
    
    net_offset = np.sum(weighted_net_offset[:,0:2], axis=0)
    
    meaningful_weights = weighted_net_offset[weighted_net_offset[:,2] > 0, 2]
    weight = 0
    if meaningful_weights.shape[0] > 0:
        weight = np.median(meaningful_weights)
    
    net_offset = np.hstack((np.around(net_offset,3), weight))
     
    #net_offset = np.median(net_displacement,axis=0) 
    
    return (point_pairs, net_offset)
    
    
def _FindTileOffsets(tiles, imageScale=None):
    '''Populates the OffsetToTile dictionary for tiles
    :param dict tiles: Dictionary mapping TileID to a tile
    :param dict imageScale: downsample level if known.  None causes it to be calculated.'''

    if imageScale is None:
        imageScale = 1.0
        
    downsample = 1.0 / imageScale

    #idx = tileset.CreateSpatialMap([t.ControlBoundingBox for t in tiles], tiles)

    CalculationCount = 0
    
    #_CalculateImageFFTs(tiles)
  
    #pool = nornir_pools.GetGlobalSerialPool()
    pool = nornir_pools.GetGlobalMultithreadingPool()
    tasks = list()
    
    layout = nornir_imageregistration.layout.Layout()
    list_tiles = list(tiles.values())
    
    for t in list_tiles:
        layout.CreateNode(t.ID, t.ControlBoundingBox.Center)
        
    print("Starting tile alignment") 
    for A,B in __iterateOverlappingTiles(list_tiles):
        #Used for debugging: __tile_offset(A, B, imageScale)
        #t = pool.add_task("Align %d -> %d %s", __tile_offset, A, B, imageScale)
        (downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment) = __Calculate_Overlapping_Regions(A,B, imageScale)
        t = pool.add_task("Align %d -> %d" % (A.ID, B.ID), __tile_offset_remote, A.ImagePath, B.ImagePath, downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment)
        
        t.A = A
        t.B = B
        tasks.append(t) 
        CalculationCount += 1
        #print("Start alignment %d -> %d" % (A.ID, B.ID))

    for t in tasks:
        offset = t.wait_return()
        
        #Figure out what offset we found vs. what offset we expected
        PredictedOffset = t.B.ControlBoundingBox.Center - t.A.ControlBoundingBox.Center
        ActualOffset = offset.peak * downsample
        
        diff = ActualOffset - PredictedOffset
        distance = np.sqrt(np.sum(diff ** 2))
        
        print("%d -> %d = %g" % (t.A.ID, t.B.ID, distance))
        
        layout.SetOffset(t.A.ID, t.B.ID, ActualOffset, offset.weight) 
        
    pool.wait_completion()
    
    print(("Total offset calculations: " + str(CalculationCount)))

    return layout

def __get_overlapping_imagespace_rect_for_tile(tile_obj, overlapping_rect):
    ''':return: Rectangle describing which region of the tile_obj image is contained in the overlapping_rect from volume space'''
    image_space_points = tile_obj.Transform.InverseTransform(overlapping_rect.Corners)    
    return spatial.BoundingPrimitiveFromPoints(image_space_points)

def __get_overlapping_image(image, overlapping_rect, excess_scalar):
    '''
    Crop the tile's image so it contains the specified rectangle
    '''
    
    scaled_rect = spatial.Rectangle.SafeRound(spatial.Rectangle.scale(overlapping_rect, excess_scalar))
    return core.CropImage(image,Xo=int(scaled_rect.BottomLeft[1]), Yo=int(scaled_rect.BottomLeft[0]), Width=int(scaled_rect.Width), Height=int(scaled_rect.Height), cval='random')
    
    #return core.PadImageForPhaseCorrelation(cropped, MinOverlap=1.0, PowerOfTwo=True)

def __Calculate_Overlapping_Regions(A,B,imageScale):
    '''
    :return: The cropped portions of A and B containing only the overlapping areas
    '''
    
    overlapping_rect = spatial.Rectangle.overlap_rect(A.ControlBoundingBox,B.ControlBoundingBox)
    
    overlapping_rect_A = __get_overlapping_imagespace_rect_for_tile(A, overlapping_rect)
    overlapping_rect_B = __get_overlapping_imagespace_rect_for_tile(B, overlapping_rect)
     
    downsampled_overlapping_rect_A = spatial.Rectangle.SafeRound(spatial.Rectangle.CreateFromBounds(overlapping_rect_A.ToArray() * imageScale))
    downsampled_overlapping_rect_B = spatial.Rectangle.SafeRound(spatial.Rectangle.CreateFromBounds(overlapping_rect_B.ToArray() * imageScale))
    
    #If the predicted alignment is perfect and we use only the overlapping regions  we would have an alignment offset of 0,0.  Therefore we add the existing offset between tiles to the result
    OffsetAdjustment = (B.ControlBoundingBox.Center - A.ControlBoundingBox.Center) * imageScale
    
    #This should ensure we never an an area mismatch
    downsampled_overlapping_rect_B = spatial.Rectangle.CreateFromPointAndArea(downsampled_overlapping_rect_B.BottomLeft, downsampled_overlapping_rect_A.Size)
    
    assert(downsampled_overlapping_rect_A.Width == downsampled_overlapping_rect_B.Width)
    assert(downsampled_overlapping_rect_A.Height == downsampled_overlapping_rect_B.Height)
    
    return (downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment)
 

def __tile_offset_remote(A_Filename, B_Filename, overlapping_rect_A, overlapping_rect_B, OffsetAdjustment):
    '''
    Return the offset required to align to image files.
    This function exists to minimize the inter-process communication
    '''
    
    A = core.LoadImage(A_Filename)
    B = core.LoadImage(B_Filename)
    
    #I tried a 1.0 overlap.  It works better for light microscopy where the reported stage position is more precise
    #For TEM the stage position can be less reliable and the 1.5 scalar produces better results
    OverlappingRegionA = __get_overlapping_image(A, overlapping_rect_A,excess_scalar=1.5)
    OverlappingRegionB = __get_overlapping_image(B, overlapping_rect_B,excess_scalar=1.5)
    
    record = core.FindOffset( OverlappingRegionA, OverlappingRegionB, FFT_Required=True)
    adjusted_record = AlignmentRecord(np.array(record.peak) + OffsetAdjustment, record.weight)
    return adjusted_record


def __tile_offset(A,B, imageScale):
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

    (downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment) = __Calculate_Overlapping_Regions(A,B,imageScale)
    
    ImageA = __get_overlapping_image(A.Image, downsampled_overlapping_rect_A)
    ImageB = __get_overlapping_image(B.Image, downsampled_overlapping_rect_B)
    
    #core.ShowGrayscale([ImageA, ImageB])
    
    record = core.FindOffset( ImageA, ImageB, FFT_Required=True)
    adjusted_record = AlignmentRecord(np.array(record.peak) + OffsetAdjustment, record.weight)
    return adjusted_record


def __iterateOverlappingTiles(list_tiles, minOverlap = 0.05):
    '''Return all tiles which overlap'''
    
    list_rects = []
    for tile in list_tiles:
        list_rects.append(tile.ControlBoundingBox)
        
    rset = spatial.RectangleSet.Create(list_rects)
    
    for (A,B) in rset.EnumerateOverlapping():
        if spatial.Rectangle.overlap(list_rects[A], list_rects[B]) >= minOverlap:
            yield (list_tiles[A],list_tiles[B])
    

def ScoreMosaicQuality(transforms, imagepaths, imageScale=None):
    '''
    Walk each overlapping region between tiles.  Subtract the 
    '''
    
    tiles = nornir_imageregistration.layout.CreateTiles(transforms, imagepaths)
    
    if imageScale is None:
        imageScale = tileModule.MostCommonScalar(transforms, imagepaths)
        
    list_tiles = list(tiles.values())
    total_score = 0
    total_pixels = 0
    
    pool = nornir_pools.GetGlobalMultithreadingPool()
    tasks = list()
    
    for A,B in __iterateOverlappingTiles(list_tiles):
        (downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment) = __Calculate_Overlapping_Regions(A,B, imageScale)
        
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
        (score, num_pixels) = t.wait_return()
        total_score += score
        total_pixels += np.prod(num_pixels)
        
    return total_score / total_pixels

def __AlignmentScoreRemote(A_Filename, B_Filename, overlapping_rect_A, overlapping_rect_B):
    OverlappingRegionA = __get_overlapping_image(A_Filename, overlapping_rect_A, excess_scalar=1.0)
    OverlappingRegionB = __get_overlapping_image(B_Filename, overlapping_rect_B, excess_scalar=1.0)
    
    OverlappingRegionA -= OverlappingRegionB
    absoluteDiff = np.fabs(OverlappingRegionA)
    return (np.sum(absoluteDiff.flat), np.prod(OverlappingRegionA.shape))

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