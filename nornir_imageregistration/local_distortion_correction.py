'''
Created on Apr 7, 2015

@author: u0490822

This module performs local distortions of images to refine alignments of mosaics and sections
'''

import nornir_imageregistration

import nornir_imageregistration.core as core
import nornir_imageregistration.spatial as spatial
import nornir_imageregistration.tile as tile
import nornir_imageregistration.tileset as tileset
import nornir_imageregistration.stos_brute
import nornir_imageregistration.transforms
import nornir_pools
import nornir_shared.histogram
import nornir_shared.plot
import numpy as np
from alignment_record import AlignmentRecord
from _ctypes import alignment
from numpy import histogram


class DistortionCorrection:
    
    def __init__(self):
        self.PointsForTile = {}

          
def CreateDisplacementGridForTile(tile):
    
    grid_dim = core.TileGridShape(overlapping_rect_A.Size, subregion_shape)
    
    
def RefineMosaic(transforms, imagepaths, imageScale=None, subregion_shape=None):
    '''
    Locate overlapping regions between tiles in a mosaic and align multiple small subregions within.  This generates a set of control points. 
    
    More than one tile may overlap, for example corners.  To solve this the set of control points is merged into a KD tree.  Points closer than a set distance (less than subregion size) are averaged to create a single offset. 
    
    Using the remaining points a mesh transform is generated for the tile. 
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
             
    for A, B in tile.IterateOverlappingTiles(list_tiles, minOverlap=0.03):
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
        
        SplitDisplacements(t.A, t.B, point_pairs)
        # offset = net_offset[0:2] + (t.OffsetAdjustment * downsample)
        # weight = net_offset[2]
        # layout.SetOffset(t.A.ID, t.B.ID, offset, weight) 
        
        # Figure out what offset we found vs. what offset we expected
        # PredictedOffset = t.B.ControlBoundingBox.Center - t.A.ControlBoundingBox.Center
        
        # diff = offset - PredictedOffset
        # distance = np.sqrt(np.sum(diff ** 2))
        
        # print("%d -> %d = %g" % (t.A.ID, t.B.ID, distance))
        
    pool.wait_completion()
    
    return (layout, tiles)


def __RefineTileAlignmentRemote(A, B, overlapping_rect_A, overlapping_rect_B, OffsetAdjustment, imageScale, subregion_shape=None):
    
    if subregion_shape is None:
        subregion_shape = np.array([128, 128])
        
    downsample = 1.0 / imageScale
        
    grid_dim = core.TileGridShape(overlapping_rect_A.Size, subregion_shape)
    
    # overlapping_rect_A = spatial.Rectangle.change_area(overlapping_rect_A, grid_dim * subregion_shape)
    # overlapping_rect_B = spatial.Rectangle.change_area(overlapping_rect_B, grid_dim * subregion_shape)
        
    overlapping_rect = spatial.Rectangle.overlap_rect(A.ControlBoundingBox, B.ControlBoundingBox)
    overlapping_rect = spatial.Rectangle.change_area(overlapping_rect, grid_dim * subregion_shape * downsample)
        
    ATransformedImageData = nornir_imageregistration.assemble_tiles.TransformTile(transform=A.Transform, imagefullpath=A.ImagePath, distanceImage=None, requiredScale=imageScale, FixedRegion=overlapping_rect.ToArray())
    BTransformedImageData = nornir_imageregistration.assemble_tiles.TransformTile(transform=B.Transform, imagefullpath=B.ImagePath, distanceImage=None, requiredScale=imageScale, FixedRegion=overlapping_rect.ToArray())
      
    # I tried a 1.0 overlap.  It works better for light microscopy where the reported stage position is more precise
    # For TEM the stage position can be less reliable and the 1.5 scalar produces better results
    # OverlappingRegionA = __get_overlapping_image(A, overlapping_rect_A,excess_scalar=1.0)
    # OverlappingRegionB = __get_overlapping_image(B, overlapping_rect_B,excess_scalar=1.0)
    A_image = core.RandomNoiseMask(ATransformedImageData.image, ATransformedImageData.centerDistanceImage < np.finfo(ATransformedImageData.centerDistanceImage.dtype).max, Copy=True)
    B_image = core.RandomNoiseMask(BTransformedImageData.image, BTransformedImageData.centerDistanceImage < np.finfo(BTransformedImageData.centerDistanceImage.dtype).max, Copy=True)
    
    # OK, create tiles from the overlapping regions
    # A_image = core.ReplaceImageExtramaWithNoise(ATransformedImageData.image)
    # B_image = core.ReplaceImageExtramaWithNoise(BTransformedImageData.image)
    # core.ShowGrayscale([A_image,B_image])
    A_tiles = core.ImageToTiles(A_image, subregion_shape, cval='random')
    B_tiles = core.ImageToTiles(B_image, subregion_shape, cval='random')
    
    # grid_dim = core.TileGridShape(ATransformedImageData.image.shape, subregion_shape)
    
    refine_dtype = np.dtype([('SourceY', 'f4'),
                           ('SourceX', 'f4'),
                           ('TargetY', 'f4'),
                           ('TargetX', 'f4'),
                           ('Weight', 'f4'),
                           ('Angle', 'f4')])
    
    point_pairs = np.empty(grid_dim, dtype=refine_dtype)
    
    net_displacement = np.empty((grid_dim.prod(), 3), dtype=np.float32)  # Amount the refine points are moved from the defaultf
    
    cell_center_offset = subregion_shape / 2.0
    
    for iRow in range(0, grid_dim[0]):
        for iCol in range(0, grid_dim[1]): 
            subregion_offset = (np.array([iRow, iCol]) * subregion_shape) + cell_center_offset  # Position subregion coordinate space
            source_tile_offset = subregion_offset + overlapping_rect_A.BottomLeft  # Position tile image coordinate space
            global_offset = OffsetAdjustment + subregion_offset  # Position subregion in tile's mosaic space
            
            if not (iRow, iCol) in A_tiles or not (iRow, iCol) in B_tiles:
                net_displacement[(iRow * grid_dim[1]) + iCol, :] = np.array([0, 0, 0])
                point_pairs[iRow, iCol] = np.array((source_tile_offset[0], source_tile_offset[1], 0 + global_offset[0], 0 + global_offset[1], 0, 0), dtype=refine_dtype)
                continue 
            
            try:
                record = core.FindOffset(A_tiles[iRow, iCol], B_tiles[iRow, iCol], FFT_Required=True)
            except:
                net_displacement[(iRow * grid_dim[1]) + iCol, :] = np.array([0, 0, 0])
                point_pairs[iRow, iCol] = np.array((source_tile_offset[0], source_tile_offset[1], 0 + global_offset[0], 0 + global_offset[1], 0, 0), dtype=refine_dtype)
                continue
            
            adjusted_record = nornir_imageregistration.AlignmentRecord(np.array(record.peak) * downsample, record.weight)
            
            # print(str(record))
            
            if np.any(np.isnan(record.peak)):
                net_displacement[(iRow * grid_dim[1]) + iCol, :] = np.array([0, 0, 0])
                point_pairs[iRow, iCol] = np.array((source_tile_offset[0], source_tile_offset[1], 0 + global_offset[0], 0 + global_offset[1], 0, record.angle), dtype=refine_dtype)
            else:
                net_displacement[(iRow * grid_dim[1]) + iCol, :] = np.array([adjusted_record.peak[0], adjusted_record.peak[1], record.weight])
                point_pairs[iRow, iCol] = np.array((source_tile_offset[0], source_tile_offset[1], adjusted_record.peak[0] + global_offset[0], adjusted_record.peak[1] + global_offset[1], record.weight, record.angle), dtype=refine_dtype)
            
    # TODO: return two sets of point pairs, one for each tile, with the offsets divided by two so both tiles are warping to improve the fit equally?
    # TODO: Try a number of angles
    # TODO: The original refine-grid assembled the image before trying to improve the fit.  Is this important or an artifact of the implementation?
    weighted_net_offset = np.copy(net_displacement)
    weighted_net_offset[:, 2] /= np.sum(net_displacement[:, 2])
    weighted_net_offset[:, 0] *= weighted_net_offset[:, 2]
    weighted_net_offset[:, 1] *= weighted_net_offset[:, 2]
    
    net_offset = np.sum(weighted_net_offset[:, 0:2], axis=0)
    
    meaningful_weights = weighted_net_offset[weighted_net_offset[:, 2] > 0, 2]
    weight = 0
    if meaningful_weights.shape[0] > 0:
        weight = np.median(meaningful_weights)
    
    net_offset = np.hstack((np.around(net_offset, 3), weight))
     
    # net_offset = np.median(net_displacement,axis=0) 
    
    return (point_pairs, net_offset)


def SplitDisplacements(A, B, point_pairs):
    '''
    :param tile A: Tile A, point_pairs are registered relative to this tile
    :param tile A: Tile B, the tile that point pairs register A onto
    :param ndarray point_pairs: A set of point pairs from a grid refinement
    '''
    
    raise NotImplementedError()


def RefineTwoImages(Transform, fixed_image, warped_image, fixed_mask=None, warped_mask=None, cell_size=(256, 256), grid_spacing=(256, 256)):
    
    if isinstance(Transform, str):
        Transform = nornir_imageregistration.factory.LoadTransform(Transform, 1)
        
    grid_spacing = np.asarray(grid_spacing, np.int32)
    cell_size = np.asarray(cell_size, np.int32)
        
    fixed_image = core.ImageParamToImageArray(fixed_image)
    warped_image = core.ImageParamToImageArray(warped_image)
    
    if fixed_mask is not None:
        fixed_mask = core.ImageParamToImageArray(fixed_mask)
        fixed_image = nornir_imageregistration.core.RandomNoiseMask(fixed_image, fixed_mask)
        
    if warped_mask is not None:
        warped_mask = core.ImageParamToImageArray(warped_mask)
        warped_image = nornir_imageregistration.core.RandomNoiseMask(warped_image, warped_mask)
        
    # Mark a grid along the fixed image, then find the points on the warped image
    grid_dims = core.TileGridShape(warped_image.shape, grid_spacing)
    
    pool = nornir_pools.GetGlobalMultithreadingPool()
    tasks = list()
    alignment_records = list()
    
    for iRow in range(grid_dims[1]):
        for iCol in range(grid_dims[0]):
            FixedPoint = np.asarray((iCol * grid_spacing[0], iRow * grid_spacing[1]), dtype=np.int32)
            WarpedPoint = Transform.Transform(FixedPoint).astype(np.int32)
            WarpedPoint = np.reshape(WarpedPoint, 2)
            
            if fixed_mask[(FixedPoint[0], FixedPoint[1])] == False:
                continue
            
            if warped_mask[(WarpedPoint[0], WarpedPoint[1])] == False:
                continue
            
            # arecord = AttemptAlignPoint(Transform, fixed_image, warped_image, FixedPoint, cell_size, anglesToSearch=[0])
            AlignTask = pool.add_task("Align %d,%d" % (iRow, iCol), AttemptAlignPoint, Transform, fixed_image, warped_image, FixedPoint, cell_size, anglesToSearch=[0])
            AlignTask.ID = (iRow, iCol)
            AlignTask.FixedPoint = FixedPoint
            AlignTask.WarpedPoint = WarpedPoint #Transform.InverseTransform(FixedPoint)
            tasks.append(AlignTask)
#             arecord.iRow = iRow
#             arecord.iCol = iCol
#             arecord.FixedPoint = FixedPoint
#             arecord.WarpedPoint = WarpedPoint
#             arecord.AdjustedWarpedPoint = WarpedPoint + arecord.peak
#             
#             alignment_records.append(arecord)
    
    for t in tasks:
        arecord = t.wait_return()
        
        erec = nornir_imageregistration.alignment_record.EnhancedAlignmentRecord(ID=(iRow, iCol), FixedPoint=t.FixedPoint, WarpedPoint=t.WarpedPoint, peak=arecord.peak, weight=arecord.weight, angle=arecord.angle, flipped_ud=arecord.flippedud)
        # arecord.ID = (iRow, iCol)
        # arecord.FixedPoint = t.FixedPoint
        # arecord.WarpedPoint = t.WarpedPoint
        # arecord.AdjustedWarpedPoint = t.WarpedPoint + arecord.peak
         
        alignment_records.append(erec)
    
    weights = np.asarray(list(map(lambda a: a.weight, alignment_records)))
    h = nornir_shared.histogram.Histogram.Init(np.min(weights), np.max(weights))
    h.Add(weights)
    nornir_shared.plot.Histogram(h, Title="Histogram of Weights", xlabel="Weight Value")
    
    return alignment_records
    # Cull the worst of the alignment records
    
    # Build a new transform using our alignment points
    # peaks = np.list(map(lambda a: a.peak, alignment_records))
    # updatedTransform = _PeakListToTransform(alignment_records)
    
    # Re-run the loop
    
    # return updatedTransform

    
def _PeakListToTransform(alignment_records):
    '''
    Converts a set of EnhancedAlignmentRecord peaks from the RefineTwoImages function into a transform
    '''
    
    FixedPoints = np.asarray(list(map(lambda a: a.FixedPoint, alignment_records)))
    WarpedPoints = np.asarray(list(map(lambda a: a.AdjustedWarpedPoint, alignment_records)))
    
    # PointPairs = np.hstack((FixedPoints, WarpedPoints))
    PointPairs = np.hstack((WarpedPoints, FixedPoints))
    # PointPairs.append((ControlY, ControlX, mappedY, mappedX))

    T = nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(PointPairs)
    
    return T

    
def AttemptAlignPoint(transform, fixedImage, warpedImage, controlpoint, alignmentArea, anglesToSearch=None):
    '''Try to use the Composite view to render the two tiles we need for alignment'''
    if anglesToSearch is None:
        anglesToSearch = np.linspace(-7.5, 7.5, 11)
        
    FixedRectangle = nornir_imageregistration.Rectangle.CreateFromPointAndArea(point=[controlpoint[0] - (alignmentArea[0] / 2.0),
                                                                                   controlpoint[1] - (alignmentArea[1] / 2.0)],
                                                                             area=alignmentArea)

    FixedRectangle = nornir_imageregistration.Rectangle.SafeRound(FixedRectangle)
    FixedRectangle = nornir_imageregistration.Rectangle.change_area(FixedRectangle, alignmentArea)
    
    # Pull image subregions 
    warpedImageROI = nornir_imageregistration.assemble.WarpedImageToFixedSpace(transform,
                            fixedImage.shape, warpedImage, botleft=FixedRectangle.BottomLeft, area=FixedRectangle.Size, extrapolate=True)

    fixedImageROI = nornir_imageregistration.core.CropImage(fixedImage.copy(), FixedRectangle.BottomLeft[1], FixedRectangle.BottomLeft[0], int(FixedRectangle.Size[1]), int(FixedRectangle.Size[0]))

    # nornir_imageregistration.core.ShowGrayscale([fixedImageROI, warpedImageROI])

    # pool = Pools.GetGlobalMultithreadingPool()

    # task = pool.add_task("AttemptAlignPoint", core.FindOffset, fixedImageROI, warpedImageROI, MinOverlap = 0.2)
    # apoint = task.wait_return()
    # apoint = core.FindOffset(fixedImageROI, warpedImageROI, MinOverlap=0.2)
    # nornir_imageregistration.ShowGrayscale([fixedImageROI, warpedImageROI], "Fixed <---> Warped")
    
    # core.ShowGrayscale([fixedImageROI, warpedImageROI])
    
    apoint = nornir_imageregistration.stos_brute.SliceToSliceBruteForce(fixedImageROI, warpedImageROI, AngleSearchRange=anglesToSearch, MinOverlap=0.25, SingleThread=True, Cluster=False, TestFlip=False)
 
    # print("Auto-translate result: " + str(apoint))
    return apoint
    
