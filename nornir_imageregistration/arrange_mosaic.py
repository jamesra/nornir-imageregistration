'''
Created on Jul 10, 2012

@author: Jamesan
'''
 
import itertools
import collections
import math
from operator import attrgetter
import os

import nornir_imageregistration
import nornir_imageregistration.assemble 
import nornir_imageregistration.assemble_tiles
import nornir_imageregistration.layout
import nornir_imageregistration.tile 
  
import nornir_imageregistration.tileset as tileset
import nornir_pools
import numpy as np

import nornir_shared.plot 

TileOverlapDetails = collections.namedtuple('TileOverlapDetails',
                                                'overlap_ID iTile overlapping_rect')

TileToOverlap = collections.namedtuple('TileToOverlap',
                                            'iTile tile_overlap')

TileOverlapFeatureScore = collections.namedtuple('TileOverlapFeatureScore',
                                                     'overlap_ID iTile image feature_score')


def __CreateTileToOverlapsDict(tile_overlaps):
    '''
    Returns a dictionary containing a list of tuples with (TileIndex, OverlapObject)
    TileIndex records if the tile is the first or second tile (A or B)
    described in the overlap object
    '''
    
    if isinstance(tile_overlaps, dict):
        tile_overlaps = list(tile_overlaps.values())
        
    tile_to_overlaps_dict = collections.defaultdict(dict)
    for tile_overlap in tile_overlaps:
        tile_to_overlaps_dict[tile_overlap.A.ID][tile_overlap.ID] = TileToOverlap(iTile=0, tile_overlap=tile_overlap)
        tile_to_overlaps_dict[tile_overlap.B.ID][tile_overlap.ID] = TileToOverlap(iTile=1, tile_overlap=tile_overlap)
        
    return tile_to_overlaps_dict


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
        max_relax_iterations = 150
    
    if max_relax_tension_cutoff is None:
        max_relax_tension_cutoff = 1.0
        
    if imageScale is None:
        imageScale = tileset.MostCommonScalar(transforms, imagepaths)

    tiles = nornir_imageregistration.tile.CreateTiles(transforms, imagepaths)
 
    tile_layout = _FindTileOffsets(tiles, excess_scalar, imageScale=imageScale)
    
    nornir_imageregistration.layout.ScaleOffsetWeightsByPopulationRank(tile_layout, min_allowed_weight=0.25, max_allowed_weight=1.0)
    nornir_imageregistration.layout.RelaxLayout(tile_layout, max_tension_cutoff=max_relax_tension_cutoff, max_iter=max_relax_iterations)
    
    # final_layout = nornir_imageregistration.layout.BuildLayoutWithHighestWeightsFirst(offsets_collection)

    # Create a mosaic file using the tile paths and transforms
    return (tile_layout, tiles)


def TranslateTiles2(transforms, imagepaths,
                    first_pass_excess_scalar=None, excess_scalar=None,
                    feature_score_threshold=None, image_scale=None,
                    min_translate_iterations=None, offset_acceptance_threshold=None,
                    max_relax_iterations=None, max_relax_tension_cutoff=None,
                    min_overlap=None, 
                    first_pass_inter_tile_distance_scale=None, inter_tile_distance_scale=None):
    '''
    Finds the optimal translation of a set of tiles to construct a larger seemless mosaic.
    :param list transforms: list of transforms for tiles
    :param list imagepaths: list of paths to tile images, must be same length as transforms list
    :param float excess_scalar: How much additional area should we pad the overlapping regions with.  Increase this value if you want larger offsets to be found.
    :param float feature_score_threshold: The minimum average power spectral density per pixel measurement required to believe there is enough texture in overlapping regions for registration algorithms 
    :param float imageScale: The downsampling of the images in imagepaths.  If None then this is calculated based on the difference in the transform and the image file dimensions
    :param int max_relax_iterations: Maximum number of iterations in the relax stage
    :param float max_relax_tension_cutoff: Stop relaxation stage if the maximum tension vector is below this value
    :param float min_overlap: The percentage of area that two tiles must overlap before being considered by the layout model
    :param float inter_tile_distance_scale: A scalar from 0 to 1.  1 indicates to trust the overlap reported by the transforms.  0 indicates to test the entire tile for overlaps.  Use this value to increase the area searched for correlations if the stage input is not reliable. 
    :return: (offsets_collection, tiles) tuple
    '''
    
    if max_relax_iterations is None:
        max_relax_iterations = 150
    
    if max_relax_tension_cutoff is None:
        max_relax_tension_cutoff = 1.0
        
    if feature_score_threshold is None:
        feature_score_threshold = 0.035
        
    if offset_acceptance_threshold is None:
        offset_acceptance_threshold = 1.0
        
    if min_translate_iterations is None:
        min_translate_iterations = 5
        
    if inter_tile_distance_scale is None:
        inter_tile_distance_scale = 1.0
    
    if first_pass_inter_tile_distance_scale is None:
        first_pass_inter_tile_distance_scale = inter_tile_distance_scale / 2
        
    if first_pass_excess_scalar is None:
        first_pass_excess_scalar = 3.0
        
    if excess_scalar is None:
        excess_scalar = 3.0
         
    if image_scale is None:
        image_scale = tileset.MostCommonScalar(transforms, imagepaths)
        
    tiles = nornir_imageregistration.tile.CreateTiles(transforms, imagepaths)
    
    if len(tiles) == 1:
        #If there is only one tile then just return it
        single_tile_layout = nornir_imageregistration.layout.Layout()
        single_tile_layout.CreateNode(tiles[0].ID, np.zeros((1,2)))
        return (single_tile_layout, tiles)
    
    minOffsetWeight = 0
    maxOffsetWeight = 1.0
    
    last_pass_overlaps = None
    translated_layout = None
    iPass = min_translate_iterations
    
    max_passes = min_translate_iterations * 4
    pass_count = 0
    inter_tile_distance_scale_this_pass = first_pass_inter_tile_distance_scale
    inter_tile_distance_scale_last_pass = inter_tile_distance_scale_this_pass
    
    first_pass_overlaps = None #The set of offsets for each tile pair from the first-pass.  Used to align layouts that are not connected.
    
    stage_reported_overlaps = None
    relaxed_layout = None
    
    while iPass >= 0:
        (distinct_overlaps, new_overlaps, updated_overlaps, removed_overlap_IDs, nonoverlapping_tile_IDs) = GenerateTileOverlaps(tiles=tiles,
                                                             existing_overlaps=last_pass_overlaps,
                                                             offset_epsilon=offset_acceptance_threshold,
                                                             image_scale=image_scale,
                                                             min_overlap=min_overlap,
                                                             inter_tile_distance_scale=inter_tile_distance_scale_this_pass)
        
        if stage_reported_overlaps is None:
            stage_reported_overlaps = {to.ID: to.offset for to in new_overlaps}
        
        new_or_updated_overlaps = list(new_overlaps)
        new_or_updated_overlaps.extend(updated_overlaps)
        # If there is nothing to update we are done
        if len(new_or_updated_overlaps) == 0:
            break
        
        # If we added or remove tile overlaps then reset loop counter
        # if (len(new_overlaps) > 0 or len(removed_overlap_IDs) > 0) and pass_count < max_passes:
        #    iPass = min_translate_iterations
        
        ScoreTileOverlaps(distinct_overlaps)
        
        # If this is the second pass remove any overlaps from the layout that no longer qualify
        if translated_layout is not None:
            for ID in removed_overlap_IDs:
                translated_layout.RemoveOverlap(ID)
        
        # Expand the area we search if we are adding and removing tiles
        inter_tile_distance_scale_this_pass = inter_tile_distance_scale
#         if pass_count < min_translate_iterations and iPass == min_translate_iterations:
#             inter_tile_distance_scale_this_pass = inter_tile_distance_scale
            
        # Recalculate all offsets if we changed the overlaps
#         if inter_tile_distance_scale_last_pass != inter_tile_distance_scale_this_pass:
#             new_or_updated_overlaps = distinct_overlaps
            
        inter_tile_distance_scale_last_pass = inter_tile_distance_scale_this_pass
                
        # Create a list of offsets requiring updates
        filtered_overlaps_needing_offsets = []
        for overlap in new_or_updated_overlaps:
            if overlap.feature_scores[0] >= feature_score_threshold and overlap.feature_scores[1] >= feature_score_threshold:
                filtered_overlaps_needing_offsets.append(overlap)
            else:
                if translated_layout is not None:
                    translated_layout.RemoveOverlap(overlap)
            
        translated_layout = _FindTileOffsets(filtered_overlaps_needing_offsets, excess_scalar=excess_scalar,
                                             imageScale=image_scale,
                                             existing_layout=translated_layout)
        
        scaled_translated_layout = translated_layout.copy()
        nornir_imageregistration.layout.ScaleOffsetWeightsByPopulationRank(scaled_translated_layout,
                                                                           min_allowed_weight=minOffsetWeight,
                                                                           max_allowed_weight=maxOffsetWeight)

#        nornir_imageregistration.layout.NormalizeOffsetWeights(scaled_translated_layout)
        
        translated_final_layouts = nornir_imageregistration.layout.BuildLayoutWithHighestWeightsFirst(scaled_translated_layout)
        #TODO: Pass the dictionary to this function that indicates tile offsets for pairs of tiles
        #translated_final_layout = nornir_imageregistration.layout.MergeDisconnectedLayoutsWithOffsets(translated_final_layouts, stage_reported_overlaps) 
        
        #Should we do a shorter pass on the first run?
        relax_iterations = max_relax_iterations
        if iPass == min_translate_iterations:
            relax_iterations = relax_iterations // 4
            if relax_iterations < 10:
                relax_iterations = max_relax_iterations // 2
                
        relaxed_layouts = []
        for layout in translated_final_layouts:
            relaxed_layout = nornir_imageregistration.layout.RelaxLayout(layout,
                                            max_iter=relax_iterations,
                                            max_tension_cutoff=max_relax_tension_cutoff)
            relaxed_layouts.append(relaxed_layout)
            
        relaxed_layout = nornir_imageregistration.layout.MergeDisconnectedLayoutsWithOffsets(relaxed_layouts, stage_reported_overlaps)
        
        relaxed_layout.UpdateTileTransforms(tiles)
        last_pass_overlaps = distinct_overlaps
        
        # Copy the relaxed layout positions back into the translated layout
        for ID, node in relaxed_layout.nodes.items():
            tnode = translated_layout.nodes[ID]
            tnode.Position = node.Position
            
        iPass = iPass - 1
        pass_count = pass_count + 1
    
    # final_layout = nornir_imageregistration.layout.BuildLayoutWithHighestWeightsFirst(offsets_collection)

    # Create a mosaic file using the tile paths and transforms
    return (relaxed_layout, tiles)


def GenerateTileOverlaps(tiles, existing_overlaps=None, offset_epsilon=1.0, image_scale=None, min_overlap=None, inter_tile_distance_scale=None):
    '''
    Create a list of TileOverlap objects for each overlapping region in the mosaic.  Assign a feature score to the regions from each image that overlap.
    :param list tiles: A dictionary of Tile objects
    :param list existing_overlaps: A list of overlaps created previously.  Scores for these offsets will be copied into the generated offsets if the difference in offset between the tiles
                                   is less than offset_epsilon.
    :param float offset_epsilon: The distance the expected offset between tiles has to change before we recalculate feature scores and registration
    :param float imageScale: The amount the images are downsampled compared to coordinates in the transforms
    :param float min_overlap: Tiles that overlap less than this amount percentage of area will not be included
    
    :return: Returns a four-component tuple composed of all found overlaps, the new overlaps, the overlaps that require updating, the deleted overlaps from the existing set, and the IDs of non-overlapping tiles. 
             None of the returned overlaps are the same objects as those in the original set
    '''
    assert(isinstance(tiles, dict))
    if image_scale is None:
        image_scale = tileset.MostCommonScalar([tile.Transform for tile in tiles.values()], [tile.ImagePath for tile in tiles.values()])
        
    if inter_tile_distance_scale is None:
        inter_tile_distance_scale = 1.0
        
    if inter_tile_distance_scale < 0 or inter_tile_distance_scale > 1.0:
        raise ValueError("inter_tile_distance_scale must be in the range 0 to 1: value was {0}".format(inter_tile_distance_scale))
        
    generated_overlaps = list(nornir_imageregistration.tile_overlap.CreateTileOverlaps(list(tiles.values()),
                                                                               image_scale,
                                                                               min_overlap=min_overlap,
                                                                               inter_tile_distance_scale=inter_tile_distance_scale))
    
    removed_offset_IDs = []
    new_overlaps = []
    updated_overlaps = []
    nonoverlapping_tile_IDs = set(tiles.keys()) #Tiles with no overlaps
    
    for overlap in generated_overlaps:
        nonoverlapping_tile_IDs -= set(overlap.ID)
        
    # new_or_updated = []
    
    # Iterate the current set of overlaps and determine if:
    # 1. The overlap is new and should be included  
    # 2. The overlap is not different in a meaningful way
    # 3. The overlap is changed and should be recalculated
    num_new = 0
    num_different = 0
    num_similiar = 0
    
    if existing_overlaps is None:
        new_overlaps.extend(generated_overlaps)
        num_new = len(generated_overlaps)
    else:
        existing_dict = {o.ID: o for o in existing_overlaps} 
        updated_dict = {o.ID: o for o in generated_overlaps} 
         
        # Remove overlaps that no longer exist so they aren't considered later
        for overlap in existing_overlaps:
            if overlap.ID not in updated_dict:
    #             to_remove = existing_dict[overlap.ID]
    #             del existing_dict[overlap.ID]
    #             existing_overlaps.remove(to_remove)
                print("Removing overlap {0}".format(str(overlap)))
                removed_offset_IDs.append(overlap.ID)
 
        for updated in generated_overlaps:
            if not updated.ID in existing_dict:
                # new_or_updated.append(updated)
                new_overlaps.append(updated)
                num_new = num_new + 1
                print("New overlap {0}".format(str(updated)))
            else:
                existing = existing_dict[updated.ID]
                
                # Compare the offsets
                delta = updated.scaled_offset - existing.scaled_offset
                distance = nornir_imageregistration.array_distance(delta)
                
                # Check whether it is significantly different
                if distance < offset_epsilon:
                    # Substantially the same, recycle the feature scores
                    updated.feature_scores = existing.feature_scores
                    num_similiar += 1
                else:
                    updated_overlaps.append(updated)
                    num_different += 1
    
    print("\n")
    print("Updated Tile Overlaps:")
    print("{0} overlaps new".format(num_new))
    print("{0} overlaps unchanged".format(num_similiar))
    print("{0} overlaps changed".format(num_different))
    print("{0} overlaps removed".format(len(removed_offset_IDs)))
    print("\n")
        
    return (generated_overlaps, new_overlaps, updated_overlaps, removed_offset_IDs, nonoverlapping_tile_IDs)


def ScoreTileOverlaps(tile_overlaps):
    '''
    Assigns feature scores to TileOverlap objects without scores.
    :param list tile_overlaps: list of TileOverlap objects
    :return: The TileOverlap object list
    '''

    tile_to_overlaps_dict = __CreateTileToOverlapsDict(tile_overlaps)
    
    tile_feature_score_list = []
    
    tasks = []
    pool = nornir_pools.GetGlobalLocalMachinePool()
    
    for tile_ID in list(tile_to_overlaps_dict.keys()):
        tile_overlaps_dict = tile_to_overlaps_dict[tile_ID]
        
        params = []  # Build a list of overlaps that need to be scored
        for (iTile, tile_overlap) in tile_overlaps_dict.values():
            if tile_overlap.feature_scores[iTile] is None:
                params.append(TileOverlapDetails(overlap_ID=tile_overlap.ID, iTile=iTile, overlapping_rect=tile_overlap.scaled_overlapping_source_rects[iTile]))
        
        # tile_feature_scores = _CalculateTileFeatures(tiles[tile_ID].ImagePath, params)
        # tile_feature_score_list.append(tile_feature_scores)
        if len(params) == 0:
            continue
        
        first_overlap = list(tile_overlaps_dict.values())[0]
        tile = first_overlap.tile_overlap.Tiles[first_overlap.iTile]
        t = pool.add_task(str(tile_ID), _CalculateTileFeatures, tile.ImagePath, params)
        tasks.append(t)
    
    for t in tasks:
        tile_feature_scores = t.wait_return()
#         
#         for score in tile_feature_scores:
#             filename = os.path.join("C:\Temp", "{0:0.05f}_{1}.tif".format(score.feature_score, score.overlap_ID))
#             pool.add_task("Save {0}".format(filename), nornir_imageregistration.SaveImage, filename, score.image)
            
        tile_feature_score_list.extend(tile_feature_scores)
        
    for score in tile_feature_score_list:
        tile_ID = score.overlap_ID[score.iTile]
        
        tile_overlaps_dict = tile_to_overlaps_dict[tile_ID]
        specific_overlap = tile_overlaps_dict[score.overlap_ID]
        specific_overlap.tile_overlap.feature_scores[score.iTile] = score.feature_score

    return tile_overlaps
      

def _CalculateTileFeatures(image_path, list_overlap_tuples, feature_coverage_score=None):
    
    #image = nornir_imageregistration.ImageParamToImageArray(image_path, dtype=np.float32)

    ImageDataList = [ TileOverlapFeatureScore(overlap_ID=overlap_ID,
                                              iTile=iTile,
                                              image=None,  # __get_overlapping_image(image, overlapping_rect, excess_scalar=1.0, cval=np.nan),
                                              feature_score=nornir_imageregistration.image_stats.__CalculateFeatureScoreSciPy__(__get_overlapping_image(image_path, overlapping_rect, excess_scalar=1.0, cval=np.nan), feature_coverage_score=feature_coverage_score)) 
                    for (overlap_ID, iTile, overlapping_rect) in list_overlap_tuples]
    
    #del image
    return ImageDataList

# 
# def RefineTranslations(transforms, imagepaths, imageScale=None, subregion_shape=None):
#     '''
#     Refine the initial translate results by registering a number of smaller regions and taking the average offset.  Then update the offsets.
#     This still produces a translation only offset
#     '''
#     if imageScale is None:
#         imageScale = 1.0
#         
#     if subregion_shape is None:
#         subregion_shape = np.array([128, 128])
#         
#     downsample = 1.0 / imageScale
#     
#     tiles = nornir_imageregistration.tile.CreateTiles(transforms, imagepaths)
#     list_tiles = list(tiles.values())
#     pool = nornir_pools.GetGlobalMultithreadingPool()
#     tasks = list()
#     
#     if imageScale is None:
#         imageScale = tileset.MostCommonScalar(transforms, imagepaths)
#     
#     layout = nornir_imageregistration.layout.Layout()    
#     for t in list_tiles:
#         layout.CreateNode(t.ID, t.FixedBoundingBox.Center)
#         
#     for A, B in nornir_imageregistration.tile.IterateOverlappingTiles(list_tiles, minOverlap=0.03):
#         # OK... add some small neighborhoods and register those...
#         (downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment) = nornir_imageregistration.tile.Tile.Calculate_Overlapping_Regions(A, B, imageScale)
# #         
#           
#         task = pool.add_task("Align %d -> %d" % (A.ID, B.ID), __RefineTileAlignmentRemote, A, B, downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment, imageScale, subregion_shape)
#         task.A = A
#         task.B = B
#         task.OffsetAdjustment = OffsetAdjustment
#         tasks.append(task)
# #          
# #         (point_pairs, net_offset) = __RefineTileAlignmentRemote(A, B, downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment, imageScale)
# #         offset = net_offset[0:2] + OffsetAdjustment
# #         weight = net_offset[2]
# #           
# #         print("%d -> %d : %s" % (A.ID, B.ID, str(net_offset)))
# #           
# #         layout.SetOffset(A.ID, B.ID, offset, weight)
#          
#         # print(str(net_offset))
#         
#     for t in tasks:
#         try:
#             (point_pairs, net_offset) = t.wait_return()
#         except Exception as e:
#             print("Could not register %d -> %d" % (t.A.ID, t.B.ID))
#             print("%s" % str(e))
#             continue 
#         
#         offset = net_offset[0:2] + (t.OffsetAdjustment * downsample)
#         weight = net_offset[2]
#         layout.SetOffset(t.A.ID, t.B.ID, offset, weight) 
#         
#         # Figure out what offset we found vs. what offset we expected
#         PredictedOffset = t.B.FixedBoundingBox.Center - t.A.FixedBoundingBox.Center
#         
#         diff = offset - PredictedOffset
#         distance = np.sqrt(np.sum(diff ** 2))
#         
#         print("%d -> %d = %g" % (t.A.ID, t.B.ID, distance))
#         
#     pool.wait_completion()
#     
#     return (layout, tiles)
            
            
def _FindTileOffsets(tile_overlaps, excess_scalar, imageScale=None, existing_layout=None):
    '''Populates the OffsetToTile dictionary for tiles
    :param list tile_overlaps: List of all tile overlaps or dictionary whose values are tile overlaps
    :param float imageScale: downsample level if known.  None causes it to be calculated.
    :param float excess_scalar: How much additional area should we pad the overlapping rectangles with.
    :return: A layout object describing the optimal adjustment for each tile to align with each neighboring tile
    '''
    
    if imageScale is None:
        imageScale = 1.0
        
    downsample = 1.0 / imageScale

    # idx = tileset.CreateSpatialMap([t.FixedBoundingBox for t in tiles], tiles)

    CalculationCount = 0
    
    # _CalculateImageFFTs(tiles)
  
    #pool = nornir_pools.GetGlobalSerialPool()
    pool = nornir_pools.GetGlobalMultithreadingPool()
    tasks = list()
    
    layout = existing_layout
    if layout is None:
        layout = nornir_imageregistration.layout.Layout()
    
    list_tile_overlaps = tile_overlaps
    if isinstance(tile_overlaps, dict):
        list_tile_overlaps = list(tile_overlaps.values())
        
    assert(isinstance(list_tile_overlaps, list))
    
    for t in list_tile_overlaps:
        if not layout.Contains(t.A.ID):
            layout.CreateNode(t.A.ID, t.A.FixedBoundingBox.Center)

        if not layout.Contains(t.B.ID):
            layout.CreateNode(t.B.ID, t.B.FixedBoundingBox.Center)
        
    print("Starting tile alignment") 
    for tile_overlap in list_tile_overlaps:
        t = pool.add_task("Align %d -> %d" % (tile_overlap.ID[0], tile_overlap.ID[1]),
                          __tile_offset_remote,
                          tile_overlap.A.ImagePath,
                          tile_overlap.B.ImagePath,
                          tile_overlap.scaled_overlapping_source_rect_A,
                          tile_overlap.scaled_overlapping_source_rect_B,
                          tile_overlap.scaled_offset,
                          excess_scalar)
        
        t.tile_overlap = tile_overlap
        tasks.append(t)
        CalculationCount += 1
        # print("Start alignment %d -> %d" % (A.ID, B.ID))

    for t in tasks:
        try:
            offset = t.wait_return()
        except FloatingPointError as e:  # Very rarely the overlapping region is entirely one color and this error is thrown.
            print("FloatingPointError: %d -> %d = %s -> Using stage coordinates." % (t.tile_overlap.A.ID, t.tile_overlap.B.ID, str(e)))
            
            # Create an alignment record using only stage position and a weight of zero 
            offset = nornir_imageregistration.AlignmentRecord(peak=t.tile_overlap.scaled_offset, weight=0)
            
        tile_overlap = t.tile_overlap
        # Figure out what offset we found vs. what offset we expected
        PredictedOffset = tile_overlap.B.FixedBoundingBox.Center - tile_overlap.A.FixedBoundingBox.Center
        ActualOffset = offset.peak * downsample
        
        diff = ActualOffset - PredictedOffset
        distance = np.sqrt(np.sum(diff ** 2))
        f_score = min(tile_overlap.feature_scores)
        final_weight = offset.weight * f_score
        print("%d -> %d = feature score: %.04g align score: %.04g Final Weight: %.04g Dist: %.04g" % (tile_overlap.A.ID, tile_overlap.B.ID, f_score, offset.weight, final_weight, distance))
        
        layout.SetOffset(tile_overlap.A.ID, tile_overlap.B.ID, ActualOffset, final_weight) 
        
    pool.wait_completion()
    
    print(("Total offset calculations: " + str(CalculationCount)))

    return layout


def __get_overlapping_image(image, overlapping_rect, excess_scalar, cval=None):
    '''
    Crop the tile's image so it contains the specified rectangle
    '''
    
    if cval is None:
        cval = 'random'
    
    if excess_scalar > 3:
        excess_scalar = 3.0
    
    Width = overlapping_rect.Width * excess_scalar
    Height = overlapping_rect.Height * excess_scalar
    scaled_rect = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(overlapping_rect.Center, [Height, Width])
    
    #scaled_rect = nornir_imageregistration.Rectangle.scale_on_center(overlapping_rect, excess_scalar)
    scaled_rect = nornir_imageregistration.Rectangle.SafeRound(scaled_rect)
    return nornir_imageregistration.CropImage(image, Xo=int(scaled_rect.BottomLeft[1]), Yo=int(scaled_rect.BottomLeft[0]), Width=int(scaled_rect.Width), Height=int(scaled_rect.Height), cval=cval)
    
    # return nornir_imageregistration.PadImageForPhaseCorrelation(cropped, MinOverlap=1.0, PowerOfTwo=True)
 

def __tile_offset_remote(A_Filename, B_Filename, scaled_overlapping_source_rect_A, scaled_overlapping_source_rect_B, OffsetAdjustment, excess_scalar):
    '''
    :param A_Filename: Path to tile A
    :param B_Filename: Path to tile B
    :param scaled_overlapping_source_rect_A: Region of overlap on tile A with tile B
    :param scaled_overlapping_source_rect_B: Region of overlap on tile B with tile A
    :param OffsetAdjustment: scaled_offset to account for the (center) position of tile B relative to tile A.  If the overlapping rectangles are perfectly aligned the reported offset would be (0,0).  OffsetAdjustment would be added to that (0,0) result to ensure Tile B remained in the same position. 
    :param float excess_scalar: How much additional area should we pad the overlapping rectangles with.
    Return the offset required to align to image files.
    This function exists to minimize the inter-process communication
    '''
    
    ShowImages = False 
    A = nornir_imageregistration.LoadImage(A_Filename)
    B = nornir_imageregistration.LoadImage(B_Filename)

# I had to add the .astype call above for DM4 support, but I recall it broke PMG input.  Leave this comment here until the tests are passing
#    A = nornir_imageregistration.LoadImage(A_Filename) #.astype(dtype=np.float16)
#    B = nornir_imageregistration.LoadImage(B_Filename) #.astype(dtype=np.float16)
    
    # I tried a 1.0 overlap.  It works better for light microscopy where the reported stage position is more precise
    # For TEM the stage position can be less reliable and the 1.5 scalar produces better results
    # For the latest version of the code that uses only the overlapping region 3 is appropriate because it allows the alignment point to be anywhere on the image without ambiguity
    OverlappingRegionA = __get_overlapping_image(A, scaled_overlapping_source_rect_A, excess_scalar=excess_scalar)
    OverlappingRegionB = __get_overlapping_image(B, scaled_overlapping_source_rect_B, excess_scalar=excess_scalar)
    
    
    if ShowImages:
        o_a = __get_overlapping_image(A, scaled_overlapping_source_rect_A, excess_scalar=1.0, cval=0)
        o_b = __get_overlapping_image(B, scaled_overlapping_source_rect_B, excess_scalar=1.0, cval=0)
        
    #nornir_imageregistration.ShowGrayscale([[OverlappingRegionA, OverlappingRegionB],[o_a,o_b]])
    OverlappingRegionA = OverlappingRegionA.astype(np.float32)
    OverlappingRegionB = OverlappingRegionB.astype(np.float32)
    
    # It is fairly common to underflow when dividing float16 images, so just warn and move on. 
    # I spent a day debugging why a mosaic was not building correctly to find the underflow 
    # issue, so don't remove it.  The underflow error removes one of the ties between a tile
    # and its neighbors.
    
    # Note error levelshould now be set in nornir_imageregistration.__init__
    # old_float_err_settings = np.seterr(under='warn')
    
    # If the entire region is a solid color, then return an alignment record with no offset and a weight of zero
    if (OverlappingRegionA.min() == OverlappingRegionA.max()) or \
        (OverlappingRegionA.max() == 0) or \
        (OverlappingRegionB.min() == OverlappingRegionB.max()) or \
        (OverlappingRegionB.max() == 0):
        return nornir_imageregistration.AlignmentRecord(peak=OffsetAdjustment, weight=0)
            
    OverlappingRegionA -= OverlappingRegionA.min()
    OverlappingRegionA /= OverlappingRegionA.max()
    
    OverlappingRegionB -= OverlappingRegionB.min()
    OverlappingRegionB /= OverlappingRegionB.max()
    
    # nornir_imageregistration.ShowGrayscale([OverlappingRegionA, OverlappingRegionB]) nornir_imageregistration.ShowGrayscale([[o_a, o_b],[OverlappingRegionA, OverlappingRegionB]])
    
    record = nornir_imageregistration.FindOffset(OverlappingRegionA, OverlappingRegionB, FFT_Required=True)#, FixedImageShape=scaled_overlapping_source_rect_A.shape, MovingImageShape=scaled_overlapping_source_rect_B.shape)
    
    #overlapping_rect_B_AdjustedToPeak = nornir_imageregistration.Rectangle.translate(scaled_overlapping_source_rect_B, -record.peak) 
    #overlapping_rect_B_AdjustedToPeak = nornir_imageregistration.Rectangle.change_area(overlapping_rect_B_AdjustedToPeak, scaled_overlapping_source_rect_A.Size)
    #median_diff = __AlignmentScoreRemote(A, B, scaled_overlapping_source_rect_A, overlapping_rect_B_AdjustedToPeak)
    #nornir_imageregistration.ShowGrayscale([[OverlappingRegionA, OverlappingRegionB], [overlapping_rect_B_AdjustedToPeak, median_diff]])
    #diff_weight = 1.0 - median_diff
    #np.seterr(**old_float_err_settings)
    
    #nornir_imageregistration.views.plot_aligned_images(record, o_a, o_b)
    adjusted_record = nornir_imageregistration.AlignmentRecord(np.array(record.peak) + OffsetAdjustment, record.weight)
    
    if ShowImages:
        nornir_imageregistration.views.plot_aligned_images(record, o_a,o_b)
        del o_a
        del o_b
        
    del A
    del B
    del OverlappingRegionA
    del OverlappingRegionB
    
    return adjusted_record

def BuildOverlappingTileDict(list_tiles, minOverlap=0.05):
    ''':return: A map of tile ID to all overlapping tile IDs'''
    
    list_rects = []
    for tile in list_tiles:
        list_rects.append(tile.FixedBoundingBox)
        
    rset = nornir_imageregistration.RectangleSet.Create(list_rects)
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
    
    for tile_overlap in nornir_imageregistration.tile_overlap.IterateTileOverlaps(list_tiles, imageScale=imageScale):
        # (downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment) = nornir_imageregistration.tile.Tile.Calculate_Overlapping_Regions(tile_overlap.A, tile_overlap.B, imageScale)
        
        # __AlignmentScoreRemote(A.ImagePath, B.ImagePath, downsampled_overlapping_rect_A, downsampled_overlapping_rect_B)
        
        t = pool.add_task("Score %d -> %d" % (tile_overlap.ID[0], tile_overlap.ID[1]),
                          __AlignmentScoreRemote,
                          tile_overlap.A.ImagePath,
                          tile_overlap.B.ImagePath,
                          tile_overlap.scaled_overlapping_source_rect_A,
                          tile_overlap.scaled_overlapping_source_rect_B)
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


def __AlignmentScoreRemote(A_Filename, B_Filename, scaled_overlapping_source_rect_A, scaled_overlapping_source_rect_B):
    '''Returns the difference between the images'''
    
    try: 
        OverlappingRegionA = __get_overlapping_image(nornir_imageregistration.ImageParamToImageArray(A_Filename, dtype=np.float16),
                                                     scaled_overlapping_source_rect_A,
                                                     excess_scalar=1.0)
        OverlappingRegionB = __get_overlapping_image(nornir_imageregistration.ImageParamToImageArray(B_Filename, dtype=np.float16),
                                                     scaled_overlapping_source_rect_B,
                                                     excess_scalar=1.0)
        
        # If the entire region is a solid color, then return the maximum score possible
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
        
        # There was data in the aligned images, but not overlapping.  So we return the maximum value
        if np.alltrue(ignoreIndicies):
            return 1.0
        
        validIndicies = np.invert(ignoreIndicies)
          
        OverlappingRegionA -= OverlappingRegionB
        absoluteDiff = np.fabs(OverlappingRegionA)
        
        # nornir_imageregistration.ShowGrayscale([OverlappingRegionA, OverlappingRegionB, absoluteDiff])
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
