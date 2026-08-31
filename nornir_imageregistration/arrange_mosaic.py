"""
Created on Jul 10, 2012

@author: Jamesan
"""

import collections
import typing
from typing import Any, Iterable, Sequence

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import cp
from nornir_imageregistration.layout import Layout
import nornir_imageregistration.phasecorrelation
import nornir_imageregistration.type_info
import nornir_pools
import nornir_shared.prettyoutput

from nornir_imageregistration.tile_overlap import TileOverlap
from nornir_imageregistration.mosaic_tileset import MosaicTileset
from nornir_imageregistration.tile import Tile

TileOverlapDetails = collections.namedtuple('TileOverlapDetails',
                                            'overlap_ID iTile overlapping_rect')

TileToOverlap = collections.namedtuple('TileToOverlap',
                                       'iTile tile_overlap')

TileOverlapFeatureScore = collections.namedtuple('TileOverlapFeatureScore',
                                                 'overlap_ID iTile image feature_score')


def CreateTileToOverlapsDict(tile_overlaps: dict[Any, TileOverlap] | Sequence[TileOverlap]) -> \
        collections.defaultdict[int, dict[int, TileToOverlap]]:
    """
    Returns a dictionary containing a list of tuples with (TileIndex, OverlapObject)
    TileIndex records if the tile is the first or second tile (A or B)
    described in the overlap object
    """

    if isinstance(tile_overlaps, dict):
        tile_overlaps = list(tile_overlaps.values())

    tile_to_overlaps_dict = collections.defaultdict(dict)
    for tile_overlap in tile_overlaps:
        tile_to_overlaps_dict[tile_overlap.A.ID][tile_overlap.ID] = TileToOverlap(iTile=0, tile_overlap=tile_overlap)
        tile_to_overlaps_dict[tile_overlap.B.ID][tile_overlap.ID] = TileToOverlap(iTile=1, tile_overlap=tile_overlap)

    return tile_to_overlaps_dict


def _CalculateImageFFTs(tiles):
    """
    Ensure all tiles have FFTs calculated and cached
    """
    pool = nornir_pools.GetGlobalLocalMachinePool()

    fft_tasks = []
    for t in tiles.values():
        task = pool.add_task("Create padded image", t.PrecalculateImages)
        task.tile = t  # type: ignore[attr-defined]
        fft_tasks.append(task)

    print("Calculating FFTs\n")
    pool.wait_completion()


def TranslateTiles(transforms, imagepaths, excess_scalar, imageScale=None, max_relax_iterations=None,
                   max_relax_tension_cutoff=None):
    """
    Finds the optimal translation of a set of tiles to construct a larger seemless mosaic.
    :param list transforms: list of transforms for tiles
    :param list imagepaths: list of paths to tile images, must be same length as transforms list
    :param float excess_scalar: How much additional area should we pad the overlapping regions with.
    :param float imageScale: The downsampling of the images in imagepaths.  If None then this is calculated based on the difference in the transform and the image file dimensions
    :param int max_relax_iterations: Maximum number of iterations in the relax stage
    :param float max_relax_tension_cutoff: Stop relaxation stage if the maximum tension vector is below this value
    :return: (offsets_collection, tiles) tuple
    """

    if max_relax_iterations is None:
        max_relax_iterations = 150

    if max_relax_tension_cutoff is None:
        max_relax_tension_cutoff = 1.0

    if imageScale is None:
        imageScale = nornir_imageregistration.tileset.MostCommonScalar(transforms, imagepaths)

    mosaic_tileset = nornir_imageregistration.mosaic_tileset.Create(transforms, imagepaths, imageScale)

    tile_layout = _FindTileOffsets(mosaic_tileset, excess_scalar, image_to_source_space_scale=imageScale)  # type: ignore[arg-type]

    nornir_imageregistration.layout.ScaleOffsetWeightsByPopulationRank(tile_layout)
    nornir_imageregistration.layout.RelaxLayout(tile_layout, max_tension_cutoff=max_relax_tension_cutoff,
                                                max_iter=max_relax_iterations)

    # final_layout = nornir_imageregistration.layout.BuildLayoutWithHighestWeightsFirst(offsets_collection)

    # Create a mosaic file using the tile paths and transforms
    return tile_layout, mosaic_tileset


def TranslateTiles2(tileset: nornir_imageregistration.mosaic_tileset.MosaicTileset,
                    config: nornir_imageregistration.settings.TranslateSettings):
    """
    Finds the optimal translation of a set of tileset to construct a larger seemless mosaic.
    :param tileset:
    :param config:
    """

    if len(tileset) == 1:
        # If there is only one tile then just return it
        single_tile_layout = nornir_imageregistration.layout.Layout()
        single_tile_layout.CreateNode(tileset[0].ID, np.zeros(2))
        return single_tile_layout, tileset

    last_pass_overlaps = None
    translated_layout = None
    iPass = config.max_translate_iterations

    pass_count = 0
    inter_tile_distance_scale_this_pass = config.inter_tile_distance_scale  # first_pass_inter_tile_distance_scale
    # inter_tile_distance_scale_last_pass = inter_tile_distance_scale_this_pass

    # first_pass_overlaps = None #The set of offsets for each tile pair from the first-pass.  Used to align layouts that are not connected.

    stage_reported_overlaps = None
    relaxed_layout = None

    while iPass >= 0:
        result = GenerateTileOverlaps(tileset=tileset,
                                      existing_overlaps=last_pass_overlaps,
                                      offset_epsilon=config.offset_acceptance_threshold,
                                      min_overlap=config.min_overlap,
                                      inter_tile_distance_scale=inter_tile_distance_scale_this_pass,
                                      exclude_diagonal_overlaps=config.exclude_diagonal_overlaps)

        distinct_overlaps = result.generated_overlaps
        new_overlaps = result.new_overlaps
        updated_overlaps = result.updated_overlaps
        removed_overlap_IDs = result.removed_offset_IDs
        non_overlapping_IDs = result.nonoverlapping_tile_IDs

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

        # If this is the second pass remove any overlaps from the layout that no longer qualify
        if translated_layout is not None:
            for ID in removed_overlap_IDs:
                translated_layout.RemoveOverlap(ID)  # type: ignore[arg-type]
            # for ID in nonoverlapping_tile_IDs:
            #    if translated_layout.nodes[ID].ConnectedIDs.shape[0] != 0:
            #        raise NornirUserException("Non-overlapping node should not have overlaps")

            # self.assertTrue(translated_layout.nodes[ID].ConnectedIDs.shape[0] == 0, "Non-overlapping node should not have overlaps")
        #                    translated_layout.RemoveNode(ID)

        inter_tile_distance_scale_this_pass = config.inter_tile_distance_scale

        # Create a list of offsets requiring updates
        filtered_overlaps_needing_offsets = []
        if not config.feature_score_calculations_required:
            filtered_overlaps_needing_offsets = new_or_updated_overlaps
        else:
            ScoreTileOverlaps(distinct_overlaps)
            NormalizeOverlapFeatureScores(distinct_overlaps)

            for overlap in new_or_updated_overlaps:
                if config.feature_score_threshold is not None:
                    if (overlap.feature_scores[0] >= config.feature_score_threshold and
                            overlap.feature_scores[1] >= config.feature_score_threshold):
                        filtered_overlaps_needing_offsets.append(overlap)
                    else:
                        if translated_layout is not None:
                            translated_layout.RemoveOverlap(overlap)
                else:
                    filtered_overlaps_needing_offsets.append(overlap)

        translated_layout = _FindTileOffsets(filtered_overlaps_needing_offsets,
                                             excess_scalar=config.excess_scalar,
                                             image_to_source_space_scale=tileset.image_to_source_space_scale,
                                             existing_layout=translated_layout,
                                             use_feature_score=config.use_feature_score,
                                             mask_extrema=config.mask_extrema)

        scaled_translated_layout = translated_layout.copy()
        # nornir_imageregistration.layout.SetUniformOffsetWeights(scaled_translated_layout)
        nornir_imageregistration.layout.NormalizeOffsetWeights(scaled_translated_layout,
                                                               min_allowed_weight=config.min_offset_weight,
                                                               max_allowed_weight=config.max_offset_weight)
        # nornir_imageregistration.layout.ScaleOffsetWeightsByPopulationRank(scaled_translated_layout,
        #                                                                    min_allowed_weight=min_offset_weight,
        #                                                                    max_allowed_weight=max_offset_weight)

        #        nornir_imageregistration.layout.NormalizeOffsetWeights(scaled_translated_layout)

        translated_final_layouts = nornir_imageregistration.layout.BuildLayoutWithHighestWeightsFirst(
            scaled_translated_layout)
        # TODO: Pass the dictionary to this function that indicates tile offsets for pairs of tileset
        # translated_final_layout = nornir_imageregistration.layout.MergeDisconnectedLayoutsWithOffsets(translated_final_layouts, stage_reported_overlaps)

        # Should we do a shorter pass on the first run?
        relax_iterations = config.max_relax_iterations
        # if iPass == min_translate_iterations:
        #     relax_iterations = relax_iterations // 4
        #     if relax_iterations < 10:
        #         relax_iterations = max_relax_iterations // 2

        relaxed_layouts = []
        for layout in translated_final_layouts:
            relaxed_layout = nornir_imageregistration.layout.RelaxLayout(layout,
                                                                         max_iter=relax_iterations,
                                                                         max_tension_cutoff=config.max_relax_tension_cutoff)
            relaxed_layouts.append(relaxed_layout)

        relaxed_layout = nornir_imageregistration.layout.MergeDisconnectedLayoutsWithOffsets(relaxed_layouts,
                                                                                             stage_reported_overlaps)
        if relaxed_layout is None:
            # Merge returns None for an empty layout_list, so an empty translated_final_layouts
            # would have made the three dereferences below fail. Leave the loop and let the
            # stage-position fallback answer instead.
            break

        relaxed_layout.TranslateToZeroOrigin()
        relaxed_layout.UpdateTileTransforms(tileset)
        last_pass_overlaps = distinct_overlaps

        # Copy the relaxed layout positions back into the translated layout
        for ID, node in relaxed_layout.nodes.items():
            tnode = translated_layout.nodes[ID]
            tnode.Position = node.Position

        iPass -= 1
        pass_count += 1

    # final_layout = nornir_imageregistration.layout.BuildLayoutWithHighestWeightsFirst(offsets_collection)

    if relaxed_layout is None:
        # No pass ever produced a layout, because the very first GenerateTileOverlaps found
        # nothing to align. Reaching the dereference below with None raised
        # "AttributeError: 'NoneType' object has no attribute 'TranslateToZeroOrigin'", which
        # said nothing about the tiles. Two tiles merely touching edge to edge is enough to
        # get here -- adjacency is not overlap -- as are tiles whose stage positions are far
        # apart, or a min_overlap set above what the mosaic actually has (#128).
        #
        # Fall back to the stage positions, matching what the single-tile branch above does
        # with its one node. That is the best available answer when nothing could be measured,
        # and it keeps a disconnected mosaic buildable instead of aborting the section. Tile
        # transforms are deliberately left untouched: no alignment was measured, so there is
        # nothing to write back.
        nornir_shared.prettyoutput.LogErr(
            f"No qualifying tile overlaps found among {len(tileset)} tiles -> Using stage "
            f"coordinates. No alignment could be measured; check that the tiles overlap and "
            f"that min_overlap ({config.min_overlap}) is not above the overlap the mosaic "
            f"actually has.")

        relaxed_layout = nornir_imageregistration.layout.Layout()
        for tile in tileset.values():
            relaxed_layout.CreateNode(tile.ID, tile.FixedBoundingBox.Center)

    # Create a mosaic file using the tile paths and transforms
    relaxed_layout.TranslateToZeroOrigin()
    return relaxed_layout, tileset


class GeneratedTileOverlapsResult(typing.NamedTuple):
    """
    A result object for the GenerateTileOverlaps function
    """
    generated_overlaps: list[TileOverlap]  # All of the overlaps found between tiles
    new_overlaps: list[TileOverlap]  # Overlaps that are new
    updated_overlaps: list[TileOverlap]  # Overlaps that existed previously and were updated
    removed_offset_IDs: set[int]  # Offsets that were removed after the new overlaps were calculated
    nonoverlapping_tile_IDs: set[int]  # Tiles that do not overlap


def GenerateTileOverlaps(tileset: nornir_imageregistration.mosaic_tileset.MosaicTileset,
                         existing_overlaps: list[TileOverlap] | None = None,
                         offset_epsilon: float = 1.0,
                         min_overlap: float | None = None,
                         inter_tile_distance_scale: float | None = None,
                         exclude_diagonal_overlaps: bool = False) \
        -> GeneratedTileOverlapsResult:
    """
    Create a list of TileOverlap objects for each overlapping region in the mosaic.  Assign a feature score to the regions from each image that overlap.
    :param inter_tile_distance_scale:
    :param exclude_diagonal_overlaps:
    :param MosaicTileset tileset: A dictionary of Tile objects or MosaicTileset
    :param existing_overlaps: A list of overlaps created previously.  Scores for these offsets will be copied into the generated offsets if the difference in offset between the tiles
                                   is less than offset_epsilon.
    :param float offset_epsilon: The distance the expected offset between tiles has to change before we recalculate feature scores and registration
    :param float min_overlap: Tiles that overlap less than this amount percentage of area will not be included

    :return: Returns a four-component tuple composed of all found overlaps, the new overlaps, the overlaps that require updating, the deleted overlaps from the existing set, and the IDs of non-overlapping tiles.
             None of the returned overlaps are the same objects as those in the original set
    """
    if not isinstance(tileset, dict):
        raise ValueError(f'tileset parameter is expected to be a dictionary of tiles or a MosaicTileset')

    if inter_tile_distance_scale is None:
        inter_tile_distance_scale = 1.0

    if inter_tile_distance_scale < 0 or inter_tile_distance_scale > 1.0:
        raise ValueError(
            "inter_tile_distance_scale must be in the range 0 to 1: value was {0}".format(inter_tile_distance_scale))

    generated_overlaps = list(nornir_imageregistration.tile_overlap.CreateTileOverlaps(tileset,
                                                                                       tileset.image_to_source_space_scale,
                                                                                       min_overlap=min_overlap,
                                                                                       inter_tile_distance_scale=inter_tile_distance_scale,
                                                                                       exclude_diagonal_overlaps=exclude_diagonal_overlaps))

    removed_offset_IDs = set()
    new_overlaps = []
    updated_overlaps = []
    nonoverlapping_tile_IDs = set(tileset.keys())  # Tiles with no overlaps

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
                removed_offset_IDs.add(overlap.ID)

        for updated in generated_overlaps:
            if not updated.ID in existing_dict:
                # new_or_updated.append(updated)
                new_overlaps.append(updated)
                num_new += 1
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

    return GeneratedTileOverlapsResult(generated_overlaps, new_overlaps, updated_overlaps, removed_offset_IDs,
                                       nonoverlapping_tile_IDs)


def ScoreTileOverlaps(tile_overlaps: Sequence[TileOverlap]):
    """
    Assigns feature scores to TileOverlap objects without scores.
    :param tile_overlaps: list of TileOverlap objects
    :return: The TileOverlap object list
    """

    tile_to_overlaps_dict = CreateTileToOverlapsDict(tile_overlaps)

    tile_feature_score_list = []

    tasks = []
    tile_to_overlaps_keys = list(tile_to_overlaps_dict.keys())

    if len(tile_overlaps) > 1:
        pool = nornir_pools.GetGlobalLocalMachinePool()
    else:
        pool = nornir_pools.GetGlobalSerialPool()

    for tile_ID in tile_to_overlaps_keys:
        tile_overlaps_dict = tile_to_overlaps_dict[tile_ID]

        overlaps = []  # Build a list of overlaps that need to be scored
        for (iTile, tile_overlap) in tile_overlaps_dict.values():
            if tile_overlap.feature_scores[iTile] is None or np.any(np.isnan(np.array(tile_overlap.feature_scores))):
                overlaps.append(TileOverlapDetails(overlap_ID=tile_overlap.ID, iTile=iTile,
                                                   overlapping_rect=tile_overlap.scaled_overlapping_source_rects[
                                                       iTile]))

        if len(overlaps) == 0:
            continue

        first_overlap = list(tile_overlaps_dict.values())[0]
        tile = first_overlap.tile_overlap.Tiles[first_overlap.iTile]
        t = pool.add_task(str(tile_ID), _CalculateTileFeatures, tile.ImagePath, overlaps)
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

        if tile_ID == specific_overlap.tile_overlap.A.ID:
            specific_overlap.tile_overlap.A_feature_score = score.feature_score
        elif tile_ID == specific_overlap.tile_overlap.B.ID:
            specific_overlap.tile_overlap.B_feature_score = score.feature_score
        else:
            raise ValueError("Tile ID {0} does not match either tile in overlap {1}".format(tile_ID,
                                                                                            specific_overlap.tile_overlap.ID))

    return tile_overlaps


def _CalculateTileFeatures(image_path: str, list_overlap_tuples: list[TileOverlapDetails], feature_coverage_score=None):
    # image = nornir_imageregistration.ImageParamToImageArray(image_path, dtype=np.float32)

    ImageDataList = [TileOverlapFeatureScore(overlap_ID=overlap_ID,
                                             iTile=iTile,
                                             image=None,
                                             # __get_overlapping_image(image, overlapping_rect, excess_scalar=1.0, cval=np.nan),
                                             feature_score=nornir_imageregistration.image_stats.__CalculateFeatureScoreSciPy__(
                                                 __get_overlapping_image(image_path, overlapping_rect,
                                                                         mask_extrema=False,
                                                                         # Do not mask extrema because we are measuring the variance and don't want random numbers injected
                                                                         excess_scalar=1.0, cval=np.nan,
                                                                         dtype=np.float16),  # type: ignore[arg-type]
                                                 feature_coverage_score=feature_coverage_score))
                     for (overlap_ID, iTile, overlapping_rect) in list_overlap_tuples]

    # del image
    return ImageDataList


def _is_scored(score: float | None) -> bool:
    """True if a feature score carries a usable measurement.

    ``TileOverlap`` initializes ``_feature_scores`` to ``(nan, nan)`` and ``ScoreTileOverlaps``
    tests for ``None``, so both mean "not scored yet" rather than "scored as zero".
    """
    return score is not None and bool(np.isfinite(score))


def NormalizeOverlapFeatureScores(tile_overlaps: Iterable[TileOverlap]):
    """
    Adds or updates a normalized_feature_score to all overlaps

    Unscored entries (``None`` or non-finite) are passed through unchanged rather than
    normalized, so they stay distinguishable from a genuine measurement downstream.

    :param tile_overlaps: iterable of TileOverlap objects
    :return: The TileOverlap object list
    """

    # Materialized because both loops below need the same elements. The annotation says
    # Iterable, and a generator silently produced no normalization at all: the first loop
    # exhausted it and the second never ran, leaving every normalized_feature_scores at its
    # previous value. The live caller passes a list, so this was latent. (#123)
    tile_overlaps = list(tile_overlaps)

    # max() over the raw tuples raised TypeError on the None scores ScoreTileOverlaps
    # explicitly allows, because None does not order against float. Reducing over only the
    # scored values also keeps a not-yet-scored (nan) overlap from participating, which
    # previously depended on argument order -- max(0, nan) is 0 but max(nan, 0) is nan. (#123)
    scored = [score for tile_overlap in tile_overlaps
              for score in tile_overlap.feature_scores if _is_scored(score)]
    max_score = max(scored) if len(scored) > 0 else 0.0

    for tile_overlap in tile_overlaps:
        if max_score > 0:
            tile_overlap.normalized_feature_scores = tuple(  # type: ignore[assignment]
                score / max_score if _is_scored(score) else score
                for score in tile_overlap.feature_scores)
        else:
            # Every overlap reported no usable texture, so max_score is 0 and the old division
            # raised ZeroDivisionError. There is no relative information to express here, and
            # normalization only ever supplies a *relative* confidence: relaxation divides
            # weights by their total, so scaling every overlap by the same constant changes
            # nothing. Uniform 1.0 therefore makes this case a no-op. Uniform 0.0 would instead
            # zero every weight, and relaxation's total_weight == 0 branch would then freeze the
            # layout entirely -- a drastic silent outcome for what is only a missing tie-break.
            # Rejecting featureless overlaps stays the job of feature_score_threshold, which
            # compares raw scores and is unaffected by this. (#123)
            tile_overlap.normalized_feature_scores = tuple(  # type: ignore[assignment]
                1.0 if _is_scored(score) else score
                for score in tile_overlap.feature_scores)


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


def _FindTileOffsets(tile_overlaps: dict[Any, TileOverlap] | Sequence[TileOverlap],
                     excess_scalar: float,
                     image_to_source_space_scale: float | None = None,
                     existing_layout: Layout | None = None,
                     use_feature_score: bool = False,
                     mask_extrema: bool = True):
    """Populates the OffsetToTile dictionary for tiles
    :param tile_overlaps: List of all tile overlaps or dictionary whose values are tile overlaps
    :param image_to_source_space_scale: downsample level if known.  None causes it to be calculated.
    :param excess_scalar: How much additional area should we pad the overlapping rectangles with.
    :param existing_layout: A layout object to update with the new offsets.  If None a new layout object is created.
    :param use_feature_score: True if the feature score of the overlapping region should be included in the results and factored into the alignment score
    :return: A layout object describing the optimal adjustment for each tile to align with each neighboring tile
    """

    if image_to_source_space_scale is None:
        image_to_source_space_scale = 1.0

    if image_to_source_space_scale < 1.0:
        raise ValueError(
            "This might be OK... but images are almost always downsampled.  This exception was added to migrate from old code to this class because at that time all scalars were positive.  For example a downsampled by 4 image must have coordinates multiplied by 4 to match the full-res source space of the transform.")

    downsample = image_to_source_space_scale

    # idx = tileset.CreateSpatialMap([t.FixedBoundingBox for t in tiles], tiles)

    CalculationCount = 0

    # _CalculateImageFFTs(tiles)

    if len(tile_overlaps) == 1 or nornir_imageregistration.UsingCupy():
        pool = nornir_pools.GetGlobalSerialPool()
    else:
        pool = nornir_pools.GetGlobalMultithreadingPool()

    tasks = list()

    layout = existing_layout if existing_layout is not None else nornir_imageregistration.layout.Layout()

    list_tile_overlaps = tile_overlaps
    if isinstance(tile_overlaps, dict):
        list_tile_overlaps = list(tile_overlaps.values())

    assert (isinstance(list_tile_overlaps, list))

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
                          excess_scalar,
                          mask_extrema=mask_extrema, )

        t.tile_overlap = tile_overlap  # type: ignore[attr-defined]
        tasks.append(t)
        CalculationCount += 1

    for t in tasks:
        tile_overlap = t.tile_overlap

        try:
            offset = t.wait_return()
        except FloatingPointError as e:  # Very rarely the overlapping region is entirely one color and this error is thrown.
            nornir_shared.prettyoutput.LogErr("FloatingPointError: %d -> %d = %s -> Using stage coordinates." % (
                t.tile_overlap.A.ID, t.tile_overlap.B.ID, str(e)))

            # Create an alignment record using only stage position and a weight of zero 
            offset = nornir_imageregistration.AlignmentRecord(peak=t.tile_overlap.scaled_offset, weight=0)
            layout.RemoveOverlap(tile_overlap)
        except ValueError as e:
            nornir_shared.prettyoutput.LogErr(
                f"Could not find overlap between:\n\t{t.tile_overlap.A.ImagePath}\n\t{t.tile_overlap.B.ImagePath,}\nConsider using a feature threshold for this section if results are poor.  This message often caused by aligning blank tiles.\n{e}")
            offset = nornir_imageregistration.AlignmentRecord(peak=t.tile_overlap.scaled_offset, weight=0)
            layout.RemoveOverlap(tile_overlap)

        if offset is not None:
            # Figure out what offset we found vs. what offset we expected
            # PredictedOffset = tile_overlap.B.FixedBoundingBox.Center - tile_overlap.A.FixedBoundingBox.Center
            ActualOffset = offset.peak * downsample

            # diff = ActualOffset - PredictedOffset
            # distance = np.sqrt(np.sum(diff ** 2))
            feature_scores = tile_overlap.normalized_feature_scores

            final_weight = offset.weight

            if use_feature_score:
                if feature_scores is None:
                    # Binding f_score only inside a "scores is not None" test left it
                    # unbound on the first such overlap (UnboundLocalError) and, worse,
                    # left the *previous* overlap's score bound for every later one,
                    # since this is a loop body. That silently scaled one overlap's
                    # weight by an unrelated overlap's texture measurement.
                    #
                    # Reaching here means the caller asked to scale by feature score
                    # without computing one. TranslateSettings.feature_score_calculations_required
                    # makes ArrangeTilesWithTranslate run ScoreTileOverlaps and
                    # NormalizeOverlapFeatureScores whenever use_feature_score is set,
                    # so this is a direct-caller contract violation, not a data case.
                    raise ValueError(
                        f"use_feature_score is set but overlap {tile_overlap.ID} has no "
                        f"normalized_feature_scores. Call NormalizeOverlapFeatureScores on "
                        f"the overlaps first, or leave use_feature_score off.")

                final_weight *= min(feature_scores)

            # print("%d -> %d = feature score: %.04g align score: %.04g Final Weight: %.04g Dist: %.04g" % (tile_overlap.A.ID, tile_overlap.B.ID, f_score, offset.weight, final_weight, distance))

            layout.SetOffset(tile_overlap.A.ID, tile_overlap.B.ID, ActualOffset, final_weight)

    pool.wait_completion()

    print(("Total offset calculations: " + str(CalculationCount)))

    return layout


def __get_overlapping_image(imageparam,
                            overlapping_rect: nornir_imageregistration.Rectangle | nornir_imageregistration.type_info.RectLike,
                            excess_scalar: float,
                            mask_extrema: bool = False,
                            cval: float | None | str | int = None,
                            dtype: np.typing.DTypeLike | None = None):
    """
    Crop the tile's image so it contains the specified rectangle
    :param bool mask_extrema: if true, mask large regions of continuous extrema regions and replace with noise
    """

    if cval is None:
        cval = 'random'

    if excess_scalar > 3:
        excess_scalar = 3.0

    Width = overlapping_rect.Width * excess_scalar  # type: ignore[union-attr]
    Height = overlapping_rect.Height * excess_scalar  # type: ignore[union-attr]
    scaled_rect = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(overlapping_rect.Center,  # type: ignore[union-attr]
                                                                                  (Height, Width))

    # scaled_rect = nornir_imageregistration.Rectangle.scale_on_center(overlapping_rect, excess_scalar)
    scaled_rect = nornir_imageregistration.Rectangle.SafeRound(scaled_rect)

    if dtype is None:
        image = nornir_imageregistration.ImageParamToImageArray(imageparam)
        dtype = image.dtype
    else:
        image = nornir_imageregistration.ImageParamToImageArray(imageparam, dtype=dtype)

    # Checked here, not on entry, and only for numeric cval.
    #
    # On entry this read `not np.issubdtype(dtype, np.floating) and np.isnan(cval)`, which had
    # two problems. np.isnan raises TypeError on the 'random' default, so any non-floating
    # dtype crashed before doing any work rather than filling with noise. And it ran before
    # the block above, so with the documented dtype=None it tested np.dtype(None) -- float64 --
    # and passed, even when the image it went on to load was uint16. It was simultaneously too
    # eager to reject and unable to catch the case it existed for (#127).
    if not isinstance(cval, str) and np.isnan(cval) \
            and not np.issubdtype(dtype, np.floating):
        raise ValueError(
            f"Cannot set cval to np.nan for non floating point dtype {np.dtype(dtype)}")

    # This is inefficient because we mask the entire image even though we only need a specific fragment.
    # However it is a pain right now to figure out how much more of the image is going to be grabbed by an expanded bounding
    # box, load that in, mask it, and then expand the boundaries a second time for the remaining area.
    # masked_fraction = 0
    if mask_extrema:
        extremaMask = nornir_imageregistration.CreateExtremaMask(image, size_cutoff=0.01)

    cropped = nornir_imageregistration.CropImage(image,
                                                 Xo=int(scaled_rect.BottomLeft[1]),
                                                 Yo=int(scaled_rect.BottomLeft[0]),
                                                 Width=int(scaled_rect.Width),
                                                 Height=int(scaled_rect.Height),
                                                 cval=cval)

    if mask_extrema:
        extremaMask_cropped = nornir_imageregistration.CropImage(extremaMask,
                                                                 Xo=int(scaled_rect.BottomLeft[1]),
                                                                 Yo=int(scaled_rect.BottomLeft[0]),
                                                                 Width=int(scaled_rect.Width),
                                                                 Height=int(scaled_rect.Height),
                                                                 cval=False)

        cropped = nornir_imageregistration.RandomNoiseMask(cropped, extremaMask_cropped, Copy=False)  # type: ignore[arg-type]

        return cropped, extremaMask_cropped
    else:
        return cropped

    # return nornir_imageregistration.pad_image_for_phase_correlation(cropped, MinOverlap=1.0, PowerOfTwo=True)


def __tile_offset_remote(A_Filename: str, B_Filename: str,
                         scaled_overlapping_source_rect_A: nornir_imageregistration.type_info.RectLike,
                         scaled_overlapping_source_rect_B,
                         OffsetAdjustment, excess_scalar,
                         mask_extrema: bool = True,
                         correlation_coefficient: float | None = None):
    """
    :param A_Filename: Path to tile A
    :param B_Filename: Path to tile B
    :param scaled_overlapping_source_rect_A: Region of overlap on tile A with tile B
    :param scaled_overlapping_source_rect_B: Region of overlap on tile B with tile A
    :param OffsetAdjustment: scaled_offset to account for the (center) position of tile B relative to tile A.  If the overlapping rectangles are perfectly aligned the reported offset would be (0,0).  OffsetAdjustment would be added to that (0,0) result to ensure Tile B remained in the same position.
    :param float excess_scalar: How much additional area should we pad the overlapping rectangles with.
    :param mask_extrema: If true, mask large regions of continuous extrema regions and replace with noise
    Return the offset required to align to image files.
    This function exists to minimize the inter-process communication
    """

    MinOverlap = 0.25
    MaxOverlap = 1
    excess_scalar = 2  # If excess_scalar > 1 and the image has bad regions we end up padding with nearly pure black or white.
    # Instead I set excess_scalar to 1 and pad the image based on the min overlap
    dtype = nornir_imageregistration.default_image_dtype()

    ShowImages = False
    A = nornir_imageregistration.LoadImage(A_Filename, dtype=dtype)
    B = nornir_imageregistration.LoadImage(B_Filename, dtype=dtype)

    # I had to add the .astype call above for DM4 support, but I recall it broke PMG input.  Leave this comment here until the tests are passing
    #    A = nornir_imageregistration.LoadImage(A_Filename) #.astype(dtype=np.float16)
    #    B = nornir_imageregistration.LoadImage(B_Filename) #.astype(dtype=np.float16)

    # I tried a 1.0 overlap.  It works better for light microscopy where the reported stage position is more precise
    # For TEM the stage position can be less reliable and the 1.5 scalar produces better results
    # For the latest version of the code that uses only the overlapping region 3 is appropriate because it allows the alignment point to be anywhere on the image without ambiguity
    (OverlappingRegionA_original, OverlappingRegionA_extremaMask) = __get_overlapping_image(A,  # type: ignore[assignment]
                                                                                            scaled_overlapping_source_rect_A,
                                                                                            excess_scalar=excess_scalar,
                                                                                            mask_extrema=mask_extrema,
                                                                                            cval='random', dtype=dtype)
    (OverlappingRegionB_original, OverlappingRegionB_extremaMask) = __get_overlapping_image(B,  # type: ignore[assignment]
                                                                                            scaled_overlapping_source_rect_B,
                                                                                            excess_scalar=excess_scalar,
                                                                                            mask_extrema=mask_extrema,
                                                                                            cval='random', dtype=dtype)

    valid_mask_fraction_A = OverlappingRegionA_extremaMask.sum() / (  # type: ignore[union-attr]
            OverlappingRegionA_extremaMask.shape[0] * OverlappingRegionA_extremaMask.shape[1])  # type: ignore[union-attr]
    valid_mask_fraction_B = OverlappingRegionB_extremaMask.sum() / (  # type: ignore[union-attr]
            OverlappingRegionB_extremaMask.shape[0] * OverlappingRegionB_extremaMask.shape[1])  # type: ignore[union-attr]
    valid_mask_fraction_scalar = min(valid_mask_fraction_A, valid_mask_fraction_B)

    OverlappingRegionA = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
        OverlappingRegionA_original,
        min_overlap=MinOverlap,
        original_shape=scaled_overlapping_source_rect_A.Dimensions)  # type: ignore[union-attr]
    OverlappingRegionB = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
        OverlappingRegionB_original,
        min_overlap=MinOverlap,
        original_shape=scaled_overlapping_source_rect_B.Dimensions)  # type: ignore[union-attr]

    if ShowImages:
        o_a = __get_overlapping_image(A, scaled_overlapping_source_rect_A, excess_scalar=1.0, cval=0, dtype=dtype)
        o_b = __get_overlapping_image(B, scaled_overlapping_source_rect_B, excess_scalar=1.0, cval=0, dtype=dtype)

    # nornir_imageregistration.ShowGrayscale([[OverlappingRegionA, OverlappingRegionB],[o_a,o_b]])
    # OverlappingRegionA = OverlappingRegionA.astype(np.float32)
    # OverlappingRegionB = OverlappingRegionB.astype(np.float32)

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

    try:
        np.seterr(
            under='ignore')  # It is common to encounter underflow for Float16, so temporarily turn off the warning here
        OverlappingRegionA -= OverlappingRegionA.min()
        OverlappingRegionA /= OverlappingRegionA.max()

        OverlappingRegionB -= OverlappingRegionB.min()
        OverlappingRegionB /= OverlappingRegionB.max()
    finally:
        np.seterr(under='warn')

    # nornir_imageregistration.ShowGrayscale([OverlappingRegionA, OverlappingRegionB]) nornir_imageregistration.ShowGrayscale([[o_a, o_b],[OverlappingRegionA, OverlappingRegionB]])

    record = nornir_imageregistration.phasecorrelation.find_offset(OverlappingRegionA,
                                                                   OverlappingRegionB,
                                                                   min_overlap=MinOverlap,
                                                                   max_overlap=MaxOverlap,
                                                                   target_shape=scaled_overlapping_source_rect_A.Dimensions,  # type: ignore[union-attr]
                                                                   source_shape=scaled_overlapping_source_rect_B.Dimensions,  # type: ignore[union-attr]
                                                                   fft_required=True,
                                                                   correlation_coefficient=correlation_coefficient)  # , FixedImageShape=scaled_overlapping_source_rect_A.shape, MovingImageShape=scaled_overlapping_source_rect_B.shape)

    # overlapping_rect_B_AdjustedToPeak = nornir_imageregistration.Rectangle.translate(scaled_overlapping_source_rect_B, -record.peak)
    # overlapping_rect_B_AdjustedToPeak = nornir_imageregistration.Rectangle.change_area(overlapping_rect_B_AdjustedToPeak, scaled_overlapping_source_rect_A.Size)
    # median_diff = __AlignmentScoreRemote(A, B, scaled_overlapping_source_rect_A, overlapping_rect_B_AdjustedToPeak)
    # nornir_imageregistration.ShowGrayscale([[OverlappingRegionA, OverlappingRegionB], [overlapping_rect_B_AdjustedToPeak, median_diff]])
    # diff_weight = 1.0 - median_diff
    # np.seterr(**old_float_err_settings)

    # nornir_imageregistration.views.plot_aligned_images(record, o_a, o_b)

    adjusted_record = nornir_imageregistration.AlignmentRecord(np.array(record.peak) + OffsetAdjustment,
                                                               record.weight * valid_mask_fraction_scalar)

    if ShowImages:
        nornir_imageregistration.views.plot_aligned_images(record, o_a, o_b)  # type: ignore[arg-type]
        del o_a
        del o_b

    del A
    del B
    del OverlappingRegionA
    del OverlappingRegionB
    del OverlappingRegionA_extremaMask
    del OverlappingRegionB_extremaMask

    return adjusted_record


def ScoreMosaicQuality(mosaicTileset):
    """
    Walk each overlapping region between tiles.  Subtract the
    """

    list_tiles = list(mosaicTileset.values())
    total_score = 0
    total_pixels = 0

    if len(mosaicTileset) <= 2:
        # This is a special case for tests to simplify debugging when there is only one pair of overlapping images
        for tile_overlap in nornir_imageregistration.tile_overlap.IterateTileOverlaps(list_tiles,
                                                                                      image_to_source_space_scale=mosaicTileset.image_to_source_space_scale):
            score = __AlignmentScoreRemote(tile_overlap.A.ImagePath,
                                           tile_overlap.B.ImagePath,
                                           tile_overlap.scaled_overlapping_source_rect_A,
                                           tile_overlap.scaled_overlapping_source_rect_B)

            return score

    else:

        pool = nornir_pools.GetGlobalMultithreadingPool()
        # pool = nornir_pools.GetGlobalSerialPool()
        tasks = list()

        for tile_overlap in nornir_imageregistration.tile_overlap.IterateTileOverlaps(list_tiles,
                                                                                      image_to_source_space_scale=mosaicTileset.image_to_source_space_scale):
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


def __AlignmentScoreRemote(A_Filename, B_Filename, scaled_overlapping_source_rect_A, scaled_overlapping_source_rect_B,
                           mask_extrema=True):
    """Returns the difference between the images.

    Accepts NumPy or CuPy crops from ``CropImage``; ops follow ``cp.get_array_module``.
    Host-only: file I/O via ``ImageParamToImageArray`` of filenames.
    """

    dtype = nornir_imageregistration.default_image_dtype()
    try:
        OverlappingRegionA, extrema_mask_OverlappingRegionA = __get_overlapping_image(  # type: ignore[assignment]
            nornir_imageregistration.ImageParamToImageArray(A_Filename,
                                                            dtype=nornir_imageregistration.default_image_dtype()),
            scaled_overlapping_source_rect_A,
            excess_scalar=1.0,
            mask_extrema=mask_extrema,
            dtype=dtype)  # type: ignore[arg-type]
        OverlappingRegionB, extrema_mask_OverlappingRegionB = __get_overlapping_image(  # type: ignore[assignment]
            nornir_imageregistration.ImageParamToImageArray(B_Filename,
                                                            dtype=nornir_imageregistration.default_image_dtype()),
            scaled_overlapping_source_rect_B,
            excess_scalar=1.0,
            mask_extrema=mask_extrema,
            dtype=dtype)  # type: ignore[arg-type]

        xp = cp.get_array_module(OverlappingRegionA)

        # If the entire region is a solid color, then return the maximum score possible
        if (OverlappingRegionA.min() == OverlappingRegionA.max()) or \
                (OverlappingRegionA.max() == 0) or \
                (OverlappingRegionB.min() == OverlappingRegionB.max()) or \
                (OverlappingRegionB.max() == 0):
            return np.finfo(dtype).max  # type: ignore[call-overload]

        OverlappingRegionA -= OverlappingRegionA.min()
        OverlappingRegionA /= OverlappingRegionA.max()

        OverlappingRegionB -= OverlappingRegionB.min()
        OverlappingRegionB /= OverlappingRegionB.max()

        extremaMask = xp.logical_and(extrema_mask_OverlappingRegionA, extrema_mask_OverlappingRegionB)  # type: ignore[arg-type]

        # ignore_indices = OverlappingRegionA == OverlappingRegionA.max()
        # ignore_indices |= OverlappingRegionA == OverlappingRegionA.min()
        # ignore_indices |= OverlappingRegionB == OverlappingRegionB.max()
        # ignore_indices |= OverlappingRegionB == OverlappingRegionB.min()

        # There was data in the aligned images, but not overlapping.  So we return the maximum value
        any_valid = xp.any(extremaMask)
        if hasattr(any_valid, 'get'):
            any_valid = any_valid.get()
        if not bool(any_valid):
            if np.issubdtype(OverlappingRegionA.dtype, np.integer):
                return np.iinfo(OverlappingRegionA.dtype).max
            else:
                return np.finfo(OverlappingRegionA.dtype).max

        valid_indices = extremaMask

        OverlappingRegionA -= OverlappingRegionB
        absoluteDiff = xp.abs(OverlappingRegionA)

        # Multiple diff by the largest masked area to compensate for the large blank area
        valid_mask_fraction_A = float(nornir_imageregistration.EnsureNumpyArray(
            extrema_mask_OverlappingRegionA.sum())) / (  # type: ignore[union-attr]
                extrema_mask_OverlappingRegionA.shape[0] * extrema_mask_OverlappingRegionA.shape[1])  # type: ignore[union-attr]
        valid_mask_fraction_B = float(nornir_imageregistration.EnsureNumpyArray(
            extrema_mask_OverlappingRegionB.sum())) / (  # type: ignore[union-attr]
                extrema_mask_OverlappingRegionB.shape[0] * extrema_mask_OverlappingRegionB.shape[1])  # type: ignore[union-attr]

        valid_mask_fraction = min(valid_mask_fraction_A, valid_mask_fraction_B)

        absoluteDiff *= valid_mask_fraction

        return float(nornir_imageregistration.EnsureNumpyArray(
            xp.mean(absoluteDiff[valid_indices])))
    except FloatingPointError as e:
        print("FloatingPointError: {0} for images\n\t{1}\n\t{2}".format(str(e), A_Filename, B_Filename))
        raise e
    finally:
        del OverlappingRegionA
        del OverlappingRegionB

        # return (np.sum(absoluteDiff[valid_indices].flat), np.sum(valid_indices))


def TranslateFiles(fileDict):
    """Translate Images expects a dictionary of images, their position and size in pixel space.  It moves the images to what it believes their optimal position is for alignment
       and returns a dictionary of the same form.
       Input: dict[ImageFileName] = [x y width height]
       Output: dict[ImageFileName] = [x y width height]"""

    # We do not want to load each image multiple time, and we do not know how many images we will get so we should not load them all at once.
    # Therefore our first action is building a matrix of each image and their overlapping counterparts
    raise NotImplementedError()


if __name__ == '__main__':
    pass
