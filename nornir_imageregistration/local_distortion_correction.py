"""
Created on Apr 7, 2015

@author: u0490822

This module performs local distortions of images to refine alignments of mosaics and sections
"""
import os
from typing import Iterable, Sequence
import numpy as np
import scipy.spatial
from numpy.typing import NDArray
import nornir_pools
import nornir_imageregistration
# import nornir_imageregistration.views.grid_data

from nornir_shared import prettyoutput

from nornir_imageregistration.transforms.triangulation import Triangulation 

# Summary type used for typing
AlignmentRecordDict = dict[tuple[int, int], nornir_imageregistration.EnhancedAlignmentRecord]
AlignmentRecordList = Sequence[nornir_imageregistration.EnhancedAlignmentRecord]
AlignmentRecordKey = tuple[int, int]


class DistortionCorrection:

    def __init__(self):
        self.PointsForTile = {}


def RefineMosaic(transforms, imagepaths, imageScale=None, subregion_shape=None):
    """
    Locate overlapping regions between tiles in a mosaic and align multiple small subregions within.  This generates a set of control points.

    More than one tile may overlap, for example corners.  To solve this the set of control points is merged into a KD tree.  points closer than a set distance (less than subregion size) are averaged to create a single offset.

    Using the remaining points a mesh transform is generated for the tile.
    """

    if subregion_shape is None:
        subregion_shape = np.array([128, 128])

    tiles = nornir_imageregistration.mosaic_tileset.Create(transforms, imagepaths,
                                                           image_to_source_space_scale=imageScale)
    list_tiles = list(tiles.values())
    pool = nornir_pools.GetGlobalMultithreadingPool()
    tasks = list()

    if imageScale is None:
        imageScale = 1.0 / tiles.image_to_source_space_scale

    layout = nornir_imageregistration.layout.Layout()
    for t in list_tiles:
        layout.CreateNode(t.ID, t.FixedBoundingBox.Center)

    for tile_overlap in nornir_imageregistration.tile_overlap.IterateTileOverlaps(list_tiles, min_overlap=0.03):
        # OK... add some small neighborhoods and register those...
        # (downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment) = nornir_imageregistration.tile.Tile.Calculate_Overlapping_Regions(A, B, imageScale)
        #
        task = pool.add_task("Align %d -> %d" % (tile_overlap.A.ID, tile_overlap.B.ID),
                             __RefineTileAlignmentRemote,
                             tile_overlap.A,
                             tile_overlap.B,
                             tile_overlap.scaled_overlapping_source_rect_A,
                             tile_overlap.scaled_overlapping_source_rect_B,
                             tile_overlap.scaled_offset,
                             imageScale,
                             subregion_shape)
        task.A = tile_overlap.A
        task.B = tile_overlap.B
        task.OffsetAdjustment = tile_overlap.scaled_offset
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
            (point_pairs, _) = t.wait_return()
        except Exception as e:
            prettyoutput.Log(f"Could not register {t.A.ID} -> {t.B.ID}")
            prettyoutput.Log(str(e))
            continue

        SplitDisplacements(t.A, t.B, point_pairs)
        # offset = net_offset[0:2] + (t.OffsetAdjustment * downsample)
        # weight = net_offset[2]
        # layout.SetOffset(t.A.ID, t.B.ID, offset, weight) 

        # Figure out what offset we found vs. what offset we expected
        # PredictedOffset = t.B.FixedBoundingBox.Center - t.A.ControlBoundingBox.Center

        # diff = offset - PredictedOffset
        # distance = np.sqrt(np.sum(diff ** 2))

        # print("%d -> %d = %g" % (t.A.ID, t.B.ID, distance))

    pool.wait_completion()

    return layout, tiles


def __RefineTileAlignmentRemote(A: nornir_imageregistration.Tile, B: nornir_imageregistration.Tile,
                                scaled_overlapping_source_rect_A, scaled_overlapping_source_rect_B, OffsetAdjustment,
                                imageScale, subregion_shape=None):
    if subregion_shape is None:
        subregion_shape = np.array([128, 128])

    downsample = 1.0 / imageScale

    grid_dim = nornir_imageregistration.TileGridShape(scaled_overlapping_source_rect_A.Size, subregion_shape)

    # scaled_overlapping_source_rect_A = nornir_imageregistration.Rectangle.change_area(scaled_overlapping_source_rect_A, grid_dim * subregion_shape)
    # scaled_overlapping_source_rect_B = nornir_imageregistration.Rectangle.change_area(scaled_overlapping_source_rect_B, grid_dim * subregion_shape)

    overlapping_rect = nornir_imageregistration.Rectangle.overlap_rect(A.FixedBoundingBox, B.FixedBoundingBox)
    overlapping_rect = nornir_imageregistration.Rectangle.change_area(overlapping_rect,
                                                                      grid_dim * subregion_shape * downsample)

    ATransformedImageData = nornir_imageregistration.assemble_tiles.TransformTile(tile=A, distanceImage=None,
                                                                                  target_space_scale=imageScale,
                                                                                  TargetRegion=overlapping_rect,
                                                                                  SingleThreadedInvoke=True)
    BTransformedImageData = nornir_imageregistration.assemble_tiles.TransformTile(tile=B, distanceImage=None,
                                                                                  target_space_scale=imageScale,
                                                                                  TargetRegion=overlapping_rect,
                                                                                  SingleThreadedInvoke=True)

    # I tried a 1.0 overlap.  It works better for light microscopy where the reported stage position is more precise
    # For TEM the stage position can be less reliable and the 1.5 scalar produces better results
    # OverlappingRegionA = __get_overlapping_image(A, scaled_overlapping_source_rect_A,excess_scalar=1.0)
    # OverlappingRegionB = __get_overlapping_image(B, scaled_overlapping_source_rect_B,excess_scalar=1.0)
    A_image = nornir_imageregistration.RandomNoiseMask(ATransformedImageData.image,
                                                       ATransformedImageData.centerDistanceImage < np.finfo(
                                                           ATransformedImageData.centerDistanceImage.dtype).max,
                                                       Copy=True)
    B_image = nornir_imageregistration.RandomNoiseMask(BTransformedImageData.image,
                                                       BTransformedImageData.centerDistanceImage < np.finfo(
                                                           BTransformedImageData.centerDistanceImage.dtype).max,
                                                       Copy=True)

    # OK, create tiles from the overlapping regions
    # A_image = nornir_imageregistration.ReplaceImageExtremaWithNoise(ATransformedImageData.image)
    # B_image = nornir_imageregistration.ReplaceImageExtremaWithNoise(BTransformedImageData.image)
    # nornir_imageregistration.ShowGrayscale([A_image,B_image])
    A_tiles = nornir_imageregistration.ImageToTiles(A_image, subregion_shape, cval='random')
    B_tiles = nornir_imageregistration.ImageToTiles(B_image, subregion_shape, cval='random')

    # grid_dim = nornir_imageregistration.TileGridShape(ATransformedImageData.image.shape, subregion_shape)

    refine_dtype = np.dtype([('SourceY', 'f4'),
                             ('SourceX', 'f4'),
                             ('TargetY', 'f4'),
                             ('TargetX', 'f4'),
                             ('Weight', 'f4'),
                             ('Angle', 'f4')])

    point_pairs = np.empty(grid_dim, dtype=refine_dtype)

    net_displacement = np.empty((grid_dim.prod(), 3),
                                dtype=np.float32)  # Amount the refine points are moved from the defaultf

    cell_center_offset = subregion_shape / 2.0

    for iRow in range(0, grid_dim[0]):
        for iCol in range(0, grid_dim[1]):
            subregion_offset = (np.array(
                [iRow, iCol]) * subregion_shape) + cell_center_offset  # Position subregion coordinate space
            source_tile_offset = subregion_offset + scaled_overlapping_source_rect_A.BottomLeft  # Position tile image coordinate space
            global_offset = OffsetAdjustment + subregion_offset  # Position subregion in tile's mosaic space

            if not (iRow, iCol) in A_tiles or not (iRow, iCol) in B_tiles:
                net_displacement[(iRow * grid_dim[1]) + iCol,:] = np.array([0, 0, 0])
                point_pairs[iRow, iCol] = np.array(
                    (source_tile_offset[0], source_tile_offset[1], 0 + global_offset[0], 0 + global_offset[1], 0, 0),
                    dtype=refine_dtype)
                continue

            try:
                record = nornir_imageregistration.FindOffset(A_tiles[iRow, iCol], B_tiles[iRow, iCol],
                                                             FFT_Required=True)
            except Exception as e:
                prettyoutput.LogErr(f'Exception on row: {iRow} col: {iCol} when finding offset:\n{e}')
                net_displacement[(iRow * grid_dim[1]) + iCol,:] = np.array([0, 0, 0])
                point_pairs[iRow, iCol] = np.array(
                    (source_tile_offset[0], source_tile_offset[1], 0 + global_offset[0], 0 + global_offset[1], 0, 0),
                    dtype=refine_dtype)
                continue

            adjusted_record = nornir_imageregistration.AlignmentRecord(np.array(record.peak) * downsample,
                                                                       record.weight)

            # print(str(record))

            if np.any(np.isnan(record.peak)):
                net_displacement[(iRow * grid_dim[1]) + iCol,:] = np.array([0, 0, 0])
                point_pairs[iRow, iCol] = np.array((source_tile_offset[0], source_tile_offset[1], 0 + global_offset[0],
                                                    0 + global_offset[1], 0, record.angle), dtype=refine_dtype)
            else:
                net_displacement[(iRow * grid_dim[1]) + iCol,:] = np.array(
                    [adjusted_record.peak[0], adjusted_record.peak[1], record.weight])
                point_pairs[iRow, iCol] = np.array((source_tile_offset[0], source_tile_offset[1],
                                                    adjusted_record.peak[0] + global_offset[0],
                                                    adjusted_record.peak[1] + global_offset[1], record.weight,
                                                    record.angle), dtype=refine_dtype)

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

    return point_pairs, net_offset


def SplitDisplacements(A, B, point_pairs):
    """
    :param B:
    :param tile A: Tile A, point_pairs are registered relative to this tile
    :param tile A: Tile B, the tile that point pairs register A onto
    :param ndarray point_pairs: A set of point pairs from a grid refinement
    """

    raise NotImplementedError()


def RefineStosFile(InputStos: str | nornir_imageregistration.StosFile,
                   OutputStosPath: str,
                   num_iterations: int | None=None,
                   cell_size: NDArray[int] | tuple[int, int]=None,
                   grid_spacing: NDArray[int] | tuple[int, int]=None,
                   angles_to_search=None,
                   final_pass_angles=None,
                   max_travel_for_finalization=None,
                   max_travel_for_finalization_improvement=None,
                   min_alignment_overlap=None,
                   min_unmasked_area: float=None,
                   SaveImages: bool=False,
                   SavePlots: bool=False,
                   **kwargs):
    """
    Refines an inputStos file and produces the OutputStos file.

    Places a regular grid of control points across the target image.  These corresponding points on the
    source image are then adjusted to create a mapping from Source To Fixed Space for the source image.
    :param final_pass_angles:
    :param max_travel_for_finalization:
    :param min_alignment_overlap:
    :param min_unmasked_area:
    :param StosFile InputStos: Either a file path or StosFile object.  This is the stosfile to be refined.
    :param OutputStosPath: Path to save the refined stos file at.
cell_size: (width, height) area of image around control points to use for registration
    :param grid_spacing: (width, height) of separation between control points on the grid
    :param array angles_to_search: An array of floats or None.  Images are rotated by the degrees indicated in the array.  The single best alignment across all angles is selected.
r plots of each iteration in the output path for debugging purposes
    """
    
    
    for k, v in kwargs.items(): 
        prettyoutput.Log(f"\tUnused parameter to RefineStosFile: {k}:{v}\n")

    outputDir = os.path.dirname(OutputStosPath)

    # Load the input stos file if it is not already loaded
    if not isinstance(InputStos, nornir_imageregistration.StosFile):
        stosDir = os.path.dirname(InputStos)
        InputStos = nornir_imageregistration.files.StosFile.Load(InputStos)
        InputStos.TryConvertRelativePathsToAbsolutePaths(stosDir)

    stosTransform = nornir_imageregistration.transforms.factory.LoadTransform(InputStos.Transform, 1)
    if stosTransform is None:
        raise ValueError(f"Could not load transform: {InputStos} - {InputStos.Transform}")

    target_image_data = nornir_imageregistration.ImagePermutationHelper(img=InputStos.ControlImageFullPath,
                                                                        mask=InputStos.ControlMaskFullPath,
                                                                        extrema_mask_size_cuttoff=None,
                                                                        dtype=nornir_imageregistration.default_image_dtype())

    source_image_data = nornir_imageregistration.ImagePermutationHelper(img=InputStos.MappedImageFullPath,
                                                                        mask=InputStos.MappedMaskFullPath,
                                                                        extrema_mask_size_cuttoff=None,
                                                                        dtype=nornir_imageregistration.default_image_dtype())

    with nornir_imageregistration.settings.GridRefinement.CreateWithPreprocessedImages(
                                                                target_img_data=target_image_data,
                                                                source_img_data=source_image_data,
                                                                num_iterations=num_iterations, cell_size=cell_size,
                                                                grid_spacing=grid_spacing,
                                                                angles_to_search=angles_to_search,
                                                                final_pass_angles=final_pass_angles,
                                                                max_travel_for_finalization=max_travel_for_finalization,
                                                                max_travel_for_finalization_improvement=max_travel_for_finalization_improvement,
                                                                min_alignment_overlap=min_alignment_overlap,
                                                                min_unmasked_area=min_unmasked_area,
                                                                single_thread_processing=False) as settings:

        output_transform = RefineTransform(stosTransform,
                                           settings,
                                           SaveImages=SaveImages,
                                           SavePlots=SavePlots,
                                           outputDir=outputDir)

        InputStos.Transform = nornir_imageregistration.transforms.ConvertTransformToGridTransform(output_transform,
                                                              source_image_shape=settings.source_image.shape,
                                                              cell_size=settings.cell_size,
                                                              grid_spacing=settings.grid_spacing)
        InputStos.Save(OutputStosPath)


def RefineTransform(stosTransform: nornir_imageregistration.ITransform,
                    settings: nornir_imageregistration.settings.GridRefinement,
                    SaveImages: bool=False,
                    SavePlots: bool=False,
                    outputDir: str=None) -> nornir_imageregistration.ITransform:
    """
    Refines a transform and returns a grid transform produced by the refinement algorithm.  This algorithm
    takes an initial transform and creates a regular grid of points.  points covered more than
    
    Places a regular grid of control points across the target image.  These corresponding points on the
    source image are then adjusted to create a mapping from Source To Fixed Space for the source image. 
    :param settings:
    :param stosTransform: The transform to refine
    """

    if (SavePlots or SaveImages) and outputDir is None:
        raise ValueError("outputDir must be specified if SavePlots or SaveImages is true.")

    # Convert inputs to numpy arrays

    final_pass = False  # True if this is the last iteration the loop will perform

    finalized_points = {}  # type: AlignmentRecordDict

    CutoffPercentilePerIteration = 10.0

    FirstPassWeightScoreCutoff = None
    FirstPassCompositeScoreCutoff = None
    FirstPassFinalizeValue = None  # The score required to finalize a control point on the first pass.
    # The first score is recorded to prevent the best scores from being finalized and then later
    # groups of poor scores looking falsely good because the correct registrations are all finalized
    first_pass_weight_distance_composite_scores = None

    transform_inclusion_percentile = 33.3  # - (CutoffPercentilePerIteration * i)
    transform_inclusion_range = 20.0
    finalize_percentile = 66.6
    finalize_range = 46.6
    updatedTransform = None  # type: nornir_imageregistration.ITransform | None

    i = 1

    while i <= settings.num_iterations:
        alignment_points = _RefineGridPointsForTwoImages(stosTransform,
                                                         settings=settings,
                                                         finalized=finalized_points)

        prettyoutput.Log(f"Pass {i} aligned {len(alignment_points)} points")

        updated_and_finalized_alignment_points = alignment_points + list(finalized_points.values())
        updated_and_finalized_weights_distance = _alignment_records_to_composite_scores(
            updated_and_finalized_alignment_points)

        # What fraction of the maximum number of iterations have been completed?
        adjustment_scalar = (i - 1) / settings.num_iterations

        transform_inclusion_percentile_this_pass = transform_inclusion_percentile
        if adjustment_scalar != 0:
            transform_inclusion_percentile_this_pass -= (transform_inclusion_range * adjustment_scalar)

        transform_inclusion_percentile_this_pass = float(np.clip(transform_inclusion_percentile_this_pass, 10.0, 100.0)) #This is a float, so don't bother with out parameter

        transform_cutoff_this_pass = np.percentile(updated_and_finalized_weights_distance[:, 2],  # Do not include finalize points because they have a distance of zero which throws off the composite scores
                                                   100.0 - transform_inclusion_percentile_this_pass)

        finalize_percentile_this_pass = finalize_percentile
        if adjustment_scalar != 0:
            finalize_percentile_this_pass -= (finalize_range * adjustment_scalar)

        finalize_percentile_this_pass = float(np.clip(finalize_percentile_this_pass, 10.0, 100.0)) #This is a float, so don't bother with out parameter

        finalize_cutoff_this_pass = np.percentile(updated_and_finalized_weights_distance[:, 0],
                                                  finalize_percentile_this_pass)

        (updatedTransform, included_alignment_records, weight_distance_composite_scores) = _PeakListToTransform(
            alignment_points,
            AlignRecordsToControlPoints(finalized_points.values()),
            percentile=transform_inclusion_percentile_this_pass,
            cutoff=transform_cutoff_this_pass)

        # if FirstPassCompositeScoreCutoff is None:
        #    FirstPassCompositeScoreCutoff = np.percentile(weight_distance_composite_scores[:, 2], 100.0 - percentile)
        #    FirstPassWeightScoreCutoff = np.percentile(weight_distance_composite_scores[:, 0], percentile)

        # if FirstPassFinalizeValue is not None:
        # cutoff_range = np.abs(FirstPassFinalizeValue - FirstPassWeightScoreCutoff)
        # fraction = i / (settings.num_iterations - 1)
        # FirstPassFinalizeValue - (cutoff_range * fraction)

        prettyoutput.Log(f'Finalize cutoff this pass: {finalize_cutoff_this_pass}')

        new_finalized_points = CalculateFinalizedAlignmentPointsMask(alignment_points,
                                                                     percentile=finalize_percentile_this_pass,
                                                                     max_travel_distance=settings.max_travel_for_finalization,
                                                                     weight_cutoff=finalize_cutoff_this_pass)

        if FirstPassFinalizeValue is None:
            FirstPassFinalizeValue = np.percentile(weight_distance_composite_scores[:, 0],
                                                   finalize_percentile_this_pass)

        if first_pass_weight_distance_composite_scores is None:
            first_pass_weight_distance_composite_scores = weight_distance_composite_scores

        new_finalized_alignments_list = list(
            filter(lambda index_item: new_finalized_points[index_item[0]], enumerate(alignment_points)))
        new_finalized_alignments_dict = {fp[1].ID: fp[1] for fp in new_finalized_alignments_list}
        new_finalization_count = len(new_finalized_alignments_dict)
        
        # remove finalized points from alignment_points
        non_final_alignment_points = list(filter(lambda r: r.ID not in new_finalized_alignments_dict, alignment_points))
        
        prettyoutput.Log(
            f"Pass {i} has locked {new_finalization_count} new points, {len(finalized_points)} of {len(updated_and_finalized_alignment_points)} are locked")

        # Check previous finalizations to see if we can do better now
        (finalized_points, improved_alignments) = TryToImproveAlignments(updatedTransform,
                                                                         finalized_points,
                                                                         settings)

        finalized_points = {**finalized_points, **new_finalized_alignments_dict}

        # (improved_finalized_dict, improved_alignments) = TryToImproveAlignments(updatedTransform,
        #                                                                         finalized_points,
        #                                                                         settings)

        # new_finalization_count = len(improved_finalized_dict)
        # finalized_points = {**finalized_points, **improved_finalized_dict}

        prettyoutput.Log(
            f"  Improved {len(improved_alignments)} finalized points using latest transform")

        if SavePlots:
            # plot_percentile_estimates(weight_distance_composite_scores[:,0])

            histogram_filename = os.path.join(outputDir, f'weight_histogram_pass{i}.png')
            nornir_imageregistration.views.PlotWeightHistogram(alignment_points, filename=histogram_filename,
                                                               transform_cutoff=transform_inclusion_percentile_this_pass / 100.0,
                                                               finalize_cutoff=finalize_percentile_this_pass / 100.0,
                                                               line_pos_list=[finalize_cutoff_this_pass])

            vector_field_filename = os.path.join(outputDir, f'Vector_field_pass{i}.png')
            nornir_imageregistration.views.PlotPeakList(non_final_alignment_points, list(finalized_points.values()),
                                                        vector_field_filename,
                                                        ylim=(0, settings.target_image.shape[1]),
                                                        xlim=(0, settings.target_image.shape[0]))
            # vector_field_filename = os.path.join(outputDir, f'Vector_field_pass_delta{i}.png')
            # nornir_imageregistration.views.PlotPeakList(alignment_points, list(finalized_points.values()),
            #                                             vector_field_filename,
            #                                             ylim=(0, settings.target_image.shape[1]),
            #                                             xlim=(0, settings.target_image.shape[0]),
            #                                             attrib='PSDDelta')

        # Update the transform with the adjusted points
        combined_records_this_pass = {a.ID: a for a in included_alignment_records}
        if len(improved_alignments) > 0:
            for item in finalized_points.items():
                combined_records_this_pass[item[0]] = item[1]

            print(f'Building transform for next round with {len(included_alignment_records)} points and {len(finalized_points)} finalized points')
            updatedTransform = nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(
                AlignRecordsToControlPoints(combined_records_this_pass.values()))

        if SaveImages:
            # InputStos.Save(os.path.join(outputDir, "UpdatedTransform_pass{0}.stos".format(i)))

            warpedToFixedImage = nornir_imageregistration.assemble.TransformStos(updatedTransform,
                                                                                 fixedImage=settings.target_image,
                                                                                 warpedImage=settings.source_image)

            Delta = warpedToFixedImage - settings.target_image
            ComparisonImage = np.abs(Delta)
            if ComparisonImage.max() != 0:
                ComparisonImage = ComparisonImage / ComparisonImage.max()

            # nornir_imageregistration.SaveImage(os.path.join(outputDir, f'delta_pass{i}.png'), ComparisonImage, bpp=8)
            # nornir_imageregistration.SaveImage(os.path.join(outputDir, f'image_pass{i}.png'), warpedToFixedImage, bpp=8)
            pool = nornir_pools.GetGlobalThreadPool()
            pool.add_task(f'delta_pass{i}.png', nornir_imageregistration.SaveImage,
                          os.path.join(outputDir, f'delta_pass{i}.png'), np.copy(ComparisonImage), bpp=8)
            pool.add_task(f'image_pass{i}.png', nornir_imageregistration.SaveImage,
                          os.path.join(outputDir, f'image_pass{i}.png'), np.copy(warpedToFixedImage), bpp=8)

        i = i + 1

        if final_pass:
            break

        if i == settings.num_iterations:
            final_pass = True

            # If we've locked 10% of the points and have not locked any new ones we are done
        if len(finalized_points) > len(updated_and_finalized_alignment_points) * 0.1 and new_finalization_count == 0:
            final_pass = True

            # If we've locked 90% of the points we are done
        if len(finalized_points) > len(updated_and_finalized_alignment_points) * 0.9:
            final_pass = True

        stosTransform = updatedTransform

        # Make one more pass to see if we can improve finalized points
    # Todo: This code remained untouched after an optimization pass.  I think it would be worth examining whether it can be improved.
    # if len(finalized_points) >= 3:
    #     final_transform = nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(
    #         AlignRecordsToControlPoints(finalized_points.values()))
    # else:
    #     final_transform = updatedTransform
    final_transform = stosTransform

    (nudged_final_points, nudged_point_keys) = TryToImproveAlignments(stosTransform, combined_records_this_pass, settings)
    prettyoutput.Log(f'Final tuning of points adjusted {len(improved_alignments)} of {len(combined_records_this_pass)} points')

    # Return a transform built from the finalized points
    if len(nudged_final_points) >= 3:
        final_transform = nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(
            AlignRecordsToControlPoints(nudged_final_points.values()))

    return final_transform


def _RefineGridPointsForTwoImages(transform: nornir_imageregistration.transforms.ITransform,
                                  finalized: AlignmentRecordDict,
                                  settings: nornir_imageregistration.settings.GridRefinement) -> list[nornir_imageregistration.EnhancedAlignmentRecord]:
    """
    Places a regular grid of control points across the target image.  These corresponding points on the
    source image are then adjusted to create a mapping from Source To Fixed Space for the source image.

    :param transform transform: Transform that maps from source to target space
    :param dict finalized: A dictionary of points, indexed by Target Space Coordinates, that are finalized and do not need to be checked
    :param nornir_imageregistration.settings.GridRefinement settings: settings to use for registrations
    """

    # Mark a grid along the fixed image, then find the points on the warped image

    grid_data = nornir_imageregistration.grid_subdivision.CenteredGridDivision(settings.source_image.shape,
                                                                               cell_size=settings.cell_size,
                                                                               grid_spacing=settings.grid_spacing,
                                                                               transform=transform)
    # grid_data = nornir_imageregistration.ITKGridDivision(settings.source_image.shape,
    #                                                                       cell_size=settings.cell_size,
    #                                                                       grid_spacing=settings.grid_spacing,
    #                                                                       transform=transform)

    # Remove finalized points from refinement consideration
    if finalized is not None and len(finalized) > 0:
        not_finalized = [tuple(grid_data.coords[i,:]) not in finalized for i in range(grid_data.coords.shape[0])]
        valid = np.asarray(not_finalized, np.bool)
        grid_data.RemoveMaskedPoints(valid)

    # grid_dims = nornir_imageregistration.TileGridShape(target_image.shape, grid_spacing)

    # Create 
    #    TargetPoints = coords * grid_spacing  # [np.asarray((iCol * grid_spacing[0], iRow * grid_spacing[1]), dtype=np.int32) for (iRow, iCol) in coords]

    # Grid dimensions round up, so if we are larger than image find out by how much and adjust the points so they are centered on the image
    #    overage = ((grid_dims * grid_spacing) - target_image.shape) / 2.0
    #    TargetPoints = np.round(TargetPoints - overage).astype(np.int64)
    # TODO, ensure fixedPoints are within the bounds of target_image
    grid_data.FilterOutofBoundsSourcePoints(settings.source_image.shape)
    grid_data.RemoveCellsUsingSourceImageMask(settings.source_mask, settings.min_unmasked_area)
    # nornir_imageregistration.views.grid_data.PlotGridPositionsAndMask(grid_data.SourcePoints, source_mask, OutputFilename=None)

    if grid_data.num_points == 0:
        # There is nothing to refine, perhaps the image is too small for the grid cell size?
        # prettyoutput.LogErr("No points meet criteria for grid refinement")
        raise ValueError("No points meet criteria for grid refinement")

    grid_data.PopulateTargetPoints(transform)
    grid_data.RemoveCellsUsingTargetImageMask(settings.target_mask, settings.min_unmasked_area)

    # nornir_imageregistration.views.grid_data.PlotGridPositionsAndMask(grid_data.TargetPoints, target_mask, OutputFilename=None)
    # grid_data.ApplyWarpedImageMask(source_mask)
    #     valid_inbounds = np.logical_and(np.all(FixedPoi4nts >= np.asarray((0, 0)), 1), np.all(TargetPoints < target_mask.shape, 1))
    #     TargetPoints = TargetPoints[valid_inbounds, :]
    #     coords = coords[valid_inbounds, :]
    #

    #         TargetPoints = TargetPoints[valid, :]
    #         coords = coords[valid, :]
    #
    #     # Filter Fixed points falling outside the mask
    #     if target_mask is not None:
    #         valid = nornir_imageregistration.index_with_array(target_mask, TargetPoints)
    #         TargetPoints = TargetPoints[valid, :]
    #         coords = coords[valid, :]
    #
    #     SourcePoints = transform.InverseTransform(TargetPoints).astype(np.int32)
    #     if source_mask is not None:
    #         valid = np.logical_and(np.all(SourcePoints >= np.asarray((0, 0)), 1), np.all(SourcePoints < source_mask.shape, 1))
    #         SourcePoints = SourcePoints[valid, :]
    #         TargetPoints = TargetPoints[valid, :]
    #         coords = coords[valid, :]
    #
    #         valid = nornir_imageregistration.index_with_array(source_mask, SourcePoints)
    #         SourcePoints = SourcePoints[valid, :]
    #         TargetPoints = TargetPoints[valid, :]
    #         coords = coords[valid, :]

    return _RefinePointsForTwoImages(transform, [tuple(row) for row in grid_data.coords], grid_data.SourcePoints,
                                     grid_data.TargetPoints, settings)


def _RefinePointsForTwoImages(transform: nornir_imageregistration.transforms.ITransform,
                              keys: list[tuple[int, int]],
                              sourcePoints: np.ndarray,
                              targetPoints: np.ndarray,
                              settings: nornir_imageregistration.settings.GridRefinement) -> list[nornir_imageregistration.EnhancedAlignmentRecord]:
    """
    Registers a set of points using regions from both images. These corresponding points on the
    target image are then adjusted to create a mapping from Source To Fixed Space for the source image.

    :param transform transform: Transform that maps from source to target space
    :param points: dict of {key:(targetpoint)} where targetpoint is a 1x2 ndarray describing a point on the target image.  These coordinates are passed through the transform to determine the point on the source image.  Key is used for return dictionary.
    :param nornir_imageregistration.settings.GridRefinement settings: settings to use for registrations
    """

    if len(keys) != targetPoints.shape[0]:
        raise ValueError("keys must have equal number of entries as points")

    nPoints = len(keys)

    pool = nornir_pools.GetGlobalMultithreadingPool()
    # pool = nornir_pools.GetGlobalThreadPool()
    tasks = list()
    alignment_records = list()

    rigid_transforms = ApproximateRigidTransformBySourcePoints(input_transform=transform, source_points=sourcePoints, cell_size=settings.cell_size)
    
    os.environ['DEBUG'] = '1'

    for i in range(nPoints):
        targetPoint = targetPoints[i,:]
        sourcePoint = sourcePoints[i,:]
        key = keys[i]
        # So... the way I'm handling refine is backwards.  I'm not sure how much it matters.
        # I'm passing the target point, but the point that is fixed in the transform I am building is the source point.
        # So the transform runs an inverse transform to obtain the source point, which may be slightly off.
        AlignTask = pool.add_task(f"Align {key}",
                                  AttemptAlignPoint,
                                  rigid_transforms[i],
                                  settings.target_image_meta, #Send the shared file to the task
                                  settings.source_image_meta, #Send the shared file to the task
                                  #settings.target_mask,
                                  #settings.source_mask,
                                  settings.target_image_stats,
                                  settings.source_image_stats,
                                  targetPoint,
                                  settings.cell_size,
                                  anglesToSearch=settings.angles_to_search,
                                  min_alignment_overlap=settings.min_alignment_overlap)
        #AlignTask = StartAttemptAlignPoint(pool,
                                           #f"Align {key}",
                                           #rigid_transforms[i],

                                           #settings.target_image,
                                           #settings.source_image,
                                           # settings.target_mask,
                                           # settings.source_mask,
                                           #settings.target_image_stats,
                                           #settings.source_image_stats,
                                           #targetPoint,
                                           #settings.cell_size,
                                           #anglesToSearch=settings.angles_to_search,
                                           #min_alignment_overlap=settings.min_alignment_overlap)

        if AlignTask is None:
            continue

        #         AlignTask = pool.add_task("Align %d,%d" % (coord[0], coord[1]),
        #                                   AttemptAlignPoint,
        #                                   Transform,
        #                                   shared_fixed_image,
        #                                   shared_warped_image,
        #                                   TargetPoints[i,:],
        #                                   cell_size,
        #                                   anglesToSearch=AnglesToSearch)

        AlignTask.ID = i
        AlignTask.key = key
        tasks.append(AlignTask)

    #             arecord.iRow = iRow
    #             arecord.iCol = iCol
    #             arecord.TargetPoint = TargetPoint
    #             arecord.WarpedPoint = WarpedPoint
    #             arecord.AdjustedWarpedPoint = WarpedPoint + arecord.peak
    #
    #             alignment_records.append(arecord)

    for t in tasks:
        arecord = t.wait_return()

        erec = nornir_imageregistration.EnhancedAlignmentRecord(ID=t.key,
                                                                TargetPoint=targetPoints[t.ID,:],
                                                                SourcePoint=sourcePoints[t.ID,:],
                                                                peak=arecord.peak,
                                                                weight=arecord.weight,
                                                                angle=arecord.angle,
                                                                flipped_ud=arecord.flippedud)

        if 'DEBUG' in os.environ:
            erec.TargetROI = arecord.TargetROI
            erec.SourceROI = arecord.SourceROI
            erec.TranslatedSourceROI = nornir_imageregistration.CropImage(erec.SourceROI, int(np.floor(-erec.peak[1])),
                                                                          int(np.floor(-erec.peak[0])),
                                                                          erec.SourceROI.shape[1], erec.SourceROI.shape[0],
                                                                          cval=float(np.median(erec.SourceROI.flat)))

        # erec.TargetPSDScore = nornir_imageregistration.image_stats.ScoreImageWithPowerSpectralDensity(t.TargetROI)
        # erec.SourcePSDScore = nornir_imageregistration.image_stats.ScoreImageWithPowerSpectralDensity(t.SourceROI)

        # erec.PSDDelta = abs(erec.TargetPSDScore - erec.SourcePSDScore)
        # erec.PSDDelta = (erec.TargetROI - np.mean(erec.TargetROI.flat)) - (
        #        erec.SourceROI - np.mean(erec.SourceROI.flat))
        # erec.PSDDelta = np.sum(np.abs(erec.PSDDelta))
        # erec.CalculatedWarpedPoint = Transform.InverseTransform(erec.AdjustedTargetPoint).reshape(2)
        # arecord.ID = (iRow, iCol)
        # arecord.TargetPoint = t.TargetPoint
        # arecord.WarpedPoint = t.WarpedPoint
        # arecord.AdjustedWarpedPoint = t.WarpedPoint + arecord.peak

        alignment_records.append(erec)
    #
    #     del shared_warped_image
    #     del shared_fixed_image

    return alignment_records
    # Cull the worst of the alignment records

    # Build a new transform using our alignment points
    # peaks = np.list(map(lambda a: a.peak, alignment_records))
    # updatedTransform = _PeakListToTransform(alignment_records)

    # Re-run the loop

    # return updatedTransform


def AlignRecordsToControlPoints(
        alignment_records: AlignmentRecordList) -> NDArray:
    """
    Convert alignment records to a numpy array of control points
    :param alignment_records: list of alignment records
    :return: ndarray of control points
    """

    SourcePoints = np.asarray(list(map(lambda a: a.SourcePoint, alignment_records)))
    TargetPoints = np.asarray(list(map(lambda a: a.AdjustedTargetPoint, alignment_records)))

    PointPairs = np.hstack((TargetPoints, SourcePoints))
    return PointPairs


def _alignment_records_to_composite_scores(
        alignment_records: AlignmentRecordList):
    """
    A helper function to produce a ndarray of measurements for alignment records
    :return: A 3xN array of [Weight Distance ((MaxWeight - Weight) * Distance)]
    """
    weights_distance = np.asarray(list(map(lambda a: (a.weight, np.sqrt(a.peak.dot(a.peak))), alignment_records)))
    max_weight_distance = np.max(weights_distance, 0)
    # I don't want a random near zero travel distance accidentally reducing a bad alignment score, so the
    # minimum travel distance is 1 for calculating the distance weight 
    floor_distances = np.maximum(1, weights_distance[:, 1])
    composite_weight = (max_weight_distance[0] - weights_distance[:, 0]) * np.sqrt(floor_distances)
    weights_distance = np.hstack((weights_distance, composite_weight.reshape((len(composite_weight), 1))))
    return weights_distance


def _PeakListToTransform(alignment_records: AlignmentRecordList,
                         fixed_points: NDArray | None=None, percentile: float=None, cutoff: float=None):
    """
    Converts a set of EnhancedAlignmentRecord peaks from the _RefineGridPointsForTwoImages function into a transform
    :param alignment_records: Records that we will include if they pass the metrics for inclusion above the cutoff percentile
    :param fixed_points: Control points that will always be included and not measured in metrics
    :param percentile: Cutoff percentile for inlcuding alignment_records, if None, all points are included
    :param float cutoff: Cutoff value, if set, percentile is ignored
    :return: (Transform, used_alignment_records, cutoff) The transform, the alignment_records used in the transform, and the cutoff value used/calculated
    """
    num_fixed = 0
    if fixed_points is not None:
        if not isinstance(fixed_points, np.ndarray):
            raise ValueError("fixed_points must be an ndarray")

        num_fixed = fixed_points.shape[0]

    num_alignments = len(alignment_records)
    if num_alignments == 0:
        raise ValueError("Need at one new alignment_record to improve a transform")

    if num_alignments + num_fixed < 3:
        raise ValueError(
            f"Need at least three points to make a transform.  Got {len(alignment_records)} alignments and {num_fixed} fixed points")

    # TargetPoints = np.asarray(list(map(lambda a: a.TargetPoint, alignment_records)))
    OriginalSourcePoints = np.asarray(list(map(lambda a: a.SourcePoint, alignment_records)))
    AdjustedTargetPoints = np.asarray(list(map(lambda a: a.AdjustedTargetPoint, alignment_records)))
    # AdjustedWarpedPoints = np.asarray(list(map(lambda a: a.AdjustedWarpedPoint, alignment_records)))
    # CalculatedWarpedPoints = np.asarray(list(map(lambda a: a.CalculatedWarpedPoint, alignment_records)))

    # With Weights big numbers are good and with peak distance generally small numbers are good.
    # To merge these scores I invert the weights to subtract them from the max weight, then multiply by distance

    weights_distance = _alignment_records_to_composite_scores(alignment_records)
    composite_score = weights_distance[:, 2]
    # WarpedPeaks = AdjustedWarpedPoints - OriginalSourcePoints

    if cutoff is None:
        cutoff = np.max(composite_score)
        if percentile is not None:
            cutoff = np.percentile(composite_score, 100.0 - percentile)

    valid_indicies = composite_score <= cutoff

    # Todo: Check that we have at least three points

    ValidFP = AdjustedTargetPoints[valid_indicies,:]
    ValidWP = OriginalSourcePoints[valid_indicies,:]

    if not np.array_equiv(ValidFP.shape, Triangulation.RemoveDuplicateControlPoints(ValidFP).shape):
        raise Exception("Duplicate fixed points detected")

    if not np.array_equiv(ValidWP.shape, Triangulation.RemoveDuplicateControlPoints(ValidWP).shape):
        raise Exception("Duplicate warped points detected")

    # See if we have enough points to build a transform.  If not include top scoring points until we have a transform
    if ValidFP.shape[0] + num_fixed < 3:
        num_needed = 3 - num_fixed
        sorted_composite_indicies = np.argsort(composite_score)
        top_alignment_indicies = sorted_composite_indicies[0:num_needed]
        ValidFP = AdjustedTargetPoints[top_alignment_indicies,:]
        ValidWP = OriginalSourcePoints[top_alignment_indicies,:]
        prettyoutput.Log(
            f'Insufficient alignments found, expanding to use top {num_needed} alignments of {num_alignments} alignments')

    # PointPairs = np.hstack((TargetPoints, SourcePoints))
    point_pairs = np.hstack((ValidFP, ValidWP))

    if fixed_points is not None and fixed_points.shape[0] > 0:
        if fixed_points.shape[1] != 4:
            raise Exception("fixed_points must have shape (N,4)")

        point_pairs = np.vstack((point_pairs, fixed_points))

    # PointPairs.append((ControlY, ControlX, mappedY, mappedX))

    prettyoutput.Log(
        f'Built transform with {point_pairs.shape[0]} of {num_alignments + num_fixed} points, including the {num_fixed} fixed points using cutoff {cutoff:g}')

    T = nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(point_pairs)

    used_alignment_records = [alignment_records[valid_item[0]] for valid_item in
                              filter(lambda item: item[1], enumerate(valid_indicies))]

    return T, used_alignment_records, weights_distance


def ConvertTransformToGridTransform(Transform: nornir_imageregistration.ITransform, source_image_shape: NDArray,
                                    cell_size: NDArray=None, grid_dims: NDArray=None,
                                    grid_spacing: NDArray=None) -> nornir_imageregistration.transforms.triangulation.Triangulation:
    """
    Converts a set of EnhancedAlignmentRecord peaks from the _RefineGridPointsForTwoImages function into a transform

    """

    grid_data = nornir_imageregistration.ITKGridDivision(source_image_shape, cell_size=cell_size,
                                                         grid_spacing=grid_spacing, grid_dims=grid_dims)
    grid_data.PopulateTargetPoints(Transform)

    point_pairs = np.hstack((grid_data.TargetPoints, grid_data.SourcePoints))

    # TODO, create a specific grid transform object that uses numpy's RegularGridInterpolator

    T = nornir_imageregistration.transforms.triangulation.Triangulation(point_pairs)
    T.gridWidth = grid_data.grid_dims[1]
    T.gridHeight = grid_data.grid_dims[0]

    return T

# def AlignmentRecordsTo2DArray(alignment_records):
#     
#     def IsFinalFunc(record):
#         record.weight = 
#     
#     #Create a 2D array of 
#     Indicies = np.hstack([np.asarray(a.ID,np.int32) for a in alignment_records])
#     
#     grid_dims = Indicies.max()
#     
#     mask = np.zeros(grid_dims, np.bool)
#     
#     for a in alignment_records:
#         mask[a.ID] = 
#           


def AlignmentRecordsToDict(alignment_records: AlignmentRecordList):
    lookup = {}
    for a in alignment_records:
        lookup[a.ID] = a

    return lookup


def plot_percentile_estimates(input_data: np.typing.NDArray, output_path: str | None, dpi: int | None):
    import matplotlib.pyplot as plt

    p = np.linspace(0, 100, 6001)
    ax = plt.gca()
    lines = [
        ('linear', '-', 'C0'),
        ('inverted_cdf', ':', 'C1'),
        # Almost the same as `inverted_cdf`:
        ('averaged_inverted_cdf', '-.', 'C1'),
        ('closest_observation', ':', 'C2'),
        ('interpolated_inverted_cdf', '--', 'C1'),
        ('hazen', '--', 'C3'),
        ('weibull', '-.', 'C4'),
        ('median_unbiased', '--', 'C5'),
        ('normal_unbiased', '-.', 'C6'),
    ]
    for method, style, color in lines:
        ax.plot(
            p, np.percentile(input_data, p, method=method),
            label=method, linestyle=style, color=color)
    ax.set(
        title='Percentiles for different methods and data',
        xlabel='Percentile',
        ylabel='Estimated percentile value')
    ax.legend()

    if output_path is not None:
        # plt.show() 
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
        plt.close()
    else:
        plt.show()


def CalculateFinalizedAlignmentPointsMask(alignment_records: AlignmentRecordList,
                                          percentile: float=0.5, max_travel_distance: float=1.0,
                                          weight_cutoff: float | None=None) -> NDArray[bool]:
    """
    :param alignment_records:
    :param weight_cutoff:
    :param percentile: Cutoff percentile for inlcuding alignment_records, if None, all points are included
an be offset before it is not eligible for finalization
    :return (array, cutoff): logical mask indicating which points meet the threshold to be finalized and cutoff value used/calculated

    """
    weights_distance = _alignment_records_to_composite_scores(alignment_records)
    # invert the weights to multiply by distance to promote low distance alignments 

    # composite_score = weights_distance[:,0] #np.prod(weights_distance,1)
    # composite_score = weights_distance[:,2]

    if weight_cutoff is None:
        if percentile is not None:
            weight_cutoff = np.percentile(weights_distance[:, 0], percentile)
        else:
            weight_cutoff = 0

    # weights = np.asarray(list(map(lambda a: a.weight, alignment_records)))
    # peak_distance = np.asarray(list(map(lambda a: np.linalg.norm(a.peak), alignment_records))) 

    # cutoff = np.percentile(weights, percentile)
    # if cutoff is None:  
    # if percentile is not None:
    # cutoff = np.percentile(composite_score, 100.0 - percentile)
    # else:
    # cutoff = np.max(composite_score) + 1

    valid_weight = weights_distance[:, 0] >= weight_cutoff
    valid_distance = weights_distance[:, 1] <= max_travel_distance

    # plot_percentile_estimates(weights_distance[:,0])

    finalize_mask = np.logical_and(valid_weight, valid_distance)

    return finalize_mask


def ApproximateRigidTransformByTargetPoints(input_transform: nornir_imageregistration.ITransform,
                                            target_points: NDArray) -> list[nornir_imageregistration.transforms.Rigid]:
    """
    Given an array of points, returns a set of rigid transforms for each point that estimate the angle and offset for those two points to align.
    """

    target_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(target_points)

    numPoints = target_points.shape[0]

    source_points = input_transform.InverseTransform(target_points)

    return ApproximateRigidTransformBySourcePoints(input_transform, source_points)
    # translate the target points by 1, and find the angle between the source points
    # offset = np.array([0, 1])
    # offset_source_points = source_points + offset
    #
    # offsets = np.tile(offset, (numPoints, 1))
    # origins = np.tile(np.array([0, 0]), (numPoints, 1))
    #
    # recalculated_target_points = input_transform.Transform(source_points)
    # offset_target_points = input_transform.Transform(offset_source_points)
    #
    # target_delta = offset_target_points - recalculated_target_points
    #
    # angles = -np.round(nornir_imageregistration.ArcAngle(origins, offsets, target_delta), 3)
    #
    # target_offsets = target_points - source_points
    #
    # output_transforms = [nornir_imageregistration.transforms.Rigid(target_offset=target_offsets[i],
    #                                                                source_rotation_center=source_points[i],
    #                                                                angle=angles[i])
    #                      for i in range(0, len(angles))]
    #
    # return output_transforms


def ApproximateRigidTransformBySourcePoints(input_transform: nornir_imageregistration.ITransform,
                                            source_points: NDArray,
                                            cell_size: NDArray | None = None) -> list[nornir_imageregistration.transforms.Rigid]:
    """
    Given an array of points, returns a set of rigid transforms for each point that estimate the angle and offset for those two points to align.
    """

    source_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(source_points)

    numPoints = source_points.shape[0]

    # translate the target points a distance, and estimage the angle to determine the rotation
    offset = None
    if cell_size is None:
        #If we don't pass a cell_size, then make a reasonable guess by measuring how far away nearest points are from first point
        if source_points.shape[0] > 1:
            estimated_cell_distance = scipy.spatial.distance.cdist(source_points[0:1, :], source_points[1:, :]).min() / 2.0
            offset = np.array((0, estimated_cell_distance))
        else:
            offset = np.array((0, 1))
    else:
        offset = cell_size / 2.0

    offset_source_points = source_points + offset

    offsets = np.tile(offset, (numPoints, 1))
    origins = np.tile(np.array([0, 0]), (numPoints, 1))

    target_points = input_transform.Transform(source_points)
    offset_target_points = input_transform.Transform(offset_source_points)

    target_delta = offset_target_points - target_points

    angles = np.round(nornir_imageregistration.ArcAngle(origins, offsets, target_delta), 3)

    target_offsets = target_points - source_points

    output_transforms = [nornir_imageregistration.transforms.Rigid(target_offset=target_offsets[i],
                                                                   source_rotation_center=source_points[i],
                                                                   angle=angles[i])
                         for i in range(0, len(angles))]

    return output_transforms


def BuildAlignmentROIs(transform: nornir_imageregistration.ITransform,
                       targetImage_param: NDArray | nornir_imageregistration.Shared_Mem_Metadata,
                       sourceImage_param: NDArray | nornir_imageregistration.Shared_Mem_Metadata,
                       target_image_stats: nornir_imageregistration.ImageStats | None,
                       source_image_stats: nornir_imageregistration.ImageStats | None,
                       target_controlpoint: NDArray | tuple[float, float],
                       alignmentArea: NDArray | tuple[float, float],
                       description: str | None=None) -> tuple[NDArray, NDArray]:
    """
    :param transform:
    :param targetImage:
    :param sourceImage:
    :param target_image_stats: if None, no noise is added to the output in masked or unmapped areas, 0 is used instead
    :param source_image_stats: if None, no noise is added to the output in masked or unmapped areas, 0 is used instead
    :param target_controlpoint:
    :param alignmentArea:
    :param description:  Entirely optional parameter describing which cell we are processing
    :return:
    """
    targetImage = nornir_imageregistration.ImageParamToImageArray(targetImage_param, dtype=nornir_imageregistration.default_image_dtype())
    sourceImage = nornir_imageregistration.ImageParamToImageArray(sourceImage_param, dtype=nornir_imageregistration.default_image_dtype())

    # Adjust the point by 0.5 if it is an odd-sized area to ensure the output is centered on the desired pixel
    target_controlpoint = target_controlpoint.astype(float, copy=False).flatten()
    adjust_mask = np.mod(alignmentArea, 2) > 0
    target_controlpoint[adjust_mask] += 0.5

    target_rectangle = nornir_imageregistration.Rectangle.CreateFromPointAndArea(
        point=(target_controlpoint[0] - (alignmentArea[0] / 2.0),
               target_controlpoint[1] - (alignmentArea[1] / 2.0)),
        area=alignmentArea)

    target_rectangle = nornir_imageregistration.Rectangle.SnapRound(target_rectangle)

    # Make sure the rectangle is the correct size, with an origin on an integer boundary
    target_rectangle = nornir_imageregistration.Rectangle.change_area(target_rectangle, alignmentArea,
                                                                      integer_origin=True)

    target_image_roi = nornir_imageregistration.CropImage(targetImage,
                                                          target_rectangle.BottomLeft[1],
                                                          target_rectangle.BottomLeft[0],
                                                          int(target_rectangle.Size[1]), int(target_rectangle.Size[0]),
                                                          cval=False if target_image_stats is None else "random",
                                                          image_stats=target_image_stats)

    # Pull image subregions
    source_image_roi = nornir_imageregistration.assemble.SourceImageToTargetSpace(transform,
                                                                                  DataToTransform=sourceImage,
                                                                                  output_botleft=target_rectangle.BottomLeft,
                                                                                  output_area=target_rectangle.Size,
                                                                                  extrapolate=True,
                                                                                  cval=False if source_image_stats is None else np.nan)

    if source_image_stats is not None:
        source_image_roi = nornir_imageregistration.RandomNoiseMask(source_image_roi,
                                                                    np.logical_not(np.isnan(source_image_roi)),
                                                                    imagestats=source_image_stats)

    nornir_imageregistration.close_shared_memory(targetImage_param)
    nornir_imageregistration.close_shared_memory(sourceImage_param)

    return target_image_roi, source_image_roi


def StartAttemptAlignPoint(pool: nornir_pools.IPool,
                           taskname: str,
                           transform: nornir_imageregistration.ITransform,
                           targetImage: NDArray,
                           sourceImage: NDArray,
                           # targetMask: NDArray,
                           # sourceMask: NDArray,
                           target_image_stats: nornir_imageregistration.ImageStats | None,
                           source_image_stats: nornir_imageregistration.ImageStats | None,
                           target_controlpoint: NDArray | tuple[float, float],
                           alignmentArea: NDArray | tuple[float, float],
                           anglesToSearch: Iterable[float] | None=None,
                           min_alignment_overlap: float=0.5) -> nornir_pools.Task | None:
    if anglesToSearch is None:
        anglesToSearch = np.linspace(-7.5, 7.5, 11)

    # target_mask_roi, source_mask_roi = BuildAlignmentROIs(transform=transform,
    #                                                         targetImage=targetMask,
    #                                                         sourceImage=sourceMask,
    #                                                         target_image_stats=None,
    #                                                         source_image_stats=None,
    #                                                         target_controlpoint=target_controlpoint,
    #                                                         alignmentArea=alignmentArea,
    #                                                         description=taskname)
    #
    # target_mask_nonzero = np.count_nonzero(target_mask_roi)
    # source_mask_nonzero = np.count_nonzero(source_mask_roi)
    # cell_area = alignmentArea.prod()
    #
    # if target_mask_nonzero / cell_area < 0.25:
    #     raise ValueError("This mask should have been found earlier")
    #
    # if source_mask_nonzero / cell_area < 0.25:
    #     raise ValueError("This mask should have been found earlier")

    target_image_roi, source_image_roi = BuildAlignmentROIs(transform=transform,
                                                            targetImage_param=targetImage,
                                                            sourceImage_param=sourceImage,
                                                            target_image_stats=target_image_stats,
                                                            source_image_stats=source_image_stats,
                                                            target_controlpoint=target_controlpoint,
                                                            alignmentArea=alignmentArea,
                                                            description=taskname)

    # Just ignore pure color regions
    if not np.any(target_image_roi != target_image_roi[0][0]):
        return None
    if not np.any(source_image_roi != source_image_roi[0][0]):
        return None

    # nornir_imageregistration.ShowGrayscale([targetImageROI, sourceImageROI])

    # pool = Pools.GetGlobalMultithreadingPool()

    # task = pool.add_task("AttemptAlignPoint", nornir_imageregistration.FindOffset, targetImageROI, sourceImageROI, MinOverlap = 0.2)
    # apoint = task.wait_return()
    # apoint = nornir_imageregistration.FindOffset(targetImageROI, sourceImageROI, MinOverlap=0.2)
    # nornir_imageregistration.ShowGrayscale([targetImageROI, sourceImageROI], "Fixed <---> Warped")

    # nornir_imageregistration.ShowGrayscale([targetImageROI, sourceImageROI])

    #     nornir_imageregistration.stos_brute.SliceToSliceBruteForce(
    #                         targetImageROI,
    #                         sourceImageROI,
    #                         AngleSearchRange=anglesToSearch,
    #                         MinOverlap=min_alignment_overlap,
    #                         SingleThread=True,
    #                         Cluster=False,
    #                         TestFlip=False)
    #
    task = pool.add_task(taskname,
                         nornir_imageregistration.stos_brute.SliceToSliceBruteForce,
                         target_image_roi,
                         source_image_roi,
                         AngleSearchRange=anglesToSearch,
                         MinOverlap=min_alignment_overlap,
                         SingleThread=True,
                         Cluster=False,
                         TestFlip=False)

    task.TargetROI = target_image_roi
    task.SourceROI = source_image_roi

    return task

def AttemptAlignPoint( transform: nornir_imageregistration.ITransform,
                       targetImage: NDArray,
                       sourceImage: NDArray,
                       # targetMask: NDArray,
                       # sourceMask: NDArray,
                       target_image_stats: nornir_imageregistration.ImageStats | None,
                       source_image_stats: nornir_imageregistration.ImageStats | None,
                       target_controlpoint: NDArray | tuple[float, float],
                       alignmentArea: NDArray | tuple[float, float],
                       anglesToSearch: Iterable[float] | None=None,
                       min_alignment_overlap: float=0.5,
                       use_cp: bool = False) -> nornir_imageregistration.AlignmentRecord | None:


    if anglesToSearch is None:
        anglesToSearch = np.linspace(-7.5, 7.5, 11)

    target_image_roi, source_image_roi = BuildAlignmentROIs(transform=transform,
                                                            targetImage_param=targetImage,
                                                            sourceImage_param=sourceImage,
                                                            target_image_stats=target_image_stats,
                                                            source_image_stats=source_image_stats,
                                                            target_controlpoint=target_controlpoint,
                                                            alignmentArea=alignmentArea,
                                                            description='')

    # Just ignore pure color regions
    if not np.any(target_image_roi != target_image_roi[0][0]):
        return None
    if not np.any(source_image_roi != source_image_roi[0][0]):
        return None

    # nornir_imageregistration.ShowGrayscale([targetImageROI, sourceImageROI])

    # pool = Pools.GetGlobalMultithreadingPool()

    # task = pool.add_task("AttemptAlignPoint", nornir_imageregistration.FindOffset, targetImageROI, sourceImageROI, MinOverlap = 0.2)
    # apoint = task.wait_return()
    # apoint = nornir_imageregistration.FindOffset(targetImageROI, sourceImageROI, MinOverlap=0.2)
    # nornir_imageregistration.ShowGrayscale([targetImageROI, sourceImageROI], "Fixed <---> Warped")

    # nornir_imageregistration.ShowGrayscale([targetImageROI, sourceImageROI])

    #     nornir_imageregistration.stos_brute.SliceToSliceBruteForce(
    #                         targetImageROI,
    #                         sourceImageROI,
    #                         AngleSearchRange=anglesToSearch,
    #                         MinOverlap=min_alignment_overlap,
    #                         SingleThread=True,
    #                         Cluster=False,
    #                         TestFlip=False)
    #
    result = nornir_imageregistration.stos_brute.SliceToSliceBruteForce(
                         target_image_roi,
                         source_image_roi,
                         AngleSearchRange=anglesToSearch,
                         MinOverlap=min_alignment_overlap,
                         SingleThread=True,
                         Cluster=False,
                         TestFlip=False)
    
    if 'DEBUG' in os.environ:
        result.TargetROI = target_image_roi
        result.SourceROI = source_image_roi

    return result


def TryToImproveAlignments(transform: nornir_imageregistration.transforms.ITransform, alignment_records: dict,
                           settings: nornir_imageregistration.settings.GridRefinement) \
                           ->tuple[AlignmentRecordDict, list[AlignmentRecordKey]]:
    """
    Given a set of alignment points, try to align the points again.  If we get a stronger score then
    replace the alignment with the higher scoring result
    
    """

    items = alignment_records.items()

    if len(items) == 0:
        return dict(), list()

    SourcePoints = np.vstack([fp[1].SourcePoint for fp in items])
    TargetPoints = np.vstack([fp[1].TargetPoint for fp in items])
    # keys = [fp.ID for fp in alignment_records]
    keys = [fp[0] for fp in items]

    refined_alignments = _RefinePointsForTwoImages(transform, keys, SourcePoints, TargetPoints, settings)
    #
    # pool = None
    # if len(alignment_points) > 8:
    #     pool = nornir_pools.GetGlobalMultithreadingPool()
    # else:
    #     pool = nornir_pools.GetGlobalSerialPool()
    #
    # tasks = {}
    # for record in alignment_points:
    #     key = tuple(record.SourcePoint)
    #
    #     # See if we can improve the final alignment
    #     # refined_align_record = nornir_imageregistration.stos_brute.SliceToSliceBruteForce(record.TargetROI,
    #     #                                                                                   record.SourceROI,
    #     #                                                                                   AngleSearchRange=settings.final_pass_angles,
    #     #                                                                                   MinOverlap=settings.min_alignment_overlap,
    #     #                                                                                   SingleThread=False,
    #     #                                                                                   Cluster=False,
    #     #                                                                                   TestFlip=False)
    #
    #     t = pool.add_task(f'{key}',
    #                         nornir_imageregistration.stos_brute.SliceToSliceBruteForce,
    #                         record.TargetROI,
    #                         record.SourceROI,
    #                         AngleSearchRange=settings.final_pass_angles,
    #                         MinOverlap=settings.min_alignment_overlap,
    #                         SingleThread=True,
    #                         Cluster=False,
    #                         TestFlip=False)
    #
    #     tasks[key] = (t, record)
    #
    # pool.wait_completion()

    output = dict()  # type: AlignmentRecordDict
    improved_alignments = []  # type: list[tuple[int, int]]
    for refined_align_record in refined_alignments:
        key = refined_align_record.ID  # type: tuple[int, int]
        record = alignment_records[key]
        # task = task_tuple[0]
        # record = task_tuple[1]
        # refined_align_record = task.wait_return()
        chosen_record = record
        magnitude = np.sqrt(refined_align_record.peak.dot(refined_align_record.peak))
        if refined_align_record.weight > record.weight and magnitude < settings.max_travel_for_finalization:
            # oldPSDDelta = record.PSDDelta
            # record = nornir_imageregistration.EnhancedAlignmentRecord(ID=record.ID,
            #                                                              TargetPoint=record.TargetPoint,
            #                                                              SourcePoint=record.SourcePoint,
            #                                                              peak=refined_align_record.peak,
            #                                                              weight=refined_align_record.weight,
            #                                                              angle=refined_align_record.angle,
            #                                                              flipped_ud=refined_align_record.flippedud)
            # record.PSDDelta = oldPSDDelta
            # record.TargetROI = record.TargetROI
            # record.SourceROI = record.SourceROI
            chosen_record = refined_align_record
            improved_alignments.append(key)

        # Create a record that is unmoving
        output[key] = nornir_imageregistration.EnhancedAlignmentRecord(chosen_record.ID,
                                                                       TargetPoint=chosen_record.AdjustedTargetPoint,
                                                                       SourcePoint=chosen_record.SourcePoint,
                                                                       peak=np.asarray((0, 0), dtype=np.float32),
                                                                       weight=chosen_record.weight, angle=0,
                                                                       flipped_ud=chosen_record.flippedud)

        # output[key].PSDDelta = chosen_record.PSDDelta

    # Close the pool to prevent threads from hanging around
    # pool.shutdown()
    return output, improved_alignments
