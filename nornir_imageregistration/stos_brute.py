"""
Created on Oct 4, 2012

@author: u0490822
"""
from typing import NamedTuple
import multiprocessing
import multiprocessing.sharedctypes
from time import sleep
import numpy as np
from numpy.typing import NDArray
from typing import Sequence
import logging

from nornir_imageregistration import AlignmentRecord
import nornir_imageregistration.phasecorrelation
from nornir_imageregistration.settings import StosBruteSettings, AngleSearchRange

# Check if cupy is available, and if it is not import thunks that refer to scipy/numpy
try:
    import cupy as cp
    import cupyx
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx

import nornir_imageregistration
import nornir_pools


# from memory_profiler import profile
def SliceToSliceBruteForce(FixedImageInput: nornir_imageregistration.ImageLike,
                           WarpedImageInput: nornir_imageregistration.ImageLike,
                           FixedImageMaskPath: nornir_imageregistration.ImageLike | None = None,
                           WarpedImageMaskPath: nornir_imageregistration.ImageLike | None = None,
                           LargestDimension: int | None = None,
                           AngleSearchRange: Sequence[float] | None = None,
                           MinOverlap: float = 0.75,
                           WarpedImageScaleFactors=None,
                           SingleThread: bool = False,
                           Cluster: bool = False,
                           TestFlip: bool = True) -> nornir_imageregistration.AlignmentRecord:
    """Given two images this function returns the rotation angle which best aligns them
       Largest dimension determines how large the images used for alignment should be.

       :param FixedImageInput:
       :param WarpedImageInput:
       :param FixedImageMaskPath:
       :param WarpedImageMaskPath:
       :param SingleThread:
       :param Cluster:
       :param TestFlip:
       :param int LargestDimension: The input images should be scaled so the largest image dimension is equal to this value, default is None
       :param float MinOverlap: The minimum amount of overlap we require in the images.  Higher values reduce false positives but may not register offset images
       :param float AngleSearchRange: A list of rotation angles to test.  Pass None for the default which is every two degrees
       :param float WarpedImageScaleFactors: Scale the warped image input by this amount before attempting registration
       """
    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

    if AngleSearchRange is not None:
        if not isinstance(AngleSearchRange, set):
            AngleSearchRange = set(AngleSearchRange)
        # if isinstance(AngleSearchRange, np.ndarray):

        if 0 not in set(AngleSearchRange):
            logger = logging.getLogger(__name__ + '.SliceToSliceBruteForce')
            logger.warning("AngleSearchRange should contain 0 degrees to ensure the best match is found")

    SingleThread = True if use_cp else SingleThread

    target_image_data = nornir_imageregistration.ImagePermutationHelper(FixedImageInput, FixedImageMaskPath)
    source_image_data = nornir_imageregistration.ImagePermutationHelper(WarpedImageInput, WarpedImageMaskPath)

    settings = StosBruteSettings(angles=AngleSearchRange,
                                 min_overlap=MinOverlap,
                                 source_image_scale_factors=WarpedImageScaleFactors,
                                 larget_dimension=LargestDimension,
                                 try_flipped=TestFlip)

    return SliceToSliceBruteForceWithPreprocessedImages(source_image_data, target_image_data, settings,
                                                        SingleThread=SingleThread, Cluster=Cluster)


def SliceToSliceBruteForceWithPreprocessedImages(source_image_data: nornir_imageregistration.ImagePermutationHelper,
                                                 target_image_data: nornir_imageregistration.ImagePermutationHelper,
                                                 settings: StosBruteSettings,
                                                 SingleThread: bool = False,
                                                 Cluster: bool = False) -> nornir_imageregistration.AlignmentRecord:
    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

    target_image = target_image_data.ImageWithMaskAsNoise
    source_image = source_image_data.ImageWithMaskAsNoise

    target_stats = target_image_data.Stats
    source_stats = source_image_data.Stats

    del target_image_data
    del source_image_data

    target_image = cp.asarraye(target_image) if use_cp and not isinstance(target_image, cp.ndarray) else target_image
    source_image = cp.asarray(source_image) if use_cp and not isinstance(source_image, cp.ndarray) else source_image

    scalar = 1.0
    if settings.larget_dimension is not None:
        scalar = nornir_imageregistration.ScalarForMaxDimension(settings.larget_dimension,
                                                                [target_image.shape, source_image.shape])
        if scalar > 1.0:
            scalar = 1.0

    if scalar != 1.0:
        target_image = nornir_imageregistration.ScaleImage(target_image, scalar)
        source_image = nornir_imageregistration.ScaleImage(source_image, scalar)

    # Replace extrema with noise
    best_match = _find_best_angle(source_image=source_image, target_image=target_image,
                                  source_stats=source_stats, target_stats=target_stats,
                                  angle_range=settings.angle_range,
                                  min_overlap=settings.min_overlap,
                                  SingleThread=SingleThread,
                                  use_cluster=Cluster)

    is_flipped = False
    if settings.try_flipped:
        # source_flipped = np.copy(source_image)
        source_flipped = np.flipud(source_image)

        best_match_flipped = _find_best_angle(source_image=source_flipped, target_image=target_image,
                                              source_stats=source_stats, target_stats=target_stats,
                                              angle_range=settings.angle_range,
                                              min_overlap=settings.min_overlap,
                                              SingleThread=SingleThread, use_cluster=Cluster)
        best_match_flipped.flippedud = True

        # Determine if the best match is flipped or not
        is_flipped = best_match_flipped.weight > best_match.weight

    if is_flipped:
        source_image = source_flipped
        best_match = best_match_flipped
    else:
        source_image = source_image

    # Note Clement - the RefinedAngleSearch list below is not centered around the current best angle
    # Default angle search range every 2 degrees
    # Old RefinedAngleSearch list: [(x * 0.1) + best_match.angle - 1.9 for x in range(0, 18)]
    # New RefinedAngleSearch list (length 39): [(x * 0.1 + best_match.angle) for x in range(-19, 20)]
    # New optional RefinedAngleSearch list (length 18): [(x * 0.2 + best_match.angle) for x in range(-9, 10)]
    if not settings.angle_range_defined():
        best_refined_match = _find_best_angle(source_image=source_image, target_image=target_image,
                                              source_stats=source_stats, target_stats=target_stats,
                                              # [(x * 0.1) + best_match.angle - 1.9 for x in range(0, 18)],
                                              angle_range=[(x * 0.2 + best_match.angle) for x in range(-9, 10)],
                                              min_overlap=settings.min_overlap, SingleThread=SingleThread)
        best_refined_match.flippedud = is_flipped
    else:
        min_step_size = 0.25
        if len(settings.angle_range) > 2:
            sorted_angles = sorted(settings.angle_range)
            iMatch = sorted_angles.index(best_match.angle)
            iBelow = iMatch - 1 if iMatch - 1 >= 0 else len(sorted_angles) - 1
            iAbove = iMatch + 1 if iMatch + 1 < len(sorted_angles) else 0
            below = sorted_angles[iMatch - 1] if iMatch - 1 >= 0 else sorted_angles[0] - np.abs(
                sorted_angles[1] - sorted_angles[0])
            above = sorted_angles[iMatch + 1] if iMatch + 1 < len(sorted_angles) else sorted_angles[
                                                                                          iMatch] + np.abs(
                sorted_angles[iMatch] - sorted_angles[iMatch - 1])
            refine_search_range = above - below
            nSteps = 20
            stepsize = refine_search_range / nSteps

            if stepsize < min_step_size:
                nSteps = int(refine_search_range / min_step_size)
                stepsize = refine_search_range / nSteps

            refined_angle_search_range = {(x * stepsize) + below for x in range(1, nSteps)}

            # Ensure we include the best match angle
            refined_angle_search_range.add(best_match.angle)

            best_refined_match = _find_best_angle(source_image=source_image, target_image=target_image,
                                                  source_stats=source_stats, target_stats=target_stats,
                                                  angle_range=np.array(list(refined_angle_search_range), float),
                                                  min_overlap=settings.min_overlap, SingleThread=SingleThread)
            best_refined_match.flippedud = is_flipped
        else:
            best_refined_match = best_match
            best_refined_match.flippedud = is_flipped

    if scalar > 1.0:
        AdjustedPeak = (best_refined_match.peak[0] * scalar, best_refined_match.peak[1] * scalar)
        best_refined_match = nornir_imageregistration.AlignmentRecord(AdjustedPeak, best_refined_match.weight,
                                                                      best_refined_match.angle, is_flipped)

    if settings.source_image_scaling_required:
        # AdjustedPeak = best_refined_match.peak * (1.0 / WarpedImageScaleFactors)
        best_refined_match = nornir_imageregistration.AlignmentRecord(best_refined_match.peak,
                                                                      best_refined_match.weight,
                                                                      best_refined_match.angle, is_flipped,
                                                                      settings.source_image_scale_factors)

    # best_refined_match.CorrectPeakForOriginalImageSize(imFixed.shape, source_image.shape)

    return best_refined_match


def ScoreOneAngle(target_original: NDArray, source_original: NDArray,
                  target_image_shape: tuple[int, int], source_image_shape: tuple[int, int],
                  angle: float,
                  target_stats: nornir_imageregistration.ImageStats | None = None,
                  source_stats: nornir_imageregistration.ImageStats | None = None,
                  target_image_prepadded: bool = True, min_overlap: float = 0.75):
    """Returns an alignment score for a fixed image and an image rotated at a specified angle"""

    # print(f'Scoring {angle} degrees')
    try:
        im_target = nornir_imageregistration.ImageParamToImageArray(target_original,
                                                                    dtype=nornir_imageregistration.default_image_dtype())
        im_source = nornir_imageregistration.ImageParamToImageArray(source_original,
                                                                    dtype=nornir_imageregistration.default_image_dtype())

        use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy
        # Use of cupy or numpy
        xp = cp.get_array_module(target_original)
        # Use of cupyx.scipy.fft or scipy.fft
        xp_scipy = cupyx.scipy.get_array_module(target_original)
        rotate = xp_scipy.ndimage.rotate

        # im_target = cp.asarray(im_target) if use_cp and not isinstance(im_target, cp.ndarray) else im_target
        # im_source = cp.asarray(im_source) if use_cp  and not isinstance(im_source, cp.ndarray)  else im_source

        # gc.set_debug(gc.DEBUG_LEAK)
        if target_stats is None:
            target_stats = nornir_imageregistration.ImageStats.CalcStats(im_target)

        if source_stats is None:
            source_stats = nornir_imageregistration.ImageStats.CalcStats(im_source)

        OKToDelimWarped = False
        if angle != 0:
            # This confused me for years, but the implementation of rotate calls affine_transform with
            # the rotation matrix.  However the docs for affine_transform state it needs to be called
            # with the inverse transform.  Hence negating the angle here.
            try:
                if use_cp:
                    im_source = rotate(im_source, axes=(0, 1), angle=-angle, cval=np.nan)
                else:
                    im_source = rotate(im_source.astype(np.float32, copy=False), axes=(0, 1), angle=-angle,
                                       cval=np.nan).astype(
                        im_source.dtype, copy=False)  # Numpy cannot rotate float16 images
            except RuntimeWarning as e:
                pass
            im_source_empty_entries = xp.isnan(im_source)
            im_source[im_source_empty_entries] = source_stats.GenerateNoise(xp.sum(im_source_empty_entries),
                                                                            dtype=im_source.dtype)
            OKToDelimWarped = True

        rotated_source = nornir_imageregistration.phasecorrelation.PadImageForPhaseCorrelation(im_source,
                                                                                               ImageMedian=source_stats.median,
                                                                                               ImageStdDev=source_stats.std,
                                                                                               MinOverlap=min_overlap)

        assert (rotated_source.shape[0] > 0)
        assert (rotated_source.shape[1] > 0)

        if not target_image_prepadded:
            padded_target = nornir_imageregistration.phasecorrelation.PadImageForPhaseCorrelation(im_target,
                                                                                                  ImageMedian=target_stats.median,
                                                                                                  ImageStdDev=target_stats.std,
                                                                                                  MinOverlap=min_overlap)
        else:
            padded_target = im_target

        # print str(padded_target.shape) + ' ' +  str(rotated_padded_source.shape)

        TargetHeight = max([padded_target.shape[0], rotated_source.shape[0]])
        TargetWidth = max([padded_target.shape[1], rotated_source.shape[1]])

        # Why is MinOverlap hard-coded to 1.0?
        # PadImageForPhaseCorrelation will always return a copy, so don't call it unless we need to
        if not np.array_equal(im_target.shape, np.array((TargetHeight, TargetWidth))):
            padded_target = nornir_imageregistration.phasecorrelation.PadImageForPhaseCorrelation(im_target,
                                                                                                  NewWidth=TargetWidth,
                                                                                                  NewHeight=TargetHeight,
                                                                                                  ImageMedian=target_stats.median,
                                                                                                  ImageStdDev=target_stats.std,
                                                                                                  MinOverlap=1.0)
            # print(f"{angle}: Padding target image to {padded_target.shape}")
        # else:
        #     print(f"{angle}: No additional padding   {padded_target.shape}")

        if np.array_equal(rotated_source.shape, np.array((TargetHeight, TargetWidth))):
            rotated_padded_source = rotated_source
        else:
            rotated_padded_source = nornir_imageregistration.phasecorrelation.PadImageForPhaseCorrelation(
                rotated_source,
                NewWidth=TargetWidth,
                NewHeight=TargetHeight,
                ImageMedian=source_stats.median,
                ImageStdDev=source_stats.std,
                MinOverlap=1.0)

        assert (np.array_equal(padded_target.shape, rotated_padded_source.shape))

        # if OKToDelimWarped:
        del im_source
        del im_target

        del rotated_source

        # if use_cp and not isinstance(padded_target, cp.ndarray):
        #     padded_target = cp.asarray(padded_target)
        #
        # if use_cp and not isinstance(rotated_padded_source, cp.ndarray):
        #     rotated_padded_source = cp.asarray(rotated_padded_source)

        correlation_image = nornir_imageregistration.phasecorrelation.ImagePhaseCorrelation(padded_target,
                                                                                            rotated_padded_source,
                                                                                            target_stats.mean,
                                                                                            source_stats.mean,
                                                                                            correlation_coefficient=1)

        del padded_target
        del rotated_padded_source

        correlation_image = xp_scipy.fft.fftshift(correlation_image)
        try:
            correlation_image -= correlation_image.min()
            # correlation_image /= correlation_image.max()
        except FloatingPointError as e:
            print(f"Floating point error: {e} for {correlation_image.min()} or {correlation_image.max()}")
            record = nornir_imageregistration.AlignmentRecord((0, 0), 0, 0)
            return record

        # Timer.Start('Find Peak')

        # Note - Clement: overlap_mask still uses numpy (and not cupy)
        overlap_mask = nornir_imageregistration.overlapmasking.GetOverlapMask(target_image_shape, source_image_shape,
                                                                              correlation_image.shape, min_overlap,
                                                                              MaxOverlap=1.0)
        if use_cp and not isinstance(overlap_mask, cp.ndarray):
            overlap_mask = cp.asarray(overlap_mask)

        (peak, weight) = nornir_imageregistration.phasecorrelation.FindPeak(correlation_image, overlap_mask)
        del overlap_mask
        del correlation_image

        record = nornir_imageregistration.AlignmentRecord(peak, weight, angle)
        return record
    finally:
        nornir_imageregistration.close_shared_memory(target_original)
        nornir_imageregistration.close_shared_memory(source_original)


def GetFixedAndWarpedImageStats(imFixed, imWarped):
    tpool = nornir_pools.GetGlobalThreadPool()

    fixedStatsTask = tpool.add_task('FixedStats', nornir_imageregistration.ImageStats.CalcStats, imFixed)
    warpedStats = nornir_imageregistration.ImageStats.CalcStats(imWarped)

    fixedStats = fixedStatsTask.wait_return()

    return fixedStats, warpedStats


def _find_best_angle(source_image: NDArray[np.floating],
                     target_image: NDArray[np.floating],
                     source_stats: nornir_imageregistration.ImageStats,
                     target_stats: nornir_imageregistration.ImageStats,
                     angle_range: NDArray[float],
                     min_overlap: float = 0.75,
                     SingleThread: bool = False,
                     use_cluster: bool = False):
    """Find the best angle to align two images.  This function can be very memory intensive.
       Setting SingleThread=True makes debugging easier"""

    try:
        Debug = False
        pool = None
        use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

        # Temporarily disable until we have  cluster pool working again.  Leaving this on eliminates shared memory which is a big optimization
        use_cluster = False

        if len(angle_range) <= 1:
            SingleThread = True

        if not SingleThread:
            if nornir_imageregistration.in_debug_mode():
                pool = nornir_pools.GetGlobalSerialPool()
            elif use_cluster:
                pool = nornir_pools.GetGlobalClusterPool()
            else:
                pool = nornir_pools.GetGlobalMultithreadingPool()

        # Preallocate lists to store results of each angle
        AngleMatchValues = list()  # type:  list[AlignmentRecord | None]
        taskList = list()  # type:  list[nornir_pools.Task | None]

        #    MaxRotatedDimension = max([max(imFixed), max(imWarped)]) * 1.4143
        #    MinRotatedDimension = max(min(imFixed), min(imWarped))
        #
        #    SmallPaddedFixed = PadImageForPhaseCorrelation(imFixed, MaxOffset=0.1)
        #    LargePaddedFixed = PadImageForPhaseCorrelation(imFixed, MaxOffset=0.1)

        padded_target = nornir_imageregistration.phasecorrelation.PadImageForPhaseCorrelation(target_image,
                                                                                              MinOverlap=min_overlap,
                                                                                              ImageMedian=target_stats.median,
                                                                                              ImageStdDev=target_stats.std)

        # Create a shared read-only memory map for the Padded fixed image

        if not (use_cluster or SingleThread):
            # temp_padded_fixed_memmap = nornir_imageregistration.CreateTemporaryReadonlyMemmapFile(padded_target)
            # temp_shared_warp_memmap = nornir_imageregistration.CreateTemporaryReadonlyMemmapFile(imWarped)

            # temp_padded_fixed_memmap.mode = 'r'  # We do not want functions we pass the memmap modifying the original data
            # temp_shared_warp_memmap.mode = 'r'  # We do not want functions we pass the memmap modifying the original data

            shared_target_metadata, shared_padded_target = nornir_imageregistration.npArrayToSharedArray(padded_target)
            shared_source_metadata, shared_source = nornir_imageregistration.npArrayToSharedArray(source_image
                                                                                                  )
            # shared_padded_target = np.save(padded_target, )
        else:
            shared_target_metadata = None
            shared_source_metadata = None
            shared_padded_target = padded_target.astype(nornir_imageregistration.default_image_dtype(),
                                                        copy=False) if not use_cp else cp.array(padded_target,
                                                                                                nornir_imageregistration.default_image_dtype())
            shared_source = source_image.astype(nornir_imageregistration.default_image_dtype(),
                                                copy=False) if not use_cp else cp.array(source_image,
                                                                                        nornir_imageregistration.default_image_dtype())

        CheckTaskInterval = 16

        source_shape = source_image.shape
        target_shape = target_image.shape
        max_task_count = multiprocessing.cpu_count() * 1.5

        for i, theta in enumerate(angle_range):
            if SingleThread:
                record = ScoreOneAngle(target_original=shared_padded_target, source_original=shared_source,
                                       target_image_shape=target_shape, source_image_shape=source_shape,
                                       angle=theta,
                                       target_stats=target_stats, source_stats=source_stats,
                                       min_overlap=min_overlap)
                AngleMatchValues.append(record)
            elif use_cluster:
                task = pool.add_task(str(theta), ScoreOneAngle,
                                     target_original=shared_padded_target, source_original=shared_source,
                                     target_image_shape=target_shape, source_image_shape=source_shape,
                                     angle=theta,
                                     target_stats=target_stats, source_stats=source_stats,
                                     min_overlap=min_overlap)
                taskList.append(task)
            else:
                task = pool.add_task(str(theta), ScoreOneAngle,
                                     target_original=shared_target_metadata, source_original=shared_source_metadata,
                                     target_image_shape=target_shape, source_image_shape=source_shape,
                                     angle=theta,
                                     target_stats=target_stats, source_stats=source_stats,
                                     min_overlap=min_overlap)
                taskList.append(task)

            if not i % CheckTaskInterval == 0:
                continue

            # I don't like this, but it lets me delete tasks before filling the queue which may save some memory.
            # No sense checking unless we've already filled the queue though
            if len(taskList) > max_task_count:
                for iTask in range(len(taskList) - 1, -1, -1):
                    if taskList[iTask].iscompleted:
                        record = taskList[iTask].wait_return()
                        AngleMatchValues.append(record)
                        del taskList[iTask]

            # TestOneAngle(shared_padded_target, shared_source, angle, None, MinOverlap)

        # taskList.sort(key=tpool.Task.name)

        while len(taskList) > 0:
            for iTask in range(len(taskList) - 1, -1, -1):
                if taskList[iTask].iscompleted:
                    record = taskList[iTask].wait_return()
                    AngleMatchValues.append(record)
                    del taskList[iTask]

            if len(taskList) > 0:
                # Wait a bit before checking the task list
                sleep(0.5)

            # print(str(record.angle) + ' = ' + str(record.peak) + ' weight: ' + str(record.weight) + '\n')

            # ShowGrayscale(NormCorrelationImage)

        # print(str(AngleMatchValues))

        # Delete the pool to ensure extra python threads do not stick around
        # if pool is not None:
        #    pool.shutdown()

        del padded_target

        BestMatch = max(AngleMatchValues, key=nornir_imageregistration.AlignmentRecord.WeightKey)
        return BestMatch
    finally:

        if shared_target_metadata is not None:
            nornir_imageregistration.unlink_shared_memory(shared_target_metadata)
        if shared_source_metadata is not None:
            nornir_imageregistration.unlink_shared_memory(shared_source_metadata)

            # os.remove(temp_shared_warp_memmap.path)
            # os.remove(temp_padded_fixed_memmap.path)


def __ExecuteProfiler():
    SliceToSliceBruteForce('C:/Src/Git/nornir-testdata/Images/0162_ds32.png',
                           'C:/Src/Git/nornir-testdata/Images/0164_ds32.png',
                           AngleSearchRange=list(range(-175, -174, 1)),
                           SingleThread=True)


if __name__ == '__main__':
    from nornir_shared import misc

    misc.RunWithProfiler("__ExecuteProfiler()", r"C:\Temp\StosBrute")
    # __ExecuteProfiler()
    pass
