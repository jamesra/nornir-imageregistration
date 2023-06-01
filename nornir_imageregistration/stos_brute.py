'''
Created on Oct 4, 2012

@author: u0490822
'''
import ctypes
import logging
import multiprocessing
import multiprocessing.sharedctypes
import os
from time import sleep

import cupyx.scipy.ndimage

import nornir_imageregistration
from numpy.fft import fftshift 
 
import nornir_pools
import numpy as np
import cupy as cp
import cupyx
from numpy.typing import NDArray
import scipy


# from memory_profiler import profile
def SliceToSliceBruteForce(FixedImageInput,
                           WarpedImageInput,
                           FixedImageMaskPath=None,
                           WarpedImageMaskPath=None,
                           LargestDimension=None,
                           AngleSearchRange: list[float] | None=None,
                           MinOverlap: float=0.75,
                           WarpedImageScaleFactors=None,
                           SingleThread: bool=False,
                           Cluster: bool=False,
                           TestFlip: bool=True,
                           use_cp: bool=False) -> nornir_imageregistration.AlignmentRecord:
    '''Given two images this function returns the rotation angle which best aligns them
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
       '''
    if AngleSearchRange is not None:
        if isinstance(AngleSearchRange, np.ndarray):
            AngleSearchRange = list(AngleSearchRange)
            
    SingleThread = True if use_cp else SingleThread
        
    # logger = logging.getLogger(__name__ + '.SliceToSliceBruteForce')
    
    WarpedImageScalingRequired = False
    if WarpedImageScaleFactors is not None:
        if hasattr(WarpedImageScaleFactors, '__iter__'):
            WarpedImageScaleFactors = nornir_imageregistration.EnsurePointsAre1DNumpyArray(WarpedImageScaleFactors)
            WarpedImageScalingRequired = any(WarpedImageScaleFactors != 1)
        else:
            WarpedImageScalingRequired = WarpedImageScaleFactors != 1
            WarpedImageScaleFactors = nornir_imageregistration.EnsurePointsAre1DNumpyArray([WarpedImageScaleFactors, WarpedImageScaleFactors])

    target_image_data = nornir_imageregistration.ImagePermutationHelper(FixedImageInput, FixedImageMaskPath)
    source_image_data = nornir_imageregistration.ImagePermutationHelper(WarpedImageInput, WarpedImageMaskPath)

    target_image = target_image_data.ImageWithMaskAsNoise
    source_image = source_image_data.ImageWithMaskAsNoise

    target_stats = target_image_data.Stats
    source_stats = source_image_data.Stats

    del target_image_data
    del source_image_data
    
    target_image = cp.asarray(target_image) if use_cp and not isinstance(target_image, cp.ndarray) else target_image
    source_image = cp.asarray(source_image) if use_cp and not isinstance(source_image, cp.ndarray) else source_image

    scalar = 1.0
    if LargestDimension is not None:
        scalar = nornir_imageregistration.ScalarForMaxDimension(LargestDimension, [target_image.shape, source_image.shape])

    if scalar < 1.0:
        target_image = nornir_imageregistration.ReduceImage(target_image, scalar)
        source_image = nornir_imageregistration.ReduceImage(source_image, scalar)

    # Replace extrema with noise
    UserDefinedAngleSearchRange = AngleSearchRange is not None
    if not UserDefinedAngleSearchRange:
        AngleSearchRange = list(range(-178, 182, 2))

    BestMatch = _find_best_angle(target_image, source_image,
                                 target_stats, source_stats,
                                 AngleSearchRange, MinOverlap=MinOverlap, SingleThread=SingleThread, use_cluster=Cluster, use_cp=use_cp)

    IsFlipped = False
    if TestFlip:
        #imWarpedFlipped = np.copy(source_image)
        imWarpedFlipped = np.flipud(source_image)

        BestMatchFlipped = _find_best_angle(target_image, imWarpedFlipped,
                                            target_stats, source_stats,
                                            AngleSearchRange, MinOverlap=MinOverlap,
                                            SingleThread=SingleThread, use_cluster=Cluster, use_cp=use_cp)
        BestMatchFlipped.flippedud = True

        # Determine if the best match is flipped or not
        IsFlipped = BestMatchFlipped.weight > BestMatch.weight

    if IsFlipped:
        imWarped = imWarpedFlipped
        BestMatch = BestMatchFlipped
    else:
        imWarped = source_image

    # Note Clement - the RefinedAngleSearch list below is not centered around the current best angle
    # Default angle search range every 2 degrees
    # Old RefinedAngleSearch list: [(x * 0.1) + BestMatch.angle - 1.9 for x in range(0, 18)]
    # New RefinedAngleSearch list (length 39): [(x * 0.1 + BestMatch.angle) for x in range(-19, 20)]
    # New optional RefinedAngleSearch list (length 18): [(x * 0.2 + BestMatch.angle) for x in range(-9, 10)]
    if not UserDefinedAngleSearchRange:
        BestRefinedMatch = _find_best_angle(target_image, imWarped,
                                            target_stats, source_stats,
                                            # [(x * 0.1) + BestMatch.angle - 1.9 for x in range(0, 18)],
                                            [(x * 0.1 + BestMatch.angle) for x in range(-19, 20)],
                                            MinOverlap=MinOverlap, SingleThread=SingleThread, use_cp=use_cp)
        BestRefinedMatch.flippedud = IsFlipped
    else:
        min_step_size = 0.25
        if len(AngleSearchRange) > 2:
            iMatch = AngleSearchRange.index(BestMatch.angle)
            iBelow = iMatch - 1 if iMatch - 1 >= 0 else len(AngleSearchRange) - 1
            iAbove = iMatch + 1 if iMatch + 1 < len(AngleSearchRange) else 0
            below = AngleSearchRange[iMatch-1] if iMatch - 1 >= 0 else AngleSearchRange[0] - np.abs(AngleSearchRange[1] - AngleSearchRange[0])
            above = AngleSearchRange[iMatch+1] if iMatch + 1 < len(AngleSearchRange) else AngleSearchRange[iMatch] + np.abs(AngleSearchRange[iMatch] - AngleSearchRange[iMatch-1]) 
            refine_search_range = above - below
            nSteps = 20
            stepsize = refine_search_range / nSteps 
            
            if stepsize < min_step_size:
                nSteps = int(refine_search_range / min_step_size)
                stepsize = refine_search_range / nSteps
            
            BestRefinedMatch = _find_best_angle(target_image, imWarped,
                                                target_stats, source_stats,
                                                [(x * stepsize) + below for x in range(1, nSteps)],
                                                MinOverlap=MinOverlap, SingleThread=SingleThread, use_cp=use_cp)
            BestRefinedMatch.flippedud = IsFlipped
        else:
            BestRefinedMatch = BestMatch
            BestRefinedMatch.flippedud = IsFlipped

    if scalar > 1.0:
        AdjustedPeak = (BestRefinedMatch.peak[0] * scalar, BestRefinedMatch.peak[1] * scalar)
        BestRefinedMatch = nornir_imageregistration.AlignmentRecord(AdjustedPeak, BestRefinedMatch.weight, BestRefinedMatch.angle, IsFlipped)
    
    if WarpedImageScalingRequired:
        # AdjustedPeak = BestRefinedMatch.peak * (1.0 / WarpedImageScaleFactors)
        BestRefinedMatch = nornir_imageregistration.AlignmentRecord(BestRefinedMatch.peak, BestRefinedMatch.weight, BestRefinedMatch.angle, IsFlipped, WarpedImageScaleFactors)

    # BestRefinedMatch.CorrectPeakForOriginalImageSize(imFixed.shape, imWarped.shape)

    return BestRefinedMatch


def ScoreOneAngle(imFixed_original: NDArray, imWarped_original: NDArray,
                  FixedImageShape: tuple[int, int], WarpedImageShape: tuple[int, int],
                  angle: float,
                  fixedStats: nornir_imageregistration.ImageStats | None = None, warpedStats: nornir_imageregistration.ImageStats | None = None,
                  FixedImagePrePadded: bool=True, MinOverlap: float=0.75, use_cp: bool | None = None):
    '''Returns an alignment score for a fixed image and an image rotated at a specified angle'''

    imFixed = nornir_imageregistration.ImageParamToImageArray(imFixed_original, dtype=nornir_imageregistration.default_image_dtype())
    imWarped = nornir_imageregistration.ImageParamToImageArray(imWarped_original, dtype=nornir_imageregistration.default_image_dtype())

    if use_cp is None:
        use_cp = isinstance(imFixed, cp.ndarray)
        
    xp = cp if use_cp else np
    rotate = cupyx.scipy.ndimage.rotate if use_cp else scipy.ndimage.rotate
    fft = cp.fft if use_cp else np.fft

    imFixed = cp.asarray(imFixed) if use_cp and not isinstance(imFixed, cp.ndarray) else imFixed
    imWarped = cp.asarray(imWarped) if use_cp  and not isinstance(imWarped, cp.ndarray)  else imWarped
 
    # gc.set_debug(gc.DEBUG_LEAK)
    if fixedStats is None:
        fixedStats = nornir_imageregistration.ImageStats.CalcStats(imFixed)

    if warpedStats is None:
        warpedStats = nornir_imageregistration.ImageStats.CalcStats(imWarped)

    OKToDelimWarped = False 
    if angle != 0:
        #This confused me for years, but the implementation of rotate calls affine_transform with
        #the rotation matrix.  However the docs for affine_transform state it needs to be called
        #with the inverse transform.  Hence negating the angle here.
        if use_cp:
            imWarped = rotate(imWarped, axes=(0, 1), angle=-angle, cval=np.nan)
        else:
            imWarped = rotate(imWarped.astype(np.float32, copy=False), axes=(0, 1), angle=-angle, cval=np.nan).astype(imWarped.dtype, copy=False) #Numpy cannot rotate float16 images
        imWarpedEmptyIndicies = xp.isnan(imWarped)
        imWarped[imWarpedEmptyIndicies] = warpedStats.GenerateNoise(xp.sum(imWarpedEmptyIndicies), dtype=imWarped.dtype, use_cp=use_cp, return_numpy=not use_cp)
        OKToDelimWarped = True

    RotatedWarped = nornir_imageregistration.PadImageForPhaseCorrelation(imWarped, ImageMedian=warpedStats.median, ImageStdDev=warpedStats.std, MinOverlap=MinOverlap)

    assert(RotatedWarped.shape[0] > 0)
    assert(RotatedWarped.shape[1] > 0)

    if not FixedImagePrePadded:
        PaddedFixed = nornir_imageregistration.PadImageForPhaseCorrelation(imFixed, ImageMedian=fixedStats.median, ImageStdDev=fixedStats.std, MinOverlap=MinOverlap)
    else:
        PaddedFixed = imFixed

    # print str(PaddedFixed.shape) + ' ' +  str(RotatedPaddedWarped.shape)

    TargetHeight = max([PaddedFixed.shape[0], RotatedWarped.shape[0]])
    TargetWidth = max([PaddedFixed.shape[1], RotatedWarped.shape[1]])

    # Why is MinOverlap hard-coded to 1.0?
    # PadImageForPhaseCorrelation will always return a copy, so don't call it unless we need to
    if False == np.array_equal(imFixed.shape, np.array((TargetHeight, TargetWidth))):
        PaddedFixed = nornir_imageregistration.PadImageForPhaseCorrelation(imFixed, NewWidth=TargetWidth, NewHeight=TargetHeight, ImageMedian=fixedStats.median, ImageStdDev=fixedStats.std, MinOverlap=1.0)
        
    if np.array_equal(RotatedWarped.shape, np.array((TargetHeight, TargetWidth))):
        RotatedPaddedWarped = RotatedWarped
    else:
        RotatedPaddedWarped = nornir_imageregistration.PadImageForPhaseCorrelation(RotatedWarped, NewWidth=TargetWidth, NewHeight=TargetHeight, ImageMedian=warpedStats.median, ImageStdDev=warpedStats.std, MinOverlap=1.0)

    assert (np.array_equal(PaddedFixed.shape, RotatedPaddedWarped.shape))

    # if OKToDelimWarped:
    del imWarped 
    del imFixed

    del RotatedWarped
    
    if use_cp and not isinstance(PaddedFixed, cp.ndarray):
        PaddedFixed = cp.asarray(PaddedFixed)
    
    if use_cp and not isinstance(RotatedPaddedWarped, cp.ndarray):
        RotatedPaddedWarped = cp.asarray(RotatedPaddedWarped)
 
    CorrelationImage = nornir_imageregistration.ImagePhaseCorrelation(PaddedFixed, RotatedPaddedWarped, fixedStats.mean, warpedStats.mean)

    del PaddedFixed
    del RotatedPaddedWarped

    CorrelationImage = fft.fftshift(CorrelationImage)
    try:
        CorrelationImage -= CorrelationImage.min()
        CorrelationImage /= CorrelationImage.max()
    except FloatingPointError as e:
        print(f"Floating point error: {e} for {CorrelationImage.min()} or {CorrelationImage.max()}")
        record = nornir_imageregistration.AlignmentRecord((0,0), 0, 0)
        return record

    # Timer.Start('Find Peak')

    OverlapMask = nornir_imageregistration.overlapmasking.GetOverlapMask(FixedImageShape, WarpedImageShape, CorrelationImage.shape, MinOverlap, MaxOverlap=1.0)
    if use_cp and not isinstance(OverlapMask, cp.ndarray):
        OverlapMask = cp.asarray(OverlapMask)

    (peak, weight) = nornir_imageregistration.FindPeak(CorrelationImage, OverlapMask)
    del OverlapMask
    del CorrelationImage
    
    nornir_imageregistration.close_shared_memory(imFixed_original)
    nornir_imageregistration.close_shared_memory(imWarped_original)

    record = nornir_imageregistration.AlignmentRecord(peak, weight, angle) 
    return record


def GetFixedAndWarpedImageStats(imFixed, imWarped):
    tpool = nornir_pools.GetGlobalThreadPool()

    fixedStatsTask = tpool.add_task('FixedStats', nornir_imageregistration.ImageStats.CalcStats, imFixed)
    warpedStats = nornir_imageregistration.ImageStats.CalcStats(imWarped)

    fixedStats = fixedStatsTask.wait_return()

    return (fixedStats, warpedStats)


def _find_best_angle(imFixed: NDArray[float],
                     imWarped: NDArray[float],
                     fixed_stats: nornir_imageregistration.ImageStats,
                     warped_stats: nornir_imageregistration.ImageStats,
                     AngleList: list[float] | None,
                     MinOverlap: float = 0.75,
                     SingleThread: bool = False,
                     use_cluster: bool = False,
                     use_cp: bool = False):
    '''Find the best angle to align two images.  This function can be very memory intensive.
       Setting SingleThread=True makes debugging easier'''

    Debug = False
    pool = None
    
    # Temporarily disable until we have  cluster pool working again.  Leaving this on eliminates shared memory which is a big optimization
    use_cluster = False
    
    if len(AngleList) <= 1:
        SingleThread = True

    if not SingleThread:
        if Debug:
            pool = nornir_pools.GetThreadPool(Poolname=None, num_threads=3)
        elif use_cluster:
            pool = nornir_pools.GetGlobalClusterPool()
        else:
            pool = nornir_pools.GetGlobalMultithreadingPool()

    AngleMatchValues = list()
    taskList = list()

#    MaxRotatedDimension = max([max(imFixed), max(imWarped)]) * 1.4143
#    MinRotatedDimension = max(min(imFixed), min(imWarped))
#
#    SmallPaddedFixed = PadImageForPhaseCorrelation(imFixed, MaxOffset=0.1)
#    LargePaddedFixed = PadImageForPhaseCorrelation(imFixed, MaxOffset=0.1)

    PaddedFixed = nornir_imageregistration.PadImageForPhaseCorrelation(imFixed,
                                                                       MinOverlap=MinOverlap,
                                                                       ImageMedian=fixed_stats.median,
                                                                       ImageStdDev=fixed_stats.std)

    # Create a shared read-only memory map for the Padded fixed image

    if not (use_cluster or SingleThread):
        #temp_padded_fixed_memmap = nornir_imageregistration.CreateTemporaryReadonlyMemmapFile(PaddedFixed)
        #temp_shared_warp_memmap = nornir_imageregistration.CreateTemporaryReadonlyMemmapFile(imWarped)

        #temp_padded_fixed_memmap.mode = 'r'  # We do not want functions we pass the memmap modifying the original data
        #temp_shared_warp_memmap.mode = 'r'  # We do not want functions we pass the memmap modifying the original data

        shared_fixed_metadata, SharedPaddedFixed = nornir_imageregistration.npArrayToReadOnlySharedArray(PaddedFixed)
        shared_warped_metadata, SharedWarped = nornir_imageregistration.npArrayToReadOnlySharedArray(imWarped)
        # SharedPaddedFixed = np.save(PaddedFixed, )
    else:
        SharedPaddedFixed = PaddedFixed.astype(nornir_imageregistration.default_image_dtype(), copy=False) if not use_cp else cp.array(PaddedFixed, nornir_imageregistration.default_image_dtype())
        SharedWarped = imWarped.astype(nornir_imageregistration.default_image_dtype(), copy=False) if not use_cp else cp.array(imWarped, nornir_imageregistration.default_image_dtype())

    CheckTaskInterval = 16

    fixed_shape = imFixed.shape
    warped_shape = imWarped.shape
    max_task_count = multiprocessing.cpu_count() * 1.5

    for i, theta in enumerate(AngleList): 
        if SingleThread:
            record = ScoreOneAngle(SharedPaddedFixed, SharedWarped, fixed_shape,
                                    warped_shape, theta, fixedStats=fixed_stats,
                                    warpedStats=warped_stats,
                                    MinOverlap=MinOverlap,
                                    use_cp=use_cp)
            AngleMatchValues.append(record)
        elif use_cluster:
            task = pool.add_task(str(theta), ScoreOneAngle, SharedPaddedFixed, SharedWarped,
                                 fixed_shape, warped_shape, theta, fixedStats=fixed_stats, warpedStats=warped_stats,
                                 MinOverlap=MinOverlap,
                                 use_cp=use_cp)
            taskList.append(task)
        else:
            task = pool.add_task(str(theta), ScoreOneAngle, shared_fixed_metadata, shared_warped_metadata,
                                 fixed_shape, warped_shape,
                                 theta,
                                 fixedStats=fixed_stats, warpedStats=warped_stats, MinOverlap=MinOverlap,
                                 use_cp=use_cp)
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

        # TestOneAngle(SharedPaddedFixed, SharedWarped, angle, None, MinOverlap)

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

    del PaddedFixed

    BestMatch = max(AngleMatchValues, key=nornir_imageregistration.AlignmentRecord.WeightKey)
    
    if not (use_cluster or SingleThread):
        nornir_imageregistration.unlink_shared_memory(shared_fixed_metadata)
        nornir_imageregistration.unlink_shared_memory(shared_warped_metadata)
        #os.remove(temp_shared_warp_memmap.path)
        #os.remove(temp_padded_fixed_memmap.path)
        
    return BestMatch


def __ExecuteProfiler():
    SliceToSliceBruteForce('C:/Src/Git/nornir-testdata/Images/0162_ds32.png',
                           'C:/Src/Git/nornir-testdata/Images/0164_ds32.png',
                            AngleSearchRange=list(range(-175, -174, 1)),
                            SingleThread=True)


if __name__ == '__main__':
    from nornir_shared import misc
    misc.RunWithProfiler("__ExecuteProfiler()", "C:\Temp\StosBrute")
    # __ExecuteProfiler()
    pass
