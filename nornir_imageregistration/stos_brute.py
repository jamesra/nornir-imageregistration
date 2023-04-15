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

import nornir_imageregistration
from numpy.fft import fftshift 
 
import nornir_pools
import numpy as np
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
                           TestFlip: bool=True) -> nornir_imageregistration.AlignmentRecord:
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
        
    # logger = logging.getLogger(__name__ + '.SliceToSliceBruteForce')
    
    WarpedImageScalingRequired = False
    if WarpedImageScaleFactors is not None:
        if hasattr(WarpedImageScaleFactors, '__iter__'):
            WarpedImageScaleFactors = nornir_imageregistration.EnsurePointsAre1DNumpyArray(WarpedImageScaleFactors)
            WarpedImageScalingRequired = any(WarpedImageScaleFactors != 1)
        else:
            WarpedImageScalingRequired = WarpedImageScaleFactors != 1
            WarpedImageScaleFactors = nornir_imageregistration.EnsurePointsAre1DNumpyArray([WarpedImageScaleFactors, WarpedImageScaleFactors])

    imFixed = None
    if isinstance(FixedImageInput, str):
        imFixed = nornir_imageregistration.LoadImage(FixedImageInput, dtype=np.float32)
        imFixedMask = None 
        if FixedImageMaskPath is not None:
            imFixedMask = nornir_imageregistration.LoadImage(FixedImageMaskPath, dtype=bool)

        imFixed = nornir_imageregistration.ReplaceImageExtremaWithNoise(imFixed, imagemask=imFixedMask, Copy=False)
    else:
        imFixed = FixedImageInput

    imWarped = None
    if isinstance(WarpedImageInput, str):
        imWarped = nornir_imageregistration.LoadImage(WarpedImageInput, dtype=np.float32)
        imWarpedMask = None
        if WarpedImageMaskPath is not None:
            imWarpedMask = nornir_imageregistration.LoadImage(WarpedImageMaskPath, dtype=bool)
        imWarped = nornir_imageregistration.ReplaceImageExtremaWithNoise(imWarped, imagemask=imWarpedMask, Copy=False)
        if WarpedImageScalingRequired:
            imWarped = nornir_imageregistration.ResizeImage(imWarped, WarpedImageScaleFactors)
    else:
        imWarped = WarpedImageInput

    scalar = 1.0
    if not LargestDimension is None:
        scalar = nornir_imageregistration.ScalarForMaxDimension(LargestDimension, [imFixed.shape, imWarped.shape])

    if scalar < 1.0:
        imFixed = nornir_imageregistration.ReduceImage(imFixed, scalar)
        imWarped = nornir_imageregistration.ReduceImage(imWarped, scalar)

    # Replace extrema with noise
    UserDefinedAngleSearchRange = not AngleSearchRange is None
    if not UserDefinedAngleSearchRange:
        AngleSearchRange = list(range(-178, 182, 2))

    BestMatch = FindBestAngle(imFixed, imWarped, AngleSearchRange, MinOverlap=MinOverlap, SingleThread=SingleThread, use_cluster=Cluster)

    IsFlipped = False
    if TestFlip:
        imWarpedFlipped = np.copy(imWarped)
        imWarpedFlipped = np.flipud(imWarpedFlipped)
    
        BestMatchFlipped = FindBestAngle(imFixed, imWarpedFlipped, AngleSearchRange, MinOverlap=MinOverlap, SingleThread=SingleThread, use_cluster=Cluster)
        BestMatchFlipped.flippedud = True

        # Determine if the best match is flipped or not
        IsFlipped = BestMatchFlipped.weight > BestMatch.weight
        
    if IsFlipped:
        imWarped = imWarpedFlipped
        BestMatch = BestMatchFlipped

    if not UserDefinedAngleSearchRange:
        BestRefinedMatch = FindBestAngle(imFixed, imWarped, [(x * 0.1) + BestMatch.angle - 1.9 for x in range(0, 18)], MinOverlap=MinOverlap, SingleThread=SingleThread)
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
            
            BestRefinedMatch = FindBestAngle(imFixed, imWarped, [(x * stepsize) + below for x in range(1, nSteps)], MinOverlap=MinOverlap, SingleThread=SingleThread)
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
                  FixedImageShape: NDArray, WarpedImageShape: NDArray,
                  angle: float,
                  fixedStats: nornir_imageregistration.ImageStats | None = None, warpedStats: nornir_imageregistration.ImageStats | None = None,
                  FixedImagePrePadded: bool=True, MinOverlap: float=0.75):
    '''Returns an alignment score for a fixed image and an image rotated at a specified angle'''

    imFixed = nornir_imageregistration.ImageParamToImageArray(imFixed_original, dtype=np.float32)
    imWarped = nornir_imageregistration.ImageParamToImageArray(imWarped_original, dtype=np.float32)

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
        imWarped = scipy.ndimage.rotate(imWarped.astype(np.float32, copy=False), axes=(0, 1), angle=-angle, cval=np.nan).astype(imWarped.dtype, copy=False) #Numpy cannot rotate float16 images
        imWarpedEmptyIndicies = np.isnan(imWarped)
        result = warpedStats.GenerateNoise(np.sum(imWarpedEmptyIndicies), dtype=imWarped.dtype)
        imWarped[imWarpedEmptyIndicies] = result
        np.clip(imWarped, a_min=warpedStats.min, a_max=warpedStats.max, out=imWarped)
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
    
    CorrelationImage = nornir_imageregistration.ImagePhaseCorrelation(PaddedFixed, RotatedPaddedWarped, fixedStats.mean, warpedStats.mean)

    del PaddedFixed
    del RotatedPaddedWarped

    CorrelationImage = scipy.fft.fftshift(CorrelationImage)
    try:
        CorrelationImage -= CorrelationImage.min()
        CorrelationImage /= CorrelationImage.max()
    except FloatingPointError as e:
        print(f"Floating point error: {e} for {CorrelationImage.min()} or {CorrelationImage.max()}")
        record = nornir_imageregistration.AlignmentRecord((0,0), 0, 0)
        return record

    # Timer.Start('Find Peak')

    OverlapMask = nornir_imageregistration.overlapmasking.GetOverlapMask(FixedImageShape, WarpedImageShape, CorrelationImage.shape, MinOverlap, MaxOverlap=1.0)
    
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


def FindBestAngle(imFixed: NDArray, imWarped: NDArray, AngleList: list[float] | None, MinOverlap: float = 0.75, SingleThread: bool = False, use_cluster: bool = False):
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

    (fixedStats, warpedStats) = GetFixedAndWarpedImageStats(imFixed, imWarped)

#    MaxRotatedDimension = max([max(imFixed), max(imWarped)]) * 1.4143
#    MinRotatedDimension = max(min(imFixed), min(imWarped))
#
#    SmallPaddedFixed = PadImageForPhaseCorrelation(imFixed, MaxOffset=0.1)
#    LargePaddedFixed = PadImageForPhaseCorrelation(imFixed, MaxOffset=0.1)

    PaddedFixed = nornir_imageregistration.PadImageForPhaseCorrelation(imFixed, MinOverlap=MinOverlap, ImageMedian=fixedStats.median, ImageStdDev=fixedStats.std)

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
        SharedPaddedFixed = PaddedFixed.astype(np.float32, copy=False)
        SharedWarped = imWarped.astype(np.float32, copy=False)

    CheckTaskInterval = 16

    fixed_shape = imFixed.shape
    warped_shape = imWarped.shape
    max_task_count = multiprocessing.cpu_count() * 1.5

    for i, theta in enumerate(AngleList): 
        if SingleThread:
            record = ScoreOneAngle(SharedPaddedFixed, SharedWarped, fixed_shape,
                                    warped_shape, theta, fixedStats=fixedStats,
                                    warpedStats=warpedStats,
                                    MinOverlap=MinOverlap)
            AngleMatchValues.append(record)
        elif use_cluster:
            task = pool.add_task(str(theta), ScoreOneAngle, SharedPaddedFixed, SharedWarped,
                                 fixed_shape, warped_shape, theta, fixedStats=fixedStats, warpedStats=warpedStats,
                                 MinOverlap=MinOverlap)
            taskList.append(task)
        else:
            task = pool.add_task(str(theta), ScoreOneAngle, shared_fixed_metadata, shared_warped_metadata, fixed_shape, warped_shape, theta, fixedStats=fixedStats, warpedStats=warpedStats, MinOverlap=MinOverlap)
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

    # print str(AngleMatchValues)
    
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
