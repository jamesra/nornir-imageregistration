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

import nornir_imageregistration.core as core
import nornir_pools
import numpy as np
import scipy.ndimage.interpolation as interpolation


# from memory_profiler import profile
def SliceToSliceBruteForce(FixedImageInput,
                           WarpedImageInput,
                           FixedImageMaskPath=None,
                           WarpedImageMaskPath=None,
                           LargestDimension=None,
                           AngleSearchRange=None,
                           MinOverlap=0.75,
                           SingleThread=False,
                           Cluster=False,
                           TestFlip=True):
    '''Given two images this function returns the rotation angle which best aligns them
       Largest dimension determines how large the images used for alignment should be'''

    logger = logging.getLogger(__name__ + '.SliceToSliceBruteForce')

    imFixed = None
    if isinstance(FixedImageInput, str):
        imFixed = core.LoadImage(FixedImageInput, FixedImageMaskPath)
    else:
        imFixed = FixedImageInput

    imWarped = None
    if isinstance(WarpedImageInput, str):
        imWarped = core.LoadImage(WarpedImageInput, WarpedImageMaskPath)
    else:
        imWarped = WarpedImageInput

    scalar = 1.0
    if not LargestDimension is None:
        scalar = core.ScalarForMaxDimension(LargestDimension, [imFixed.shape, imWarped.shape])

    if scalar < 1.0:
        imFixed = core.ReduceImage(imFixed, scalar)
        imWarped = core.ReduceImage(imWarped, scalar)

    # Replace extrema with noise
    imFixed = core.ReplaceImageExtramaWithNoise(imFixed, ImageMedian=0.5, ImageStdDev=0.25)
    imWarped = core.ReplaceImageExtramaWithNoise(imWarped, ImageMedian=0.5, ImageStdDev=0.25)

    UserDefinedAngleSearchRange = not AngleSearchRange is None
    if not UserDefinedAngleSearchRange:
        AngleSearchRange = list(range(-180, 180, 2))

    BestMatch = FindBestAngle(imFixed, imWarped, AngleSearchRange, MinOverlap=MinOverlap, SingleThread=SingleThread, Cluster=Cluster)

    IsFlipped = False
    if TestFlip:
        imWarpedFlipped = np.copy(imWarped)
        imWarpedFlipped = np.flipud(imWarpedFlipped)
    
        BestMatchFlipped = FindBestAngle(imFixed, imWarpedFlipped, AngleSearchRange, MinOverlap=MinOverlap, SingleThread=SingleThread, Cluster=Cluster)
        BestMatchFlipped.flippedud = True

        # Determine if the best match is flipped or not
        IsFlipped = BestMatchFlipped.weight > BestMatch.weight
        
    if IsFlipped:
        imWarped = imWarpedFlipped
        BestMatch = BestMatchFlipped

    if not UserDefinedAngleSearchRange:
        BestRefinedMatch = FindBestAngle(imFixed, imWarped, [(x * 0.1) + BestMatch.angle - 1 for x in range(0, 20)], MinOverlap=MinOverlap, SingleThread=SingleThread)
        BestRefinedMatch.flippedud = IsFlipped
    else:
        BestRefinedMatch = BestMatch

    if scalar > 1.0:
        AdjustedPeak = (BestRefinedMatch.peak[0] * scalar, BestRefinedMatch.peak[1] * scalar)
        BestRefinedMatch = nornir_imageregistration.AlignmentRecord(AdjustedPeak, BestRefinedMatch.weight, BestRefinedMatch.angle, IsFlipped)

   # BestRefinedMatch.CorrectPeakForOriginalImageSize(imFixed.shape, imWarped.shape)

    return BestRefinedMatch


def ScoreOneAngle(imFixed, imWarped, FixedImageShape, WarpedImageShape, angle, fixedStats=None, warpedStats=None, FixedImagePrePadded=True, MinOverlap=0.75):
    '''Returns an alignment score for a fixed image and an image rotated at a specified angle'''

    imFixed = core.ImageParamToImageArray(imFixed)
    imWarped = core.ImageParamToImageArray(imWarped)

    # gc.set_debug(gc.DEBUG_LEAK)
    if fixedStats is None:
        fixedStats = core.ImageStats.CalcStats(imFixed)

    if warpedStats is None:
        warpedStats = core.ImageStats.CalcStats(imWarped)

    OKToDelimWarped = False 
    if angle != 0:
        imWarped = interpolation.rotate(imWarped, axes=(1, 0), angle=angle, cval=np.nan)
        imWarpedEmptyIndicies = np.isnan(imWarped)
        imWarped[imWarpedEmptyIndicies] = warpedStats.GenerateNoise(np.sum(imWarpedEmptyIndicies))
        OKToDelimWarped = True


    RotatedWarped = core.PadImageForPhaseCorrelation(imWarped, ImageMedian=warpedStats.median, ImageStdDev=warpedStats.std, MinOverlap=MinOverlap)

    assert(RotatedWarped.shape[0] > 0)
    assert(RotatedWarped.shape[1] > 0)

    if not FixedImagePrePadded:
        PaddedFixed = core.PadImageForPhaseCorrelation(imFixed, ImageMedian=fixedStats.median, ImageStdDev=fixedStats.std, MinOverlap=MinOverlap)
    else:
        PaddedFixed = imFixed

    # print str(PaddedFixed.shape) + ' ' +  str(RotatedPaddedWarped.shape)

    TargetHeight = max([PaddedFixed.shape[0], RotatedWarped.shape[0]])
    TargetWidth = max([PaddedFixed.shape[1], RotatedWarped.shape[1]])

    PaddedFixed = core.PadImageForPhaseCorrelation(imFixed, NewWidth=TargetWidth, NewHeight=TargetHeight, ImageMedian=fixedStats.median, ImageStdDev=fixedStats.std, MinOverlap=1.0)
    RotatedPaddedWarped = core.PadImageForPhaseCorrelation(RotatedWarped, NewWidth=TargetWidth, NewHeight=TargetHeight, ImageMedian=warpedStats.median, ImageStdDev=warpedStats.std, MinOverlap=1.0)

    #if OKToDelimWarped:
    del imWarped

    del imFixed

    del RotatedWarped

    assert(PaddedFixed.shape == RotatedPaddedWarped.shape)

    CorrelationImage = core.ImagePhaseCorrelation(PaddedFixed, RotatedPaddedWarped)

    del PaddedFixed
    del RotatedPaddedWarped

    CorrelationImage = fftshift(CorrelationImage)
    CorrelationImage -= CorrelationImage.min()
    CorrelationImage /= CorrelationImage.max()

    # Timer.Start('Find Peak')

    OverlapMask = nornir_imageregistration.overlapmasking.GetOverlapMask(FixedImageShape, WarpedImageShape, CorrelationImage.shape, MinOverlap, MaxOverlap=1.0)
    (peak, weight) = core.FindPeak(CorrelationImage, OverlapMask)
    del OverlapMask

    del CorrelationImage

    record = nornir_imageregistration.AlignmentRecord(peak, weight, angle)

    return record


def GetFixedAndWarpedImageStats(imFixed, imWarped):
    tpool = nornir_pools.GetGlobalThreadPool()

    fixedStatsTask = tpool.add_task('FixedStats', core.ImageStats.CalcStats, imFixed)
    warpedStats = core.ImageStats.CalcStats(imWarped)

    fixedStats = fixedStatsTask.wait_return()

    return (fixedStats, warpedStats)


def FindBestAngle(imFixed, imWarped, AngleList, MinOverlap=0.75, SingleThread=False, Cluster=False):
    '''Find the best angle to align two images.  This function can be very memory intensive.
       Setting SingleThread=True makes debugging easier'''

    Debug = False
    pool = None
    
    # Temporarily disable until we have  cluster pool working again.  Leaving this on eliminates shared memory which is a big optimization
    Cluster = False
    
    if len(AngleList) == 0:
        SingleThread = True

    if not SingleThread:
        if Debug:
            pool = nornir_pools.GetThreadPool(Poolname=None, num_threads=3)
        elif Cluster:
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

    PaddedFixed = core.PadImageForPhaseCorrelation(imFixed, MinOverlap=MinOverlap, ImageMedian=fixedStats.median, ImageStdDev=fixedStats.std)

    # Create a shared read-only memory map for the Padded fixed image

    if not (Cluster or SingleThread):
        temp_padded_fixed_memmap = core.CreateTemporaryReadonlyMemmapFile(PaddedFixed)
        temp_shared_warp_memmap = core.CreateTemporaryReadonlyMemmapFile(imWarped)

        temp_padded_fixed_memmap.mode = 'r' #We do not want functions we pass the memmap modifying the original data
        temp_shared_warp_memmap.mode = 'r' #We do not want functions we pass the memmap modifying the original data

        # SharedPaddedFixed = core.npArrayToReadOnlySharedArray(PaddedFixed)
        # SharedWarped = core.npArrayToReadOnlySharedArray(imWarped)
        # SharedPaddedFixed = np.save(PaddedFixed, )
    else:
        SharedPaddedFixed = PaddedFixed
        SharedWarped = imWarped

    CheckTaskInterval = 16

    fixed_shape = imFixed.shape
    warped_shape = imWarped.shape

    for i, theta in enumerate(AngleList):

        if SingleThread:
            record = ScoreOneAngle(SharedPaddedFixed, SharedWarped, fixed_shape, warped_shape, theta, fixedStats=fixedStats, warpedStats=warpedStats, MinOverlap=MinOverlap)
            AngleMatchValues.append(record)
        else:
            task = pool.add_task(str(theta), ScoreOneAngle, temp_padded_fixed_memmap, temp_shared_warp_memmap, fixed_shape, warped_shape, theta, fixedStats=fixedStats, warpedStats=warpedStats, MinOverlap=MinOverlap)
            taskList.append(task)

        if not i % CheckTaskInterval == 0:
            continue

        # I don't like this, but it lets me delete tasks before filling the queue which may save some memory.
        # No sense checking unless we've already filled the queue though
        if len(taskList) > multiprocessing.cpu_count() * 1.5:
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
    if pool is not None:
        pool.wait_completion()

    del PaddedFixed

    if not (Cluster or SingleThread):
        os.remove(temp_shared_warp_memmap.path)
        os.remove(temp_padded_fixed_memmap.path)
        # del SharedPaddedFixed
        # del SharedWarped

    BestMatch = max(AngleMatchValues, key=nornir_imageregistration.AlignmentRecord.WeightKey)
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
