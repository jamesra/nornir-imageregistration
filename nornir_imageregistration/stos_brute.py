'''
Created on Oct 4, 2012

@author: u0490822
'''
import ctypes
import logging
import multiprocessing.sharedctypes
import os

from numpy.fft import fftshift

import nornir_imageregistration.core as core
import nornir_pools as pools
import numpy as np
import scipy.ndimage.interpolation as interpolation

from time import sleep


# from memory_profiler import profile
def SliceToSliceBruteForce(FixedImageInput,
                           WarpedImageInput,
                           FixedImageMaskPath=None,
                           WarpedImageMaskPath=None,
                           LargestDimension=None,
                           AngleSearchRange=None,
                           MinOverlap=0.75):
    '''Given two images this function returns the rotation angle which best aligns them
       Largest dimension determines how large the images used for alignment should be'''

    logger = logging.getLogger('IrTools.SliceToSliceBruteForce')

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

    # Replace extrama with noise
    imFixed = core.ReplaceImageExtramaWithNoise(imFixed, ImageMedian=0.5, ImageStdDev=0.25)
    imWarped = core.ReplaceImageExtramaWithNoise(imWarped, ImageMedian=0.5, ImageStdDev=0.25)

    UserDefinedAngleSearchRange = not AngleSearchRange is None
    if not UserDefinedAngleSearchRange:
        AngleSearchRange = range(-180, 180, 2)

    BestMatch = FindBestAngle(imFixed, imWarped, AngleSearchRange)

    # Find the best match

    if not UserDefinedAngleSearchRange:
        BestRefinedMatch = FindBestAngle(imFixed, imWarped, [(x * 0.1) + BestMatch.angle - 1 for x in range(0, 20)])
    else:
        BestRefinedMatch = BestMatch

    BestRefinedMatch.peak = (BestRefinedMatch.peak[0] * scalar, BestRefinedMatch.peak[1] * scalar)
   # BestRefinedMatch.CorrectPeakForOriginalImageSize(imFixed.shape, imWarped.shape)

    return BestRefinedMatch



def ScoreOneAngle(imFixed, imWarped, angle, FixedImagePrePadded=True, MinOverlap=0.75):
    '''Returns an alignment score for a fixed image and an image rotated at a specified angle'''

    try:
        RotatedWarped = None
        OKToDelimWarped = False
        if angle != 0:
            imWarped = interpolation.rotate(imWarped, axes=(1, 0), angle=angle)
            OKToDelimWarped = True

        RotatedWarped = core.PadImageForPhaseCorrelation(imWarped, MinOverlap=MinOverlap)

        assert(RotatedWarped.shape[0] > 0)
        assert(RotatedWarped.shape[1] > 0)

        if not FixedImagePrePadded:
            PaddedFixed = core.PadImageForPhaseCorrelation(imFixed, MinOverlap=MinOverlap)
        else:
            PaddedFixed = imFixed

        # print str(PaddedFixed.shape) + ' ' +  str(RotatedPaddedWarped.shape)

        TargetHeight = max([PaddedFixed.shape[0], RotatedWarped.shape[0]])
        TargetWidth = max([PaddedFixed.shape[1], RotatedWarped.shape[1]])

        PaddedFixed = core.PadImageForPhaseCorrelation(imFixed, NewWidth=TargetWidth, NewHeight=TargetHeight, MinOverlap=1.0)
        RotatedPaddedWarped = core.PadImageForPhaseCorrelation(RotatedWarped, NewWidth=TargetWidth, NewHeight=TargetHeight, MinOverlap=1.0)

        if OKToDelimWarped:
            del imWarped

        del RotatedWarped

        assert(PaddedFixed.shape == RotatedPaddedWarped.shape)

        CorrelationImage = core.ImagePhaseCorrelation(PaddedFixed, RotatedPaddedWarped)

        del PaddedFixed
        del RotatedPaddedWarped

        CorrelationImage = fftshift(CorrelationImage)
        NormCorrelationImage = CorrelationImage - CorrelationImage.min()
        NormCorrelationImage /= NormCorrelationImage.max()

        del CorrelationImage

        # Timer.Start('Find Peak')
        (peak, weight) = core.FindPeak(NormCorrelationImage)

        del NormCorrelationImage

        record = core.AlignmentRecord(angle=angle, peak=peak, weight=weight)
    except Exception as e:
        print "Exception calculating angle: " + str(angle)
        print str(e)
        raise e

    return record


def FindBestAngle(imFixed, imWarped, AngleList, MinOverlap=0.75):
    '''Find the best angle to align two images.  This function can be very memory intensive'''

    SingleThread = False
    Debug = False

    pool = pools.GetGlobalMultithreadingPool()
    if Debug:
        pool = pools.GetThreadPool(Poolname=None, num_threads=3)

    AngleMatchValues = list()
    taskList = list()

#    MaxRotatedDimension = max([max(imFixed), max(imWarped)]) * 1.4143
#    MinRotatedDimension = max(min(imFixed), min(imWarped))
#
#    SmallPaddedFixed = PadImageForPhaseCorrelation(imFixed, MaxOffset=0.1)
#    LargePaddedFixed = PadImageForPhaseCorrelation(imFixed, MaxOffset=0.1)

    PaddedFixed = core.PadImageForPhaseCorrelation(imFixed, MinOverlap=MinOverlap)

    # Create a shared read-only memory map for the Padded fixed image
    SharedPaddedFixed = core.npArrayToReadOnlySharedArray(PaddedFixed)
    SharedWarped = core.npArrayToReadOnlySharedArray(imWarped)

    CheckTaskInterval = 16

    for i, theta in enumerate(AngleList):

        if SingleThread:
            record = ScoreOneAngle(SharedPaddedFixed, SharedWarped, theta, MinOverlap=MinOverlap)
            AngleMatchValues.append(record)
        else:
            task = pool.add_task(str(theta), ScoreOneAngle, SharedPaddedFixed, SharedWarped, theta, MinOverlap=MinOverlap)
            taskList.append(task)

        if not i % CheckTaskInterval == 0:
            continue

        # I don't like this, but it lets me delete tasks before filling the queue which may save some memory
        for iTask in range(len(taskList) - 1, 0, -1):
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

    del PaddedFixed
    del SharedPaddedFixed
    del SharedWarped

    BestMatch = max(AngleMatchValues, key=core.AlignmentRecord.WeightKey)
    return BestMatch


def __ExecuteProfiler():
    SliceToSliceBruteForce('C:/Src/Git/nornir-testdata/Images/0162_ds32.png',
                           'C:/Src/Git/nornir-testdata/Images/0164_ds32.png',
                           AngleSearchRange=range(-175, -174, 1))

if __name__ == '__main__':

    from nornir_shared import misc
    misc.RunWithProfiler("__ExecuteProfiler()", "C:\Temo\StosBrute.pr")
    # __ExecuteProfiler()
    pass
