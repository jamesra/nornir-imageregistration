'''
Created on Jun 26, 2012

@author: James Anderson
'''

from collections import deque
import logging
import multiprocessing
import os
import subprocess
import sys

import numpy
from pylab import median, mean, std, sqrt, imread, ceil, floor, mod
import scipy.misc
import scipy.ndimage
import scipy.ndimage.measurements
import scipy.stats

import nornir_pools
import nornir_shared.histogram
import nornir_shared.images as images
import nornir_shared.prettyoutput as PrettyOutput

import nornir_imageregistration.core
import nornir_imageregistration.im_histogram_parser  
from numpy import int32
import matplotlib.pyplot as plt


class ImageStats():
    '''A container for image statistics'''
    
    @property
    def median(self):
        return self._median

    @median.setter
    def median(self, val):
        self._median = val
        
    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, val):
        self._mean = val

    @property
    def std(self):
        return self._std
    
    @std.setter
    def std(self, val):
        self._std = val
    
    @property
    def min(self):
        return self._min
    
    @min.setter
    def min(self, val):
        self._min = val
    
    @property
    def max(self):
        return self._max
    
    @max.setter
    def max(self, val):
        self._max = val

    def __init__(self):
        self._median = None
        self._mean = None
        self._stddev = None

    def __getstate__(self):
        d = {}
        d['_median'] = self._median 
        d['_mean'] = self._mean
        d['_std'] = self._std
        d['_min'] = self._min
        d['_max'] = self._max
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        
    @classmethod
    def CalcStats(cls, image):
        return ImageStats.Create(image)

    @classmethod
    def Create(cls, image):
        '''Returns an object with the mean,median,std.dev of an image,
           this object is attached to the image object and only calculated once'''

#        I removed this cache in the image object of the statistics.  I believe 
#        Python 3 had issues with it.  If there are performance problems we 
#        should add it back
#         try:
#             cachedVal = image.__IrToolsImageStats__
#             if cachedVal is not None:
#                 return cachedVal
#         except AttributeError:
#             pass
        
        obj = ImageStats()
        if image.dtype is not numpy.float64: #Use float 64 to ensure accurate statistical results
            image = image.astype(dtype=numpy.float64)
            
        flatImage = image.flat 
        obj._median = numpy.median(flatImage)
        obj._mean = numpy.mean(flatImage)
        obj._std = numpy.std(flatImage)
        obj._max = numpy.max(flatImage)
        obj._min = numpy.min(flatImage)
        
        del flatImage
         
#        image.__IrtoolsImageStats__ = obj
        return obj
    
    def GenerateNoise(self, shape):
        '''
        Generate random data of shape with the specified mean and standard deviation.  Returned values will not be less than min or greater than max
        :param array shape: Shape of the returned array 
        '''
        data = (numpy.random.randn(shape.astype(numpy.int64)).astype(numpy.float32) * self.std) + self.median
    
        if self.median - (self.std * 2) < self.min:
            data[data < self.min] = self.min
        
        if self.median + (self.std * 2) > self.max:
            data[data > self.max] = self.max
             
        return data


def Prune(filenames, MaxOverlap=None):

    if isinstance(filenames, str):
        listfilenames = [filenames]
    else:
        listfilenames = filenames

    # logger = logging.getLogger('irtools.prune')

    if MaxOverlap is None:
        MaxOverlap = 0

    assert isinstance(listfilenames, list)

    FilenameToResult = __InvokeFunctionOnImageList__(listfilenames, Function=__PruneFileSciPy__, MaxOverlap=MaxOverlap)

    # Convert results to a float
    for k in FilenameToResult.keys():
        FilenameToResult[k] = float(FilenameToResult[k])

    if isinstance(filenames, str):
        return list(FilenameToResult.items())[0]
    else:
        return FilenameToResult

def __InvokeFunctionOnImageList__(listfilenames, Function=None, Pool=None, **kwargs):
    '''Return a number indicating how interesting the image is using SciPy
       '''

    if Pool is None:
        TPool = nornir_pools.GetGlobalMultithreadingPool()
    else:
        TPool = Pool

    TileToScore = dict()
    tasklist = []
    for filename in listfilenames:
        task = TPool.add_task('Calc Feature Score: ' + os.path.basename(filename), Function, filename, **kwargs)
        task.filename = filename
        tasklist.append(task)

    TPool.wait_completion()

    numTasks = len(tasklist)
    iTask = 0
    for task in tasklist:
        Result = task.wait_return()
        iTask = iTask + 1
        if Result is None:
            PrettyOutput.LogErr('No return value for ' + task.filename)
            continue

#         if Result[0] is None:
#             PrettyOutput.LogErr('No filename for ' + task.name)
#             continue

        PrettyOutput.CurseProgress("ImageStats", iTask, numTasks)

        filename = task.filename
        TileToScore[filename] = Result

    return TileToScore


def __PruneFileSciPyV2__(image, cell_size=None, **kwargs):
    
    Im = nornir_imageregistration.ImageParamToImageArray(image, dtype=numpy.float32)
    Im_filtered = scipy.ndimage.filters.median_filter(Im, size=3)
    sx = scipy.ndimage.sobel(Im_filtered, axis=0, mode='nearest')
    sy = scipy.ndimage.sobel(Im_filtered, axis=1, mode='nearest')
    sob = numpy.hypot(sx,sy)
# #     
#     fft = numpy.fft.rfft2(Im_filtered)
#     fft = numpy.fft.fftshift(fft)
#     amp = numpy.abs(fft)
#     
# #     
#     logamp = numpy.log(amp) ** 2 
#     logampflat = numpy.asarray(logamp.flat)
#     aboveMedian = numpy.median(logampflat)
#     score = numpy.mean(logampflat[logampflat > aboveMedian])

    #score = numpy.max(Im_filtered.flat) - numpy.min(Im_filtered.flat)
    
    #score = numpy.var(Im_filtered.flat)
    #score = numpy.percentile(sob.flat, 90)
    #score = numpy.max(sob.flat)
    score = numpy.mean(sob.flat)
    #score = numpy.median(sob.flat) - numpy.percentile(sob.flat, 10)
    #mode = numpy.stats.mode(sob.flat)
    
    #p10 = numpy.percentile(Im_filtered.flat, 10)
    #p90 = numpy.percentile(Im_filtered.flat, 90)
    #med = numpy.median(sob.flat)
    
    #score = (p90 - p10)
    
    #score = numpy.median(sob.flat) - numpy.percentile(sob.flat, 10))
    
#     if score < .025:
#         nornir_imageregistration.ShowGrayscale([Im, Im_filtered, sob], title=str(score))
#         plt.figure()
#         plt.hist(sob.flat, bins=100)
#         a = 4
    return score

#     finite_subset = numpy.asarray(Im[numpy.isfinite(Im)].flat, dtype=numpy.float32)
#     if len(finite_subset) < 3:
#         return 0
# 
#     return numpy.std(finite_subset)
         
    if cell_size is None:
        cell_size = numpy.max(numpy.vstack((numpy.asarray(numpy.asarray(Im.shape) / 64, dtype=numpy.int32), numpy.asarray((64,64),dtype=numpy.int32))),0) 
     
    grid = nornir_imageregistration.CenteredGridDivision(Im.shape, cell_size=cell_size)
     
    StdDevList = []
     
    for iPoint in range(0, grid.num_points):
        rect = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(grid.SourcePoints[iPoint,:], grid.cell_size)
        subset = nornir_imageregistration.CropImageRect(image, rect,cval=numpy.nan)
        finite_subset = subset[numpy.isfinite(subset)].flat
        if len(finite_subset) < 3:
            continue
         
        std_val = numpy.std(numpy.asarray(finite_subset, dtype=numpy.float32))
        StdDevList.append(std_val)
         
    #val = numpy.percentile(StdDevList, q=90)
 
    #val = numpy.max(StdDevList)
    #val = numpy.mean(StdDevList)
    val = numpy.median(StdDevList)    
    return val
    

def __PruneFileSciPy__(filename, MaxOverlap=0.15, **kwargs):
    '''Returns a prune score for a single file
        Args:
           MaxOverlap = 0 to 1'''

    #TODO: This function should be updated to use the grid_subdivision module to create cells.  It should be used in the mosaic tile translation code to eliminate featureless 
    #overlap regions of adjacent tiles

    # logger = logging.getLogger('irtools.prune')
    # logger = multiprocessing.log_to_stderr()

    if MaxOverlap > 0.5:
        MaxOverlap = 0.5
# 
#     if not os.path.exists(filename):
#         # logger.error(filename + ' not found when attempting prune')
#         # PrettyOutput.LogErr(filename + ' not found when attempting prune')
#         return None

    Im = nornir_imageregistration.ImageParamToImageArray(filename)
    (Height, Width) = Im.shape

    StdDevList = []
    #MeanList = []

    MaxDim = Height
    if Width > Height:
        MaxDim = Width

    SampleSize = int(ceil(MaxDim / 32))

    VertOverlapPixelRange = int(MaxOverlap * float(Height))
    HorzOverlapPixelRange = int(MaxOverlap * float(Width))

    MaskTopBorder = VertOverlapPixelRange + (SampleSize - (mod(VertOverlapPixelRange, 32)))
    MaskBottomBorder = Height - VertOverlapPixelRange - (SampleSize - (mod(VertOverlapPixelRange, 32)))

    MaskLeftBorder = HorzOverlapPixelRange + (SampleSize - (mod(HorzOverlapPixelRange, 32)))
    MaskRightBorder = Width - HorzOverlapPixelRange - (SampleSize - (mod(HorzOverlapPixelRange, 32)))

    # Calculate the top
    for iHeight in range(0, MaskTopBorder - (SampleSize - 1), SampleSize):
        for iWidth in range(0, Width - 1, SampleSize):
            StdDev = numpy.std(Im[iHeight:iHeight + SampleSize, iWidth:iWidth + SampleSize])
            StdDevList.append(StdDev)
            # Im[iHeight:iHeight+SampleSize,iWidth:iWidth+SampleSize] = 0

    # Calculate the sides
    for iHeight in range(MaskTopBorder, MaskBottomBorder, SampleSize):
        for iWidth in range(0, MaskLeftBorder - (SampleSize - 1), SampleSize):
            StdDev = numpy.std(Im[iHeight:iHeight + SampleSize, iWidth:iWidth + SampleSize])
            StdDevList.append(StdDev)
            # Im[iHeight:iHeight+SampleSize,iWidth:iWidth+SampleSize] = 0.25

        for iWidth in range(MaskRightBorder, Width - SampleSize, SampleSize):
            StdDev = numpy.std(Im[iHeight:iHeight + SampleSize, iWidth:iWidth + SampleSize])
            StdDevList.append(StdDev)
            # Im[iHeight:iHeight+SampleSize,iWidth:iWidth+SampleSize] = 0.5

    # Calculate the bottom
    for iHeight in range(MaskBottomBorder, Height - SampleSize, SampleSize):
        for iWidth in range(0, Width - 1, SampleSize):
            StdDev = numpy.std(Im[iHeight:iHeight + SampleSize, iWidth:iWidth + SampleSize])
            StdDevList.append(StdDev)
            # Im[iHeight:iHeight+SampleSize,iWidth:iWidth+SampleSize] = 0.75

    del Im
    # nornir_imageregistration.core.ShowGrayscale(Im)
    return sum(StdDevList)

def Histogram(filenames, Bpp=None, Scale=None, numBins=None):
    '''Returns a single histogram built by combining histograms of all images
       If scale is not none the images are scaled before the histogram is collected'''

    if isinstance(filenames, str):
        listfilenames = [filenames]
    else:
        listfilenames = filenames

    numTiles = len(listfilenames)
    if numTiles == 0:
        return dict()

    if Bpp is None:
        Bpp = images.GetImageBpp(listfilenames[0])
        
    

    assert isinstance(listfilenames, list)

    FilenameToTask = {} 
    local_machine_pool = nornir_pools.GetGlobalLocalMachinePool()
    for f in listfilenames:
        (root, ext) = os.path.splitext(f)
        if ext == '.npy':
            task = __HistogramFileSciPy__(f, Bpp=Bpp, Scale=Scale)
        else:
            task = __HistogramFileImageMagick__(f, ProcPool=local_machine_pool, Bpp=Bpp, Scale=Scale)
        FilenameToTask[f] = task

    # maxVal = (1 << Bpp) - 1

    if numBins is None:
        numBins = 256
        if Bpp > 8:
            numBins = 1024
    else:
        assert(isinstance(numBins, int))

    local_machine_pool.wait_completion()

    OutputMap = {}
    minVal = None
    maxVal = None
    for f in list(FilenameToTask.keys()):
        task = FilenameToTask[f]
        taskOutput = task.wait_return()
        lines = taskOutput.splitlines()

        OutputMap[f] = lines

        (fminVal, fmaxVal) = nornir_imageregistration.im_histogram_parser .MinMaxValues(lines)
        if minVal is None:
            minVal = fminVal
        else:
            minVal = min(minVal, fminVal)

        if maxVal is None:
            maxVal = fmaxVal
        else:
            maxVal = max(maxVal, fmaxVal)

    threadTasks = []
     
    thread_pool = nornir_pools.GetGlobalThreadPool()
    for f in list(OutputMap.keys()):
        threadTask = thread_pool.add_task(f, nornir_imageregistration.im_histogram_parser .Parse, OutputMap[f], minVal=minVal, maxVal=maxVal, numBins=numBins)
        threadTasks.append(threadTask)
        
    HistogramComposite = nornir_shared.histogram.Histogram.Init(minVal=minVal, maxVal=maxVal, numBins=numBins)
    for t in threadTasks:
        hist = t.wait_return()
        HistogramComposite.AddHistogram(hist.Bins)
        # histogram = IMHistogramOutput.Parse(taskOutput, minVal=minVal, maxVal=maxVal, numBins=numBins)

        # FilenameToResult[f] = [histogram, None, None]

    del threadTasks

    # FilenameToResult = __InvokeFunctionOnImageList__(listfilenames, Function=__HistogramFileImageMagick__, Pool=nornir_pools.GetGlobalThreadPool(), ProcPool = nornir_pools.GetGlobalClusterPool(), Bpp=Bpp, Scale=Scale)#, NumSamples=SamplesPerImage)

#    maxVal = 1 << Bpp
#    numBins = 256
#    if Bpp > 8:
#        numBins = 1024


    # Sum all of the result arrays together
#    for filename in listfilenames:
#        if filename in FilenameToResult:
#            Result = FilenameToResult[filename]
#            histogram = Result[0]
# #
# #            if HistogramComposite is None:
# #                HistogramComposite = numpy.zeros(histogram.shape, dtype=numpy.int0)
# #
# #            HistogramComposite = numpy.add(HistogramComposite, histogram)
#
#            HistogramComposite.AddHistogram(histogram.Bins)


    return HistogramComposite


def __Get_Histogram_For_Image_From_ImageMagick(filename, Bpp=None, Scale=None):
    
    Cmd = __CreateImageMagickCommandLineForHistogram(filename, Scale) 
    raw_output = __HistogramFileImageMagick__(filename, ProcPool, Bpp, Scale)
    
    
 

def __HistogramFileSciPy__(filename, Bpp=None, NumSamples=None, numBins=None, Scale=None):
    '''Return the histogram of an image'''

    Im = None
    if isinstance(filename, str):
        Im = nornir_imageregistration.core.LoadImage(filename)
    else:
        Im = filename
        
    (Height, Width) = Im.shape
    NumPixels = Width * Height
    
    if(not Scale is None):
        if(Scale != 1.0):
            Im = scipy.misc.imresize(Im, size=Scale, interp='nearest') 

    # ImOneD = reshape(Im, Width * Height, 1)
    ImOneD = Im.flat

    if Bpp is None:
        Bpp = images.GetImageBpp(filename)

    if NumSamples is None:
        NumSamples = Height * Width
    elif NumSamples > Height * Width:
        NumSamples = Height * Width

    StepSize = int(float(NumPixels) / float(NumSamples))

    if StepSize > 1:
        Samples = numpy.random.random_integers(0, NumPixels - 1, NumSamples)
        ImOneD = ImOneD[Samples]

    if numBins is None:
        numBins = 256
        if Bpp > 8:
            numBins = 1024
    else:
        assert(isinstance(numBins, int))

    #[histogram_array, low_range, binsize] = numpy.histogram(ImOneD, bins=numBins, range =[0, 1])
    [histogram_array, bin_edges] = numpy.histogram(ImOneD, bins=numBins, range =[0, 1])
    
    histogram_obj = nornir_shared.histogram.Histogram.FromArray(histogram_array, bin_edges[0], bin_edges[1] - bin_edges[0])
    
    return histogram_obj

def __CreateImageMagickCommandLineForHistogram(filename, Scale):
    CmdTemplate = "magick convert %(filename)s -filter point -scale %(scale)g%% -define histogram:unique-colors=true -format %%c histogram:info:- && exit"
    return CmdTemplate % {'filename' : filename, 'scale' : Scale * 100}
    

def __HistogramFileImageMagick__(filename, ProcPool, Bpp=None, Scale=None):

    if Scale is None:
        Scale = 1

    # We only scale down, so if it is over 1 assume it is a percentage
    if Scale > 1:
        Scale = Scale / 100.0

    if Scale > 1:
        Scale = 1

    Cmd = __CreateImageMagickCommandLineForHistogram(filename, Scale)
    task = ProcPool.add_process(os.path.basename(filename), Cmd, shell=True)

    return task


# if __name__ == '__main__':
# 
#     Histogram = Histogram('C:\\Buildscript\\IrTools\\RawTile.png')
# 
#     import cProfile
#     import pstats
# 
#     score = Prune('C:\\Buildscript\\IrTools\\RawTile.png', 0.1)
#     PrettyOutput.Log("Score: " + str(score))
# 
#     ProfilePath = 'C:\\Buildscript\\IrTools\\BuildProfile.pr'
# 
#     ProfileDir = os.path.dirname(ProfilePath)
#     if not os.path.exists(ProfileDir):
# 
#         os.makedirs(ProfileDir)
# 
#     try:
#         cProfile.run("__PruneFileSciPy__('C:\\Buildscript\\IrTools\\RawTile.png', 0.1)", ProfilePath)
#     finally:
#         if not os.path.exists(ProfilePath):
#             PrettyOutput.LogErr("No profile file found" + ProfilePath)
#             sys.exit()
# 
#         pr = pstats.Stats(ProfilePath)
#         if not pr is None:
#             pr.sort_stats('time')
#             print(str(pr.print_stats(.05)))
# 





