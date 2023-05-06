'''
Created on Jun 26, 2012

@author: James Anderson
'''

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from numpy.typing import NDArray, DTypeLike
from PIL import Image

import numpy
import cupy as cp
from pylab import ceil, mod

import nornir_pools
import nornir_shared.histogram
import nornir_shared.images as images
import nornir_shared.prettyoutput as PrettyOutput

import nornir_imageregistration


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
        self._std = None
        self._min = None
        self._max = None
        
    def __str__(self):
        return f'mean: {self._mean} std: {self._std} min: {self._min} max: {self._max}'

    def __getstate__(self):
        d = {'_median': self._median, '_mean': self._mean, '_std': self._std, '_min': self._min, '_max': self._max}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        
    @classmethod
    def CalcStats(cls, image) -> self:
        return ImageStats.Create(image)

    @classmethod
    def Create(cls, image: NDArray) -> self:
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

        use_cp = isinstance(image, cp.ndarray)
        
        obj = ImageStats()
        image = nornir_imageregistration.ImageParamToImageArray(image, dtype=numpy.float64)
        #if image.dtype is not numpy.float64:  # Use float 64 to ensure accurate statistical results
        #    image = image.astype(dtype=numpy.float64)

        xp = cp if use_cp else numpy

        flatImage = image.ravel() if use_cp else image.flat

        obj._median = xp.median(flatImage)
        obj._mean = xp.mean(flatImage)
        obj._std = xp.std(flatImage)
        obj._max = xp.max(flatImage)
        obj._min = xp.min(flatImage)
        
        del flatImage
         
#        image.__IrtoolsImageStats__ = obj
        return obj
    
    def GenerateNoise(self, shape:np.ndarray, dtype: DTypeLike, use_cp: bool | None = None, return_numpy: bool = True):
        '''
        Generate random data of shape with the specified mean and standard deviation.  Returned values will not be less than min or greater than max
        :param array shape: Shape of the returned array 
        '''
        
        size = None
        height = 1
        width = 1
        one_d_result = False
        if isinstance(shape, int) or isinstance(shape, np.int32):
            size = shape
            height = shape
            width = 1
            one_d_result = True
        elif isinstance(shape, np.ndarray) or isinstance(shape, cp.ndarray):
            shape_shape = shape.shape
            one_d_result = len(shape_shape) == 0
            height = int(shape) if one_d_result else shape_shape[0]
            width = shape[1] if not one_d_result else 1
            size = int(shape) if one_d_result else shape_shape
        else:
            one_d_result = len(shape) == 1
            height = shape[0] if not one_d_result else int(shape)
            width = shape[1] if not one_d_result else 1
            size = int(shape) if one_d_result else shape_shape
            
        if use_cp is None:
            use_cp = width * height > 4092
        
        xp = cp if use_cp else numpy
        data = ((xp.random.standard_normal(size) * self.std) + self.median).astype(float, copy=False)
        xp.clip(data, self.min, self.max, out=data) # Ensure random data doesn't change range of the image
        
        if return_numpy and isinstance(data, cp.ndarray):
            return data.get()
        
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


def ScoreImageWithPowerSpectralDensity(image):
    
    # Find all NaN values and replace with median value
    adjustment_value = numpy.mean(image[numpy.isfinite(image)].flat)
    image[numpy.isfinite(image) == False] = adjustment_value
    
    # Adjust image to have median value of zero, makes the PSD numbers more human-readable and possibly avoids floating point precision issues
    Im_centered = image - adjustment_value   
    fft = numpy.fft.rfft2(Im_centered)
    # fft = numpy.fft.fftshift(fft) 
    total_amp = numpy.sum(numpy.abs(fft))
    score = total_amp / numpy.prod(Im_centered.shape)
    return score


def __CalculateFeatureScoreSciPy__(image, cell_size=None, feature_coverage_percent=None, **kwargs):
    '''
    Calculates a score indicating the amount of texture available for our phase correlation algorithm to use for alignment
    :param image: The image to score, either an ndarray or filename
    :param tuple cell_size: The dimensions of the subregions that will be evaluated across the image.
    :param float feature_coverage_percent: A value from 0 - 100 indicating what percentage of the image should contain textures scoring at or above the returned value.
     
    '''
    
    if feature_coverage_percent is None:
        feature_coverage_percent = 75
    else:
        feature_coverage_percent = 100 - feature_coverage_percent
        assert(feature_coverage_percent <= 100 and feature_coverage_percent >= 0)
    
    Im = nornir_imageregistration.ImageParamToImageArray(image, dtype=nornir_imageregistration.default_image_dtype())
# #     Im_filtered = scipy.ndimage.filters.median_filter(Im, size=3)
# #     sx = scipy.ndimage.sobel(Im_filtered, axis=0, mode='nearest')
# #     sy = scipy.ndimage.sobel(Im_filtered, axis=1, mode='nearest')
# #     sob = numpy.hypot(sx,sy)
# #
      
#     
# #     
#     logamp = numpy.log(amp) ** 2 
#     logampflat = numpy.asarray(logamp.flat)
#     aboveMedian = numpy.median(logampflat)
#     score = numpy.mean(logampflat[logampflat > aboveMedian])

    # score = numpy.max(Im_filtered.flat) - numpy.min(Im_filtered.flat)
    
    # score = numpy.var(Im_filtered.flat)
    # score = numpy.percentile(sob.flat, 90)
    # score = numpy.max(sob.flat)
# #    score = numpy.mean(sob.flat)
    # score = numpy.median(sob.flat) - numpy.percentile(sob.flat, 10)
    # mode = numpy.stats.mode(sob.flat)
    
    # p10 = numpy.percentile(Im_filtered.flat, 10)
    # p90 = numpy.percentile(Im_filtered.flat, 90)
    # med = numpy.median(sob.flat)
    
    # score = (p90 - p10)
    
    # score = numpy.median(sob.flat) - numpy.percentile(sob.flat, 10))
    
#     if score < .025:
#         nornir_imageregistration.ShowGrayscale([Im, Im_filtered, sob], title=str(score))
#         plt.figure()
#         plt.hist(sob.flat, bins=100)
#         a = 4
# #     return score

#     finite_subset = numpy.asarray(Im[numpy.isfinite(Im)].flat, dtype=numpy.float32)
#     if len(finite_subset) < 3:
#         return 0
# 
#     return numpy.std(finite_subset)
         
    if cell_size is None:
        # cell_size = numpy.max(numpy.vstack((numpy.asarray(numpy.asarray(Im.shape) / 64, dtype=numpy.int32), numpy.asarray((64,64),dtype=numpy.int32))),0) 
        cell_size = numpy.asarray((64, 64), dtype=numpy.int32)
     
    grid = nornir_imageregistration.CenteredGridDivision(Im.shape, cell_size=cell_size)
    
    cell_area = numpy.prod(cell_size)
     
    score_list = []
     
    for iPoint in range(0, grid.num_points):
        rect = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(grid.SourcePoints[iPoint, :], grid.cell_size)
        subset = nornir_imageregistration.CropImageRect(Im, rect, cval=numpy.nan)
        finite_subset = subset[numpy.isfinite(subset)].flat
        if len(finite_subset) < (cell_area / 2.0):
            continue
         
        std_val = ScoreImageWithPowerSpectralDensity(subset)
         
        # std_val = numpy.var(numpy.asarray(finite_subset, dtype=numpy.float32))
        # std_val = numpy.percentile(finite_subset, q=90) - numpy.percentile(finite_subset, q=10) 
        score_list.append(std_val)
        
        del subset
        
    del Im

    if len(score_list) == 0:
        return 0
    elif len(score_list) == 1:
        return score_list[0]
    else:
        val = numpy.percentile(score_list, q=feature_coverage_percent)
     
        # val = numpy.max(score_list)
        # val = numpy.mean(score_list) #Median was less reliable when using the range of intensity values as a measure
        return val
    

def __PruneFileSciPy__(filename, MaxOverlap=0.15, **kwargs):
    '''Returns a prune score for a single file
        Args:
           MaxOverlap = 0 to 1'''

    # TODO: This function should be updated to use the grid_subdivision module to create cells.  It should be used in the mosaic tile translation code to eliminate featureless 
    # overlap regions of adjacent tiles

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
    # MeanList = []

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
    # nornir_imageregistration.ShowGrayscale(Im)
    return sum(StdDevList)


def Histogram(filenames: str | Sequence[str], Bpp: int | None = None, Scale=None, **kwargs) -> nornir_shared.histogram.Histogram:
    '''Returns a single histogram built by combining histograms of all images
       If scale is not none the images are scaled before the histogram is collected'''

    if isinstance(filenames, str):
        listfilenames = [filenames]
    elif isinstance(filenames, Sequence):
        listfilenames = filenames
    else:
        raise ValueError("filenames has unexpected type ")

    numTiles = len(listfilenames)
    if numTiles == 0:
        raise ValueError("Cannot histogram with no input files")

    if Bpp is None:
        Bpp = nornir_shared.images.GetImageBpp(listfilenames[0])

    assert isinstance(listfilenames, list)

    FilenameToTask = {} 
    if len(listfilenames) > 2:
        pool = nornir_pools.GetGlobalLocalMachinePool()
    else:
        pool = nornir_pools.GetGlobalSerialPool()
    
    for f in listfilenames:
        # (root, ext) = os.path.splitext(f)
        # __HistogramFilePillow__(f, Bpp=Bpp, Scale=Scale)
        task = pool.add_task(f, __HistogramFileSciPy__, f, Bpp=Bpp, Scale=Scale, **kwargs)
#         if ext == '.npy':
#             task = __HistogramFileSciPy__(f, Bpp=Bpp, Scale=Scale)
#         else:
#             #task = __HistogramFilePillow__(f, ProcPool=pool, Bpp=Bpp, Scale=Scale)
#             task = pool.add_task(f, __HistogramFilePillow__,f, Bpp=Bpp, Scale=Scale)
#             #task = __HistogramFileImageMagick__(f, ProcPool=pool, Bpp=Bpp, Scale=Scale)
        FilenameToTask[f] = task
        
    minVal = None
    maxVal = None
    histlist = []
    numBins = None
    for f in list(FilenameToTask.keys()):
        task = FilenameToTask[f]
        try:
            h = task.wait_return()
        except IOError as e:
            PrettyOutput.Log("File not found " + f)
            continue
        
        histlist.append(h)
#         lines = taskOutput.splitlines()
# 
#         OutputMap[f] = lines
# 
#         (fminVal, fmaxVal) = nornir_imageregistration.im_histogram_parser.MinMaxValues(lines)
        if minVal is None:
            minVal = h.MinValue
        else:
            minVal = min(minVal, h.MinValue)

        if maxVal is None:
            maxVal = h.MaxValue
        else:
            maxVal = max(maxVal, h.MaxValue)
            
        numBins = len(h.Bins)
# 
#     threadTasks = []
#      
#     thread_pool = nornir_pools.GetGlobalThreadPool()
#     for f in list(OutputMap.keys()):
#         threadTask = thread_pool.add_task(f, nornir_imageregistration.im_histogram_parser.Parse, OutputMap[f], minVal=minVal, maxVal=maxVal, numBins=numBins)
#         threadTasks.append(threadTask)
#         
    HistogramComposite = nornir_shared.histogram.Histogram.Init(minVal=minVal, maxVal=maxVal, numBins=numBins)
    for h in histlist:
        # hist = t.wait_return()
        HistogramComposite.AddHistogram(h)
        # histogram = IMHistogramOutput.Parse(taskOutput, minVal=minVal, maxVal=maxVal, numBins=numBins)

        # FilenameToResult[f] = [histogram, None, None]

    if Bpp > 8:
        HistogramComposite = nornir_shared.histogram.Histogram.Trim(HistogramComposite)
    # del threadTasks

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
 

def __HistogramFileSciPy__(filename, Bpp=None, NumSamples=None, numBins=None, Scale=None, MinVal=None, MaxVal=None):
    '''Return the histogram of an image'''

    Im = None
    with Image.open(filename, mode='r') as img:
        img_I = img.convert("I")
        Im = numpy.asarray(img_I)
        # dims = numpy.asarray(img.size).astype(dtype=numpy.float32)
         
    (Height, Width) = Im.shape
    NumPixels = Width * Height
    
    if MinVal is None:
        MinVal = 0
        
    if MaxVal is None:
        if Bpp is None:
            Bpp = images.GetImageBpp(filename)
        
        assert(isinstance(Bpp, int))
        MaxVal = (1 << Bpp) - 1
        
    if numBins is None:
        numBins = (MaxVal - MinVal) + 1
    else:
        assert(isinstance(numBins, int))
        if numBins > (MaxVal - MinVal) + 1:
            numBins = (MaxVal - MinVal) + 1
         
    # if(not Scale is None):
    #    if(Scale != 1.0):
    #        Im = scipy.misc.imresize(Im, size=Scale, interp='nearest') 

    # ImOneD = reshape(Im, Width * Height, 1)
    
    ImOneD = Im.flat

    if NumSamples is None:
        NumSamples = Height * Width
    elif NumSamples > Height * Width:
        NumSamples = Height * Width

    StepSize = int(float(NumPixels) / float(NumSamples))

    if StepSize > 1:
        Samples = numpy.random.random_integers(0, NumPixels - 1, NumSamples)
        ImOneD = ImOneD[Samples]

    # [histogram_array, low_range, binsize] = numpy.histogram(ImOneD, bins=numBins, range =[0, 1])
    #In numpy's histogram, the max value must be at the end of the last bin, so for a 256 grayscale image MinVal=0 MaxVal=256
    [histogram_array, bin_edges] = numpy.histogram(ImOneD, bins=numBins, range=[MinVal, MaxVal+1])
    binWidth = bin_edges[1] - bin_edges[0] #(MaxVal - MinVal) / len(histogram_array)
    assert(binWidth > 0)
    histogram_obj = nornir_shared.histogram.Histogram.FromArray(histogram_array, bin_edges[0], binWidth)
    
    return histogram_obj


def __CreateImageMagickCommandLineForHistogram(filename, Scale):
    CmdTemplate = "magick convert %(filename)s -filter point -scale %(scale)g%% -define histogram:unique-colors=true -format %%c histogram:info:- && exit"
    return CmdTemplate % {'filename' : filename, 'scale' : Scale * 100}

    
def __HistogramFilePillow__(filename, Bpp=None, Scale=None):

    if Scale is None:
        Scale = 1

    # We only scale down, so if it is over 1 assume it is a percentage
    if Scale > 1:
        Scale = Scale / 100.0

    if Scale > 1:
        Scale = 1
        
    im = Image.open(filename).convert('I')
    histogram_array = im.histogram()
    binWidth = (1 << Bpp) // len(histogram_array) 
     
    histogram_obj = nornir_shared.histogram.Histogram.FromArray(histogram_array, 0, binWidth)

    return histogram_obj


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

