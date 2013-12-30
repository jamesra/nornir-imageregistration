'''
scipy image arrays are indexed [y,x]
'''

import ctypes
import logging
import multiprocessing
import os

from PIL import Image
import numpy
import pylab
import scipy.ndimage.measurements
import scipy.stats

from alignment_record import *
import scipy.ndimage.interpolation as interpolation


# from memory_profiler import profile
logger = logging.getLogger('IrTools.core')

class ImageStats(object):
    '''A container for image statistics'''

    @property
    def median(self):
        return self._median

    @median.setter
    def median(self, val):
        self._median = val

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, val):
        self._std = val

    def __init__(self):
        self._median = None
        self._std = None

    @classmethod
    def CalcStats(cls, image):
        image1D = image.flat
        obj = ImageStats()
        obj.median = median(image1D)
        obj.std = std(image1D)
        return obj


def ApproxEqual(A, B, epsilon=None):

    if epsilon is None:
        epsilon = 0.01

    return np.abs(A - B) < epsilon

def ScalarForMaxDimension(max_dim, shapes):
    '''Returns the scalar value to use so the largest dimensions in a list of shapes has the maximum value'''
    shapearray = None
    if not isinstance(shapes, list):
         shapearray = np.array(shapes)
    else:
        shapeArrays = map(np.array, shapes)
        shapearray = np.hstack(shapeArrays)

    maxVal = float(np.max(shapearray))

    return max_dim / maxVal

def ReduceImage(image, scalar):
    return interpolation.zoom(image, scalar)

def ShowGrayscale(imageList):
    '''
    :param list imageList: A list or single ndimage to be displayed with imshow
    '''
    
    if isinstance(imageList, list):
        fig, axeslist = plt.subplots(1, len(imageList))
        # fig = figure()
        for i, image in enumerate(imageList):
            if isinstance(image, numpy.ndarray):
                # ax = fig.add_subplot(101 + ((len(imageList) - (i)) * 10))
                ax = axeslist[i]
                ax.imshow(image, cmap=gray(), figure=fig)
    elif isinstance(imageList, numpy.ndarray):
        imshow(imageList, cmap=gray())
    else:
        return

    show()


def ROIRange(start, count, maxVal, minVal=0):
    '''Returns a range that falls within the limits, but contains count entries.'''

    r = None
    if maxVal - minVal < count:
        return None

    if start < minVal:
        r = range(minVal, minVal + count)
    elif start + count >= maxVal:
        r = range(maxVal - count, maxVal)
    else:
        r = range(start, start + count)

    return r

def ConstrainedRange(start, count, maxVal, minVal=0):
    '''Returns a range that falls within min/max limits.'''

    end = start + count
    r = None
    if maxVal - minVal < count:
        return range(minVal, maxVal)

    if start < minVal:
        r = range(minVal, end)
    elif end >= maxVal:
        r = range(start, maxVal)
    else:
        r = range(start, end)

    return r


def ExtractROI(image, center, area):
    '''Returns an ROI around a center point with the area, if the area passes a boundary the ROI
       maintains the same area, but is shifted so the entire area remains in the image.
       USES NUMPY (Y,X) INDEXING'''

    x_range = ROIRange(area[1], (center - area[1]) / 2.0, maxVal=image.shape[1])
    y_range = ROIRange(area[0], (center - area[0]) / 2.0, maxVal=image.shape[0])

    ROI = image(y_range, x_range)

    return ROI


def CropImage(imageparam, Xo, Yo, Width, Height, background=None):
    '''
       Crop the image at the passed bounds and returns the cropped ndarray.
       If the requested area is outside the bounds of the array then the correct region is returned
       with a background color set
       
       :param ndarray imageparam: An ndarray image to crop.  A string containing a path to an image is also acceptable.e
       :param int Xo: X origin for crop
       :param int Yo: Y origin for crop
       :param int Width: New width of image
       :param int Height: New height of image
       :param int background: default value for regions outside the original image boundaries.  Defaults to 0.
       :return: Cropped image
       :rtype: ndarray
       '''

    image = None
    if isinstance(imageparam, str):
        image = LoadImage(imageparam)
    else:
        image = imageparam

    if image is None:
        return None

    in_startY = Yo
    in_startX = Xo
    in_endX = Xo + Width
    in_endY = Yo + Height

    out_startY = 0
    out_startX = 0
    out_endX = Width
    out_endY = Height

    if in_startY < 0:
        out_startY = -in_startY
        in_startY = 0

    if in_startX < 0:
        out_startX = -in_startX
        in_startX = 0

    if in_endX > image.shape[1]:
        in_endX = image.shape[1]
        out_endX = out_startX + (in_endX - in_startX)

    if in_endY > image.shape[0]:
        in_endY = image.shape[0]
        out_endY = out_startY + (in_endY - in_startY)

    cropped = None
    if background is None:
        cropped = zeros((Height, Width), dtype=image.dtype)
    else:
        cropped = ones((Height, Width), dtype=image.dtype) * background
        
    cropped[out_startY:out_endY, out_startX:out_endX] = image[in_startY:in_endY, in_startX:in_endX]

    return cropped


def npArrayToReadOnlySharedArray(npArray):
    '''Returns a shared memory array for a numpy array.  Used to reduce memory footprint when passing parameters to multiprocess pools'''
    SharedBase = multiprocessing.sharedctypes.RawArray(ctypes.c_float, npArray.shape[0] * npArray.shape[1])
    SharedArray = np.ctypeslib.as_array(SharedBase)
    SharedArray = SharedArray.reshape(npArray.shape)
    np.copyto(SharedArray, npArray)
    return SharedArray

def GenRandomData(height, width, mean, standardDev):
    '''
    Generate random data of shape with the specified mean and standard deviation
    '''
    image = (numpy.random.randn(height, width).astype(numpy.float32) * standardDev) + mean

    if mean - (standardDev * 2) < 0:
        image = abs(image)
    return image


def GetImageSize(ImageFullPath):
    '''
    :returns: Image (height, width)
    :rtype: tuple
    '''

    if not os.path.exists(ImageFullPath):
        return None

    try:
        im = Image.open(ImageFullPath)
        return (im.size[1], im.size[0])
    except IOError:
        return None


def ForceGrayscale(image):
    '''
    :param: ndarray with 3 dimensions
    :returns: grayscale data 
    :rtype: ndarray with 2 dimensions'''

    if len(image.shape) > 2:
        image = image[:, :, 0]
        return np.squeeze(image)

    return image

# @profile
def LoadImage(ImageFullPath, ImageMaskFullPath=None, MaxDimension=None):
    
    '''
    Loads an image, masks it, and removes extrema pixels.
    
    :param str ImageFullPath: Path to image
    :param str ImageMaskFullPath: Path to mask, dimension should match input image
    :param MaxDimension: Limit the largest dimension of the returned image to this size.  Downsample if necessary.
    :returns: Loaded image.  Masked areas and extrema pixel values are replaced with gaussian noise matching the median and std. dev. of the unmasked image.
    :rtype: ndimage
    '''
    if(not os.path.isfile(ImageFullPath)):
        logger.error('File does not exist: ' + ImageFullPath)
        return None

    image = imread(ImageFullPath)
    image = ForceGrayscale(image)

    if not MaxDimension is None:
        scalar = ScalarForMaxDimension(MaxDimension, image.shape)
        if scalar < 1.0:
            image = ReduceImage(image, scalar)

    image_mask = None

    if(not ImageMaskFullPath is None):
        if(not os.path.isfile(ImageMaskFullPath)):
            logger.error('Fixed image mask file does not exist: ' + ImageMaskFullPath)
        else:
            image_mask = imread(ImageMaskFullPath)
            if not MaxDimension is None:
                scalar = ScalarForMaxDimension(MaxDimension, image_mask.shape)
                if scalar < 1.0:
                    image_mask = ReduceImage(image_mask, scalar)

            assert((image.shape == image_mask.shape))
            image = RandomNoiseMask(image, image_mask)

    return image


def RandomNoiseMask(image, Mask, ImageMedian=None, ImageStdDev=None, Copy=False):
    '''
    Fill the masked area with random noise with gaussian distribution about the image
    mean and with standard deviation matching the image's standard deviation
    
    :param ndimage image: Input image
    :param ndimage mask: Mask, zeros are replaced with noise.  Ones pull values from input image
    :param float ImageMedian: Mean of noise distribution, calculated from image if none
    :param float ImageStdDev: Standard deviation of noise distribution, calculated from image if none
    :param bool Copy: Returns a copy of input image if true, otherwise write noise to the input image
    :rtype: ndimage
    '''

    assert(image.shape == Mask.shape)

    Height = image.shape[0]
    Width = image.shape[1]

    MaskedImage = image.copy()
    Image1D = MaskedImage.flat
    Mask1D = Mask.flat

    iZero1D = Mask1D == 0
    if(ImageMedian is None or ImageStdDev is None):
        # Create masked array for accurate stats

        MaskedImage1D = ma.masked_array(Image1D, iZero1D)

        if(ImageMedian is None):
            ImageMedian = numpy.median(MaskedImage1D)
        if(ImageStdDev is None):
            ImageStdDev = numpy.std(MaskedImage1D)

    iNonZero1D = np.transpose(nonzero(Mask1D))

    NumMaskedPixels = (Width * Height) - len(iNonZero1D)
    if(NumMaskedPixels == 0):
        return image

    NoiseData = GenRandomData(1, NumMaskedPixels, ImageMedian, ImageStdDev)

    # iZero1D = transpose(nonzero(iZero1D))
    Image1D[iZero1D] = NoiseData

    # NewImage = reshape(Image1D, (Height, Width), 2)

    return MaskedImage


def ReplaceImageExtramaWithNoise(image, ImageMedian=None, ImageStdDev=None):
    '''
    Replaced the min/max values in the image with random noise.  This is useful when aligning images composed mostly of dark or bright regions
    '''

    Image1D = image.flat

    (minima, maxima, iMin, iMax) = scipy.ndimage.measurements.extrema(Image1D)

    maxima_index = np.transpose((Image1D == maxima).nonzero())
    minima_index = np.transpose((Image1D == minima).nonzero())

    if(ImageMedian is None or ImageStdDev is None):
        if(ImageMedian is None):
            ImageMedian = numpy.median(Image1D)
        if(ImageStdDev is None):
            ImageStdDev = numpy.std(Image1D)

    num_pixels = len(maxima_index) + len(minima_index)

    OutputImage = numpy.copy(image)


    if num_pixels > 0:
        OutputImage1d = OutputImage.flat
        randData = GenRandomData(num_pixels, 1, ImageMedian, ImageStdDev)
        OutputImage1d[maxima_index] = randData[0:len(maxima_index)]
        OutputImage1d[minima_index] = randData[len(maxima_index):]



    return OutputImage


def NearestPowerOfTwoWithOverlap(val, overlap=1.0):
    '''
    :return: Same as DimensionWithOverlap, but output dimension is increased to the next power of two for faster FFT operations
    '''

    if overlap > 1.0:
        overlap = 1.0

    if overlap < 0.0:
        overlap = 0.0

    # Figure out the minimum dimension to accomodate the requested overlap
    MinDimension = DimensionWithOverlap(val, overlap)

    # Figure out the power of two dimension
    NewDimension = math.pow(2, math.ceil(math.log(MinDimension, 2)))
    return NewDimension


def DimensionWithOverlap(val, overlap=1.0):
    '''
    :param float val: Original dimension
    :param float overlap: Amount of overlap possible between images, from 0 to 1
    :returns: Required dimension size to unambiguously determine the offset in an fft image
    '''

    # An overlap of 50% is half of the image, so we don't need to expand the image to find the peak in the correct quadrant
    if overlap >= 0.5:
        return val

    overlap += 0.5

    return val + (val * (1.0 - overlap) * 4.0)

# @profile
def PadImageForPhaseCorrelation(image, MinOverlap=.05, ImageMedian=None, ImageStdDev=None, NewWidth=None, NewHeight=None, PowerOfTwo=True):
    '''
    Prepares an image for use with the phase correlation operation.  Padded areas are filled with noise matching the histogram of the 
    original image.  Optionally the min/max pixels can also replaced be replaced with noise using FillExtremaWithNoise
    
    :param ndarray image: Input image
    :param float MinOverlap: Minimum overlap allowed between the input image and images it will be registered to
    :param float ImageMean: Median value of noise, calculated or pulled from cache if none
    :param float ImageStdDev: Standard deviation of noise, calculated or pulled from cache if none
    :param int NewWidth: Pad input image to this dimension if not none
    :param int NewHeight: Pad input image to this dimension if not none
    :param bool PowerOfTwo: Pad the image to a power of two if true
    :return: An image with the input image centered surrounded by noise
    :rtype: ndimage
    
    
    '''
    Size = image.shape

    Height = Size[0]
    Width = Size[1]

    if(NewHeight is None):
        if PowerOfTwo:
            NewHeight = NearestPowerOfTwoWithOverlap(Height, MinOverlap)
        else:
            NewHeight = DimensionWithOverlap(Height, MinOverlap)  #  # Height + (Height * (1 - MinOverlap))  # + 1

    if(NewWidth is None):
        if PowerOfTwo:
            NewWidth = NearestPowerOfTwoWithOverlap(Width, MinOverlap)
        else:
            NewWidth = DimensionWithOverlap(Width, MinOverlap)  #  # Height + (Height * (1 - MinOverlap))  # + 1

    if(Width == NewWidth and Height == NewHeight):
        return np.copy(image)

    if(ImageMedian is None or ImageStdDev is None):
        Image1D = image.flat

        if(ImageMedian is None):
            ImageMedian = median(Image1D)
        if(ImageStdDev is None):
            ImageStdDev = std(Image1D)

    PaddedImage = numpy.zeros((NewHeight, NewWidth), dtype=numpy.float16)

    PaddedImageXOffset = floor((NewWidth - Width) / 2.0)
    PaddedImageYOffset = floor((NewHeight - Height) / 2.0)

    # Copy image into padded image
    PaddedImage[PaddedImageYOffset:PaddedImageYOffset + Height, PaddedImageXOffset:PaddedImageXOffset + Width] = image[:, :]

    if not Width == NewWidth:
        LeftBorder = GenRandomData(NewHeight, PaddedImageXOffset, ImageMedian, ImageStdDev)
        RightBorder = GenRandomData(NewHeight, NewWidth - (Width + PaddedImageXOffset), ImageMedian, ImageStdDev)

        PaddedImage[:, 0:PaddedImageXOffset] = LeftBorder
        PaddedImage[:, Width + PaddedImageXOffset:] = RightBorder

        del LeftBorder
        del RightBorder

    if not Height == NewHeight:

        TopBorder = GenRandomData(PaddedImageYOffset, Width, ImageMedian, ImageStdDev)
        BottomBorder = GenRandomData(NewHeight - (Height + PaddedImageYOffset), Width, ImageMedian, ImageStdDev)

        PaddedImage[0:PaddedImageYOffset, PaddedImageXOffset:PaddedImageXOffset + Width] = TopBorder
        PaddedImage[PaddedImageYOffset + Height:, PaddedImageXOffset:PaddedImageXOffset + Width] = BottomBorder

        del TopBorder
        del BottomBorder

    return PaddedImage

# @profile
def ImagePhaseCorrelation(FixedImage, MovingImage):
    '''
    Returns the phase shift correlation of the FFT's of two images. 
    
    Dimensions of Fixed and Moving images must match
    
    :param ndarray FixedImage: grayscale image
    :param ndarray MovingImage: grayscale image
    :returns: Correlation image of the FFT's.  Light pixels indicate the phase is well aligned at that offset.
    :rtype: ndimage
    
    '''

    if(not (FixedImage.shape == MovingImage.shape)):
        # TODO, we should pad the smaller image in this case to allow the comparison to continue
        raise ValueError("ImagePhaseCorrelation: Fixed and Moving image do not have same dimension")

#    FFTMan = FFTWManager.GetFFTManager()
#
#    FFTPlan = FFTMan.GetPlan(FixedImage.shape)
#
#    #ShowGrayscale(MovingImage)
#
#    #It is not possible to multi-thread scipy fft2 calls at this time 7/2012
# #
#    FFTFixed = FFTPlan.fft(FixedImage)
#    FFTMoving  = FFTPlan.fft(MovingImage)


    #--------------------------------
    # This is here in case this function ever needs to be revisited.  Scipy is a lot faster working with in-place operations so this
    # code has been obfuscated more than I like
    # FFTFixed = fftpack.rfft2(FixedImage)
    # FFTMoving = fftpack.rfft2(MovingImage)
    # conjFFTFixed = conj(FFTFixed)
    # Numerator = conjFFTFixed * FFTMoving
    # Divisor = abs(conjFFTFixed * FFTMoving)
    # T = Numerator / Divisor
    # CorrelationImage = real(fftpack.irfft2(T))
    #--------------------------------

    FFTFixed = fftpack.rfft2(FixedImage)
    FFTMoving = fftpack.rfft2(MovingImage)

    conjFFTFixed = conj(FFTFixed)
    del FFTFixed

    conjFFTFixed *= FFTMoving
    del FFTMoving

    conjFFTFixed /= abs(conjFFTFixed)  # Numerator / Divisor

    CorrelationImage = real(fftpack.irfft2(conjFFTFixed))
    del conjFFTFixed

    return CorrelationImage
    # SmallCorrelationImage = CorrelationImage.astype(np.float32)
    # del CorrelationImage
    # return SmallCorrelationImage


# @profile
def FindPeak(image, Cutoff=0.995, MinOverlap=0, MaxOverlap=1):
    '''
    Find the offset of the strongest response in a phase correlation image
    
    :param ndimage image: grayscale image
    :param float Cutoff: Percentile used to threshold image.  Values below the percentile are ignored
    :param float MinOverlap: Minimum overlap allowed
    :param float MaxOverlap: Maximum overlap allowed
    :return: Offset of peak from image center and sum of pixels values at peak
    :rtype: (tuple, float)
    '''
    
    # CutoffValue = ImageIntensityAtPercent(image, Cutoff)

    CutoffValue = scipy.stats.scoreatpercentile(image, per=Cutoff * 100.0)

    ThresholdImage = scipy.stats.threshold(image, threshmin=CutoffValue, threshmax=None, newval=0)
    # ShowGrayscale(ThresholdImage)

    [LabelImage, NumLabels] = scipy.ndimage.measurements.label(ThresholdImage)
    LabelSums = scipy.ndimage.measurements.sum(ThresholdImage, LabelImage, range(0, NumLabels))
    PeakValueIndex = LabelSums.argmax()
    PeakCenterOfMass = scipy.ndimage.measurements.center_of_mass(ThresholdImage, LabelImage, PeakValueIndex)
    PeakStrength = LabelSums[PeakValueIndex]

    del LabelImage
    del ThresholdImage
    del LabelSums

    # center_of_mass returns results as (y,x)
    Offset = (image.shape[0] / 2.0 - PeakCenterOfMass[0], image.shape[1] / 2.0 - PeakCenterOfMass[1])
    # Offset = (Offset[0], Offset[1])

    return (Offset, PeakStrength)


def CropNonOverlapping(FixedImageSize, MovingImageSize, CorrelationImage, MinOverlap=0.0, MaxOverlap=1.0):
    ''' '''

    if not FixedImageSize == MovingImageSize:
        return CorrelationImage


def CreateOverlapMask(FixedImageSize, MovingImageSize, MinOverlap=0.0, MaxOverlap=1.0):
    '''Defines a mask that determines which peaks should be considered'''

    MaxWidth = FixedImageSize[1] + MovingImageSize[1]
    MaxHeight = FixedImageSize[0] + MovingImageSize[0]

    mask = np.ones([MaxHeight, MaxWidth], dtype=np.Bool)



def FindOffset(FixedImage, MovingImage, MinOverlap=0.0, MaxOverlap=1.0):
    '''return an alignment record describing how the images overlap. The alignment record indicates how much the 
       moving image must be rotated and translated to align perfectly with the FixedImage'''

    # Find peak requires both the fixed and moving images have equal size
    assert((FixedImage.shape[0] == MovingImage.shape[0]) and (FixedImage.shape[1] == MovingImage.shape[1]))

    CorrelationImage = ImagePhaseCorrelation(FixedImage, MovingImage)
    CorrelationImage = fftshift(CorrelationImage)

    # Crop the areas that cannot overlap

    CorrelationImage -= CorrelationImage.min()
    CorrelationImage /= CorrelationImage.max()

    # Timer.Start('Find Peak')
    (peak, weight) = FindPeak(CorrelationImage, MinOverlap=MinOverlap, MaxOverlap=MaxOverlap)

    del CorrelationImage

    record = AlignmentRecord(peak=peak, weight=weight)

    return record

def ImageIntensityAtPercent(image, Percent=0.995):
    '''Returns the intensity of the Cutoff% most intense pixel in the image'''
    NumPixels = image.size


#   Sorting the list is a more correct and straightforward implementation, but using numpy.histogram is about 1 second faster
#   image1D = numpy.sort(image, axis=None)
#   targetIndex = math.floor(float(NumPixels) * Percent)
#
#   val = image1D[targetIndex]
#
#   del image1D
#   return val

    NumBins = 1024
    [histogram, binEdge] = numpy.histogram(image, bins=NumBins)

    PixelNum = float(NumPixels) * Percent
    CumulativePixelsInBins = 0
    CutOffHistogramValue = None
    for iBin in range(0, len(histogram)):
        if CumulativePixelsInBins > PixelNum:
            CutOffHistogramValue = binEdge[iBin]
            break

        CumulativePixelsInBins += histogram[iBin]

    if CutOffHistogramValue is None:
        CutOffHistogramValue = binEdge[-1]

    return CutOffHistogramValue


if __name__ == '__main__':

    FilenameA = 'C:\\BuildScript\\Test\\Images\\400.png'
    FilenameB = 'C:\\BuildScript\\Test\\Images\\401.png'
    OutputDir = 'C:\\Buildscript\\Test\\Results\\'

    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)


    def TestPhaseCorrelation(imA, imB):


        # import TaskTimer
        # Timer = TaskTimer.TaskTimer()
        # Timer.Start('Correlate One Pair')

        # Timer.Start('Pad image One Pair')
        FixedA = PadImageForPhaseCorrelation(imA)
        MovingB = PadImageForPhaseCorrelation(imB)

        record = FindOffset(FixedA, MovingB)
        print str(record)

        stos = record.ToStos(FilenameA, FilenameB)

        stos.Save(os.path.join(OutputDir, "TestPhaseCorrelation.stos"))

        # Timer.End('Find Peak', False)

        # Timer.End('Correlate One Pair', False)

        # print(str(Timer))

        # ShowGrayscale(NormCorrelationImage)
        return

    def SecondMain():


        imA = imread(FilenameA)
        imB = imread(FilenameB)

        for i in range(1, 5):
            print(str(i))
            TestPhaseCorrelation(imA, imB)

    import cProfile
    import pstats
    cProfile.run('SecondMain()', 'CoreProfile.pr')
    pr = pstats.Stats('CoreProfile.pr')
    pr.sort_stats('time')
    print str(pr.print_stats(.5))

