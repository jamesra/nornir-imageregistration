'''
Created on Jul 6, 2012

@author: Jamesan

Remember that the scipy image arrays are indexed [y,x]
'''

import logging
import scipy.stats
import scipy.ndimage.measurements
import scipy.ndimage.interpolation as interpolation
import multiprocessing
import ctypes
import os
from alignment_record import *
import numpy
from pylab import *


from PIL import Image

logger = logging.getLogger('IrTools.core')

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


def ExtractROI(Image, center, area):
    '''Returns an ROI around a center point with the area, if the area passes a boundary the ROI
       maintains the same area, but is shifted so the entire area remains in the image.
       USES NUMPY (Y,X) INDEXING'''

    x_range = ROIRange(area[1], (center - area[1]) / 2.0, maxVal=Image.shape[1])
    y_range = ROIRange(area[0], (center - area[0]) / 2.0, maxVal=Image.shape[0])

    ROI = Image(y_range, x_range)

    return ROI


def CropImage(imageparam, Xo, Yo, Width, Height, background=None):
    '''Crop the image at the passed bounds and returns the cropped ndarray.
       IF the requested area is outside the bounds of the array then the correct region is returned
       with a background color set'''

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

    cropped = zeros((Height, Width), dtype=image.dtype)

    cropped[out_startY:out_endY, out_startX:out_endX] = image[in_startY:in_endY, in_startX:in_endX]

    return cropped


def npArrayToReadOnlySharedArray(npArray):
    '''Returns a shared memory array for a numpy array'''
    SharedBase = multiprocessing.sharedctypes.RawArray(ctypes.c_float, npArray.shape[0] * npArray.shape[1])
    SharedArray = np.ctypeslib.as_array(SharedBase)
    SharedArray = SharedArray.reshape(npArray.shape)
    np.copyto(SharedArray, npArray)
    return SharedArray

def GenRandomData(height, width, mean, standardDev):
    '''Generate random data of shape with the specified mean and standard deviation'''
    Image = (scipy.randn(height, width).astype(numpy.float32) * standardDev) + mean;

    if mean - (standardDev * 2) < 0:
        Image = abs(Image)
    return Image


def GetImageSize(ImageFullPath):
    '''Returns image size as (height,width)'''

    if not os.path.exists(ImageFullPath):
        return None

    try:
        im = Image.open(ImageFullPath)
        return (im.size[1], im.size[0])
    except IOError:
        return None


def ForceGrayscale(image):
    '''Ensure an image is greyscale'''
    
    if len(image.shape) > 2:
        image = image[:,:,0]
        return np.squeeze(image)
    
    return image

def LoadImage(ImageFullPath, ImageMaskFullPath = None, MaxDimension = None):
    '''Loads an image, masks it, and removes extrema pixels.
       This is a helper function for registering images'''
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

def RandomNoiseMask(Image, Mask, ImageMedian=None, ImageStdDev=None, Copy=False):
    '''Fill the masked area with random noise with gaussian distribution about the image
       mean and with standard deviation matching the image's standard deviation'''


    assert(Image.shape == Mask.shape)

    Height = Image.shape[0];
    Width = Image.shape[1];

    MaskedImage = Image.copy()
    Image1D = MaskedImage.flat
    Mask1D = Mask.flat

    iZero1D = Mask1D == 0;
    if(ImageMedian is None or ImageStdDev is None):
        # Create masked array for accurate stats

        MaskedImage1D = ma.masked_array(Image1D, iZero1D);

        if(ImageMedian is None):
            ImageMedian = numpy.median(MaskedImage1D);
        if(ImageStdDev is None):
            ImageStdDev = numpy.std(MaskedImage1D);

    iNonZero1D = np.transpose(nonzero(Mask1D))

    NumMaskedPixels = (Width * Height) - len(iNonZero1D)
    if(NumMaskedPixels == 0):
        return Image;

    NoiseData = GenRandomData(1, NumMaskedPixels, ImageMedian, ImageStdDev);

    # iZero1D = transpose(nonzero(iZero1D));
    Image1D[iZero1D] = NoiseData;

    # NewImage = reshape(Image1D, (Height, Width), 2)

    return MaskedImage


def ReplaceImageExtramaWithNoise(Image, ImageMedian=None, ImageStdDev=None):
    '''Replaced the min/max values in the image with random noise.  This is useful when aligning images composed mostly of dark or bright regions'''

    Image1D = Image.flat

    (minima, maxima, iMin, iMax) = scipy.ndimage.measurements.extrema(Image1D)

    maxima_index = np.transpose((Image1D == maxima).nonzero())
    minima_index = np.transpose((Image1D == minima).nonzero())

    if(ImageMedian is None or ImageStdDev is None):
        if(ImageMedian is None):
            ImageMedian = numpy.median(Image1D);
        if(ImageStdDev is None):
            ImageStdDev = numpy.std(Image1D);

    num_pixels = len(maxima_index) + len(minima_index)

    OutputImage = numpy.copy(Image)


    if num_pixels > 0:
        OutputImage1d = OutputImage.flat
        randData = GenRandomData(num_pixels, 1, ImageMedian, ImageStdDev)
        OutputImage1d[maxima_index] = randData[0:len(maxima_index)]
        OutputImage1d[minima_index] = randData[len(maxima_index):]



    return OutputImage


def PadImageForPhaseCorrelation(Image, MinOverlap=.05, ImageMedian=None, ImageStdDev=None, NewWidth=None, NewHeight=None):
    '''Prepares an image for use with the phase correllation operation.  Padded areas are filled with noise matching the histogram of the 
       original image.  Optionally the min/max pixels can also replaced be replaced with noise using FillExtremaWithNoise'''
    Size = Image.shape;

    Height = Size[0]
    Width = Size[1]

#
#    if MinWidth is None:
#        MinWidth = Width;
#
#    if MinWidth < Width:
#        MinWidth = Width;
#
#    if MinHeight is None:
#        MinHeight = Height;
#
#    if MinHeight < Height:
#        MinHeight = Height;
#
    if(NewHeight is None):
        NewHeight = Height + (Height * (1 - MinOverlap)) + 1;

    if(NewWidth is None):
        NewWidth = Width + (Width * (1 - MinOverlap)) + 1;

    # Round up size to nearest power of 2
    NewHeight = math.pow(2, math.ceil(math.log(NewHeight, 2)));
    NewWidth = math.pow(2, math.ceil(math.log(NewWidth, 2)));

    if(Width == NewWidth and Height == NewHeight):
        return Image;

    if(ImageMedian is None or ImageStdDev is None):
        Image1D = Image.flat

        if(ImageMedian is None):
            ImageMedian = median(Image1D);
        if(ImageStdDev is None):
            ImageStdDev = std(Image1D);

    PaddedImage = numpy.zeros((NewHeight, NewWidth), dtype=numpy.float32);

    PaddedImageXOffset = floor((NewWidth - Width) / 2);
    PaddedImageYOffset = floor((NewHeight - Height) / 2);

    # Copy image into padded image
    PaddedImage[PaddedImageYOffset:PaddedImageYOffset + Height, PaddedImageXOffset:PaddedImageXOffset + Width] = Image[:, :]

    if not Width == NewWidth:
        LeftBorder = GenRandomData(NewHeight, PaddedImageXOffset, ImageMedian, ImageStdDev);
        RightBorder = GenRandomData(NewHeight, NewWidth - (Width + PaddedImageXOffset), ImageMedian, ImageStdDev);

        PaddedImage[:, 0:PaddedImageXOffset] = LeftBorder
        PaddedImage[:, Width + PaddedImageXOffset:] = RightBorder

    if not Height == NewHeight:

        TopBorder = GenRandomData(PaddedImageYOffset, Width, ImageMedian, ImageStdDev);
        BottomBorder = GenRandomData(NewHeight - (Height + PaddedImageYOffset), Width, ImageMedian, ImageStdDev);

        PaddedImage[0:PaddedImageYOffset, PaddedImageXOffset:PaddedImageXOffset + Width] = TopBorder
        PaddedImage[ PaddedImageYOffset + Height:, PaddedImageXOffset:PaddedImageXOffset + Width] = BottomBorder

    return PaddedImage;


def ImagePhaseCorrelation(FixedImage, MovingImage):
    '''Returns the phase shift correlation of the FFT's of two images. 
       Light pixels indicate the phase is well aligned at that offset'''

    if(not (FixedImage.shape == MovingImage.shape)):
        # TODO, we should pad the smaller image in this case to allow the comparison to continue
        raise ValueError("ImagePhaseCorrelation: Fixed and Moving image do not have same dimension");

#    FFTMan = FFTWManager.GetFFTManager();
#
#    FFTPlan = FFTMan.GetPlan(FixedImage.shape);
#
#    #ShowGrayscale(MovingImage)
#
#    #It is not possible to multi-thread scipy fft2 calls at this time 7/2012
# #
#    FFTFixed = FFTPlan.fft(FixedImage)
#    FFTMoving  = FFTPlan.fft(MovingImage)

    FFTFixed = fftpack.fft2(FixedImage);
    FFTMoving = fftpack.fft2(MovingImage);

    conjFFTFixed = conj(FFTFixed);

    Numerator = conjFFTFixed * FFTMoving;
    Divisor = abs(conjFFTFixed * FFTMoving);

    T = Numerator / Divisor;

    # Correlation = FFTPlan.ifft(T);
    Correlation = fftpack.ifft2(T);

    CorrelationImage = real(Correlation);

    return CorrelationImage;


def FindPeak(Image, Cutoff=0.995, MinOverlap=0, MaxOverlap=1):
    CutoffValue = ImageIntensityAtPercent(Image, Cutoff);

    ThresholdImage = scipy.stats.threshold(Image, threshmin=CutoffValue, threshmax=None, newval=0);
    # ShowGrayscale(ThresholdImage)

    [LabelImage, NumLabels] = scipy.ndimage.measurements.label(ThresholdImage);
    LabelSums = scipy.ndimage.measurements.sum(ThresholdImage, LabelImage, range(0, NumLabels));
    PeakValueIndex = LabelSums.argmax()
    PeakCenterOfMass = scipy.ndimage.measurements.center_of_mass(ThresholdImage, LabelImage, PeakValueIndex)

    # center_of_mass returns results as (y,x)
    Offset = (Image.shape[0] / 2.0 - PeakCenterOfMass[0], Image.shape[1] / 2.0 - PeakCenterOfMass[1])
    # Offset = (Offset[0], Offset[1])

    return (Offset, LabelSums[PeakValueIndex]);


def FindOffset(FixedImage, MovingImage, MinOverlap=0.0, MaxOverlap=1.0):
    '''return an alignment record describing how the images overlap. The alignment record indicates how much the 
       moving image must be rotated and translated to align perfectly with the FixedImage'''

    # Find peak requires both the fixed and moving images have equal size
    assert((FixedImage.shape[0] == MovingImage.shape[0]) and (FixedImage.shape[1] == MovingImage.shape[1]));

    CorrelationImage = ImagePhaseCorrelation(FixedImage, MovingImage);
    CorrelationImage = fftshift(CorrelationImage);

    NormCorrelationImage = CorrelationImage - CorrelationImage.min();
    NormCorrelationImage /= NormCorrelationImage.max();

    # Timer.Start('Find Peak');

    (peak, weight) = FindPeak(NormCorrelationImage, MinOverlap=MinOverlap, MaxOverlap=MaxOverlap);
    record = AlignmentRecord(peak=peak, weight=weight);

    return record;

def ImageIntensityAtPercent(Image, Percent=0.995):
    '''Returns the intensity of the Cutoff% most intense pixel in the image'''
    NumPixels = Image.size;
    # FlatSortedImage = sort(reshape(Image,NumPixels, 1));
    # CutoffIndex = int(NumPixels * Percent);
    # CutoffValue = FlatSortedImage[CutoffIndex];

    NumBins = 200;
    [histogram, binEdge] = numpy.histogram(Image, bins=NumBins)

    PixelNum = float(NumPixels) * Percent;
    CumulativePixelsInBins = 0;
    CutOffHistogramValue = None;
    for iBin in range(0, len(histogram)):
        if CumulativePixelsInBins > PixelNum:
            CutOffHistogramValue = binEdge[iBin];
            break;

        CumulativePixelsInBins = CumulativePixelsInBins + histogram[iBin];

    if CutOffHistogramValue is None:
        CutOffHistogramValue = binEdge[-1];

    return CutOffHistogramValue;


if __name__ == '__main__':

    FilenameA = 'C:\\BuildScript\\Test\\Images\\400.png';
    FilenameB = 'C:\\BuildScript\\Test\\Images\\401.png';
    OutputDir = 'C:\\Buildscript\\Test\\Results\\';

    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir);


    def TestPhaseCorrelation(imA, imB):


        # import TaskTimer;
        # Timer = TaskTimer.TaskTimer()
        # Timer.Start('Correlate One Pair');

        # Timer.Start('Pad Image One Pair');
        FixedA = PadImageForPhaseCorrelation(imA);
        MovingB = PadImageForPhaseCorrelation(imB);

        record = FindOffset(FixedA, MovingB);
        print str(record)

        stos = record.ToStos(FilenameA, FilenameB);

        stos.Save(os.path.join(OutputDir, "TestPhaseCorrelation.stos"));

        # Timer.End('Find Peak', False);

        # Timer.End('Correlate One Pair', False);

        # print(str(Timer));

        # ShowGrayscale(NormCorrelationImage)
        return

    def SecondMain():


        imA = imread(FilenameA);
        imB = imread(FilenameB);

        for i in range(1, 5):
            print(str(i));
            TestPhaseCorrelation(imA, imB);

    import cProfile
    import pstats
    cProfile.run('SecondMain()', 'CoreProfile.pr');
    pr = pstats.Stats('CoreProfile.pr');
    pr.sort_stats('time');
    print str(pr.print_stats(.5));

