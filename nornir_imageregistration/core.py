'''
scipy image arrays are indexed [y,x]
'''

import ctypes
import logging
import math
import multiprocessing
import os

from PIL import Image
import numpy.fft
import scipy.ndimage.measurements
import scipy.stats
import scipy.misc
import nornir_shared.images as shared_images

import matplotlib.pyplot as plt
import nornir_imageregistration
import numpy as np
import numpy.fft.fftpack as fftpack
import scipy.ndimage.interpolation as interpolation
import multiprocessing.sharedctypes


import collections

#In a remote process we need errors raised, otherwise we crash for the wrong reason and debugging is tougher. 
np.seterr(all='raise')
    
# from memory_profiler import profile
logger = logging.getLogger(__name__)

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
        obj = ImageStats()
        obj.median = np.median(image.flat)
        obj.std = np.std(image.flat)
        return obj
    
def array_distance(array):
    '''Convert an Mx2 array into a Mx1 array of euclidean distances'''
    if array.ndim == 1:
        return np.sqrt(np.sum(array ** 2)) 
    
    return np.sqrt(np.sum(array ** 2,1))
    
def GetBitsPerPixel(File): 
    return shared_images.GetImageBpp(File)

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
        shapeArrays = list(map(np.array, shapes))
        shapearray = np.hstack(shapeArrays)

    maxVal = float(np.max(shapearray))

    return max_dim / maxVal

def ReduceImage(image, scalar):
    return interpolation.zoom(image, scalar)

def _GridLayoutDims(imagelist):
    '''Given a list of N items, returns the number of rows & columns to display the list.  Dimensions will always be wider than they are tall or equal in dimension
    '''

    numImages = len(imagelist)
    width = math.ceil(math.sqrt(numImages))
    height = math.ceil(numImages / width)

    if height > width:
        tempH = height
        height = width
        height = tempH

    return (int(height), int(width))

def ShowGrayscale(imageList, title=None):
    '''
    :param list imageList: A list or single ndimage to be displayed with imshow
    :param str title: Informative title for the figure, for example expected test results
    '''
    
    if not title is None:
        plt.title(title)
        plt.tight_layout(pad=1.0)    
        
    if isinstance(imageList, np.ndarray):
        plt.imshow(imageList, cmap=plt.gray())
    elif isinstance(imageList, collections.Iterable):

        if len(imageList) == 1:
            plt.imshow(imageList[0], cmap=plt.gray())
        else: 
            height, width = _GridLayoutDims(imageList)
            fig, axeslist = plt.subplots(height, width)
            fig.suptitle(title)

            for i, image in enumerate(imageList):
                # fig = figure()
                if isinstance(image, np.ndarray):
                    # ax = fig.add_subplot(101 + ((len(imageList) - (i)) * 10))
                    iRow = i // width
                    iCol = (i - (iRow * width)) % width

                    print("Row %d Col %d" % (iRow, iCol))

                    if height > 1:
                        ax = axeslist[iRow, iCol ]
                    else:
                        ax = axeslist[iCol]

                    ax.imshow(image, cmap=plt.gray(), figure=fig)  
    else:
        return

    plt.show()
    plt.clf()


def ROIRange(start, count, maxVal, minVal=0):
    '''Returns a range that falls within the limits, but contains count entries.'''

    r = None
    if maxVal - minVal < count:
        return None

    if start < minVal:
        r = list(range(minVal, minVal + count))
    elif start + count >= maxVal:
        r = list(range(maxVal - count, maxVal))
    else:
        r = list(range(start, start + count))

    return r

def ConstrainedRange(start, count, maxVal, minVal=0):
    '''Returns a range that falls within min/max limits.'''

    end = start + count
    r = None
    if maxVal - minVal < count:
        return list(range(minVal, maxVal))

    if start < minVal:
        r = list(range(minVal, end))
    elif end >= maxVal:
        r = list(range(start, maxVal))
    else:
        r = list(range(start, end))

    return r


def ExtractROI(image, center, area):
    '''Returns an ROI around a center point with the area, if the area passes a boundary the ROI
       maintains the same area, but is shifted so the entire area remains in the image.
       USES NUMPY (Y,X) INDEXING'''

    x_range = ROIRange(area[1], (center - area[1]) / 2.0, maxVal=image.shape[1])
    y_range = ROIRange(area[0], (center - area[0]) / 2.0, maxVal=image.shape[0])

    ROI = image(y_range, x_range)

    return ROI

def ChangeImageDownsample(image, input_downsample, output_downsample):
    scale_factor = int(input_downsample) / output_downsample
    desired_size = scipy.array(image.shape) * scale_factor
    return scipy.misc.imresize(image, (int(desired_size[0]), int(desired_size[1])))

def ResizeImage(image, scalar):
    '''Change image size by scalar'''
    
    interp = 'bilinear'
    if scalar < 1.0:
        interp = 'bicubic'

    new_size = np.array(image.shape, dtype=np.float) * scalar
    
    return scipy.misc.imresize(image, np.array(new_size, dtype=np.int64), interp=interp)

def CropImage(imageparam, Xo, Yo, Width, Height, cval=None):
    '''
       Crop the image at the passed bounds and returns the cropped ndarray.
       If the requested area is outside the bounds of the array then the correct region is returned
       with a background color set
       
       :param ndarray imageparam: An ndarray image to crop.  A string containing a path to an image is also acceptable.e
       :param int Xo: X origin for crop
       :param int Yo: Y origin for crop
       :param int Width: New width of image
       :param int Height: New height of image
       :param int cval: default value for regions outside the original image boundaries.  Defaults to 0.  Use 'random' to fill with random noise matching images statistical profile
       
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
    
#     if not isinstance(Width, int):
#         Width = int(Width)
#     
#     if not isinstance(Height, int):
#         Height = int(Height)
        
    assert(isinstance(Width, int))
    assert(isinstance(Height, int))

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
    rMask = None
    if cval is None:
        cropped = np.zeros((Height, Width), dtype=image.dtype)
    elif cval == 'random':
        rMask = np.zeros((Height, Width), dtype=np.bool)
        rMask[out_startY:out_endY, out_startX:out_endX] = True
        cropped = np.ones((Height, Width), dtype=image.dtype)
    else:
        cropped = np.ones((Height, Width), dtype=image.dtype) * cval

    cropped[out_startY:out_endY, out_startX:out_endX] = image[in_startY:in_endY, in_startX:in_endX]
    
    if not rMask is None:
        RandomNoiseMask(cropped, rMask, Copy=False)

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
    image = (np.random.randn(height, width).astype(np.float32) * standardDev) + mean

    if mean - (standardDev * 2) < 0:
        image = abs(image)
    return image


def GetImageSize(ImageFullPath):
    '''
    :returns: Image (height, width)
    :rtype: tuple
    '''

    # if not os.path.exists(ImageFullPath):
        # raise ValueError("%s does not exist" % (ImageFullPath))
        
    (root, ext) = os.path.splitext(ImageFullPath)
    
    image = None
    try:
        if ext == '.npy':
            image = _LoadImageByExtension(ImageFullPath)
            return image.shape
        else:
            image = Image.open(ImageFullPath)
            return (image.size[1], image.size[0])
    except IOError:
        raise IOError("Unable to read size from %s" % (ImageFullPath))
    finally:
        del image

def ForceGrayscale(image):
    '''
    :param: ndarray with 3 dimensions
    :returns: grayscale data 
    :rtype: ndarray with 2 dimensions'''

    if len(image.shape) > 2:
        image = image[:, :, 0]
        return np.squeeze(image)

    return image

def _Image_To_Uint8(image):
    '''Converts image to uint8.  If input image uses floating point the image is scaled to the range 0-255'''
    if image.dtype == np.float32 or image.dtype == np.float16:
        image = image * 255.0

    if image.dtype == np.bool:
        image = image.astype(np.uint8) * 255
    else:
        image = image.astype(np.uint8)
        
    return image

def SaveImage(ImageFullPath, image, **kwargs):
    '''Saves the image as greyscale with no contrast-stretching'''

    (root, ext) = os.path.splitext(ImageFullPath)
    if ext == '.jp2':
        SaveImage_JPeg2000(ImageFullPath, image,  **kwargs)
    elif ext == '.npy':
        np.save(ImageFullPath, image)
    else:
        Uint8_image = _Image_To_Uint8(image)
        del image
        
        im = Image.fromarray(Uint8_image)
        im.save(ImageFullPath)
    

def SaveImage_JPeg2000(ImageFullPath, image, tile_dim=None):
    '''Saves the image as greyscale with no contrast-stretching'''
    
    if tile_dim is None:
        tile_dim = (512,512)
        
    Uint8_image = _Image_To_Uint8(image)
    del image
        
    im = Image.fromarray(Uint8_image)
    im.save(ImageFullPath, tile_size=tile_dim)

#     
# def SaveImage_JPeg2000_Tile(ImageFullPath, image, tile_coord, tile_dim=None):
#     '''Saves the image as greyscale with no contrast-stretching'''
#     
#     if tile_dim is None:
#         tile_dim = (512,512)
# 
#     if image.dtype == np.float32 or image.dtype == np.float16:
#         image = image * 255.0
# 
#     if image.dtype == np.bool:
#         image = image.astype(np.uint8) * 255
#     else:
#         image = image.astype(np.uint8)
# 
#     im = Image.fromarray(image)
#     im.save(ImageFullPath, tile_offset=tile_coord, tile_size=tile_dim)
#     

def _LoadImageByExtension(ImageFullPath, bpp=8):
    (root, ext) = os.path.splitext(ImageFullPath)
    
    image = None
    if ext == '.npy':
        image = np.load(ImageFullPath, 'c') 
    else:
        image = plt.imread(ImageFullPath)
        if bpp == 1:
            image = image.astype(np.bool)
        else:
            image = ForceGrayscale(image)
        
    return image

# @profile
def LoadImage(ImageFullPath, ImageMaskFullPath=None, MaxDimension=None):

    '''
    Loads an image converts to greyscale, masks it, and removes extrema pixels.
    
    :param str ImageFullPath: Path to image
    :param str ImageMaskFullPath: Path to mask, dimension should match input image
    :param MaxDimension: Limit the largest dimension of the returned image to this size.  Downsample if necessary.
    :returns: Loaded image.  Masked areas and extrema pixel values are replaced with gaussian noise matching the median and std. dev. of the unmasked image.
    :rtype: ndimage
    '''
    if(not os.path.isfile(ImageFullPath)):
        logger.error('File does not exist: ' + ImageFullPath)
        return None
    
    (root, ext) = os.path.splitext(ImageFullPath)
    
    image = _LoadImageByExtension(ImageFullPath) 

    if not MaxDimension is None:
        scalar = ScalarForMaxDimension(MaxDimension, image.shape)
        if scalar < 1.0:
            image = ReduceImage(image, scalar)

    image_mask = None

    if(not ImageMaskFullPath is None):
        if(not os.path.isfile(ImageMaskFullPath)):
            logger.error('Fixed image mask file does not exist: ' + ImageMaskFullPath)
        else:
            image_mask = _LoadImageByExtension(ImageMaskFullPath, bpp=1)
            if not MaxDimension is None:
                scalar = ScalarForMaxDimension(MaxDimension, image_mask.shape)
                if scalar < 1.0:
                    image_mask = ReduceImage(image_mask, scalar)

            assert((image.shape == image_mask.shape))
            image = RandomNoiseMask(image, image_mask)

    return image


def NormalizeImage(image):
    '''Adjusts the image to have a range of 0 to 1.0'''

    miniszeroimage = image - image.min()
    scalar = (1.0 / miniszeroimage.max())

    if np.isinf(scalar).all():
        scalar = 1.0

    return miniszeroimage * scalar

def TileGridShape(source_image_shape, tile_size):
    '''Given an image and tile size, return the dimensions of the grid'''
    
    if not isinstance(tile_size, np.ndarray):
        tile_shape = np.asarray(tile_size)
    else:
        tile_shape = tile_size
    
    return np.ceil(source_image_shape / tile_shape).astype(np.int32)
     
def ImageToTiles(source_image, tile_size, grid_shape=None, cval=0):
    '''
    :param ndarray source_image: Image to cut into tiles
    :param array tile_size: Shape of each tile
    :param array grid_shape: Dimensions of grid, if None the grid is large enough to reproduce the source_image with zero padding if needed
    :param object cval: Fill value for images that are padded.  Default is zero.  Use 'random' to generate random noise
    :return: Dictionary of images indexed by tuples
    '''    
    #Build the output dictionary
    grid = {}
    for (iRow, iCol, tile) in ImageToTilesGenerator(source_image, tile_size):
        grid[iRow,iCol] = tile
        
    return grid  


def ImageToTilesGenerator(source_image, tile_size, grid_shape=None, cval=0):
    '''An iterator generating all tiles for an image
    :param array tile_size: Shape of each tile
    :param array grid_shape: Dimensions of grid, if None the grid is large enough to reproduce the source_image with zero padding if needed
    :param object cval: Fill value for images that are padded.  Default is zero.  Use 'random' to generate random noise
    :return: (iCol,iRow, tile_image)
    ''' 
    grid_shape = TileGridShape(source_image.shape, tile_size)
    
    (required_shape) = grid_shape * tile_size 
    
    source_image_padded = CropImage(source_image, Xo=0, Yo=0, Width=int(math.ceil(required_shape[1])), Height=int(math.ceil(required_shape[0])), cval=0)
    
    #Build the output dictionary
    StartY = 0 
    EndY = tile_size[0]
    
    for iRow in range(0, int(grid_shape[0])):
        
        StartX = 0
        EndX = tile_size[1]
    
        for iCol in range(0, int(grid_shape[1])):
            yield (iRow, iCol, source_image_padded[StartY:EndY,StartX:EndX])
        
            StartX += tile_size[1]
            EndX += tile_size[1]    
        
        StartY += tile_size[0]
        EndY += tile_size[0]
        

def GetImageTile(source_image, iRow, iCol, tile_size):
    StartY = tile_size[0] * iRow
    EndY = StartY + tile_size[0]
    StartX = tile_size[1] * iCol
    EndX = StartX + tile_size[1]
    
    return source_image[StartY:EndY,StartX:EndX]


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

    MaskedImage = image
    if Copy:
        MaskedImage = image.copy()

    Mask1D = Mask.flat

    iMasked = Mask1D == 0
    
    NumMaskedPixels = np.sum(iMasked)
    if(NumMaskedPixels == 0):
        return MaskedImage
   
    Image1D = MaskedImage.flat
    
    #iUnmasked = numpy.logical_not(iMasked)
    if(ImageMedian is None or ImageStdDev is None):
        # Create masked array for accurate stats
        
        numUnmaskedPixels = len(Image1D) - NumMaskedPixels 
        if numUnmaskedPixels <= 2:
            if numUnmaskedPixels == 0:
                raise ValueError("Entire image is masked, cannot calculate median or standard deviation")
            else:
                raise ValueError("All but %d pixels are masked, cannot calculate standard deviation" % ())
         
        #Bit of a backward convention here.
        #Need to use float64 so that sum does not return an infinite value
        UnmaskedImage1D = np.ma.masked_array(Image1D, iMasked, dtype=numpy.float64)
         
        if(ImageMedian is None):
            ImageMedian = np.median(UnmaskedImage1D)
        if(ImageStdDev is None):
            ImageStdDev = np.std(UnmaskedImage1D)
            
        del UnmaskedImage1D
 
    NoiseData = GenRandomData(1, NumMaskedPixels, ImageMedian, ImageStdDev)

    # iMasked = transpose(nonzero(iMasked))
    Image1D[iMasked] = NoiseData

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
            ImageMedian = np.median(Image1D)
        if(ImageStdDev is None):
            ImageStdDev = np.std(Image1D)

    num_pixels = len(maxima_index) + len(minima_index)

    OutputImage = np.copy(image)
    
    if num_pixels > 0:
        OutputImage1d = OutputImage.flat
        randData = GenRandomData(num_pixels, 1, ImageMedian, ImageStdDev)
        OutputImage1d[maxima_index] = randData[0:len(maxima_index)]
        OutputImage1d[minima_index] = randData[len(maxima_index):]

    return OutputImage

def NearestPowerOfTwo(val):
    return math.pow(2, math.ceil(math.log(val, 2)))

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

    return val + (val * (1.0 - overlap) * 2.0)

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
            ImageMedian = np.median(Image1D)
        if(ImageStdDev is None):
            ImageStdDev = np.std(Image1D)

    PaddedImage = np.zeros((NewHeight, NewWidth), dtype=np.float16)

    PaddedImageXOffset = np.floor((NewWidth - Width) / 2.0)
    PaddedImageYOffset = np.floor((NewHeight - Height) / 2.0)

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
    
    return FFTPhaseCorrelation(FFTFixed, FFTMoving, True) 
    
    
def FFTPhaseCorrelation(FFTFixed, FFTMoving, delete_input=False):
    '''
    Returns the phase shift correlation of the FFT's of two images. 
    
    Dimensions of Fixed and Moving images must match
    
    :param ndarray FixedImage: grayscale image
    :param ndarray MovingImage: grayscale image
    :returns: Correlation image of the FFT's.  Light pixels indicate the phase is well aligned at that offset.
    :rtype: ndimage
    
    '''

    if(not (FFTFixed.shape == FFTMoving.shape)):
        # TODO, we should pad the smaller image in this case to allow the comparison to continue
        raise ValueError("ImagePhaseCorrelation: Fixed and Moving image do not have same dimension")
 

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

    conjFFTFixed = np.conjugate(FFTFixed)
    if delete_input:
        del FFTFixed

    conjFFTFixed *= FFTMoving
    
    if delete_input:
        del FFTMoving   

    conjFFTFixed /= np.absolute(conjFFTFixed)  # Numerator / Divisor

    CorrelationImage = np.real(fftpack.irfft2(conjFFTFixed))
    del conjFFTFixed

    return CorrelationImage 


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
    LabelSums = scipy.ndimage.measurements.sum(ThresholdImage, LabelImage, list(range(0, NumLabels)))
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

    raise NotImplementedError()


def FindOffset(FixedImage, MovingImage, MinOverlap=0.0, MaxOverlap=1.0, FFT_Required=True):
    '''return an alignment record describing how the images overlap. The alignment record indicates how much the 
       moving image must be rotated and translated to align perfectly with the FixedImage
       '''

    # Find peak requires both the fixed and moving images have equal size
    assert((FixedImage.shape[0] == MovingImage.shape[0]) and (FixedImage.shape[1] == MovingImage.shape[1]))
    
    CorrelationImage = None
    if FFT_Required:
        CorrelationImage = ImagePhaseCorrelation(FixedImage, MovingImage)
    else:
        CorrelationImage = FFTPhaseCorrelation(FixedImage, MovingImage, delete_input=False)
        
    CorrelationImage = np.fft.fftshift(CorrelationImage)

    # Crop the areas that cannot overlap

    CorrelationImage -= CorrelationImage.min()
    CorrelationImage /= CorrelationImage.max()

    # Timer.Start('Find Peak')
    (peak, weight) = FindPeak(CorrelationImage, MinOverlap=MinOverlap, MaxOverlap=MaxOverlap)

    del CorrelationImage

    record = nornir_imageregistration.AlignmentRecord(peak=peak, weight=weight)

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
    [histogram, binEdge] = np.histogram(image, bins=NumBins)

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
        print(str(record))

        stos = record.ToStos(FilenameA, FilenameB)

        stos.Save(os.path.join(OutputDir, "TestPhaseCorrelation.stos"))

        # Timer.End('Find Peak', False)

        # Timer.End('Correlate One Pair', False)

        # print(str(Timer))

        # ShowGrayscale(NormCorrelationImage)
        return

    def SecondMain():


        imA = plt.imread(FilenameA)
        imB = plt.imread(FilenameB)

        for i in range(1, 5):
            print((str(i)))
            TestPhaseCorrelation(imA, imB)

    import cProfile
    import pstats
    cProfile.run('SecondMain()', 'CoreProfile.pr')
    pr = pstats.Stats('CoreProfile.pr')
    pr.sort_stats('time')
    print(str(pr.print_stats(.5)))

