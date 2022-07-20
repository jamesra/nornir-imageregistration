"""
scipy image arrays are indexed [y,x]
"""

import ctypes
import logging
import math
import multiprocessing.sharedctypes
import os
import tempfile
import warnings

from PIL import Image

import scipy.misc
import scipy.ndimage.measurements

import nornir_pools
import nornir_imageregistration
import nornir_shared.images
import nornir_shared.prettyoutput as prettyoutput

import matplotlib.pyplot as plt

import numpy as np
# import numpy.fft.fftpack as fftpack
import scipy.fftpack as fftpack  # Cursory internet research suggested Scipy was faster at this time.  Untested. 
import scipy.ndimage.interpolation as interpolation

from collections.abc import Iterable

# Disable decompression bomb protection since we are dealing with huge images on purpose
Image.MAX_IMAGE_PIXELS = None


# from memory_profiler import profile


class memmap_metadata(object):
    """meta-data for a memmap array"""

    @property
    def path(self):
        return self._path

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        # Default to copy-on-write
        if value is None:
            self._mode = 'c'
            return

        if not isinstance(value, str):
            raise ValueError("Mode must be a string and one of the allowed memmap mode strings, 'r','r+','w+','c'")

        self._mode = value

    def __init__(self, path, shape, dtype, mode=None):
        self._path = path
        self._shape = shape
        self._dtype = dtype
        self._mode = None
        self.mode = mode


def ravel_index(idx, shp):
    """
    Convert a nx2 numpy array of coordinates into array indicies

    The arrays we expect are in this shape [[X1,Y1],
                                    [X2,Y2],
                                    [XN,YN]]
    """
    if shp[0] == 1:
        return idx[:, 1]

    if idx.shape[0] != shp[-1]:
        idx = np.transpose(idx)

    return np.transpose(np.concatenate((np.asarray(shp[1:])[::-1].cumprod()[::-1], [1])).dot(idx))


def index_with_array(image, indicies):
    """
    Returns values from image at the coordinates
    :param ndarray image: Image to index into
    :param ndarray indicies: nx2 array of pixel coordinates
    """

    return np.take(image, ravel_index(indicies, image.shape))
    # return np.reshape(values, (len(values),1))


def array_distance(array):
    """Convert an Mx2 array into a Mx1 array of euclidean distances"""
    if array.ndim == 1:
        return np.sqrt(np.sum(array ** 2))

    return np.sqrt(np.sum(array ** 2, 1))


# def GetBitsPerPixel(File): 
#    return shared_images.GetImageBpp(File)


def ApproxEqual(A, B, epsilon=None):
    if epsilon is None:
        epsilon = 0.01

    return np.abs(A - B) < epsilon


def ImageParamToImageArray(imageparam, dtype=None):
    image = None
    if isinstance(imageparam, np.ndarray):
        if dtype is None:
            image = imageparam
        elif np.issubdtype(imageparam.dtype, np.integer) and np.issubdtype(dtype, np.floating):
            # Scale image to 0.0 to 1.0
            image = imageparam.astype(dtype) / np.iinfo(imageparam.dtype).max
            image.astype(dtype=dtype)
        else:
            image = imageparam.astype(dtype=dtype)

    elif isinstance(imageparam, str):
        image = LoadImage(imageparam, dtype=dtype)

    elif isinstance(imageparam, memmap_metadata):
        if dtype is None:
            dtype = imageparam.dtype

        image = np.memmap(imageparam.path, dtype=imageparam.dtype, mode=imageparam.mode, shape=imageparam.shape)
        if dtype != imageparam.dtype:
            image = image.astype(dtype=dtype)

    if image is None:
        raise ValueError("Image param %s is not a numpy array or image file" % (str(imageparam)))

    return image


def ScalarForMaxDimension(max_dim, shapes):
    """Returns the scalar value to use so the largest dimensions in a list of shapes has the maximum value"""
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


def ExtractROI(image, center, area):
    """Returns an ROI around a center point with the area, if the area passes a boundary the ROI
       maintains the same area, but is shifted so the entire area remains in the image.
       USES NUMPY (Y,X) INDEXING"""

    half_area = area / 2.0
    x_range = SafeROIRange(center - half_area[1], area[1], maxVal=image.shape[1])
    y_range = SafeROIRange(center - half_area[0], area[0], maxVal=image.shape[0])

    ROI = image(y_range, x_range)

    return ROI


def SafeROIRange(start, count, maxVal, minVal=0):
    """
    Returns a range cropped within min and max values, but always attempts to have count entries in the ROI.
    If minVal or maxVal would crop the list then start is shifted to ensure the resulting value has the correct number of entries.
    :param int start: Starting value
    :param int count: Number of items in the list, incremented by 1, to return.
    :param int maxVal: Maximum value allowed to be returned.  Output list will be cropped if it equals or exceeds this value.
    :param int minVal: Minimum value allowed to be returned.  Output list will be cropped below this value.
    :return:  [start start+1, start+2, ..., start+count]
    """

    if count == 0:
        return list()

    if maxVal < minVal:
        raise ValueError(f"maxVal must be greater than minVal. {maxVal} > {minVal}");

    if maxVal - minVal < count:
        raise ValueError(
            f"Not enough room to return a ROI of requested size.  maxVal - minVal must be >= count. {maxVal} - {minVal} >= {count}");

    r = None

    if start < minVal:
        r = list(range(minVal, minVal + count))
    elif start + count >= maxVal:
        r = list(range(maxVal - count, maxVal))
    else:
        r = list(range(start, start + count))

    return r


def ConstrainedRange(start, count, maxVal, minVal=0):
    """Returns a range that falls within min/max limits."""

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


def _ShrinkNumpyImageFile(InFile, OutFile, Scalar):
    image = nornir_imageregistration.LoadImage(InFile)
    resized_image = nornir_imageregistration.ResizeImage(image, Scalar)
    nornir_imageregistration.SaveImage(OutFile, resized_image)


def _ShrinkPillowImageFile(InFile, OutFile, Scalar, **kwargs):
    resample = kwargs.pop('resample', None)

    if resample is None:
        resample = resample = Image.BILINEAR
        if Scalar < 1.0:
            resample = Image.LANCZOS

    with Image.open(InFile, mode='r') as img:

        dims = np.asarray(img.size).astype(dtype=np.float32)
        desired_dims = dims * Scalar
        desired_dims = np.around(desired_dims).astype(dtype=np.int64)

        shrunk_img = img.resize(size=desired_dims, resample=resample)
        img.close()
        del img

        shrunk_img.save(OutFile, **kwargs)
        shrunk_img.close()
        del shrunk_img

    return None


# Shrinks the passed image file, return procedure handle of invoked command
def Shrink(InFile, OutFile, Scalar, **kwargs):
    """Shrinks the passed image file.  If Pool is not None the
       task is returned. kwargs are passed on to Pillow's image save function
       :param str InFile: Path to input file
       :param str OutFile: Path to output file
       :param float ShrinkFactor: Multiplier for image dimensions
    """
    (root, ext) = os.path.splitext(InFile)
    if ext == '.npy':
        _ShrinkNumpyImageFile(InFile, OutFile, Scalar, **kwargs)
    else:
        _ShrinkPillowImageFile(InFile, OutFile, Scalar, **kwargs)


def ResizeImage(image, scalar):
    """Change image size by scalar"""

    original_min = image.min()
    original_max = image.max()

    order = 2
    if isinstance(scalar, float) and scalar < 1.0:
        order = 3
    elif hasattr(scalar, "__iter__"):
        scalar = nornir_imageregistration.EnsurePointsAre1DNumpyArray(scalar)
        order = 3 if any([s < 1.0 for s in scalar]) else 2

    # new_size = np.array(image.shape, dtype=np.float) * scalar

    return scipy.ndimage.zoom(image, zoom=scalar, order=order).clip(original_min, original_max)


def _ConvertSingleImage(input_image_param, Flip=False, Flop=False, Bpp=None, Invert=False, MinMax=None, Gamma=None):
    """Converts a single image according to the passed parameters using Numpy"""

    image = ImageParamToImageArray(input_image_param)
    original_dtype = image.dtype
    max_possible_int_val = None

    # max_possible_float_val = 1.0

    NeedsClip = False

    # After lots of pain it is simplest to ensure all images are represented by floats before operating on them
    if nornir_imageregistration.IsIntArray(original_dtype):
        max_possible_int_val = nornir_imageregistration.ImageMaxPixelValue(image)
        image = image.astype(np.float32) / max_possible_int_val

    if Flip is not None and Flip:
        image = np.flipud(image)

    if Flop is not None and Flop:
        image = np.fliplr(image)

    if MinMax is not None:
        (min_val, max_val) = MinMax

        if nornir_imageregistration.IsIntArray(original_dtype) == True:
            min_val = min_val / max_possible_int_val
            max_val = max_val / max_possible_int_val

        if min_val is None:
            min_val = 0

        if max_val is None:
            max_val = 1.0

        max_minus_min = max_val - min_val
        image = image - min_val
        image = image / max_minus_min

        NeedsClip = True

    if Gamma is None:
        Gamma = 1.0

    if Gamma != 1.0:
        image = np.float_power(image, 1.0 / Gamma, where=image >= 0)
        NeedsClip = True

    if NeedsClip:
        np.clip(image, a_min=0, a_max=1.0, out=image)

    if Invert is not None and Invert:
        image = 1.0 - image

    if nornir_imageregistration.IsIntArray(original_dtype) == True:
        image = image * max_possible_int_val

    image = image.astype(original_dtype)

    return image


def _ConvertSingleImageToFile(input_image_param, output_filename, Flip=False, Flop=False, InputBpp=None, OutputBpp=None,
                              Invert=False, MinMax=None, Gamma=None):
    image = _ConvertSingleImage(input_image_param,
                                Flip=Flip,
                                Flop=Flop,
                                Bpp=InputBpp,
                                Invert=Invert,
                                MinMax=MinMax,
                                Gamma=Gamma)

    if OutputBpp is None:
        OutputBpp = InputBpp

    (_, ext) = os.path.splitext(output_filename)
    if ext.lower() == '.png':
        nornir_imageregistration.SaveImage(output_filename, image, bpp=OutputBpp, optimize=True)
    else:
        nornir_imageregistration.SaveImage(output_filename, image, bpp=OutputBpp)
    return


def ConvertImagesInDict(ImagesToConvertDict, Flip=False, Flop=False, InputBpp=None, OutputBpp=None, Invert=False,
                        bDeleteOriginal=False, RightLeftShift=None, AndValue=None, MinMax=None, Gamma=None):
    """
    The key and value in the dictionary have the full path of an image to convert.
    MinMax is a tuple [Min,Max] passed to the -level parameter if it is not None
    RightLeftShift is a tuple containing a right then left then return to center shift which should be done to remove useless bits from the data
    I do not use an and because I do not calculate ImageMagick's quantum size yet.
    Every image must share the same colorspace

    :return: True if images were converted
    :rtype: bool
    """

    if len(ImagesToConvertDict) == 0:
        return False

    if InputBpp is None:
        for k in ImagesToConvertDict.keys():
            if os.path.exists(k):
                InputBpp = nornir_shared.images.GetImageBpp(k)
                break

    prettyoutput.CurseString('Stage', "ConvertImagesInDict")

    if MinMax is not None:
        if MinMax[0] > MinMax[1]:
            raise ValueError("Invalid MinMax parameter passed to ConvertImagesInDict")

    num_threads = multiprocessing.cpu_count() * 2
    if num_threads > len(ImagesToConvertDict):
        num_threads = len(ImagesToConvertDict) + 1

    pool = nornir_pools.GetMultithreadingPool("ConvertImagesInDict", num_threads=num_threads)
    # pool = nornir_pools.GetGlobalSerialPool()
    tasks = []

    for (input_image, output_image) in ImagesToConvertDict.items():
        task = pool.add_task("{0} -> {1}".format(input_image, output_image),
                             _ConvertSingleImageToFile,
                             input_image_param=input_image,
                             output_filename=output_image,
                             Flip=Flip,
                             Flop=Flop,
                             InputBpp=InputBpp,
                             OutputBpp=OutputBpp,
                             Invert=Invert,
                             MinMax=MinMax,
                             Gamma=Gamma)
        tasks.append(task)

    while len(tasks) > 0:
        t = tasks.pop(0)
        try:
            t.wait()
        except Exception as e:
            prettyoutput.LogErr("Failed to convert " + t.name)

    if bDeleteOriginal:
        for (input_image, output_image) in ImagesToConvertDict.items():
            if input_image != output_image:
                pool.add_task("Delete {0}".format(input_image), os.remove, input_image)

        while len(tasks) > 0:
            t = tasks.pop(0)
            try:
                t.wait()
            except OSError as e:
                prettyoutput.LogErr("Unable to delete {0}\n{1}".format(t.name, e))
                pass
            except IOError as e:
                prettyoutput.LogErr("Unable to delete {0}\n{1}".format(t.name, e))
                pass

    if not pool is None:
        pool.wait_completion()
        pool.shutdown()
        pool = None

    del tasks


def CropImageRect(imageparam, bounding_rect, cval=None):
    return CropImage(imageparam, Xo=int(bounding_rect[1]), Yo=int(bounding_rect[0]), Width=int(bounding_rect.Width),
                     Height=int(bounding_rect.Height), cval=cval)


def CropImage(imageparam, Xo, Yo, Width, Height, cval=None):
    """
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
       """

    image = ImageParamToImageArray(imageparam)

    if image is None:
        return None

    #     if not isinstance(Width, int):
    #         Width = int(Width)
    #
    #     if not isinstance(Height, int):
    #         Height = int(Height)

    assert (isinstance(Width, int))
    assert (isinstance(Height, int))

    if Width < 0:
        raise ValueError("Negative dimensions are not allowed")

    if Height < 0:
        raise ValueError("Negative dimensions are not allowed")

    image_rectangle = nornir_imageregistration.Rectangle([0, 0, image.shape[0], image.shape[1]])
    crop_rectangle = nornir_imageregistration.Rectangle.CreateFromPointAndArea([Yo, Xo], [Height, Width])

    overlap_rectangle = nornir_imageregistration.Rectangle.overlap_rect(image_rectangle, crop_rectangle)

    in_startY = Yo
    in_startX = Xo
    in_endX = Xo + Width
    in_endY = Yo + Height

    out_startY = 0
    out_startX = 0
    out_endX = Width
    out_endY = Height

    if overlap_rectangle is None:
        out_startY = 0
        out_startX = 0
        out_endX = 0
        out_endY = 0

        in_startY = Yo
        in_startX = Xo
        in_endX = Xo
        in_endY = Yo
    else:
        (in_startY, in_startX) = overlap_rectangle.BottomLeft
        (in_endY, in_endX) = overlap_rectangle.TopRight

        (out_startY, out_startX) = overlap_rectangle.BottomLeft - crop_rectangle.BottomLeft
        (out_endY, out_endX) = np.array([out_startY, out_startX]) + overlap_rectangle.Size

    # To correct a numpy warning, convert values to int
    in_startX = int(in_startX)
    in_startY = int(in_startY)
    in_endX = int(in_endX)
    in_endY = int(in_endY)

    out_startX = int(out_startX)
    out_startY = int(out_startY)
    out_endX = int(out_endX)
    out_endY = int(out_endY)

    # Create mask
    rMask = None
    if cval == 'random':
        rMask = np.zeros((Height, Width), dtype=np.bool)
        rMask[out_startY:out_endY, out_startX:out_endX] = True

    # Create output image
    cropped = None
    if cval is None:
        cropped = np.zeros((Height, Width), dtype=image.dtype)
    elif cval == 'random':
        cropped = np.ones((Height, Width), dtype=image.dtype)
    else:
        cropped = np.ones((Height, Width), dtype=image.dtype) * cval

    cropped[out_startY:out_endY, out_startX:out_endX] = image[in_startY:in_endY, in_startX:in_endX]

    if not rMask is None:
        return RandomNoiseMask(cropped, rMask, Copy=False)

    return cropped


def npArrayToReadOnlySharedArray(npArray):
    """Returns a shared memory array for a numpy array.  Used to reduce memory footprint when passing parameters to multiprocess pools"""
    SharedBase = multiprocessing.sharedctypes.RawArray(ctypes.c_float, npArray.shape[0] * npArray.shape[1])
    SharedArray = np.ctypeslib.as_array(SharedBase)
    SharedArray = SharedArray.reshape(npArray.shape)
    np.copyto(SharedArray, npArray)
    return SharedArray


def CreateTemporaryReadonlyMemmapFile(npArray):
    with tempfile.NamedTemporaryFile(suffix='.memmap', delete=False) as hFile:
        TempFullpath = hFile.name
        hFile.close()
    memImage = np.memmap(TempFullpath, dtype=npArray.dtype, shape=npArray.shape, mode='w+')
    memImage[:] = npArray[:]
    memImage.flush()
    del memImage
    # np.save(TempFullpath, npArray)
    return memmap_metadata(path=TempFullpath, shape=npArray.shape, dtype=npArray.dtype)


def GenRandomData(height, width, mean, standardDev, min_val, max_val):
    """
    Generate random data of shape with the specified mean and standard deviation
    """
    image = (np.random.standard_normal((int(height), int(width))).astype(np.float32) * standardDev) + mean

    np.clip(image, a_min=min_val, a_max=max_val, out=image)
    return image


def GetImageSize(image_param: str):
    """
    :param image_param str: Either a path to an image file, an ndarray, or a list
    of paths/ndimages
    :returns: The image's (height, width) or [(height,width),...] for a list
    :rtype: tuple
    """

    if isinstance(image_param, str):
        return nornir_shared.images.GetImageSize(image_param)
    elif isinstance(image_param, np.ndarray):
        return image_param.shape
    elif isinstance(image_param, Iterable):
        return [GetImageSize(i) for i in image_param]

    raise ValueError(f'Unexpected image argument {image_param}')


def ForceGrayscale(image: np.ndarray):
    """
    :param: ndarray with 3 dimensions
    :returns: grayscale data
    :rtype: ndarray with 2 dimensions"""

    if len(image.shape) > 2:
        image = image[:, :, 0]
        return np.squeeze(image)

    return image


def _Image_To_Uint8(image):
    """Converts image to uint8.  If input image uses floating point the image is scaled to the range 0-255"""
    if image.dtype == np.uint8:
        return image

    elif image.dtype == np.bool:
        image = image.astype(np.uint8) * 255

    elif nornir_imageregistration.IsFloatArray(image.dtype):
        iMax = image.max()
        if iMax <= 1:
            image = image * 255.0
        else:
            pass
            # image = #(255.0 / iMax)
    elif nornir_imageregistration.IsIntArray(image.dtype):
        iMax = image.max()
        if iMax > 255:
            image = image / (iMax / 255.0)

    image = image.astype(np.uint8)

    return image


def OneBit_img_from_bool_array(data):
    """
    Covers for pillow bug with bit images
    https://stackoverflow.com/questions/50134468/convert-boolean-numpy-array-to-pillow-image
    """
    size = data.shape[::-1]

    if data.dtype == np.bool:
        return Image.frombytes(mode='1', size=size, data=np.packbits(data, axis=1))
    else:
        return Image.frombytes(mode='1', size=size, data=np.packbits(data > 0, axis=1))


def uint16_img_from_uint16_array(data):
    """
    Covers for pillow bug with bit images
    https://github.com/python-pillow/Pillow/issues/2970
    """
    assert (nornir_imageregistration.IsIntArray(data))

    size = data.shape[::-1]
    img = Image.new("I", size=data.T.shape)
    img.frombytes(data.tobytes(), 'raw', 'I;16')
    return img


def uint16_img_from_float_array(image):
    """
    Covers for pillow bug with bit images
    https://github.com/python-pillow/Pillow/issues/2970
    """
    assert (nornir_imageregistration.IsFloatArray(image))
    iMax = image.max()
    if iMax <= 1:
        image = image * (1 << 16) - 1
    else:
        pass

    return image.astype(np.uint16)


def SaveImage(ImageFullPath, image, bpp=None, **kwargs):
    """Saves the image as greyscale with no contrast-stretching
    :param str ImageFullPath: The filename to save
    :param ndarray image: The image data to save
    :param int bpp: The bit depth to save, if the image data bpp is higher than this value it will be reduced.  Otherwise only the bpp required to preserve the image data will be used. (8-bit data will not be upsampled to 16-bit)
    """
    dirname = os.path.dirname(ImageFullPath)
    if dirname is not None and len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)

    if bpp is None:
        bpp = nornir_imageregistration.ImageBpp(image)
        if bpp > 16:
            prettyoutput.LogErr(
                "Saving image at 32 bits-per-pixel, check SaveImageParameters for efficiency:\n{0}".format(
                    ImageFullPath))

    if bpp > 8:
        # Ensure we even have the data to bother saving a higher bit depth
        detected_bpp = nornir_imageregistration.ImageBpp(image)
        if detected_bpp < bpp:
            bpp = detected_bpp

    (root, ext) = os.path.splitext(ImageFullPath)
    if ext == '.jp2':
        SaveImage_JPeg2000(ImageFullPath, image, **kwargs)
    elif ext == '.npy':
        np.save(ImageFullPath, image)
    else:
        if np.issubdtype(image.dtype, bool) or bpp == 1:
            # Covers for pillow bug with bit images
            # https://stackoverflow.com/questions/50134468/convert-boolean-numpy-array-to-pillow-image
            # im = Image.fromarray(image.astype(np.uint8) * 255, mode='L').convert('1')
            im = OneBit_img_from_bool_array(image)
        elif bpp == 8:
            Uint8_image = _Image_To_Uint8(image)
            del image
            im = Image.fromarray(Uint8_image, mode="L")
        elif nornir_imageregistration.IsFloatArray(image):
            # TODO: I believe Pillow-SIMD finally added the ability to save I;16 for 16bpp PNG images 
            if image.dtype == np.float16:
                image = image.astype(np.float32)

            im = Image.fromarray(image * ((1 << bpp) - 1))
            im = im.convert('I')
        else:
            if bpp < 32:
                if ext.lower() == '.png':
                    im = uint16_img_from_uint16_array(image)
                else:
                    im = Image.fromarray(image, mode="I;{0}".format(bpp))
            else:
                im = Image.fromarray(image, mode="I".format(bpp))

        im.save(ImageFullPath, **kwargs)

    return


def SaveImage_JPeg2000(ImageFullPath, image, tile_dim=None):
    """Saves the image as greyscale with no contrast-stretching"""

    if tile_dim is None:
        tile_dim = (512, 512)

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


def _LoadImageByExtension(ImageFullPath, dtype):
    """
    Loads an image file and returns an ndarray of dtype
    :param dtype dtype: Numpy datatype of returned array. If the type is a float then the returned array is in the range 0 to 1.  Otherwise it is whatever pillow and numpy decide.
    """
    (root, ext) = os.path.splitext(ImageFullPath)

    image = None
    try:
        if ext == '.npy':
            image = np.load(ImageFullPath, 'c').astype(dtype)
        else:
            # image = plt.imread(ImageFullPath)
            with Image.open(ImageFullPath) as im:

                expected_dtype = nornir_imageregistration.pillow_helpers.dtype_for_pillow_image(im)
                image = np.array(im, dtype=expected_dtype)
                max_pixel_val = nornir_imageregistration.ImageMaxPixelValue(image)

                if dtype is not None:
                    if nornir_imageregistration.IsIntArray(image.dtype) and nornir_imageregistration.IsFloatArray(
                            dtype):
                        # Ensure we remap values to the range of 0 to 1 without loss before converting to desired floating type
                        # if image.dtype.itemsize == dtype.itemsize: #Check if we need to bump up the item size
                        image = image / image.max()

                    image = image.astype(dtype)
                #                 else:
                #                     #Reduce to smallest integer type that can hold the data
                #                     if im.mode[0] == 'I' and (np.issubdtype(image.dtype, np.int32) or np.issubdtype(image.dtype, np.uint32)):
                #                         (min_val, max_val) = im.getextrema()
                #                         smallest_dtype = np.uint32
                #                         if max_val <= 65535:
                #                             smallest_dtype = np.uint16
                #                         if max_val <= 255:
                #                             smallest_dtype = np.uint8
                #
                #                         image = image.astype(smallest_dtype)
                #
                #                     dtype = image.dtype

                # Ensure data is in the range 0 to 1 for floating types
                elif nornir_imageregistration.IsFloatArray(dtype):

                    if im.mode[0] == 'F':
                        (_, im_max_val) = im.getextrema()
                        if im_max_val <= 1.0:
                            return image

                    max_val = max_pixel_val
                    if max_val > 0:
                        image = image / max_val

                im.close()

    except IOError as E:
        prettyoutput.LogErr("IO error loading image {0}\n{1}".format(ImageFullPath, str(E)))
        raise E
    except Exception as E:
        prettyoutput.LogErr("Unexpected exception loading image {0}\n{1}".format(ImageFullPath, str(E)))
        raise E

    return image


# @profile
def LoadImage(ImageFullPath, ImageMaskFullPath=None, MaxDimension=None, dtype=None):
    """
    Loads an image converts to greyscale, masks it, and removes extrema pixels.

    :param str ImageFullPath: Path to image
    :param str ImageMaskFullPath: Path to mask, dimension should match input image
    :param MaxDimension: Limit the largest dimension of the returned image to this size.  Downsample if necessary.
    :returns: Loaded image.  Masked areas and extrema pixel values are replaced with gaussian noise matching the median and std. dev. of the unmasked image.
    :rtype: ndimage
    """
    if not os.path.isfile(ImageFullPath):
        # logger = logging.getLogger(__name__)
        prettyoutput.LogErr('File does not exist: ' + ImageFullPath)
        raise IOError("Unable to load image: %s" % ImageFullPath)

    (root, ext) = os.path.splitext(ImageFullPath)

    image = _LoadImageByExtension(ImageFullPath, dtype)

    if not MaxDimension is None:
        scalar = ScalarForMaxDimension(MaxDimension, image.shape)
        if scalar < 1.0:
            image = ReduceImage(image, scalar)

    image_mask = None

    if not ImageMaskFullPath is None:
        if not os.path.isfile(ImageMaskFullPath):
            # logger = logging.getLogger(__name__)
            prettyoutput.LogErr('Fixed image mask file does not exist: ' + ImageMaskFullPath)
        else:
            image_mask = _LoadImageByExtension(ImageMaskFullPath, np.bool)
            if not MaxDimension is None:
                scalar = ScalarForMaxDimension(MaxDimension, image_mask.shape)
                if scalar < 1.0:
                    image_mask = ReduceImage(image_mask, scalar)

            assert (image.shape == image_mask.shape)
            image = RandomNoiseMask(image, image_mask)

    return image


def NormalizeImage(image):
    """Adjusts the image to have a range of 0 to 1.0"""

    miniszeroimage = image - image.min()
    scalar = (1.0 / miniszeroimage.max())

    if np.isinf(scalar).all():
        scalar = 1.0

    typecode = 'f%d' % image.dtype.itemsize
    return (miniszeroimage * scalar).astype(typecode)


def TileGridShape(source_image_shape, tile_size):
    """Given an image and tile size, return the dimensions of the grid"""

    if isinstance(source_image_shape, nornir_imageregistration.Rectangle):
        source_image_shape = source_image_shape.shape

    if not isinstance(tile_size, np.ndarray):
        tile_shape = np.asarray(tile_size)
    else:
        tile_shape = tile_size

    return np.ceil(source_image_shape / tile_shape).astype(np.int32)


def ImageToTiles(source_image, tile_size, grid_shape=None, cval=0):
    """
    :param ndarray source_image: Image to cut into tiles
    :param array tile_size: Shape of each tile
    :param array grid_shape: Dimensions of grid, if None the grid is large enough to reproduce the source_image with zero padding if needed
    :param object cval: Fill value for images that are padded.  Default is zero.  Use 'random' to generate random noise
    :return: Dictionary of images indexed by tuples
    """
    # Build the output dictionary
    grid = {}
    for (iRow, iCol, tile) in ImageToTilesGenerator(source_image, tile_size):
        grid[iRow, iCol] = tile

    return grid


def ImageToTilesGenerator(source_image, tile_size, grid_shape=None, coord_offset=None, cval=0):
    """An iterator generating all tiles for an image
    :param array tile_size: Shape of each tile
    :param array grid_shape: Dimensions of grid, if None the grid is large enough to reproduce the source_image with zero padding if needed
    :param tuple coord_offset: Add this amount to coordinates returned by this function, used if the image passed is part of a larger image
    :param object cval: Fill value for images that are padded.  Default is zero.  Use 'random' to generate random noise
    :return: (iCol,iRow, tile_image)
    """
    source_image = ImageParamToImageArray(source_image)

    grid_shape = TileGridShape(source_image.shape, tile_size)

    if coord_offset is None:
        coord_offset = (0, 0)

    (required_shape) = grid_shape * tile_size

    source_image_padded = None
    if not np.array_equal(source_image.shape, required_shape):
        source_image_padded = CropImage(source_image,
                                        Xo=0, Yo=0,
                                        Width=int(math.ceil(required_shape[1])),
                                        Height=int(math.ceil(required_shape[0])),
                                        cval=0)
    else:
        source_image_padded = source_image

    # nornir_imageregistration.ShowGrayscale(source_image_padded)

    # Build the output dictionary
    StartY = 0
    EndY = tile_size[0]

    for iRow in range(grid_shape[0]):

        StartX = 0
        EndX = tile_size[1]

        for iCol in range(grid_shape[1]):
            t = (iRow + coord_offset[0], iCol + coord_offset[1], source_image_padded[StartY:EndY, StartX:EndX])
            # nornir_imageregistration.ShowGrayscale(tile)
            (yield t)

            StartX += tile_size[1]
            EndX += tile_size[1]

        StartY += tile_size[0]
        EndY += tile_size[0]

    return


def GetImageTile(source_image, iRow, iCol, tile_size):
    StartY = tile_size[0] * iRow
    EndY = StartY + tile_size[0]
    StartX = tile_size[1] * iCol
    EndX = StartX + tile_size[1]

    return source_image[StartY:EndY, StartX:EndX]


def RandomNoiseMask(image, Mask, imageStats=None, Copy=False):
    """
    Fill the masked area with random noise with gaussian distribution about the image
    mean and with standard deviation matching the image's standard deviation.  Mask
    pixels that are False will be replaced with random noise

    :param ndimage image: Input image
    :param ndimage Mask: Mask, zeros are replaced with noise.  Ones pull values from input image
    :param ImageStats imageStats: Image stats.  Calculated from image if none    
    :param bool Copy: Returns a copy of input image if true, otherwise write noise to the input image
    :rtype: ndimage
    """

    assert (image.shape == Mask.shape)

    MaskedImage = image
    if Copy:
        MaskedImage = image.copy()
  
    iPixelsToReplace = Mask.flat == 0

    numInvalidPixels = np.sum(iPixelsToReplace)

    Image1D = MaskedImage.flat 
    numValidPixels = len(Image1D) - numInvalidPixels
    if numInvalidPixels == 0:
        # Entire image is masked, there is no noise to create
        return MaskedImage
    
    if imageStats is None:
        # Create masked array for accurate stats
        if numValidPixels == 0:
            raise ValueError("Entire image is masked, cannot calculate median or standard deviation")
            # return MaskedImage
        elif numValidPixels <= 2:
            raise ValueError(f"All but {numValidPixels} pixels are masked, cannot calculate statistics")
    
        UnmaskedImage1D = np.ma.masked_array(Image1D, iPixelsToReplace, dtype=np.float64).compressed()
        imageStats = nornir_imageregistration.ImageStats.Create(UnmaskedImage1D)
        del UnmaskedImage1D

    NoiseData = imageStats.GenerateNoise(np.array((1, numInvalidPixels)))

    # iPixelsToReplace = transpose(nonzero(iPixelsToReplace))
    Image1D[iPixelsToReplace] = NoiseData

    # NewImage = reshape(Image1D, (Height, Width), 2)

    return MaskedImage


def CreateExtremaMask(image: np.ndarray, mask: np.ndarray=None, size_cutoff=0.001, minima=None, maxima=None):
    """
    Returns a mask for features above a set size that are at max or min pixel value
    :param numpy.ndarray mask: Pixels we wish to not include in the analysis
    :param size_cutoff: Determines how large a continous region must be before it is masked. If 0 to 1 this is a fraction of total area.  If > 1 it is an absolute count of pixels. If None all min/max are masked regardless of size
    """
    # (minima, maxima, iMin, iMax) = scipy.ndimage.measurements.extrema(image)
     
    if mask is not None:
        image = np.ma.masked_array(image, np.logical_not(mask))
 
    if minima is None:
        minima = image.min()

    if maxima is None:
        maxima = image.max()

    extrema_mask = np.logical_or(image == maxima, image == minima)
    
    if mask is not None:
        extrema_mask = np.logical_or(extrema_mask, np.logical_not(mask))

    if size_cutoff is None:
        return extrema_mask
    else:
        (extrema_mask_label, nLabels) = scipy.ndimage.label(extrema_mask)
        if nLabels == 0:  # If there are no labels, do not mask anything
            return np.ones(image.shape, extrema_mask.dtype)

        label_sums = scipy.ndimage.measurements.sum_labels(extrema_mask, extrema_mask_label, list(range(0, nLabels)))

        cutoff_value = None
        # if cutoff value is less than one treat it as a fraction of total area
        if size_cutoff <= 1.0:
            cutoff_value = np.prod(image.shape) * size_cutoff
        elif isinstance(size_cutoff, int) == False:
            warnings.warn(
                f"Expecting an integer to specify min area of labels to mask in CreateExtremaMask.  Got {size_cutoff}.")
            cutoff_value = size_cutoff
        else:
            cutoff_value = size_cutoff

        labels_to_save = label_sums < cutoff_value
        if np.any(labels_to_save):
            cutoff_labels = np.nonzero(labels_to_save)
            extrema_mask_minus_small_features = np.isin(extrema_mask_label, cutoff_labels)

        # nornir_imageregistration.ShowGrayscale((image, extrema_mask, extrema_mask_minus_small_features))

        return extrema_mask_minus_small_features


def ReplaceImageExtremaWithNoise(image: np.ndarray, imagemask: np.ndarray = None, imageStats= None,
                                 size_cutoff: float = 0.001, Copy=True):
    """
    Replaced the min/max values in the image with random noise.  This is useful when aligning images composed mostly of dark or bright regions. 
    It is usually best to pass None for statistical parameters since the function will calculate the statistics with the extrema removed.
    :param numpy.ndarray imagemask: Additional pixels we wish to be included in the extrema mask
    :param float ImageMedian: Median pixel value, if known, to use for random noise output.  Will be calculated if not passed.  
    :param float ImageStdDev: Standard Deviation of pixel value, if known, to use for random noise output.  Will be calculated if not passed
    :param size_cutoff: 0 to 1.0, determines how large a continuos min or max region must be before it is masked. If None all min/max are masked regardless of size.  Defaults to 0.001, None will mask all min/max
    """

    # If profiling shows this is slow there are older implementations in git
    mask = CreateExtremaMask(image, imagemask, size_cutoff=size_cutoff)
      
    noised_image = nornir_imageregistration.RandomNoiseMask(image, mask, imageStats, Copy=Copy)
    return noised_image


def NearestPowerOfTwo(val):
    return math.pow(2, math.ceil(math.log(val, 2)))


def NearestPowerOfTwoWithOverlap(val, overlap=1.0):
    """
    :param float overlap: Minimum amount of overlap possible between images, from 0 to 1.  Values greater than 0.5 require no increase to image size.
    :return: Same as DimensionWithOverlap, but output dimension is increased to the next power of two for faster FFT operations
    """

    if overlap is None:
        overlap = 0.0

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
    """
    :param float val: Original dimension
    :param float overlap: Amount of overlap possible between images, from 0 to 1
    :returns: Required dimension size to unambiguously determine the offset in an fft image
    """

    # An overlap of 50% is half of the image, so we don't need to expand the image to find the peak in the correct quadrant
    if overlap >= 0.5:
        return val

    overlap += 0.5

    return val + (val * (1.0 - overlap) * 2.0)


# @profile
def PadImageForPhaseCorrelation(image, MinOverlap=.05, ImageMedian=None, ImageStdDev=None, OriginalShape=None,
                                NewWidth=None, NewHeight=None, PowerOfTwo=True, AlwaysCopy=True):
    """
    Prepares an image for use with the phase correlation operation.  Padded areas are filled with noise matching the histogram of the
    original image.

    :param ndarray image: Input image
    :param float MinOverlap: Minimum overlap allowed between the input image and images it will be registered to
    :param float ImageMean: Median value of noise, calculated or pulled from cache if none
    :param float ImageStdDev: Standard deviation of noise, calculated or pulled from cache if none
    :param tuple OriginalShape: The original size of the image, if None the shape of the input image is used.  Set this if the image has been previously been padded to prevent over-padding
    :param int NewWidth: Pad input image to this dimension if not none
    :param int NewHeight: Pad input image to this dimension if not none
    :param bool PowerOfTwo: Pad the image to a power of two if true
    :param bool AlwaysCopy: If true, always copy the image even if no padding is needed
    :return: An image with the input image centered surrounded by noise
    :rtype: ndimage


    """
    MinVal = image.min()
    MaxVal = image.max()

    Height = image.shape[0]
    Width = image.shape[1]
    OriginalHeight = Height
    OriginalWidth = Width

    if OriginalShape is not None:
        OriginalWidth = OriginalShape[1]
        OriginalHeight = OriginalShape[0]

    if NewHeight is None:
        if PowerOfTwo:
            NewHeight = NearestPowerOfTwoWithOverlap(OriginalHeight, MinOverlap)
        else:
            NewHeight = DimensionWithOverlap(OriginalHeight,
                                             MinOverlap)  # # Height + (Height * (1 - MinOverlap))  # + 1

    if NewWidth is None:
        if PowerOfTwo:
            NewWidth = NearestPowerOfTwoWithOverlap(OriginalWidth, MinOverlap)
        else:
            NewWidth = DimensionWithOverlap(OriginalWidth, MinOverlap)  # # Height + (Height * (1 - MinOverlap))  # + 1

    if NewWidth < image.shape[
        1]:  # If we need a smaller size than we already are (from padding an image a 2nd time) then keep current size
        NewWidth = image.shape[1]

    if NewHeight < image.shape[0]:
        NewHeight = image.shape[0]

    if Width >= NewWidth and Height >= NewHeight:
        if AlwaysCopy:
            return np.copy(image)
        else:
            return image

    if ImageMedian is None or ImageStdDev is None:
        Image1D = image.astype(np.float64).flat

        if ImageMedian is None:
            ImageMedian = np.median(Image1D)
        if ImageStdDev is None:
            ImageStdDev = np.std(Image1D)

    desired_type = np.float16
    if np.finfo(desired_type).max < MaxVal:
        desired_type = np.float32

    PaddedImage = np.zeros((int(NewHeight), int(NewWidth)), dtype=desired_type)

    PaddedImageXOffset = int(np.floor((NewWidth - Width) / 2.0))
    PaddedImageYOffset = int(np.floor((NewHeight - Height) / 2.0))

    # Copy image into padded image
    PaddedImage[PaddedImageYOffset:PaddedImageYOffset + Height, PaddedImageXOffset:PaddedImageXOffset + Width] = image[
                                                                                                                 :, :]

    if not Width == NewWidth:
        LeftBorder = GenRandomData(NewHeight, PaddedImageXOffset, ImageMedian, ImageStdDev, MinVal, MaxVal)
        RightBorder = GenRandomData(NewHeight, NewWidth - (Width + PaddedImageXOffset), ImageMedian, ImageStdDev,
                                    MinVal, MaxVal)

        PaddedImage[:, 0:PaddedImageXOffset] = LeftBorder
        PaddedImage[:, Width + PaddedImageXOffset:] = RightBorder

        del LeftBorder
        del RightBorder

    if not Height == NewHeight:
        TopBorder = GenRandomData(PaddedImageYOffset, Width, ImageMedian, ImageStdDev, MinVal, MaxVal)
        BottomBorder = GenRandomData(NewHeight - (Height + PaddedImageYOffset), Width, ImageMedian, ImageStdDev, MinVal,
                                     MaxVal)

        PaddedImage[0:PaddedImageYOffset, PaddedImageXOffset:PaddedImageXOffset + Width] = TopBorder
        PaddedImage[PaddedImageYOffset + Height:, PaddedImageXOffset:PaddedImageXOffset + Width] = BottomBorder

        del TopBorder
        del BottomBorder

    return PaddedImage


# @profile
def ImagePhaseCorrelation(FixedImage, MovingImage):
    """
    Returns the phase shift correlation of the FFT's of two images.

    Dimensions of Fixed and Moving images must match

    :param ndarray FixedImage: grayscale image
    :param ndarray MovingImage: grayscale image
    :returns: Correlation image of the FFT's.  Light pixels indicate the phase is well aligned at that offset.
    :rtype: ndimage

    """

    if not (FixedImage.shape == MovingImage.shape):
        # TODO, we should pad the smaller image in this case to allow the comparison to continue
        raise ValueError("ImagePhaseCorrelation: Fixed and Moving image do not have same dimension")

    # --------------------------------
    # This is here in case this function ever needs to be revisited.  Scipy is a lot faster working with in-place operations so this
    # code has been obfuscated more than I like
    # FFTFixed = fftpack.rfft2(FixedImage)
    # FFTMoving = fftpack.rfft2(MovingImage)
    # conjFFTFixed = conj(FFTFixed)
    # Numerator = conjFFTFixed * FFTMoving
    # Divisor = abs(conjFFTFixed * FFTMoving)
    # T = Numerator / Divisor
    # CorrelationImage = real(fftpack.irfft2(T))
    # --------------------------------

    FFTFixed = fftpack.fft2(FixedImage - np.mean(FixedImage.flat))
    FFTMoving = fftpack.fft2(MovingImage - np.mean(MovingImage.flat))

    return FFTPhaseCorrelation(FFTFixed, FFTMoving, True)


def FFTPhaseCorrelation(FFTFixed, FFTMoving, delete_input=False):
    """
    Returns the phase shift correlation of the FFT's of two images.

    Dimensions of Fixed and Moving images must match

    :param ndarray FixedImage: grayscale image
    :param ndarray MovingImage: grayscale image
    :returns: Correlation image of the FFT's.  Light pixels indicate the phase is well aligned at that offset.
    :rtype: ndimage

    """

    if not (FFTFixed.shape == FFTMoving.shape):
        # TODO, we should pad the smaller image in this case to allow the comparison to continue
        raise ValueError("ImagePhaseCorrelation: Fixed and Moving image do not have same dimension")

    # --------------------------------
    # This is here in case this function ever needs to be revisited.  Scipy is a lot faster working with in-place operations so this
    # code has been obfuscated more than I like
    # FFTFixed = fftpack.rfft2(FixedImage)
    # FFTMoving = fftpack.rfft2(MovingImage)
    # conjFFTFixed = conj(FFTFixed)
    # Numerator = conjFFTFixed * FFTMoving
    # Divisor = abs(conjFFTFixed * FFTMoving)
    # T = Numerator / Divisor
    # CorrelationImage = real(fftpack.irfft2(T))
    # --------------------------------

    conjFFTFixed = np.conjugate(FFTFixed)
    if delete_input:
        del FFTFixed

    conjFFTFixed *= FFTMoving

    if delete_input:
        del FFTMoving

    abs_conjFFTFixed = np.absolute(conjFFTFixed)

    # wht_expon = -0.65
    # wht_mask = conjFFTFixed > 0
    # wht_scales = np.power(abs_conjFFTFixed[wht_mask], wht_expon)
    # conjFFTFixed[wht_mask] *= wht_scales

    # if np.any(abs_conjFFTFixed == 0):
    # raise ValueError("Zero found in conjugation of FFT, is the image a single value?")

    # Based on talk with Art Wetzel, apparently wht_expon = -1 is Phase Correlation.  0 is Pierson Correlation 
    mask = abs_conjFFTFixed > 1e-5
    # conjFFTFixed[wht_mask] /= wht_scales  # Numerator / Divisor
    # conjFFTFixed[mask] /= abs_conjFFTFixed[mask]
    conjFFTFixed[mask] /= np.power(abs_conjFFTFixed[mask], 0.65)
    del mask

    # wht_expon_adjustment = np.power(np.absolute(conjFFTFixed[mask]), wht_expon)
    # conjFFTFixed[mask] *= wht_expon_adjustment
    # wht_mask = conjFFTFixed > 1e-5
    # conjFFTFixed[wht_mask] *= np.power(conjFFTFixed[wht_mask], -0.65)
    # del wht_expon_adjustment
    del abs_conjFFTFixed

    CorrelationImage = np.real(fftpack.ifft2(conjFFTFixed))
    del conjFFTFixed

    return CorrelationImage


# @profile
def FindPeak(image, OverlapMask=None, Cutoff=None):
    """
    Find the offset of the strongest response in a phase correlation image

    :param ndimage image: grayscale image
    :param float Cutoff: Percentile used to threshold image.  Values below the percentile are ignored
    :param ndimage OverlapMask: Mask describing which pixels are eligible
    :return: scaled_offset of peak from image center and sum of pixels values at peak
    :rtype: (tuple, float)
    """

    if Cutoff is None:
        Cutoff = 0.995
    #        num_pixels = np.prod(image.shape)

    #        if (1.0 - Cutoff) * num_pixels > 1000:
    #            Cutoff = 1.0 - (1000.0 / num_pixels)

    # CutoffValue = ImageIntensityAtPercent(image, Cutoff)

    # CutoffValue = scipy.stats.scoreatpercentile(image, per=Cutoff * 100.0)
    ThresholdImage = np.copy(image)

    if OverlapMask is not None:
        CutoffValue = np.percentile(image[OverlapMask], q=Cutoff * 100.0)
        ThresholdImage[OverlapMask == False] = 0
    else:
        CutoffValue = np.percentile(image, q=Cutoff * 100.0)

    ThresholdImage[ThresholdImage < CutoffValue] = 0

    # ThresholdImage = scipy.stats.threshold(image, threshmin=CutoffValue, threshmax=None, newval=0)
    # nornir_imageregistration.ShowGrayscale([image, OverlapMask, ThresholdImage])

    [LabelImage, NumLabels] = scipy.ndimage.measurements.label(ThresholdImage)
    #The first interesting label starts at 1, 0 is the background
    LabelSums = scipy.ndimage.sum_labels(ThresholdImage, LabelImage, list(range(1, NumLabels+1))) 
    if LabelSums.sum() == 0:  # There are no peaks identified
        scaled_offset = (np.asarray(image.shape, dtype=np.float32) / 2.0)
        PeakStrength = 0
        return scaled_offset, PeakStrength
    else:
        PeakValueIndex = LabelSums.argmax()
        PeakStrength = LabelSums[PeakValueIndex]
        #Because we offset the sum_labels call by 1, we must do the same for the PeakValueIndex
        PeakCenterOfMass = scipy.ndimage.measurements.center_of_mass(ThresholdImage, LabelImage, PeakValueIndex+1)
        
        # print(f'{LabelSums.shape} Labels -> {PeakStrength} Peak')

        del LabelImage
        del ThresholdImage
        del LabelSums

        # center_of_mass returns results as (y,x)
        # scaled_offset = (image.shape[0] / 2.0 - PeakCenterOfMass[0], image.shape[1] / 2.0 - PeakCenterOfMass[1])
        scaled_offset = (np.asarray(image.shape, dtype=np.float32) / 2.0) - PeakCenterOfMass
        # scaled_offset = (scaled_offset[0], scaled_offset[1])

        return scaled_offset, PeakStrength


def CropNonOverlapping(FixedImageSize, MovingImageSize, CorrelationImage, MinOverlap=0.0, MaxOverlap=1.0):
    """ """

    if not FixedImageSize == MovingImageSize:
        return CorrelationImage


def FindOffset(FixedImage, MovingImage, MinOverlap=0.0, MaxOverlap=1.0, FFT_Required=True, FixedImageShape=None,
               MovingImageShape=None):
    """return an alignment record describing how the images overlap. The alignment record indicates how much the
       moving image must be rotated and translated to align perfectly with the FixedImage.

       If adjusting control points the peak can be added to the fixed image's control point, or subtracted from the
       warped image's control point (accounting for any transform used to create the warped image) to align the images.

       :param ndarray FixedImage:  Target space we are registering into
       :param ndarray MovingImage: Source space we are coming from
       :param float MinOverlap: The minimum amount of overlap by area the registration must have
       :param float MaxOverlap: The maximum amount of overlap by area the registration must have
       :param bool FFT_Required: True by default, if False the input images are in FFT space already
       :param tuple FixedImageShape: Defaults to None, if specified it contains the size of the fixed image before padding.  Used to calculate mask for valid overlap values.
       :param tuple MovingImageShape: Defaults to None, if specified it contains the size of the moving image before padding.  Used to calculate mask for valid overlap values.
       """

    # nornir_imageregistration.ShowGrayscale([FixedImage, MovingImage])

    if FixedImageShape is None:
        FixedImageShape = FixedImage.shape

    if MovingImageShape is None:
        MovingImageShape = MovingImage.shape

    # Find peak requires both the fixed and moving images have equal size
    if not ((FixedImage.shape[0] == MovingImage.shape[0]) and (FixedImage.shape[1] == MovingImage.shape[1])):
        # Pad the smaller image to the appropriate size
        (desired_height, desired_width) = (
            max((FixedImage.shape[0], MovingImage.shape[0])), max((FixedImage.shape[1], MovingImage.shape[1])))
        FixedImage = PadImageForPhaseCorrelation(FixedImage, MinOverlap=1, NewWidth=desired_width,
                                                 NewHeight=desired_height, AlwaysCopy=False)
        MovingImage = PadImageForPhaseCorrelation(MovingImage, MinOverlap=1, NewWidth=desired_width,
                                                  NewHeight=desired_height, AlwaysCopy=False)

    CorrelationImage = None
    if FFT_Required:
        CorrelationImage = ImagePhaseCorrelation(FixedImage, MovingImage)
    else:
        CorrelationImage = FFTPhaseCorrelation(FixedImage, MovingImage, delete_input=False)

    CorrelationImage = fftpack.fftshift(CorrelationImage)

    # Crop the areas that cannot overlap 
    CorrelationImage -= CorrelationImage.min()
    CorrelationImage /= CorrelationImage.max()

    # Timer.Start('Find Peak')
    OverlapMask = nornir_imageregistration.GetOverlapMask(FixedImageShape, MovingImageShape, CorrelationImage.shape,
                                                          MinOverlap, MaxOverlap)
    (peak, weight) = FindPeak(CorrelationImage, OverlapMask)

    del CorrelationImage

    record = nornir_imageregistration.AlignmentRecord(peak=peak, weight=weight)

    return record


def ImageIntensityAtPercent(image, Percent=0.995):
    """Returns the intensity of the Cutoff% most intense pixel in the image"""
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

    os.makedirs(OutputDir, exist_ok=True)


    def TestPhaseCorrelation(imA, imB):

        # import TaskTimer
        # Timer = TaskTimer.TaskTimer()
        # Timer.Start('Correlate One Pair')

        # Timer.Start('Pad image One Pair')
        FixedA = PadImageForPhaseCorrelation(imA)
        MovingB = PadImageForPhaseCorrelation(imB)

        record = FindOffset(FixedA, MovingB, FixedImageShape=imA.shape, MovingImage=imB.shape)
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
