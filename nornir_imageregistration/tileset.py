'''
Created on Nov 18, 2013

@author: u0490822
'''

import copy
import logging
import os

import PIL
from scipy import stats
from scipy.misc import imsave

import nornir_imageregistration.transforms.utils as tutils
import nornir_pools as pools
import numpy as np

from . import core
from . import core


# from nornir_imageregistration.files.mosaicfile import MosaicFile
# from nornir_imageregistration.mosaic import Mosaic
class ShadeCorrectionTypes(object):
    BRIGHTFIELD = 0
    DARKFIELD = 1


def __DetermineTransformScale(transform, imageSize):
    '''Returns a scalar that can be applied to the transform to make the transform bounds match the image dimensions'''

    width = transform.MappedBoundingBox.Width
    height = transform.MappedBoundingBox.Height

    if core.ApproxEqual(imageSize[0], height,epsilon=1.1) and core.ApproxEqual(imageSize[1], width,epsilon=1.1):
        return 1.0
    else:
        heightScale = (imageSize[0] / height)
        widthScale = (imageSize[1] / width)

        if(core.ApproxEqual(heightScale, widthScale)):
            return heightScale
        else:
            return None


def MostCommonScalar(transforms, imagepaths):
    '''Compare the image size encoded in the transforms to the image sizes on disk. 
       Return the most common scale factor required to make the transforms match the image dimensions'''

    scales = []

    for i, transform in enumerate(transforms):
        imagefullpath = imagepaths[i]

        size = core.GetImageSize(imagefullpath)

        if size is None:
            continue

        scales.append(__DetermineTransformScale(transform, size))

    if len(scales) == 0:
        raise Exception("No image sizes available to determine scale for assemble")

    return stats.mode(scales)[0]


def __CompositeTiles(imagepaths, func):
    '''Takes two images, merges, and returns the max image
    
    :param list imagepaths: list of paths to images
    :param str outputpath: Path of output image
    :param function func: function taking two images as arguments.  Called on image pairs.
    '''

    stack = copy.copy(imagepaths)

    CompositeImage = core.LoadImage(stack.pop())

    while len(stack) > 0:

        imageA = core.LoadImage(stack.pop())
        CompositeImage = func(CompositeImage, imageA)

        del imageA

    return CompositeImage


def __CalculateBrightfieldShadeImage(imagepaths):
     InitialCorrection = __CompositeTiles(imagepaths, func=np.maximum)


     # AddValue = 1.0 - np.max(InitialCorrection)
     # ZerodCorrectionImage = InitialCorrection + AddValue

     # Invert the Correction so that bright areas have nothing added
     # InvertedCorrection = np.abs(ZerodCorrectionImage - 1)

     # return InvertedCorrection
     return InitialCorrection


def __CalculateDarkfieldShadeImage(imagepaths):
     InitialTile = __CompositeTiles(imagepaths, func=np.minimum)

     ZerodCorrectionImage = InitialTile - np.min(InitialTile)

     return ZerodCorrectionImage


def CalculateShadeImage(imagepaths, type=None):

    # Find the min or max of the tiles depending on type
    if type == ShadeCorrectionTypes.BRIGHTFIELD:
       return __CalculateBrightfieldShadeImage(imagepaths)
    elif type == ShadeCorrectionTypes.DARKFIELD:
       return __CalculateDarkfieldShadeImage(imagepaths)

    return None


def __CorrectBrightfieldShading(imagepaths, shadeimage, outputpath):

    outputPaths = []

    # nshadeimage = shadeimage - shadeimage.min()
    # nshadeimage = NormalizeImage(shadeimage)
    # nshadeimage[np.isinf(shadeimage)] = 1.0

    # How much do we need to scale nonmax pixel values so the maximum pixel value is uniform across the entire image
    imagescalar = shadeimage / shadeimage.max()
    # imagescalar[np.isinf(imagescalar)] = 1.0

    for imagepath in imagepaths:
        image = core.LoadImage(imagepath)

        imageFilename = os.path.basename(imagepath)
        outputFilename = os.path.join(outputpath, imageFilename)

        # Shadeimage is the max of all tiles.  Figure out what the multiplier is for each pixel.
        # invertedimage = 1.0 - shadeimage
        # zerodshadeimage = invertedimage - invertedimage.min()

        # Make max shading value a 1 so the brightest pixel is unchanged
        # multiplierimage = zerodshadeimage.max() - zerodshadeimage

        # NormalizeImage(shadeimage)

        # shadeimage = shadeimage * (shadeimage / 1.0)

        correctedimage = image / imagescalar

        correctedimage[np.isinf(correctedimage)] = 0
        correctedimage[correctedimage > 1.0] = 1.0
        correctedimage[correctedimage < 0] = 0

        core.SaveImage(outputFilename, correctedimage)

        del image
        del correctedimage

        outputPaths.append(outputFilename)

    return outputPaths


def __CorrectDarkfieldShading(imagepaths, shadeimage, outputpath):

    outputPaths = []

    for imagepath in imagepaths:
        image = core.LoadImage(imagepath)

        imageFilename = os.path.basename(imagepath)
        outputFilename = os.path.join(outputpath, imageFilename)

        correctedimage = image - shadeimage
        core.SaveImage(outputFilename, correctedimage)

        outputPaths.append(outputFilename)

        del image
        del correctedimage

    return outputPaths


def ShadeCorrect(imagepaths, shadeimagepath, outputpath, type=None):

    shadeimage = None
    if isinstance(shadeimagepath, str):
        shadeimage = core.LoadImage(shadeimagepath)
    elif isinstance(shadeimagepath, np.ndarray):
        shadeimage = shadeimagepath

    if type == ShadeCorrectionTypes.BRIGHTFIELD:
        return __CorrectBrightfieldShading(imagepaths, shadeimage, outputpath)
    elif type == ShadeCorrectionTypes.DARKFIELD:
        return __CorrectDarkfieldShading(imagepaths, shadeimage, outputpath)

if __name__ == '__main__':
    pass