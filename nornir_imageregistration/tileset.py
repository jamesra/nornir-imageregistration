'''
Created on Nov 18, 2013

@author: u0490822
'''

import copy
import logging
import os
 
from scipy import stats 

import nornir_pools
import nornir_imageregistration
import nornir_imageregistration.transforms.utils as tutils
import nornir_shared.images
# import nornir_pools as pools
import numpy as np

   
# from nornir_imageregistration.files.mosaicfile import MosaicFile
# from nornir_imageregistration.mosaic import Mosaic
class ShadeCorrectionTypes(object):
    BRIGHTFIELD = 0
    DARKFIELD = 1


def __DetermineTransformScale(transform, imageSize):
    '''Returns a scalar that can be applied to the transform to make the transform bounds match the image dimensions'''

    if not hasattr(transform, 'MappedBoundingBox'):
        return 1.0 
    
    width = transform.MappedBoundingBox.Width + 1 #Account for zero origin by adding 1
    height = transform.MappedBoundingBox.Height + 1 #Account for zero origin by adding 1

    if nornir_imageregistration.ApproxEqual(imageSize[0], height, epsilon=1.1) and nornir_imageregistration.ApproxEqual(imageSize[1], width, epsilon=1.1):
        return 1.0
    else:
        heightScale = (imageSize[0] / height)
        widthScale = (imageSize[1] / width)
        
        if(nornir_imageregistration.ApproxEqual(heightScale, widthScale)):
            return heightScale
        else:
            raise ValueError(f"Mismatch between heightScale and widthScale. {heightScale} vs {widthScale}")


def MostCommonScalar(transforms, imagepaths):
    '''Compare the image size encoded in the transforms to the image sizes on disk. 
       Return the most common scale factor required to make the transforms match the image dimensions'''

    raise DeprecationWarning("Examine why we are determining scale from a set of images.  Attempt to replace with mosaic_tileset's attribute for image to source space scale")
    scales = []

    for i, transform in enumerate(transforms):
        imagefullpath = imagepaths[i]

        try:
            size = nornir_shared.images.GetImageSize(imagefullpath)
        except IOError:
            continue 
        
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

    CompositeImage = nornir_imageregistration.LoadImage(stack.pop())

    while len(stack) > 0:

        imageA = nornir_imageregistration.LoadImage(stack.pop())
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


def CalculateShadeImage(imagepaths, correction_type=None):

    # Find the min or max of the tiles depending on type
    if correction_type == ShadeCorrectionTypes.BRIGHTFIELD:
        return __CalculateBrightfieldShadeImage(imagepaths)
    elif correction_type == ShadeCorrectionTypes.DARKFIELD:
        return __CalculateDarkfieldShadeImage(imagepaths)

    return None


def __CorrectBrightfieldShadingOneImage(input_fullpath, output_fullpath, imagescalar, bpp):
    max_pixel_value = ((1 << bpp) - 1)
    image = nornir_imageregistration.LoadImage(input_fullpath)
    image = image.astype(np.float16) / max_pixel_value
     
    correctedimage = image / imagescalar
    del image
    
    correctedimage[np.isinf(correctedimage)] = 0
    np.clip(correctedimage, a_min=0, a_max=1.0, out=correctedimage)
    correctedimage = correctedimage * max_pixel_value

    nornir_imageregistration.SaveImage(output_fullpath, correctedimage, bpp=bpp)
    del correctedimage 


def __CorrectBrightfieldShading(imagepaths, shadeimage, outputpath, bpp=None):

    outputPaths = []
    if bpp is None:
        bpp = nornir_shared.images.GetImageBpp(imagepaths[0])
        
    #max_pixel_value = ((1 << bpp) - 1)
    # nshadeimage = shadeimage - shadeimage.min()
    # nshadeimage = NormalizeImage(shadeimage)
    # nshadeimage[np.isinf(shadeimage)] = 1.0

    # How much do we need to scale nonmax pixel values so the maximum pixel value is uniform across the entire image
    imagescalar = shadeimage / shadeimage.max()
    imagescalar.setflags(write=False)
    # imagescalar[np.isinf(imagescalar)] = 1.0
    
    pool = nornir_pools.GetGlobalSerialPool()
    tasks = []
    for imagepath in imagepaths:
        imageFilename = os.path.basename(imagepath)
        outputFilename = os.path.join(outputpath, imageFilename)
        
        t = pool.add_task(imageFilename, __CorrectBrightfieldShadingOneImage, imagepath, outputFilename, imagescalar, bpp=bpp)
        t.output_fullpath = outputFilename

        # Shadeimage is the max of all tiles.  Figure out what the multiplier is for each pixel.
        # invertedimage = 1.0 - shadeimage
        # zerodshadeimage = invertedimage - invertedimage.min()

        # Make max shading value a 1 so the brightest pixel is unchanged
        # multiplierimage = zerodshadeimage.max() - zerodshadeimage

        # NormalizeImage(shadeimage)

        # shadeimage = shadeimage * (shadeimage / 1.0)

#         correctedimage = image / imagescalar
# 
#         correctedimage[np.isinf(correctedimage)] = 0
#         np.clip(correctedimage, a_min=0, a_max=1.0, out=correctedimage)
#         
#         correctedimage = correctedimage * max_pixel_value
# 
#         nornir_imageregistration.SaveImage(outputFilename, correctedimage, bpp=bpp)
# 
#         del image
#         del correctedimage

        outputPaths.append(outputFilename)
        
    while len(tasks) > 0:
        t = tasks.pop(0)
        t.wait()
        outputPaths.append(t.output_fullpath)
        assert(os.path.exists(t.output_fullpath))

    return outputPaths


def __CorrectDarkfieldShading(imagepaths, shadeimage, outputpath, bpp=None):

    outputPaths = []
    if bpp is None:
        bpp = nornir_shared.images.GetImageBpp(imagepaths[0])

    for imagepath in imagepaths:
        image = nornir_imageregistration.LoadImage(imagepath)

        imageFilename = os.path.basename(imagepath)
        outputFilename = os.path.join(outputpath, imageFilename)

        correctedimage = image - shadeimage
        nornir_imageregistration.SaveImage(outputFilename, correctedimage, bpp=bpp)

        outputPaths.append(outputFilename)

        del image
        del correctedimage

    return outputPaths


def ShadeCorrect(imagepaths, shadeimagepath, outputpath, correction_type=None, bpp=None):

    shadeimage = nornir_imageregistration.ImageParamToImageArray(shadeimagepath)
     
    if correction_type == ShadeCorrectionTypes.BRIGHTFIELD:
        return __CorrectBrightfieldShading(imagepaths, shadeimage, outputpath, bpp=bpp)
    elif correction_type == ShadeCorrectionTypes.DARKFIELD:
        return __CorrectDarkfieldShading(imagepaths, shadeimage, outputpath, bpp=bpp)


if __name__ == '__main__':
    pass
