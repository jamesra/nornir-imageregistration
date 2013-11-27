'''
Created on Nov 18, 2013

@author: u0490822
'''

import numpy as np
import core
from scipy import stats
# from nornir_imageregistration.files.mosaicfile import MosaicFile
# from nornir_imageregistration.mosaic import Mosaic
import core
import os
import logging
import transforms.utils as tutils
import nornir_pools as pools
import copy


def __DetermineScale(transform, imageSize):
    '''Returns a scalar that can be applied to the transform to make the transform bounds match the image dimensions'''

    width = transform.MappedBoundingBoxWidth
    height = transform.MappedBoundingBoxHeight

    if core.ApproxEqual(imageSize[0], height) and core.ApproxEqual(imageSize[1], width):
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

        scales.append(__DetermineScale(transform, size))

    if len(scales) == 0:
        raise Exception("No image sizes available to determine scale for assemble")

    return stats.mode(scales)[0]



if __name__ == '__main__':
    pass