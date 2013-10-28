'''
Created on Oct 28, 2013

@author: u0490822

Deals with assembling images composed of mosaics or dividing images into tiles
'''

import assemble
import numpy as np
from io.mosaicfile import MosaicFile
from mosaic import Mosaic


def __ArgToMosaic(arg):

    mosaicObj = None
    if isinstance(arg, str):
        mosaicObj = MosaicFile.Load(arg)
    elif isinstance(arg, MosaicFile):
        mosaicObj = Mosaic.LoadFromMosaicFile(arg)
    elif isinstance(arg, Mosaic):
        mosaicObj = Mosaic
    else:
        raise Exception("Unexpected argument type for mosaic")

    return mosaicObj


def MosaicToImage(mosaicArg):
    
    mosaic = __ArgToMosaic(mosaicArg)
    if mosaic is None:
        raise Exception("Invalid mosaic argument to MosaicToImage " + str(mosaicArg))
    
   # for ImageTransformPair in mosaic.ImageToTransform.items():
        # TransformedTile =
    
    
#def TilesToImage(transforms, imagepaths, ):
    

if __name__ == '__main__':
    pass