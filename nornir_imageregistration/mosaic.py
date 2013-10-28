'''
Created on Mar 29, 2013

@author: u0490822
'''

from io.mosaicfile import MosaicFile
import transforms.factory as tfactory
import transforms.utils as tutils



class Mosaic(object):
    '''
    Maps images into a mosaic with a transform
    '''

    @classmethod 
    def LoadFromMosaicFile(cls, mosaicfile):
        '''Return a dictionary mapping tiles to transform objects'''

        if isinstance(mosaicfile, str):
            mosaicfile = MosaicFile.Load(mosaicfile)
            if mosaicfile is None:
                raise Exception("Expected valid mosaic file path")
        elif not isinstance(mosaicfile, MosaicFile):
            raise Exception("Expected valid mosaic file path or object")

        ImageToTransform = {}
        for (k, v) in mosaicfile.ImageToTransform.items():
            ImageToTransform[k] = tfactory.LoadTransform(v, pixelSpacing=1.0)

        return Mosaic(ImageToTransform)

    def __init__(self, ImageToTransform):
        '''
        Constructor
        '''

        self.ImageToTransform = ImageToTransform
        self.ImageScale = 1


    @property
    def FixedBoundingBox(self):
        '''Calculate the bounding box of the warped position for a set of transforms'''

        return tutils.FixedBoundingBox(self.ImageToTransform.values())

    @property
    def MappedBoundingBox(self):
        '''Calculate the bounding box of the warped position for a set of transforms'''

        return tutils.MappedBoundingBox(self.ImageToTransform.values())


    @classmethod
    def TranslateLayout(cls, Images, Positions, ImageScale = 1):
        '''Creates a layout for the provided images at the provided
           It is assumed that Positions are not scaled, but the image size may be scaled'''

        raise Exception("Not implemented")
