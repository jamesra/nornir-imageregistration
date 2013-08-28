'''
Created on Mar 29, 2013

@author: u0490822
'''

from IO.mosaicfile import MosaicFile
import IrTools.Transforms

class Mosaic(object):
    '''
    Maps images into a mosaic with a transform
    '''


    def __init__(self):
        '''
        Constructor
        '''

        self.ImageToTransform = {}
        self.ImageScale = 1

    @classmethod
    def LoadFromMosaicFile(cls, mosaicFile):
        '''Creates a layout from a .mosaic file, mosaicFile can be a path or a IrTools.IO.MosaicFile instance'''

        if isinstance(mosaicFile, str):
            mfile = MosaicFile.Load(mosaicFile)
        elif isinstance(mosaicFile, MosaicFile):
            mfile = mosaicFile
        else:
            return None

        #Create a transform for each tile

        for imagename in mfile.ImageToTransform.keys():
            transformString = mfile.ImageToTransform[imagename]

    @classmethod
    def TranslateLayout(cls, Images, Positions, ImageScale = 1):
        '''Creates a layout for the provided images at the provided
           It is assumed that Positions are not scaled, but the image size may be scaled'''

        mosaic = Mosaic()

        mosaic.ImageScale = ImageScale

