'''
Created on Mar 29, 2013

@author: u0490822
'''

from nornir_imageregistration.io.mosaicfile import MosaicFile
import transforms.factory as tfactory
import transforms.utils as tutils
import assemble_tiles as at
import numpy as np
import os



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
        '''Calculate the bounding box of the warped position for a set of transforms
           (minX, minY, maxX, maxY)'''

        return tutils.FixedBoundingBox(self.ImageToTransform.values())

    @property
    def MappedBoundingBox(self):
        '''Calculate the bounding box of the warped position for a set of transforms
           (minX, minY, maxX, maxY)'''

        return tutils.MappedBoundingBox(self.ImageToTransform.values())

    @property
    def FixedBoundingBoxWidth(self):
        return tutils.FixedBoundingBoxWidth(self.ImageToTransform.values())

    @property
    def FixedBoundingBoxHeight(self):
        return tutils.FixedBoundingBoxHeight(self.ImageToTransform.values())

    @property
    def MappedBoundingBoxWidth(self):
        return tutils.MappedBoundingBoxWidth(self.ImageToTransform.values())

    @property
    def MappedBoundingBoxHeight(self):
        return tutils.MappedBoundingBoxHeight(self.ImageToTransform.values())


    def TileFullPaths(self, tilesDir):
        '''Return a list of full paths to the tile for each transform'''
        return [os.path.join(tilesDir, x) for x in self.ImageToTransform.keys()]



    def TranslateToZeroOrigin(self):
        '''Ensure that the transforms in the mosaic do not map to negative coordinates'''

        (minX, minY, maxX, maxY) = self.FixedBoundingBox

        for t in self.ImageToTransform.values():
            t.TranslateFixed((-minY, -minX))

    @classmethod
    def TranslateLayout(cls, Images, Positions, ImageScale=1):
        '''Creates a layout for the provided images at the provided
           It is assumed that Positions are not scaled, but the image size may be scaled'''

        raise Exception("Not implemented")



    def AssembleTiles(self, tilesPath, parallel=True):
        '''Create a single large mosaic'''

        # Allocate a buffer for the tiles
        tilesPath = [os.path.join(tilesPath, x) for x in self.ImageToTransform.keys()]

        if parallel:
            return at.TilesToImageParallel(self.ImageToTransform.values(), tilesPath)
        else:
            return at.TilesToImage(self.ImageToTransform.values(), tilesPath)



