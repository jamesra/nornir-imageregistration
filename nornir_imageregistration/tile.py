'''
Created on Feb 21, 2014

@author: u0490822
'''

import logging
import os

import nornir_imageregistration.spatial as spatial
import nornir_imageregistration.core as core
import numpy as np


def IterateOverlappingTiles(tiles, minOverlap = 0.0):
    '''
    :param list tiles: A list of tile objects
    :param float minOverlap: Tiles must have this percentage of overlap to be returned by the generator
    :return: A generator returning pairs of tiles which overlap
    '''
            
    for (A,B) in itertools.combinations(tiles.values(), 2):
        if spatial.rectangle.Rectangle.overlap(A.ControlBoundingBox, B.ControlBoundingBox) >= minOverlap:
            yield (A,B)
            
def BuildSortedTileControlBoundingRects(tiles):
    rects = np.array((len(tiles), 5))
    
    for i, tile in enumerate(tiles):
        rects[i,:] = np.hstack( tile.ControlBoundingBox.ToArray(), np.array([tile.ID]))
        
    return rects 
             

class Tile(object):
    '''
    A combination of a transform and a path to an image on disk.  Image will be loaded on demand
    ''' 
    __nextID = 0

    @property
    def MappedBoundingBox(self):
        return self._transform.MappedBoundingBox
 
    @property
    def ControlBoundingBox(self):
        return self._transform.FixedBoundingBox

    @property
    def OriginalImageSize(self):
        dims = self.MappedBoundingBox
        return (dims[spatial.iRect.MaxY] - dims[spatial.iRect.MinY], dims[spatial.iRect.MaxX] - dims[spatial.iRect.MinY])

    @property
    def WarpedImageSize(self):
        dims = self.ControlBoundingBox
        return (dims[spatial.iRect.MaxY] - dims[spatial.iRect.MinY], dims[spatial.iRect.MaxX] - dims[spatial.iRect.MinY])

    @property
    def Transform(self):
        return self._transform

    @property
    def Image(self):
        if self._image is None:
            self._image = core.LoadImage(self._imagepath)
        
        return self._image
        
    @property
    def PaddedImage(self):
        if self._paddedimage is None:
            self._paddedimage = core.PadImageForPhaseCorrelation(self.Image)

        return self._paddedimage

    @property
    def ImagePath(self):
        return self._imagepath

    @property
    def FFTImage(self):
        if self._fftimage is None:
            self._fftimage = np.fft.rfft2(self.PaddedImage)

        return self._fftimage
    
    def PrecalculateImages(self):
        temp = self.FFTImage.shape

    @property
    def ID(self):
        return self._ID

    @classmethod
    def CreateTiles(cls, transforms, imagepaths):

        tiles = []
        for i, t in enumerate(transforms):

            if not os.path.exists(imagepaths[i]):
                log = logging.getLogger(__name__ + ".CreateTiles")
                log.error("Missing tile: " + imagepaths[i])
                continue

            tile = Tile(t, imagepaths[i], i)
            tiles.append(tile)

        return tiles

    def __init__(self, transform, imagepath, ID=None):

        global __nextID

        self._transform = transform
        self._imagepath = imagepath
        self._image = None
        self._paddedimage = None
        self._fftimage = None

        if ID is None:
            self._ID = Tile.__nextID
            Tile.__nextID += 1
        else:
            self._ID = ID

    def __str__(self):
        return "%d: %s" % (self._ID, self._imagepath)