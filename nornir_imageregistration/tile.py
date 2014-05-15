'''
Created on Feb 21, 2014

@author: u0490822
'''

import logging
import os

import nornir_imageregistration.core as core
import numpy as np


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
        return (dims[3] - dims[1], dims[2] - dims[0])

    @property
    def WarpedImageSize(self):
        dims = self.ControlBoundingBox
        return (dims[3] - dims[1], dims[2] - dims[0])

    @property
    def Transform(self):
        return self._transform

    @property
    def Image(self):
        if self._image is None:
            img = core.LoadImage(self._imagepath)
            self._image = core.PadImageForPhaseCorrelation(img)
            del img

        return self._image

    @property
    def ImagePath(self):
        return self._imagepath

    @property
    def FFTImage(self):
        if self._fftimage is None:
            self._fftimage = np.fft.rfft2(self.Image)

        return self._fftimage

    @property
    def ID(self):
        return self._ID

    @classmethod
    def CreateTiles(cls, transforms, imagepaths):

        tiles = []
        for i, t in enumerate(transforms):

            if not os.path.exists(imagepaths[i]):
                log = logging.getLogger("CreateTiles")
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
        self._fftimage = None

        if ID is None:
            self._ID = Tile.__nextID
            Tile.__nextID += 1
        else:
            self._ID = ID

    def __str__(self):
        return "%d: %s" % (self._ID, self._imagepath)