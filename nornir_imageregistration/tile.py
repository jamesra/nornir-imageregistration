'''
Created on Feb 21, 2014

@author: u0490822
'''

import logging
import os

import nornir_imageregistration.spatial as spatial
import nornir_imageregistration.core as core
import numpy as np
import itertools


def CreateTiles(transforms, imagepaths):
    '''Create tiles from pairs of transforms and image paths
    :param transform transforms: List of N transforms
    :param str imagepaths: List of N paths to image files
    :return: List of N tile objects
    '''

    tiles = {}
    for i, t in enumerate(transforms):

        if not os.path.exists(imagepaths[i]):
            log = logging.getLogger(__name__ + ".CreateTiles")
            log.error("Missing tile: " + imagepaths[i])
            continue

        tile = Tile(t, imagepaths[i], i)
        tiles[tile.ID] = tile
        
    return tiles


def IterateOverlappingTiles(list_tiles, minOverlap = 0.05):
    '''Return all tiles which overlap'''
    
    list_rects = []
    for tile in list_tiles:
        list_rects.append(tile.ControlBoundingBox)
        
    rset = spatial.RectangleSet.Create(list_rects)
    
    for (A,B) in rset.EnumerateOverlapping():
        if spatial.Rectangle.overlap(list_rects[A], list_rects[B]) >= minOverlap:
            yield (list_tiles[A],list_tiles[B])
            

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
    
    
    def Get_Overlapping_Imagespace_Rect(self, overlapping_rect):
        ''':return: Rectangle describing which region of the tile_obj image is contained in the overlapping_rect from volume space'''
        image_space_points = self.Transform.InverseTransform(overlapping_rect.Corners)    
        return spatial.BoundingPrimitiveFromPoints(image_space_points)


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
    
    
    @classmethod
    def Calculate_Overlapping_Regions(cls, A, B, imageScale):
        '''
        :return: The rectangle describing the overlapping portions of tile A and B in the destination (volume) space
        '''
        
        overlapping_rect = spatial.Rectangle.overlap_rect(A.ControlBoundingBox, B.ControlBoundingBox)
        
        overlapping_rect_A = A.Get_Overlapping_Imagespace_Rect(overlapping_rect)
        overlapping_rect_B = B.Get_Overlapping_Imagespace_Rect(overlapping_rect)
         
        downsampled_overlapping_rect_A = spatial.Rectangle.SafeRound(spatial.Rectangle.CreateFromBounds(overlapping_rect_A.ToArray() * imageScale))
        downsampled_overlapping_rect_B = spatial.Rectangle.SafeRound(spatial.Rectangle.CreateFromBounds(overlapping_rect_B.ToArray() * imageScale))
        
        #If the predicted alignment is perfect and we use only the overlapping regions  we would have an alignment offset of 0,0.  Therefore we add the existing offset between tiles to the result
        OffsetAdjustment = (B.ControlBoundingBox.Center - A.ControlBoundingBox.Center) * imageScale
        
        #This should ensure we never an an area mismatch
        downsampled_overlapping_rect_B = spatial.Rectangle.CreateFromPointAndArea(downsampled_overlapping_rect_B.BottomLeft, downsampled_overlapping_rect_A.Size)
        
        assert(downsampled_overlapping_rect_A.Width == downsampled_overlapping_rect_B.Width)
        assert(downsampled_overlapping_rect_A.Height == downsampled_overlapping_rect_B.Height)
         
        return (downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment)

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
            
    def __getstate__(self):
        odict = {}
        odict['_transform'] = self._transform
        odict['_imagepath'] = self._imagepath
        odict['_ID'] = self._ID

        return odict

    def __setstate__(self, dictionary):         
        self.__dict__.update(dictionary)
        self._image = None
        self._paddedimage = None
        self._fftimage = None

    def __str__(self):
        return "%d: %s" % (self._ID, self._imagepath)