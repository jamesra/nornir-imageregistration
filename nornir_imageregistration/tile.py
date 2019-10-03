'''
Created on Feb 21, 2014

@author: u0490822
'''

import logging
import os 
import nornir_imageregistration 
import numpy as np


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
        #Try to use the tile's number if it is encoded in the filename, otherwise just use enumeration number
        try:
            tile_ID = int(os.path.splitext(os.path.basename(imagepaths[i]))[0])
        except:
            tile_ID = i
            
        tile = Tile(t, imagepaths[i], tile_ID)
        tiles[tile.ID] = tile
        
    return tiles


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
        DeprecationWarning("ControlBoundingBox")
        return self._transform.FixedBoundingBox
       
    @property
    def FixedBoundingBox(self):
        return self._transform.FixedBoundingBox

    @property
    def OriginalImageSize(self):
        dims = self.MappedBoundingBox
        return (dims[nornir_imageregistration.iRect.MaxY] - dims[nornir_imageregistration.iRect.MinY], dims[nornir_imageregistration.iRect.MaxX] - dims[nornir_imageregistration.iRect.MinY])

    @property
    def WarpedImageSize(self):
        dims = self.FixedBoundingBox
        return (dims[nornir_imageregistration.iRect.MaxY] - dims[nornir_imageregistration.iRect.MinY], dims[nornir_imageregistration.iRect.MaxX] - dims[nornir_imageregistration.iRect.MinY])

    @property
    def Transform(self):
        '''A string encoding our tile's transform'''
        return self._transform
    
    @Transform.setter
    def Transform(self,val):
        '''A string encoding our tile's transform'''
        self._transform = val

    @property
    def Image(self):
        if self._image is None:
            self._image = nornir_imageregistration.LoadImage(self._imagepath)
        
        return self._image
        
    @property
    def PaddedImage(self):
        if self._paddedimage is None:
            self._paddedimage = nornir_imageregistration.PadImageForPhaseCorrelation(self.Image)

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
    
    def __str__(self):
        return str(self.ID) 
    
    def Get_Overlapping_Source_Rect(self, overlapping_target_rect):
        ''':return: Rectangle describing which region of the tile_obj image is contained in the overlapping_rect from volume space'''
        source_space_points = self.Transform.InverseTransform(overlapping_target_rect.Corners)    
        return nornir_imageregistration.BoundingPrimitiveFromPoints(source_space_points)


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
    
    def EnsureTileHasMappedBoundingBox(self, image_scale, cached_tile_shape = None):
        '''
        If our tile does not have a mapped bounding box, define it using the tile's image dimensions
        :param float image_scale: Downsample factor of image files
        :param tuple cached_tile_shape: If not None, use these dimensions for the mapped bounding box instead of loading the image to obtain the shape
        '''
        
        if self._transform.MappedBoundingBox is None:
            if cached_tile_shape is not None: 
                mapped_bbox_shape = cached_tile_shape
            else:
                mapped_bbox_shape = nornir_imageregistration.GetImageSize(self._imagepath)
                mapped_bbox_shape = np.array(mapped_bbox_shape, dtype=np.int32) * (1.0 / image_scale)
                
            self._transform.MappedBoundingBox = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0,0), mapped_bbox_shape)
            return mapped_bbox_shape
        
        return None

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
         
        if not isinstance(self._ID, int):
            raise TypeError("Tile ID must be an integer: {0}".format(ID))
            
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

    def __repr__(self):
        return "%d: %s" % (self._ID, self._imagepath)
