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

        tile = Tile(t, imagepaths[i], i)
        tiles[tile.ID] = tile
        
    return tiles


def IterateOverlappingTiles(list_tiles, min_overlap=None):
    '''Return all tiles which overlap'''
    
    list_rects = [tile.ControlBoundingBox for tile in list_tiles]        
    rset = nornir_imageregistration.RectangleSet.Create(list_rects)
    
    for (A, B) in rset.EnumerateOverlapping():
        if min_overlap is None:
            yield (list_tiles[A], list_tiles[B])
        elif nornir_imageregistration.Rectangle.overlap(list_rects[A], list_rects[B]) > min_overlap:
            yield (list_tiles[A], list_tiles[B])
            
def IterateTileOverlaps(list_tiles, imageScale=1.0, min_overlap=None):
    yield from CreateTileOverlaps(list_tiles, imageScale, min_overlap)
            
def CreateTileOverlaps(list_tiles, imageScale=1.0, min_overlap=None):
    for (A, B) in IterateOverlappingTiles(list_tiles,min_overlap=min_overlap):
        overlap_data = TileOverlap(A,B,imageScale=imageScale)
        if overlap_data.has_overlap:
            yield overlap_data
            

class TileOverlap(object):
    '''
    Describes properties of the overlapping regions of two tiles
    ''' 
    
    iA = 0
    iB = 1
    
    @property
    def has_overlap(self):
        return not (self._overlapping_rects[0] is None or self._overlapping_rects[1] is None)
    
    @property
    def ID(self):
        '''ID tuple of (A.ID, B.ID)'''
        assert(self._Tiles[0].ID < self._Tiles[1].ID)
        return (self._Tiles[0].ID, self._Tiles[1].ID)
    
    @property
    def Tiles(self):
        '''[A,B]'''
        return self._Tiles
    
    @property
    def A(self):
        '''Tile object'''
        return self._Tiles[0]
    
    @property
    def B(self):
        '''Tile object'''
        return self._Tiles[1]
    
    @property
    def feature_scores(self):
        '''float tuple indicating how much texture is available in the overlap region for registration'''
        return self._feature_scores
    
    @feature_scores.setter
    def feature_scores(self, val):
        self._feature_scores = val.copy()
    
    @property
    def A_feature_score(self):
        '''float value indicating how much texture is available in the overlap region for registration'''
        return self._feature_scores[0]
    
    @property
    def B_feature_score(self):
        '''float value indicating how much texture is available in the overlap region for registration'''
        return self._feature_scores[1]
    
    @A_feature_score.setter
    def A_feature_score(self, val):
        self._feature_scores[0] = val
    
    @B_feature_score.setter
    def B_feature_score(self, val):
        self._feature_scores[1] = val
    
    @property
    def Offset(self):
        '''
        The result of B.Center - A.Center.
        '''
        return self._offset
    
    @property
    def overlapping_rects(self):
        '''
        Tuple of rectangles (A,B) describing the overlap of both tiles in volume (target) space
        '''
        return self._overlapping_rects
    
    @property
    def overlapping_rect_A(self):
        '''
        Rectangle describing the overlap of both tiles in volume (target) space
        '''
        return self._overlapping_rects[0]
    
    @property
    def overlapping_rect_B(self):
        '''
        Rectangle describing the overlap of both tiles in volume (target) space
        '''
        return self._overlapping_rects[1]
    
     
    @property
    def get_scaled_overlapping_rects(self, scale_factor):
        '''
        Scale the area of overlap by the provided scale_factor, however
        do not scale the overlapping rectangles into areas that can't overlap
        ''' 
    
    def overlap(self):
        '''
        :return: 0 to 1 float indicating the overlapping rectangle area divided by largest tile area
        '''
        if self._overlap is None:
            if self._overlapping_rect_A is None or self._overlapping_rect_B is None:
                self._overlap = 0
            else:
                self._overlap = nornir_imageregistration.Rectangle.overlap_normalized(self.A,self.B)
            
        return self._overlap
    
    
    def get_expand_overlap_rects(self, scale_factor):
        raise NotImplemented()
            
    def __init__(self, A, B, imageScale):
        
        if imageScale is None:
            imageScale = 1.0
             
        #Ensure ID's used are from low to high
        if A.ID > B.ID:
            temp = A
            A = B
            B = temp
            
        self._imageScale = imageScale
        self._Tiles = (A,B)
        self._feature_scores = [None, None]
        self._overlap = None
        (overlapping_rect_A, overlapping_rect_B, self._offset) = TileOverlap.Calculate_Overlapping_Regions(A,B, imageScale=imageScale)
        self._overlapping_rects =  (overlapping_rect_A, overlapping_rect_B)
        
    def __repr__(self):
        area_scale = self._imageScale ** 2
        val = "({0},{1}) ({2:02f}%,{3:02f}%)".format(self.ID[0], self.ID[1],
                                                     float((self._overlapping_rects[0].Area / (self.A.ControlBoundingBox.Area * area_scale)) * 100.0),
                                                     float((self._overlapping_rects[1].Area / (self.B.ControlBoundingBox.Area * area_scale)) * 100.0))
                                             
        
        if self.feature_scores[0] is not None and self.feature_scores[1] is not None:
            val = val + " {0}".format(str(self.feature_scores))
        
        
            
        return val
    
    @staticmethod
    def Calculate_Overlapping_Regions(A, B, imageScale):
        '''
        :return: The rectangle describing the overlapping portions of tile A and B in the destination (volume) space
        '''
        
        overlapping_rect = nornir_imageregistration.Rectangle.overlap_rect(A.ControlBoundingBox, B.ControlBoundingBox)
        if overlapping_rect is None:
            return (None, None, None)
        
        overlapping_rect_A = A.Get_Overlapping_Imagespace_Rect(overlapping_rect)
        overlapping_rect_B = B.Get_Overlapping_Imagespace_Rect(overlapping_rect)
         
        downsampled_overlapping_rect_A = nornir_imageregistration.Rectangle.SafeRound(nornir_imageregistration.Rectangle.CreateFromBounds(overlapping_rect_A.ToArray() * imageScale))
        downsampled_overlapping_rect_B = nornir_imageregistration.Rectangle.SafeRound(nornir_imageregistration.Rectangle.CreateFromBounds(overlapping_rect_B.ToArray() * imageScale))
        
        # If the predicted alignment is perfect and we use only the overlapping regions  we would have an alignment offset of 0,0.  Therefore we add the existing offset between tiles to the result
        OffsetAdjustment = (B.ControlBoundingBox.Center - A.ControlBoundingBox.Center) * imageScale
        
        # This should ensure we never have an an area mismatch
        downsampled_overlapping_rect_B = nornir_imageregistration.Rectangle.CreateFromPointAndArea(downsampled_overlapping_rect_B.BottomLeft, downsampled_overlapping_rect_A.Size)
        
        assert(downsampled_overlapping_rect_A.Width == downsampled_overlapping_rect_B.Width)
        assert(downsampled_overlapping_rect_A.Height == downsampled_overlapping_rect_B.Height)
         
        return (downsampled_overlapping_rect_A, downsampled_overlapping_rect_B, OffsetAdjustment)
    

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
        return (dims[nornir_imageregistration.iRect.MaxY] - dims[nornir_imageregistration.iRect.MinY], dims[nornir_imageregistration.iRect.MaxX] - dims[nornir_imageregistration.iRect.MinY])

    @property
    def WarpedImageSize(self):
        dims = self.ControlBoundingBox
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
    
    def Get_Overlapping_Imagespace_Rect(self, overlapping_rect):
        ''':return: Rectangle describing which region of the tile_obj image is contained in the overlapping_rect from volume space'''
        image_space_points = self.Transform.InverseTransform(overlapping_rect.Corners)    
        return nornir_imageregistration.BoundingPrimitiveFromPoints(image_space_points)


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
