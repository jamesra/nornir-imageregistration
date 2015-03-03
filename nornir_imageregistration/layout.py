import logging
import os

import nornir_imageregistration.transforms.factory as tfactory
import nornir_imageregistration.tile
import numpy as np

from . import alignment_record
from . import core
from . import spatial


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

        tile = nornir_imageregistration.tile.Tile(t, imagepaths[i], i)
        tiles[tile.ID] = tile
        
    return tiles



class LayoutPosition(object):
    '''This is an anchor with a number of springs of a certain length attached.  In our use the anchor is a tile and the spring size
       and strength is determined by the offset to overlap an adjacent tile
       
       Offsets is a numpy array of the form [[ID Y X Weight]]
    '''
    
    iOffsetID = 0
    iOffsetY = 1 
    iOffsetX = 2 
    iOffsetWeight = 3
    
    #offset_dtype = np.dtype([('ID', np.int32), ('Y', np.float32), ('X', np.float32), ('Weight', np.float32)])
     
    @property
    def ID(self):
        return self._ID
    
    @property
    def Position(self):
        '''Our position in the layout'''
        return self._position
    
    @Position.setter
    def Position(self, value):
        '''Our position in the layout'''
        if not isinstance(value, np.ndarray):
            self._position = np.array(value)
        else:
            self._position = value
        
        assert(self._position.ndim == 1)
        return 
        
    @property
    def KnownOffsetIDs(self):
        return self._OffsetArray[:,LayoutPosition.iOffsetID]
    
    def SetOffset(self, ID, offset, weight):
        '''Set the offset for the specified Layout position ID.  
           This means that when we subtract our position from the other ID's position we hope to obtain this offset value. 
        '''
        
        new_row = np.array((ID, offset[0], offset[1], weight))#, dtype=LayoutPosition.offset_dtype, ndmin=2)
        iKnown = self.KnownOffsetIDs == ID
        if np.any(iKnown):
            #Update a row
            self._OffsetArray[iKnown] = new_row            
        else:
            #Insert a new row
            self._OffsetArray = np.vstack((self._OffsetArray, new_row))
            if self._OffsetArray.ndim == 1:
                self._OffsetArray = np.reshape(self._OffsetArray, (1, self._OffsetArray.shape[0]))
                
            self._OffsetArray = np.sort(self._OffsetArray, 0)
            
        return
     
    @property
    def NetTensionVector(self):
        '''The direction of the vector this tile wants to move after summing all of the offsets'''
        
        #TODO, take weight into account
        return np.sum(self._OffsetArray[:,LayoutPosition.iOffsetY:LayoutPosition.iOffsetX+1],0)
     
    def __init__(self, ID, position, *args, **kwargs):
        
        self._ID = ID 
        self.Position = position
        self._OffsetArray = np.empty((0,4)) #dtype=LayoutPosition.offset_dtype)
        

class SpringLayout(object):
    '''Arranges tiles in 2D space to form a mosaic'''
    
    def __init__(self):
        
        return 


class TileLayout(object):
    '''Arranges tiles in 2D space that form a mosaic.'''

    @property
    def TileIDs(self):
        '''List of tile IDs used in the layout'''
        return list(self._TileToTransform.keys())

    def Contains(self, ID):
        ''':rtype: bool
           :return: True if layout contains the ID
        '''
        return ID in self.TileToTransform

    def GetTileOrigin(self, ID):
        '''Position of the bottom left corner, (0,0), of the tile in the mosaic
           :rtype: numpy.Array with point
           ''' 
        refTransform = self.TileToTransform[ID]
        return refTransform.FixedBoundingBox.BottomLeft
        
    @property
    def Tiles(self):
        '''Tiles contained within this layout'''
        return self._tiles

    def __init__(self, tiles):
        self._tiles = tiles
        self.TileToTransform = dict()


    def _AddIsolatedTile(self, tile):
        '''Add a tile with no overlap to other tiles'''

        if tile.ID in self.TileToTransform:
            return False

        self.TileToTransform[tile.ID] = TranslationTransformForAlignmentRecord((tile.MappedBoundingBox[spatial.iRect.MaxY],
                                                                               tile.MappedBoundingBox[spatial.iRect.MaxX]), arecord=None)

        return True

    def AddTileViaAlignment(self, arecord):

        if not arecord.FixedTileID in self.TileToTransform:
            refTile = self._tiles[arecord.FixedTileID]
            self._AddIsolatedTile(refTile)

        # Figure out the reference tiles position in the ctontrol (mosaic) space.  Add our offset to that
        (minY, minX) = self.GetTileOrigin(arecord.FixedTileID)

        movingTile = self._tiles[arecord.MovingTileID]
        arecord.translate((minY, minX))

        newTransform = TranslationTransformForAlignmentRecord((movingTile.MappedBoundingBox[spatial.iRect.MaxY],
                                                              movingTile.MappedBoundingBox[spatial.iRect.MaxX]), arecord)
        self.TileToTransform[arecord.MovingTileID] = newTransform


    @classmethod
    def MergeLayouts(cls, layoutA, layoutB, offset):
        '''
        Merge B with A by translating all B transforms by offset.
        Then update the dictionary of A
        '''

        for t in list(layoutB.values()):
            t.translate(offset)

        layoutA.TileToTransform.update(layoutB.TileToTransform)


    @classmethod
    def MergeLayoutsWithAlignmentRecord(cls, layoutA, layoutB, arecord):
        '''
        Merge B with A by translating all B transforms by offset.
        Then update the dictionary of A
        '''

        FixedLayout = TileLayout.GetLayoutForID([layoutA, layoutB], arecord.FixedTileID)
        MovingLayout = TileLayout.GetLayoutForID([layoutA, layoutB], arecord.MovingTileID)

        assert(not FixedLayout is None and not MovingLayout is None)

        (fixedMinY, fixedMinX) = FixedLayout.GetTileOrigin(arecord.FixedTileID)
        (movingMinY, movingMinX) = MovingLayout.GetTileOrigin(arecord.MovingTileID)

        # arecord.translate((fixedMinY, fixedMinX))
        ExpectedMovingTilePosition = arecord.peak + np.array([fixedMinY, fixedMinX])

        MovingPositionDifference = np.array([ExpectedMovingTilePosition[0] - movingMinY, ExpectedMovingTilePosition[1] - movingMinX])

        for t in list(MovingLayout.TileToTransform.values()):
            t.TranslateFixed(MovingPositionDifference)

        FixedLayout.TileToTransform.update(MovingLayout.TileToTransform)


    @classmethod
    def GetLayoutForID(cls, listLayouts, ID):
        '''Given a list of tile layouts, returns the layout containing the given ID'''

        if listLayouts is None:
            return None

        for layout in listLayouts:
            if layout.Contains(ID):
                return layout

        return None

def TranslationTransformForAlignmentRecord(OriginalImageSize, arecord=None):
    '''
    :param Tile tile: A tile object
    :param AlignmentRecord arecord: An alignment record
    :return: A transform positioning the tile in 2D space
    '''
    if arecord is None:
        arecord = alignment_record.AlignmentRecord((0, 0), 0, 0)

    return tfactory.CreateRigidTransform(OriginalImageSize, OriginalImageSize, 0, arecord.peak)