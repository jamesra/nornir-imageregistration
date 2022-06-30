
import nornir_imageregistration 
import numpy as np 

def _IterateOverlappingTiles(list_tiles, min_overlap=None):
    '''Return all tiles which overlap'''
    
    list_rects = [tile.FixedBoundingBox for tile in list_tiles]        
    rset = nornir_imageregistration.RectangleSet.Create(list_rects)
    
    for (A, B) in rset.EnumerateOverlapping():
        if min_overlap is None:
            yield (list_tiles[A], list_tiles[B])
        elif nornir_imageregistration.Rectangle.overlap(list_rects[A], list_rects[B]) > min_overlap:
            yield (list_tiles[A], list_tiles[B])
            
def IterateTileOverlaps(list_tiles, image_to_source_space_scale=1.0, min_overlap=None, inter_tile_distance_scale=1.0):
    yield from CreateTileOverlaps(list_tiles, image_to_source_space_scale, min_overlap, inter_tile_distance_scale=inter_tile_distance_scale)
            
def CreateTileOverlaps(list_tiles, image_to_source_space_scale=1.0, min_overlap=None, inter_tile_distance_scale=1.0):
    '''
    :param float imageScale: Downsample factor of image files
    :param float min_overlap: 0 to 1.0 indicating amount of area that must overlap between tiles
    :param float inter_tile_distance_scale: When tiles overlap, scale the distance between them by this factor.  Used to increase the area of overlap used for registration for cases where the input positions are noisy 
    '''
    if isinstance(list_tiles, dict):
        list_tiles = list(list_tiles.values())
    
    for (A, B) in _IterateOverlappingTiles(list_tiles,min_overlap=min_overlap):
        overlap_data = TileOverlap(A,B,image_to_source_space_scale=image_to_source_space_scale, inter_tile_distance_scale=inter_tile_distance_scale)
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
        return not (self._scaled_overlapping_source_rects[0] is None or self._scaled_overlapping_source_rects[1] is None)
    
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
    def offset(self):
        '''
        The result of B.Center - A.Center.
        '''
        return self._offset
    
    @property
    def scaled_offset(self):
        '''
        The result of B.Center - A.Center.
        '''
        return self._offset * self._imageScale
    
    @property
    def overlapping_target_rect(self):
        '''
        Rectangle describing the overlap in volume (target) space
        '''
        return self._overlapping_target_rect
    
    @property
    def overlapping_source_rects(self):
        '''
        Tuple of rectangles (A,B) describing the overlap of both tiles in tile image (source) space
        '''
        return self._overlapping_source_rects
    
    @property
    def overlapping_source_rect_A(self):
        '''
        Rectangle describing the overlap of both tiles in tile image (source) space
        '''
        return self._overlapping_source_rects[0]
    
    @property
    def overlapping_source_rect_B(self):
        '''
        Rectangle describing the overlap of both tiles in tile image (source) space
        '''
        return self._overlapping_source_rects[1]
    
    @property
    def scaled_overlapping_source_rects(self):
        '''
        Tuple of rectangles (A,B) describing the overlap of both tiles in tile image (source) space
        '''
        return self._scaled_overlapping_source_rects
    
    @property
    def scaled_overlapping_source_rect_A(self):
        '''
        Rectangle describing the overlap of both tiles in tile image (source) space
        '''
        return self._scaled_overlapping_source_rects[0]
    
    @property
    def scaled_overlapping_source_rect_B(self):
        '''
        Rectangle describing the overlap of both tiles in tile image (source) space
        '''
        return self._scaled_overlapping_source_rects[1]
    
     
    @property
    def overlap(self):
        '''
        :return: 0 to 1 float indicating the overlapping rectangle area divided by largest tile area
        '''
        if self._overlap is None:
            if self.scaled_overlapping_source_rect_A is None or self.scaled_overlapping_source_rect_B is None:
                self._overlap = 0
            else:
                self._overlap = nornir_imageregistration.Rectangle.overlap_normalized(self.A.FixedBoundingBox,self.B.FixedBoundingBox)
            
        return self._overlap
    
    
    def get_expanded_overlap_rects(self, scale_factor):
        raise NotImplemented()
            
    def __init__(self, A, B, image_to_source_space_scale, inter_tile_distance_scale=1.0):
        
        if image_to_source_space_scale is None:
            image_to_source_space_scale = 1.0
            
        if image_to_source_space_scale < 1:
            raise ValueError("This might be OK... but images are almost always downsampled.  This exception was added to migrate from old code to this class because at that time all scalars were positive.  For example a downsampled by 4 image must have coordinates multiplied by 4 to match the full-res source space of the transform.")

             
        #Ensure ID's used are from low to high
        if A.ID > B.ID:
            temp = A
            A = B
            B = temp
            
        self._imageScale = 1.0 / image_to_source_space_scale
        self._Tiles = (A,B)
        self._feature_scores = [None, None]
        self._overlap = None
        (overlapping_rect_A, overlapping_rect_B, self._overlapping_target_rect, self._offset) = TileOverlap.Calculate_Overlapping_Regions(A, B,
                                                                                                                                          inter_tile_distance_scale=inter_tile_distance_scale)
        self._overlapping_source_rects = (overlapping_rect_A, overlapping_rect_B)
        
        self._scaled_overlapping_source_rects = TileOverlap.scale_overlapping_rects(overlapping_rect_A, overlapping_rect_B, scalar=1.0 / image_to_source_space_scale)
        
    def __repr__(self):
        area_scale = self._imageScale ** 2
        val = "({0},{1}) ({2:02f}%,{3:02f}%)".format(self.ID[0], self.ID[1],
                                                     float((self._scaled_overlapping_source_rects[0].Area / (self.A.FixedBoundingBox.Area * area_scale)) * 100.0),
                                                     float((self._scaled_overlapping_source_rects[1].Area / (self.B.FixedBoundingBox.Area * area_scale)) * 100.0))
                                             
        
        if self.feature_scores[0] is not None and self.feature_scores[1] is not None:
            val = val + " {0}".format(str(self.feature_scores))
        
        
            
        return val
    
    @staticmethod
    def Calculate_Overlapping_Regions(A, B, inter_tile_distance_scale=1.0):
        '''
        :param tile A:
        :param tile B:
        :param float inter_tile_distance_scale: A value from 0 to 1 that scales the distance between the centers of the two tiles.  A value of 0 considers the full images when registering.  A value of 1.0 only considers the overlapping regions according to the tile transforms.  Reduce this value in early registration passes to increase the search area.
        :return: A tuple with:
            1. The rectangle describing the overlapping region in source (image) space of A
            2. The rectangle describing the overlapping region in source (image) space of B
            3. The rectangle describing the overlapping portions of tile A and B in the destination (volume) space
            4. The offset adjustment from A to B, how much to add to the center of A to get the center of B
        '''
        A_target_bbox = A.FixedBoundingBox
        B_target_bbox = B.FixedBoundingBox
        OffsetAdjustment = (B.FixedBoundingBox.Center - A.FixedBoundingBox.Center)
        original_offset = OffsetAdjustment
        
        if inter_tile_distance_scale != 1.0:
            
            dist = np.sqrt(np.sum(np.power(original_offset, 2)))
            scaled_offset = original_offset * inter_tile_distance_scale
            new_B_target_bbox_center = A_target_bbox.Center + scaled_offset #B_target_bbox.Center - (scaled_offset / 2)
            new_A_target_bbox_center = A_target_bbox.Center #A_target_bbox.Center + (scaled_offset / 2)
            OffsetAdjustment = new_B_target_bbox_center - new_A_target_bbox_center
             
            moved_B_target_bbox = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(new_B_target_bbox_center, B_target_bbox.Dimensions)
            
            moved_A_target_bbox = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(new_A_target_bbox_center, A_target_bbox.Dimensions)
            
            A_target_bbox = nornir_imageregistration.Rectangle.Union(moved_A_target_bbox, A_target_bbox)
            B_target_bbox = nornir_imageregistration.Rectangle.Union(moved_B_target_bbox, B_target_bbox)
        
        overlapping_target_rect = nornir_imageregistration.Rectangle.overlap_rect(A_target_bbox, B_target_bbox)
        if overlapping_target_rect is None:
            return (None, None, None,None)
        
        overlapping_rect_A = A.Get_Overlapping_Source_Rect(overlapping_target_rect)
        overlapping_rect_B = B.Get_Overlapping_Source_Rect(overlapping_target_rect)
        
        overlapping_rect_B = nornir_imageregistration.Rectangle.translate(overlapping_rect_B, original_offset - OffsetAdjustment)
        #overlapping_target_rect = nornir_imageregistration.Rectangle.translate(overlapping_target_rect, original_offset - OffsetAdjustment)
        
        return (overlapping_rect_A, overlapping_rect_B, overlapping_target_rect, OffsetAdjustment)
    
    @staticmethod
    def scale_overlapping_rects(overlapping_rect_A, overlapping_rect_B, scalar):
         
        downsampled_overlapping_rect_A = nornir_imageregistration.Rectangle.SafeRound(nornir_imageregistration.Rectangle.CreateFromBounds(overlapping_rect_A.ToArray() * scalar))
        downsampled_overlapping_rect_B = nornir_imageregistration.Rectangle.SafeRound(nornir_imageregistration.Rectangle.CreateFromBounds(overlapping_rect_B.ToArray() * scalar))
        
        # If the predicted alignment is perfect and we use only the overlapping regions  we would have an alignment offset of 0,0.  Therefore we add the existing offset between tiles to the result
        #OffsetAdjustment = OffsetAdjustment * imageScale
        
        # This should ensure we never have an an area mismatch
        downsampled_overlapping_rect_B = nornir_imageregistration.Rectangle.CreateFromPointAndArea(downsampled_overlapping_rect_B.BottomLeft, downsampled_overlapping_rect_A.Size)
        
        assert(downsampled_overlapping_rect_A.Width == downsampled_overlapping_rect_B.Width)
        assert(downsampled_overlapping_rect_A.Height == downsampled_overlapping_rect_B.Height)

        return (downsampled_overlapping_rect_A, downsampled_overlapping_rect_B)
    
    @staticmethod
    def Calculate_Largest_Possible_Region(A,B):
        A_B_vector = A.FixedBoundingBox.Center - B.FixedBoundingBox.Center
        A_B_distance = np.sqrt(np.sum((A_B_vector ** 2)))
        
        
        pass