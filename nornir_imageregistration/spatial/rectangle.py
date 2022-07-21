'''

Rectangles are represented as (MinY, MinX, MaxY, MaxZ)
Points are represented as (Y,X)

'''

import collections.abc
import numpy as np
from .indicies import iPoint, iRect, iArea
from .converters import BoundingPrimitiveFromPoints
   
def RaiseValueErrorOnInvalidBounds(bounds):
    if not IsValidBoundingBox(bounds):
        raise ValueError("Negative dimensions are not allowed")

def IsValidBoundingBox(bounds):
    '''Raise a value error if the bounds have negative dimensions'''
    return bounds[iRect.MinX] < bounds[iRect.MaxX] and bounds[iRect.MinY] < bounds[iRect.MaxY]

def IsValidRectangleInputArray(bounds):
    if bounds.ndim == 1:
        return bounds.size == 4
    elif bounds.ndim > 1:
        return bounds.shape[1] == 4
     
class RectangleSet():
    '''A set of rectangles with minor optimization to identify overlapping rectangles in the set'''
    
    rect_dtype = np.dtype([('MinY', 'f4'), ('MinX', 'f4'), ('MaxY', 'f4'), ('MaxX', 'f4'), ('ID', 'u8')])
    # active_dtype is used for lists sorting the start and end position of rectangles
    active_dtype = np.dtype([('Value', 'f4'), ('ID', 'u8'), ('InBounds', 'u1')])
    
    @classmethod
    def _create_bounds_array(cls, rects):
        '''Create a single numpy array containing the boundaries for each rectangle, with an additional 5th column containing  the index of the rectangle in the original set'''
        rect_array = np.empty(len(rects), dtype=cls.rect_dtype)        
        for (i, rect) in enumerate(rects):
            input_rect = rect.BoundingBox
            rect_array[i] = (input_rect[0], input_rect[1], input_rect[2], input_rect[3], i)
            
        return rect_array
    
    @classmethod
    def _create_sweep_arrays(cls, rect_array):
        '''
        Create lists that sort the beginning and end values for rectangles along each axis
        '''
        x_sweep_array = np.empty(len(rect_array) * 2, dtype=cls.active_dtype)
        y_sweep_array = np.empty(len(rect_array) * 2, dtype=cls.active_dtype)
        
        #degenerate_x = []
        #degenerate_y = []
        
        for i, rect in enumerate(rect_array):
            x_sweep_array[i * 2] = (rect['MinX'], rect['ID'], True)
            x_sweep_array[(i * 2) + 1] = (rect['MaxX'], rect['ID'], False)
            y_sweep_array[i * 2] = (rect['MinY'], rect['ID'], True)
            y_sweep_array[(i * 2) + 1] = (rect['MaxY'], rect['ID'], False)
            
            if rect['MinX'] >= rect['MaxX']:
                raise ValueError(f"Degenerate rectangle with zero width not supported: {rect}")
                #degenerate_x.append(rect['ID'])
            if rect['MinY'] >= rect['MaxY']:
                raise ValueError(f"Degenerate rectangle with zero height not supported: {rect}")
                #degenerate_y.append(rect['ID'])
            
        x_sweep_array = np.sort(x_sweep_array, order=('Value', 'InBounds'))
        y_sweep_array = np.sort(y_sweep_array, order=('Value', 'InBounds'))
        
        #Handle degenerate cases by reverse sorting the Active column so they are in-bounds before out of bounds
        
         
        return (x_sweep_array, y_sweep_array)
    
    def __init__(self, rects_array):    
        self._rects_array = rects_array
        
        (self.x_sweep_array, self.y_sweep_array) = RectangleSet._create_sweep_arrays(self._rects_array) 
    
    @classmethod
    def Create(cls, rects):
        
        rects_array = cls._create_bounds_array(rects)
        rset = RectangleSet(rects_array)
        return rset
    
    def _AddOverlapPairToDict(self, OverlapDict, ID, MatchingID):
        
        if ID in OverlapDict:
            OverlapDict[ID].add(MatchingID)
        
        if MatchingID in OverlapDict:
            OverlapDict[MatchingID].add(ID)


    def BuildTileOverlapDict(self):
        
        OverlapDict = {}
        
        for (ID, MatchingID) in self.EnumerateOverlapping():
            self._AddOverlapPairToDict(OverlapDict, ID, MatchingID)
        
        return OverlapDict
    
    def Intersect(self, rect):
        '''
        :returns: all rectangles in the set that intersect the provided rectangle 
        '''
        
        rect = Rectangle.PrimitiveToRectange(rect)
        
        x_intersections = frozenset(self._scan_sweep_array_for_all_intersections(self.x_sweep_array,
                                                                       min_val=rect.MinX, 
                                                                       max_val=rect.MaxX))
        if x_intersections:
            y_intersections = frozenset(self._scan_sweep_array_for_all_intersections(self.y_sweep_array,
                                                                       min_val=rect.MinY, 
                                                                       max_val=rect.MaxY))
            
            return x_intersections & y_intersections
        else:
            return frozenset([])
        
        
    
    @classmethod
    def _scan_sweep_array_for_all_intersections(cls, sweep_array, min_val:float, max_val:float):
        '''
        Given a min and max value, return all intersecting rectangle IDs for the 
        given axis.
        :returns: A generator, which may yield duplicates, of all intersecting entries in the sweep array
        '''
        
        #This function is very chatty in that it returns rectangles more than once, that is OK, resist the urge to optimize.
        #Use a set if you need unique results 
        potential_intersections = set()
        
        for entry in sweep_array:
            axis_coordinate = entry['Value']
            
            if axis_coordinate < min_val: #Before we reach min_val track all rectangles that start in case they span the entire range
                if entry['InBounds']:
                    potential_intersections.add(entry['ID'])
                else:
                    potential_intersections.remove(entry['ID'])
            else:
                if axis_coordinate > max_val: #Check if we have exited the interesting range
                    break #No need to scan the remainder of the rectangles, none will be in range
                else:
                    ID = entry['ID']
                    if ID in potential_intersections:
                        potential_intersections.remove(ID)
                        
                    yield ID
        
        yield from potential_intersections #Return the intersections that could span the entire range
        
                
    
    def EnumerateOverlapping(self):
        '''
        :return: A set of tuples containing the indicies of overlapping rectangles passed to the Create function
        '''
        
        OverlapSet = {}
        for (ID, overlappingIDs) in RectangleSet.SweepAlongAxis(self.x_sweep_array):
            OverlapSet[ID] = overlappingIDs
                
        for (ID, overlappingIDs) in RectangleSet.SweepAlongAxis(self.y_sweep_array):
            OverlapSet[ID] &= overlappingIDs
        
        returned_overlaps = set()
        for (ID, overlappingIDs) in OverlapSet.items():
            for MatchingID in overlappingIDs:
                if MatchingID != ID:
                    key = None
                    if ID < MatchingID:
                        key = (ID, MatchingID)
                    else:
                        key = (MatchingID, ID)
                        
                    if key in returned_overlaps:
                        continue
                    
                    returned_overlaps.add(key)
                    yield key
    
    @staticmethod
    def SweepAlongAxis(sweep_array):
        '''
        :param ndarray sweep_array: Array of active_dtype 
        :return: A set of tuples containing the indicies of overlapping rectangles on the axis
        '''
        if sweep_array is None:
            raise ValueError("sweep_array must not be None")
         
        ActiveSet = set() 
        overlaps_for_ID = {}
        IDs_to_yield_on_sweep_move = []
        for i_x in range(0, len(sweep_array)):
            sweep_line_moves = True
            active_state_changes = True
            entry_is_active = sweep_array[i_x]['InBounds']
             
            ID = sweep_array[i_x]['ID']
            if i_x + 1 < len(sweep_array): 
                sweep_line_moves = sweep_array[i_x + 1]['Value'] != sweep_array[i_x]['Value']
                active_state_changes = sweep_array[i_x + 1]['InBounds'] != sweep_array[i_x]['InBounds']            
            if entry_is_active:
                ActiveSet.add(ID)
                overlaps_for_ID[ID] = ActiveSet.copy()
                
                for active_entry_ID in ActiveSet:
                    overlaps_for_ID[active_entry_ID].add(ID) 
            else:   
                ActiveSet.remove(ID)
                IDs_to_yield_on_sweep_move.append(ID)
                
            if sweep_line_moves or active_state_changes:
                for YieldID in IDs_to_yield_on_sweep_move:
                    yield (YieldID, overlaps_for_ID[YieldID])
                    del overlaps_for_ID[YieldID]
                    
                IDs_to_yield_on_sweep_move.clear()
        
        for YieldID in IDs_to_yield_on_sweep_move:
            yield (YieldID, overlaps_for_ID[YieldID])
            del overlaps_for_ID[YieldID]
            
        IDs_to_yield_on_sweep_move.clear()
        return
    
                         
    def __str__(self):
        return str(self._rects_array)


class Rectangle(object):
    '''
    Defines a 2D rectangle
    '''
    
    @property
    def MinX(self):
        return self._bounds[iRect.MinX]
    
    @property
    def MaxX(self):
        return self._bounds[iRect.MaxX]
    
    @property
    def MinY(self):
        return self._bounds[iRect.MinY]
    
    @property
    def MaxY(self):
        return self._bounds[iRect.MaxY]

    @property
    def Width(self):
        return self._bounds[iRect.MaxX] - self._bounds[iRect.MinX]

    @property
    def Height(self):
        return self._bounds[iRect.MaxY] - self._bounds[iRect.MinY]

    @property
    def BottomLeft(self):
        return np.array([self._bounds[iRect.MinY], self._bounds[iRect.MinX]])

    @property
    def TopLeft(self):
        return np.array([self._bounds[iRect.MaxY], self._bounds[iRect.MinX]])

    @property
    def BottomRight(self):
        return np.array([self._bounds[iRect.MinY], self._bounds[iRect.MaxX]])

    @property
    def TopRight(self):
        return np.array([self._bounds[iRect.MaxY], self._bounds[iRect.MaxX]])
    
    @property
    def Corners(self):
        return np.vstack((self.BottomLeft,
                             self.TopLeft,
                             self.TopRight,
                             self.BottomRight))
    
    @property
    def Center(self):
        return self.BottomLeft + ((self.TopRight - self.BottomLeft) / 2.0)
    
    @property
    def Area(self):
        return self.Width * self.Height
    
    @property
    def Dimensions(self):
        '''
        The [height, width] of the rectangle
        '''
        
        return np.asarray([self.Height, self.Width], np.float64)
    
    @property
    def shape(self):
        '''
        The [height, width] of the rectangle
        '''
        
        return np.ceil(np.asarray([self.Height, self.Width])).astype(np.int64)
    
    @property
    def Size(self):
        return self.TopRight - self.BottomLeft

    @property
    def BoundingBox(self):
        return self._bounds
    
    
    def __eq__(self, other):
        if isinstance(other,Rectangle):
            return np.array_equal(self._bounds, other._bounds)
        elif isinstance(other, np.array):
            return np.array_equal(self._bounds, other)
        
        return False;
    
    def __ne__(self, other):
        return self.__eq__(other) == False
            
    def __getitem__(self, i):
        return self._bounds.__getitem__(i)

    def __setitem__(self, i, sequence):
        self._bounds.__setitem__(i, sequence)

    def __getslice__(self, i, j):
        return self._bounds.__getslice__(i, j)

    def __setslice__(self, i, j, sequence):
        self._bounds.__setslice__(i, j, sequence)

    def __delslice__(self, i, j, sequence):
        raise Exception("Spatial objects should not have elements deleted from the array")
    
    def __repr__(self):
        return "x:{0}g y:{1}g w:{2}g h:{3}g".format(self.BottomLeft[1], self.BottomLeft[0], self.Width, self.Height)

    def __init__(self, bounds):
        '''
        :param object bounds: An ndarray or iterable of [bottom left top right] OR an existing Rectangle object to copy. 
        '''
        if bounds is None:
            raise ValueError("bounds for Rectangle must not be None")
        
        if isinstance(bounds, np.ndarray):
            if not IsValidRectangleInputArray(bounds):
                raise ValueError("Invalid input to Rectangle constructor.  Expected four elements (MinY,MinX,MaxY,MaxX): {!r}".format(bounds))
            
            self._bounds = bounds.astype(np.float64)
        elif isinstance(bounds, Rectangle):
            self._bounds = bounds.ToArray()
        else:
            self._bounds = np.array(bounds, dtype=np.float64)
            
        if not Rectangle._AreBoundsValid(self._bounds):
            raise ValueError("Invalid input to Rectangle constructor.  Expected four elements (MinY,MinX,MaxY,MaxX): {!r}".format(self._bounds))
        
        return 
    
    def __getstate__(self):
        d = {}
        d['_bounds'] = self._bounds 
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)

    def ToArray(self):
        return self._bounds.copy()
    
    def ToTuple(self):
        return (self._bounds[iRect.MinY],
                self._bounds[iRect.MinX],
                self._bounds[iRect.MaxY],
                self._bounds[iRect.MaxX])
        
    def copy(self):
        return Rectangle.CreateFromBounds(self._bounds.copy())

    @classmethod
    def Union(cls, *args):
        '''
        :param other: Either a 2D array for a point, a 4D array for a rectangle, or a rectangle object
        :rtype: Rectangle
        :returns: The rectangle describing the bounding box of both shapes
        ''' 
        
        if len(args) == 0:
            raise ValueError("*args must have at least one Rectangle")
        elif len(args) == 1:
            if isinstance(args[0], Rectangle):
                return args[0]
            elif isinstance(args[0], collections.abc.Iterable):
                return cls.Union(*(args[0]))
            else: 
                return Rectangle.PrimitiveToRectange(args[0])
        else:
            r = Rectangle.PrimitiveToRectange(args[0])
            mbb = r._bounds.copy()
            
            for r in args:
                mbb[0:2] = np.min((mbb[0:2], r._bounds[0:2]),0)
                mbb[2:] = np.max((mbb[2:], r._bounds[2:]),0)
                
            return Rectangle(mbb)
    
    @classmethod
    def Intersect(cls, A, B):
        '''
        Returns the intersection of two triangles
        :param other: Either a 2D array for a point, a 4D array for a rectangle, or a rectangle object
        :rtype: Rectangle
        :returns: The rectangle describing the bounding box of both shapes
        ''' 

        A = Rectangle.PrimitiveToRectange(A)
        B = Rectangle.PrimitiveToRectange(B)

        if not cls.contains(A, B):
            return None

        minX = max((A.BoundingBox[iRect.MinX], B.BoundingBox[iRect.MinX]))
        minY = max((A.BoundingBox[iRect.MinY], B.BoundingBox[iRect.MinY]))
        maxX = min((A.BoundingBox[iRect.MaxX], B.BoundingBox[iRect.MaxX]))
        maxY = min((A.BoundingBox[iRect.MaxY], B.BoundingBox[iRect.MaxY]))
        
        if minX > maxX or minY > maxY:
            return None
        
        return Rectangle.CreateFromBounds((minY, minX, maxY, maxX))

    @classmethod
    def _AreBoundsValid(cls, bounds):
        return isinstance(bounds, np.ndarray) and \
               bounds.size == 4 and \
               np.issubdtype(bounds.dtype, np.floating) 
               
    @staticmethod
    def CreateBoundingRectangleForPoints(points):
        '''
        Create a rectangle bounding the passed array of points
        :param tuple points: ndarray of (Y,X)
        :rtype: Rectangle
        '''
        
        if not isinstance(points, np.ndarray):
            points = np.asarray(points)
            
        rect = BoundingPrimitiveFromPoints(points)
        if not isinstance(rect, Rectangle):
            raise ValueError("CreateBoundingRectangleForPoints expects a 2D array of points")
        
        return rect


    @staticmethod
    def CreateFromCenterPointAndArea(point, area):
        '''
        Create a rectangle whose center is point and the requested area
        :param tuple point: (Y,X)
        :param tuple area: (Height, Area)
        :rtype: Rectangle
        '''
        if not isinstance(area, np.ndarray):
            area = np.asarray(area)
            
        half_area = area / 2.0
            
        return Rectangle(bounds=(point[iPoint.Y] - half_area[iArea.Height], point[iPoint.X] - half_area[iArea.Width], point[iPoint.Y] + half_area[iArea.Height], point[iPoint.X] + half_area[iArea.Width]))
    
    @staticmethod
    def CreateFromPointAndArea(point, area):
        '''
        Create a rectangle whose bottom left origin is at point with the requested area
        :param tuple point: (Y,X)
        :param tuple area: (Height, Area)
        :rtype: Rectangle
        '''
        return Rectangle(bounds=(point[iPoint.Y], point[iPoint.X], point[iPoint.Y] + area[iArea.Height], point[iPoint.X] + area[iArea.Width]))

    @staticmethod
    def CreateFromBounds(Bounds):
        '''
        :param tuple Bounds: (MinY,MinX,MaxY,MaxX)
        '''
        # return Rectangle(Bounds[1], Bounds[0], Bounds[3], Bounds[2])
        return Rectangle(Bounds)

    @classmethod
    def PrimitiveToRectange(cls, primitive):
        '''Primitive can be a list of (Y,X) or (MinY, MinX, MaxY, MaxX) or a Rectangle'''

        if isinstance(primitive, Rectangle):
            return primitive
        
        if isinstance(primitive, np.void):
            if primitive.dtype == RectangleSet.rect_dtype:
                return Rectangle((primitive[0], primitive[1], primitive[2], primitive[3]))

        if len(primitive) == 2:
            Warning("This constructor appears odd, investigate this path")
            return Rectangle(primitive[0], primitive[1], primitive[0], primitive[1])
        elif len(primitive) == 4:
            return Rectangle(primitive)
        else:
            raise ValueError("Unknown primitve type %s" % str(primitive))

    @classmethod
    def contains(cls, A, B):
        '''If len == 2 primitive is a point,
           if len == 4 primitive is a rect [left bottom right top]'''

        A = Rectangle.PrimitiveToRectange(A)
        B = Rectangle.PrimitiveToRectange(B)

        if(A.BoundingBox[iRect.MaxX] <= B.BoundingBox[iRect.MinX] or
           A.BoundingBox[iRect.MinX] >= B.BoundingBox[iRect.MaxX] or
           A.BoundingBox[iRect.MaxY] <= B.BoundingBox[iRect.MinY] or
           A.BoundingBox[iRect.MinY] >= B.BoundingBox[iRect.MaxY]):

            return False

        return True
    
    @classmethod
    def max_overlap_dimensions(cls, A, B):
        '''
        :returns: The maximum distance the rectangles can overlap on each axis
        '''
        return np.min(np.vstack((A.Dimensions, B.Dimensions)),0)
        
    @classmethod
    def max_overlap_area(cls, A,B):
        '''
        :returns: The maximum area the rectangles can overlap
        '''
        return np.prod(cls.max_overlap_dimensions(A, B))
    
    @classmethod
    def overlap_rect(cls, A, B):
        '''
        :rtype: Rectangle
        :returns: The rectangle describing the overlapping regions of rectangles A and B
        '''
        A = Rectangle.PrimitiveToRectange(A)
        B = Rectangle.PrimitiveToRectange(B)
        
        if not cls.contains(A, B):
            return None
        
        minX = max((A.BoundingBox[iRect.MinX], B.BoundingBox[iRect.MinX]))
        minY = max((A.BoundingBox[iRect.MinY], B.BoundingBox[iRect.MinY]))
        maxX = min((A.BoundingBox[iRect.MaxX], B.BoundingBox[iRect.MaxX]))
        maxY = min((A.BoundingBox[iRect.MaxY], B.BoundingBox[iRect.MaxY]))
        
        return Rectangle.CreateFromBounds((minY, minX, maxY, maxX))
    
    @classmethod 
    def overlap(cls, A, B):
        '''
        :rtype: float
        :returns: 0 to 1 indicating area of A overlapped by B
        '''
       
        overlapping_rect = cls.overlap_rect(A, B)
        if overlapping_rect is None:
            return 0.0
          
        return overlapping_rect.Area / A.Area
    
    @classmethod 
    def overlap_normalized(cls, A, B):
        '''
        :rtype: float
        :returns: 0 to 1 indicating overlapping rect divided by maximum possible overlapping rectangle
        '''
       
        overlapping_rect = cls.overlap_rect(A, B)
        if overlapping_rect is None:
            return 0.0
          
        return overlapping_rect.Area / Rectangle.max_overlap_area(A, B)
    
    @classmethod
    def translate(cls, A, offset):
        '''
        :return: A copy of the rectangle translated by the specified amount
        '''
        if not isinstance(offset, np.ndarray):
            offset = np.array(offset)
            
        translated_rect = Rectangle(A._bounds.copy())
        translated_rect[iRect.MinY] += offset[0]
        translated_rect[iRect.MaxY] += offset[0]
        translated_rect[iRect.MinX] += offset[1]
        translated_rect[iRect.MaxX] += offset[1]
        
        return translated_rect
        
    
    @classmethod
    def scale_on_center(cls, A, scale):
        '''
        Return a rectangle with the same center, but scaled in total area
        '''
        
        new_size = A.Size * scale
        return cls.change_area(A, new_size)
    
    @classmethod
    def scale_on_origin(cls, A, scale):
        '''
        Return a rectangle scaled in total area relative to the (0,0) origin
        '''
        
        return cls.CreateFromBounds(A.BoundingBox * scale)
    
    @classmethod
    def change_area(cls, A, new_size, integer_origin=False):
        '''
        :param bool integer_origin: If true, the bottom left will remain as integers, prevents a rectangle shifting 0.5 units to keep the center perfectly positioned
        :Returns: A rectangle with the area of new_shape, but the same center
        '''
        if not isinstance(new_size, np.ndarray):
            new_size = np.array(new_size)
        bottom_left = A.Center - (new_size / 2.0)
        
        if integer_origin:
            bottom_left = np.floor(bottom_left)
            
        return cls.CreateFromPointAndArea(bottom_left, new_size)
    
    @classmethod
    def SafeRound(cls, A):
        '''Round the rectangle bounds to the nearest integer, increasing in area and never decreasing. 
           The bottom left corner is rounded down and the upper right corner is rounded up.
           This is useful to prevent the case where two images have rectangles that are later scaled, but precision and rounding issues
           cause them to have mismatched bounding boxes'''
        
        bottomleft = np.floor(A.BottomLeft)
        topright = bottomleft + np.ceil(A.Size)
        
        if topright[0] < A.TopRight[0]:
            topright[0] += 1
        
        if topright[1] < A.TopRight[1]:
            topright[1] += 1
        
        return cls.CreateFromPointAndArea(bottomleft, topright - bottomleft)        
         

    def __str__(self):
        return "MinX: %g MinY: %g MaxX: %g MaxY: %g" % (self._bounds[iRect.MinX], self._bounds[iRect.MinY], self._bounds[iRect.MaxX], self._bounds[iRect.MaxY])
