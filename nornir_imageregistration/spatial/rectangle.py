'''

Rectangles are represented as (MinY, MinX, MaxY, MaxZ)
Points are represented as (Y,X)

'''

import numpy as np

from .indicies import *


def RaiseValueErrorOnInvalidBounds(bounds):
    if not IsValidBoundingBox(bounds):
        raise ValueError("Negative dimensions are not allowed")

def IsValidBoundingBox(bounds):
    '''Raise a value error if the bounds have negative dimensions'''
    return bounds[iRect.MinX] < bounds[iRect.MaxX] and bounds[iRect.MinY] < bounds[iRect.MaxY]

class RectangleSet():
    '''A set of rectangles'''
    
    rect_dtype = np.dtype([('MinY', 'f4'), ('MinX', 'f4'),('MaxY', 'f4'),('MaxX','f4'), ('ID', 'u8')])
    #active_dtype is used for lists sorting the start and end position of rectangles
    active_dtype = np.dtype([('Value', 'f4'), ('ID','u8'), ('Active', 'u1')])
    
    @classmethod
    def _create_bounds_array(cls, rects):
        '''Create a single numpy array for each rectangle, with the index of the rectangle in the original set'''
        rect_array = np.empty(len(rects), dtype=cls.rect_dtype)        
        for i, rect in enumerate(rects):
            input_rect = rects[i].BoundingBox
            rect_array[i] = (input_rect[0], input_rect[1], input_rect[2],input_rect[3], i)
            
        return rect_array
    
    @classmethod
    def _create_sweep_arrays(cls, rect_array):
        '''
        Create lists that sort the beginning and end values for rectangles along each axis
        '''
        
        x_sweep_array = np.empty(len(rect_array) * 2, dtype=cls.active_dtype)
        y_sweep_array = np.empty(len(rect_array) * 2, dtype=cls.active_dtype)
        
        for i, rect in enumerate(rect_array):
            x_sweep_array[i*2] = (rect['MinX'],rect['ID'],True)
            x_sweep_array[(i*2)+1] = (rect['MaxX'],rect['ID'],False)
            y_sweep_array[i*2] = (rect['MinY'],rect['ID'],True)
            y_sweep_array[(i*2)+1] = (rect['MaxY'],rect['ID'],False)
            
        x_sweep_array = np.sort(x_sweep_array, order=('Value','Active'))
        y_sweep_array = np.sort(y_sweep_array, order=('Value','Active'))
        
        return (x_sweep_array, y_sweep_array)
            
    
    def __init__(self, rects_array):    
        self._rects_array = rects_array#np.sort(rects_array, order=('MinX', 'MinY'))
        
        (self.x_sweep_array, self.y_sweep_array) = RectangleSet._create_sweep_arrays(self._rects_array) 
        #self._minx_sorted = np.sort(self._rects_array, order=('MinX', 'MaxX'))
        #self._miny_sorted = np.sort(self._rects_array, order=('MinY', 'MaxY'))
        #self._maxx_sorted = np.sort(self._rects_array, order=('MaxX', 'MinX'))
        #self._maxy_sorted = np.sort(self._rects_array, order=('MaxY', 'MinY'))
    
    @classmethod
    def Create(cls, rects):
        
        rects_array = cls._create_bounds_array(rects)
        rset = RectangleSet(rects_array)
        return rset
    
    def EnumerateOverlapping(self):
        '''
        :return: A set of tuples containing the indicies of overlapping rectangles passed to the Create function
        '''
        
        OverlapSet = {}
        for (ID,overlappingIDs) in RectangleSet.SweepAlongAxis(self.x_sweep_array):
            OverlapSet[ID] = overlappingIDs
            
        for (ID,overlappingIDs) in RectangleSet.SweepAlongAxis(self.y_sweep_array):
            OverlapSet[ID] &= overlappingIDs
        
        for (ID,overlappingIDs) in OverlapSet.items():
            for MatchingID in overlappingIDs:
                if MatchingID != ID:
                    yield (ID, MatchingID)
                        
    
    @classmethod
    def SweepAlongAxis(cls, sweep_array):
        '''
        :param ndarray sweep_array: Array of active_dtype 
        :return: A set of tuples containing the indicies of overlapping rectangles on the axis
        '''
        ActiveSet = set()
        last_value = None
        IDsToYield = []
        for i_x in range(0,len(sweep_array)):
            NextIsDifferent = True
            
            ID = sweep_array[i_x]['ID'] 
            if i_x + 1 < len(sweep_array):
                NextIsDifferent = sweep_array[i_x + 1]['Value'] != sweep_array[i_x]['Value'] and sweep_array[i_x + 1]['Active'] != sweep_array[i_x]['Active']
            
            if sweep_array[i_x]['Active']:
                ActiveSet.add(ID)
                IDsToYield.append(ID)
            else:   
                ActiveSet.remove(ID)
                
            if NextIsDifferent:
                for YieldID in IDsToYield:
                    yield (YieldID, ActiveSet.copy())
                IDsToYield.clear()
                         
    def __str__(self):
        return str(self._rects_array)


class Rectangle(object):
    '''
    
    '''

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
        return self.BottomLeft + ((self.TopRight - self.BottomLeft)  / 2.0)
    
    @property
    def Area(self):
        return self.Width * self.Height
    
    @property
    def Size(self):
        return self.TopRight - self.BottomLeft

    @property
    def BoundingBox(self):
        return self._bounds

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

    def __init__(self, bounds):
        '''
        Constructor, bounds = [left bottom right top]
        '''
 
        self._bounds = bounds

    def ToArray(self):
        return np.array(self._bounds)
    
    def ToTuple(self):
        return (self._bounds[iRect.MinY],
                self._bounds[iRect.MinX],
                self._bounds[iRect.MaxY],
                self._bounds[iRect.MaxX])

    @classmethod
    def CreateFromPointAndArea(cls, point, area):
        '''
        :param tuple point: (Y,X)
        :param tuple area: (Height, Area)
        :rtype: Rectangle
        '''
        return Rectangle(bounds=(point[iPoint.Y], point[iPoint.X], point[iPoint.Y] + area[iArea.Height], point[iPoint.X] + area[iArea.Width]))

    @classmethod
    def CreateFromBounds(cls, Bounds):
        '''
        :param tuple Bounds: (MinY,MinX,MaxY,MaxX)
        '''
        # return Rectangle(Bounds[1], Bounds[0], Bounds[3], Bounds[2])
        return Rectangle(Bounds)

    @classmethod
    def PrimitiveToRectange(cls, primitive):
        '''Privitive can be a list of (Y,X) or (MinY, MinX, MaxY, MaxX) or a Rectangle'''

        if isinstance(primitive, Rectangle):
            return primitive

        if len(primitive) == 2:
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
    def overlap_rect(cls, A, B):
        '''
        :rtype: Rectangle
        :returns: The rectangle describing the overlapping regions of rectangles A and B
        '''
        A = Rectangle.PrimitiveToRectange(A)
        B = Rectangle.PrimitiveToRectange(B)
        
        if not cls.contains(A,B):
            return None
        
        minX = max((A.BoundingBox[iRect.MinX], B.BoundingBox[iRect.MinX]))
        minY = max((A.BoundingBox[iRect.MinY], B.BoundingBox[iRect.MinY]))
        maxX = min((A.BoundingBox[iRect.MaxX], B.BoundingBox[iRect.MaxX]))
        maxY = min((A.BoundingBox[iRect.MaxY], B.BoundingBox[iRect.MaxY]))
        
        return Rectangle.CreateFromBounds((minY,minX,maxY,maxX))
    
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
    def scale(cls, A, scale):
        '''
        Return a rectangle with the same center, but scaled in total area
        '''
        
        new_size = A.Size * scale
        bottom_left = A.Center - (new_size / 2.0)
        return cls.CreateFromPointAndArea(bottom_left, new_size)
    
    @classmethod
    def SafeRound(cls, A):
        '''Round the rectangle bounds to the nearest integer, increasing in area and never decreasing. 
           The bottom left corner is rounded down and the upper right corner is rounded up.
           This is useful to prevent the case where two images have rectangles that are later scaled, but precision and rounding issues
           cause them to have mismatched bounding boxes'''
        
        bottomleft = np.floor(A.BottomLeft)
        topright = bottomleft + np.ceil(A.Size)
        
        return cls.CreateFromPointAndArea(bottomleft, topright-bottomleft)        
         

    def __str__(self):
        return "MinX: %g MinY: %g MaxX: %g MaxY: %g" % (self._bounds[iRect.MinX], self._bounds[iRect.MinY], self._bounds[iRect.MaxX], self._bounds[iRect.MaxY])
