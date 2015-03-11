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

        if(A.BoundingBox[iRect.MaxX] < B.BoundingBox[iRect.MinX] or
           A.BoundingBox[iRect.MinX] > B.BoundingBox[iRect.MaxX] or
           A.BoundingBox[iRect.MaxY] < B.BoundingBox[iRect.MinY] or
           A.BoundingBox[iRect.MinY] > B.BoundingBox[iRect.MaxY]):

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
