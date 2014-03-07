'''

Rectangles are represented as (MinY, MinX, MaxY, MaxZ)
Points are represented as (Y,X)

'''

from indicies import *

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
    def BoundingBox(self):
        return self._bounds

    def __init__(self, bounds):
        '''
        Constructor, bounds = [left bottom right top]
        '''

        self._bounds = bounds

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
    def contains(cls, rect, primitive):
        '''If len == 2 primitive is a point,
           if len == 4 primitive is a rect [left bottom right top]'''
        if isinstance(primitive, Rectangle):
            primitive = primitive.BoundingBox

        IsPoint = len(primitive) == 2
        IsRect = len(primitive) == 4

        if(IsPoint):
            point = primitive
            if point[iPoint.X] < rect[iRect.MinX] or \
                point[iPoint.X] > rect[iRect.MaxX] or \
                point[iPoint.Y] < rect[iRect.MinY] or \
                point[iPoint.Y] > rect[iRect.MaxY]:
                    return False

            return True
        if(IsRect):
            if(primitive[iRect.MaxX] < rect.BoundingBox[iRect.MinX] or
                   primitive[iRect.MinX] > rect.BoundingBox[iRect.MaxX] or
                   primitive[iRect.MaxY] < rect.BoundingBox[iRect.MinY] or
                   primitive[iRect.MinY] > rect.BoundingBox[iRect.MaxY]):
                    return False
            return True

    def __str__(self):
        return "MinX: %g MinY: %g MaxX: %g MaxY: %g" % (self._bounds[iRect.MinX], self._bounds[iRect.MinY], self._bounds[iRect.MaxX], self._bounds[iRect.MaxY])
