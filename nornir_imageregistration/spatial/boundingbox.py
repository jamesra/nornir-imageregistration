'''

Rectangles are represented as (MinY, MinX, MaxY, MaxZ)
Points are represented as (Y,X)

'''

import numpy as np

from .indicies import *


class BoundingBox(object):
    '''
    
    '''

    @property
    def Width(self):
        return self._bounds[iBox.MaxX] - self._bounds[iBox.MinX]

    @property
    def Height(self):
        return self._bounds[iBox.MaxY] - self._bounds[iBox.MinY]

    @property
    def Depth(self):
        return self._bounds[iBox.MaxY] - self._bounds[iBox.MinY]

    @property
    def BottomLeftFront(self):
        return np.array([self._bounds[iBox.MinZ], self._bounds[iBox.MinY], self._bounds[iBox.MinX]])
    @property
    def TopRightBack(self):
        return np.array([self._bounds[iBox.MaxZ], self._bounds[iBox.MaxY], self._bounds[iBox.MaxX]])

    @property
    def BoundingBox(self):
        return self._bounds

    def __init__(self, bounds):
        '''
        Constructor, bounds = [left bottom right top]
        '''

        self._bounds = bounds

    @classmethod
    def CreateFromPointAndVolume(cls, point, vol):
        '''
        :param tuple point: (Z,Y,X)
        :param tuple volume: (Depth, Height, Area)
        :rtype: BoundingBox
        '''
        return BoundingBox(bounds=(point[iPoint3.Z], point[iPoint3.Y], point[iPoint3.X], point[iPoint3.Z] + vol[iVolume.Depth], point[iPoint3.Y] + vol[iVolume.Height], point[iPoint.X] + vol[iVolume.Width]))

    @classmethod
    def CreateFromBounds(cls, Bounds):
        '''
        :param tuple Bounds: (MinZ,MinY,MinX,MaxZ,MaxY,MaxX)
        '''
        return BoundingBox(Bounds)

    @classmethod
    def PrimitiveToBox(cls, primitive):
        '''Privitive can be a list of (Z,Y,X) or (MinZ, MinY, MinX, MaxZ, MaxY, MaxX) or a BoundingBox'''

        if isinstance(primitive, BoundingBox):
            return primitive

        if len(primitive) == 3:
            return BoundingBox(primitive[0], primitive[1], primitive[0], primitive[1])
        elif len(primitive) == 6:
            return BoundingBox(primitive)
        else:
            raise ValueError("Unknown primitve type %s" % str(primitive))

    @classmethod
    def contains(cls, A, B):
        '''If len == 2 primitive is a point,
           if len == 4 primitive is a rect [left bottom right top]'''

        A = BoundingBox.PrimitiveToBox(A)
        B = BoundingBox.PrimitiveToBox(B)

        if(A.BoundingBox[iBox.MaxZ] < B.BoundingBox[iBox.MinZ] or
           A.BoundingBox[iBox.MinZ] > B.BoundingBox[iBox.MaxZ] or
           A.BoundingBox[iBox.MaxX] < B.BoundingBox[iBox.MinX] or
           A.BoundingBox[iBox.MinX] > B.BoundingBox[iBox.MaxX] or
           A.BoundingBox[iBox.MaxY] < B.BoundingBox[iBox.MinY] or
           A.BoundingBox[iBox.MinY] > B.BoundingBox[iBox.MaxY]):

            return False

        return True

    def __str__(self):
        return "MinX: %g MinY: %g MinZ: %g MaxX: %g MaxY: %g MaxZ %g" % (self._bounds[iBox.MinX], self._bounds[iBox.MinY], self._bounds[iBox.MinZ], self._bounds[iBox.MaxX], self._bounds[iBox.MaxY], self._bounds[iBox.MaxZ])
