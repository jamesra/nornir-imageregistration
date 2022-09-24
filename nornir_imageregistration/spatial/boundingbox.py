'''

Rectangles are represented as (MinY, MinX, MaxY, MaxZ)
Points are represented as (Y,X)

'''

from typing import *

import numpy as np
from numpy.typing import * 
 
import nornir_imageregistration.spatial
from nornir_imageregistration.spatial import *
from nornir_imageregistration.spatial.indicies import *


class BoundingBox(object):
    '''
    Defines a 3D box
    '''

    @property
    def Width(self) -> float:
        return self._bounds[iBox.MaxX] - self._bounds[iBox.MinX]

    @property
    def Height(self) -> float:
        return self._bounds[iBox.MaxY] - self._bounds[iBox.MinY]

    @property
    def Depth(self) -> float:
        return self._bounds[iBox.MaxZ] - self._bounds[iBox.MinZ]

    @property
    def BottomLeftFront(self) -> NDArray:
        return np.array([self._bounds[iBox.MinZ], self._bounds[iBox.MinY], self._bounds[iBox.MinX]])
    @property
    def TopRightBack(self) -> NDArray:
        return np.array([self._bounds[iBox.MaxZ], self._bounds[iBox.MaxY], self._bounds[iBox.MaxX]])

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

    def ToArray(self) -> NDArray:
        return np.array(self._bounds)
    
    def ToTuple(self) -> (float, float, float, float, float, float):
        return (self._bounds[iBox.MinZ],
                self._bounds[iBox.MinY],
                self._bounds[iBox.MinX],
                self._bounds[iBox.MaxZ],
                self._bounds[iBox.MaxY],
                self._bounds[iBox.MaxX])

    @property
    def RectangleXY(self) -> Rectangle:
        '''Returns a rectangle based on the XY plane of the box'''
        return Rectangle.CreateFromBounds((self._bounds[iBox.MinY],
                                          self._bounds[iBox.MinX],
                                          self._bounds[iBox.MaxY],
                                          self._bounds[iBox.MaxX]))

    def __init__(self, bounds):
        '''
        Constructor, bounds = [left bottom right top]
        '''

        self._bounds = bounds


    @classmethod
    def CreateFromPoints(cls, points) -> BoundingBox:
        boundingArray = BoundsArrayFromPoints(points)
        return BoundingBox(bounds=boundingArray)

    @classmethod
    def CreateFromPointAndVolume(cls, point, vol) -> BoundingBox:
        '''
        :param vol: (Depth, Height, Area)
        :param tuple point: (Z,Y,X)
        '''

        return BoundingBox(bounds=(point[iPoint3.Z], point[iPoint3.Y], point[iPoint3.X], point[iPoint3.Z] + vol[iVolume.Depth], point[iPoint3.Y] + vol[iVolume.Height], point[iPoint.X] + vol[iVolume.Width]))

    @classmethod
    def CreateFromBounds(cls, Bounds) -> BoundingBox:
        '''
        :param tuple Bounds: (MinZ,MinY,MinX,MaxZ,MaxY,MaxX)
        '''
        return BoundingBox(Bounds)

    @classmethod
    def PrimitiveToBox(cls, primitive) -> BoundingBox:
        '''Privitive can be a list of (Y,X) or (Z,Y,X) or (MinZ, MinY, MinX, MaxZ, MaxY, MaxX) or a BoundingBox'''

        if isinstance(primitive, BoundingBox):
            return primitive

        if len(primitive) == 2:
            return BoundingBox((0, primitive[0], primitive[1], 0, primitive[0], primitive[1]))
        elif len(primitive) == 3:
            return BoundingBox((primitive[0], primitive[1], primitive[2], primitive[0], primitive[1], primitive[2]))
        elif len(primitive) == 6:
            return BoundingBox(primitive)
        else:
            raise ValueError("Unknown primitve type %s" % str(primitive))

    @classmethod
    def contains(cls, A: BoundingBox, B: BoundingBox) -> bool:
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
