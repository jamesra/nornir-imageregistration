
# from . import index
from .indicies import *
from .point import *
from .rectangle import Rectangle, RectangleSet, RaiseValueErrorOnInvalidBounds, IsValidBoundingBox
from .boundingbox import BoundingBox 
import numpy as np

def BoundsArrayFromPoints(points):
    '''
    :param ndarray points: (Z?,Y,X) 3xN or 2xN array of points
    :return: (minZ, minY, minX, maxZ, maxY, maxX) or (minY, minX, maxY, maxX)'''

    min_point = np.min(points, 0)
    max_point = np.max(points, 0)

    if(points.shape[1] == 2):
        return np.array((min_point[iPoint.Y], min_point[iPoint.X], max_point[iPoint.Y], max_point[iPoint.X]))
    elif(points.shape[1] == 3):
        return  np.array((min_point[iPoint3.Z], min_point[iPoint3.Y], min_point[iPoint3.X], max_point[iPoint3.Z], max_point[iPoint3.Y], max_point[iPoint3.X]))
    else:
        raise Exception("PointBoundingBox: Unexpected number of dimensions in point array" + str(points.shape))

def BoundingPrimitiveFromPoints(points):
    '''Return either a rectangle or bounding box for a set of points'''
    
    bounds = BoundsArrayFromPoints(points)
    if bounds.shape[0] == 4:
        return Rectangle.CreateFromBounds(bounds)
    if bounds.shape[0] == 7:
        return BoundingBox.CreateFromBounds(bounds)

    raise ValueError("Expected either 4 or 6 bounding values")