'''
Created on Apr 26, 2019

@author: u0490822
'''

import numpy
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.spatial import BoundingBox, Rectangle, iPoint, iPoint3


def ArcAngle(origin: NDArray[numpy.floating], A: NDArray[numpy.floating], B: NDArray[numpy.floating]) -> NDArray[
    numpy.floating]:
    '''
    :return: The angle, in radians, between A to B as observed from the origin 
    '''

    A = nornir_imageregistration.EnsurePointsAre2DNumpyArray(A)
    B = nornir_imageregistration.EnsurePointsAre2DNumpyArray(B)
    origin = nornir_imageregistration.EnsurePointsAre2DNumpyArray(origin)

    translated_A = A - origin  # Do not use inplace operation, we do not want to modify input arrays
    translated_B = B - origin
    AnglesA = numpy.arctan2(translated_A[:, 0], translated_A[:, 1])
    AnglesB = numpy.arctan2(translated_B[:, 0], translated_B[:, 1])
    angle = AnglesB - AnglesA

    lessthanpi = angle < -numpy.pi
    angle[lessthanpi] = angle[lessthanpi] + (numpy.pi * 2)

    greaterthanpi = angle > numpy.pi
    angle[greaterthanpi] = angle[greaterthanpi] - (numpy.pi * 2)

    return angle


def BoundsArrayFromPoints(points):
    '''
    :param ndarray points: (Z?,Y,X) 3xN or 2xN array of points
    :return: (minZ, minY, minX, maxZ, maxY, maxX) or (minY, minX, maxY, maxX)'''

    min_point = numpy.min(points, 0)
    max_point = numpy.max(points, 0)

    if points.shape[1] == 2:
        return numpy.array((min_point[iPoint.Y], min_point[iPoint.X], max_point[iPoint.Y], max_point[iPoint.X]))
    elif points.shape[1] == 3:
        return numpy.array((min_point[iPoint3.Z], min_point[iPoint3.Y], min_point[iPoint3.X], max_point[iPoint3.Z],
                            max_point[iPoint3.Y], max_point[iPoint3.X]))
    else:
        raise Exception("PointBoundingBox: Unexpected number of dimensions in point array" + str(points.shape))


def BoundingPrimitiveFromPoints(
        points: NDArray) -> Rectangle | BoundingBox:
    '''Return either a rectangle or bounding box for a set of points'''

    if not isinstance(points, numpy.ndarray):
        points = points.get()

    bounds = BoundsArrayFromPoints(points)
    if bounds.shape[0] == 4:
        return Rectangle.CreateFromBounds(bounds)
    if bounds.shape[0] == 7:
        return BoundingBox.CreateFromBounds(bounds)

    raise ValueError("Expected either 4 or 6 bounding values")


def BoundingRectangleFromPoints(points: NDArray) -> Rectangle:
    '''Return either a rectangle box for a set of points.  If the set is 3D, return the XY bounds'''

    if not isinstance(points, numpy.ndarray):
        points = points.get()

    bounds = BoundsArrayFromPoints(points)
    if bounds.shape[0] == 4:
        return Rectangle.CreateFromBounds(bounds)
    if bounds.shape[0] == 7:
        return BoundingBox.CreateFromBounds(numpy.hstack((bounds[1:3], bounds[4:6])))

    raise ValueError("Expected either 4 or 6 bounding values")


def BoundingBoxFromPoints(
        points: NDArray) -> BoundingBox:
    '''Return either a rectangle or bounding box for a set of points'''

    if not isinstance(points, numpy.ndarray):
        points = points.get()

    bounds = BoundsArrayFromPoints(points)
    if bounds.shape[0] != 7:
        raise ValueError("Expected 7 bounding values")

    return BoundingBox.CreateFromBounds(bounds)
