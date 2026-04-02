'''
Created on Apr 26, 2019

@author: u0490822
'''

import numpy
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.spatial.rectangle import Rectangle
from nornir_imageregistration.spatial.indices import iPoint, iPoint3


def ArcAngle(origin: NDArray[numpy.floating], A: NDArray[numpy.floating], B: NDArray[numpy.floating]) -> NDArray[
    numpy.floating]:
    """Return the signed angle, in radians, from A to B as observed from the origin.

    :param origin: Reference point (2D or broadcastable to Nx2).
    :param A: Start points (Nx2).
    :param B: End points (Nx2).
    :return: Angle in radians, in [-pi, pi].
    """

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
        points: NDArray) -> Rectangle | "BoundingBox":
    """Return a Rectangle (2D) or BoundingBox (3D) enclosing the given points.

    :param points: Nx2 (XY) or Nx3 (ZYX) array of points.
    :return: Rectangle for 2D points, BoundingBox for 3D points.
    :raises ValueError: If bounds length is not 4 or 6.
    """
    from nornir_imageregistration.spatial.boundingbox import BoundingBox

    if not isinstance(points, numpy.ndarray):
        points = points.get()

    bounds = BoundsArrayFromPoints(points)
    if bounds.shape[0] == 4:
        return Rectangle.CreateFromBounds(bounds)
    if bounds.shape[0] == 6:
        return BoundingBox.CreateFromBounds(bounds)

    raise ValueError("Expected either 4 or 6 bounding values")


def BoundingRectangleFromPoints(points: NDArray) -> Rectangle:
    """Return the axis-aligned rectangle enclosing the points; for 3D points, use XY bounds only.

    :param points: Nx2 (XY) or Nx3 (ZYX) array of points.
    :return: Rectangle (2D bounds); for 3D input, (MinY, MinX, MaxY, MaxX).
    :raises ValueError: If bounds length is not 4 or 6.
    """

    if not isinstance(points, numpy.ndarray):
        points = points.get()

    bounds = BoundsArrayFromPoints(points)
    if bounds.shape[0] == 4:
        return Rectangle.CreateFromBounds(bounds)
    if bounds.shape[0] == 6:
        return Rectangle.CreateFromBounds(numpy.hstack((bounds[1:3], bounds[4:6])))

    raise ValueError("Expected either 4 or 6 bounding values")


def BoundingBoxFromPoints(
        points: NDArray) -> "BoundingBox":
    """Return the 3D axis-aligned bounding box enclosing the points.

    :param points: Nx3 array of points (Z, Y, X or similar 3D).
    :return: BoundingBox with 6 values (MinZ, MinY, MinX, MaxZ, MaxY, MaxX).
    :raises ValueError: If points are not 3D (expected 6 bounding values).
    """
    from nornir_imageregistration.spatial.boundingbox import BoundingBox

    if not isinstance(points, numpy.ndarray):
        points = points.get()

    bounds = BoundsArrayFromPoints(points)
    if bounds.shape[0] != 6:
        raise ValueError("Expected 6 bounding values")

    return BoundingBox.CreateFromBounds(bounds)
