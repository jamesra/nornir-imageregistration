'''
Created on Apr 26, 2019

@author: u0490822
'''

import numpy
import nornir_imageregistration
from nornir_imageregistration.spatial.indicies import iPoint, iPoint3  

def ArcAngle(origin, A, B):
    '''
    :return: The angle, in radians, between A to B as observed from the origin 
    '''

    A = nornir_imageregistration.EnsurePointsAre2DNumpyArray(A)
    B = nornir_imageregistration.EnsurePointsAre2DNumpyArray(B)
    origin = nornir_imageregistration.EnsurePointsAre2DNumpyArray(origin)

    A = A - origin
    B = B - origin
    AnglesA = numpy.arctan2(A[:, 0], A[:, 1])
    AnglesB = numpy.arctan2(B[:, 0], B[:, 1])
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

    if(points.shape[1] == 2):
        return numpy.array((min_point[iPoint.Y], min_point[iPoint.X], max_point[iPoint.Y], max_point[iPoint.X]))
    elif(points.shape[1] == 3):
        return  numpy.array((min_point[iPoint3.Z], min_point[iPoint3.Y], min_point[iPoint3.X], max_point[iPoint3.Z], max_point[iPoint3.Y], max_point[iPoint3.X]))
    else:
        raise Exception("PointBoundingBox: Unexpected number of dimensions in point array" + str(points.shape))

def BoundingPrimitiveFromPoints(points):
    '''Return either a rectangle or bounding box for a set of points'''
    
    bounds = BoundsArrayFromPoints(points)
    if bounds.shape[0] == 4:
        return nornir_imageregistration.Rectangle.CreateFromBounds(bounds)
    if bounds.shape[0] == 7:
        return nornir_imageregistration.BoundingBox.CreateFromBounds(bounds)

    raise ValueError("Expected either 4 or 6 bounding values")