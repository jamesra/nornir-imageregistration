'''
Created on Aug 31, 2018

@author: u0490822
'''

import numpy
from . import triangulation, utils
from numpy import arctan2

# /// <summary>
# /// Angle of arc between A & B with Origin
# /// </summary>
# /// <param name="Origin"></param>
# /// <param name="A"></param>
# /// <param name="B"></param>
# /// <returns></returns>
# static public double ArcAngle(GridVector2 Origin, GridVector2 A, GridVector2 B)
# {
#     A = A - Origin;
#     B = B - Origin;
#     double AngleA = Math.Atan2(A.Y, A.X);
#     double AngleB = Math.Atan2(B.Y, B.X);
#     return AngleB - AngleA; 
# }


def ArcAngle(origin, A, B):
    '''
    :return: The angle, in radians, between A to B as observed from the origin 
    '''

    A = utils.EnsurePointsAre2DNumpyArray(A)
    B = utils.EnsurePointsAre2DNumpyArray(B)
    origin = utils.EnsurePointsAre2DNumpyArray(origin)

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


def TriangleAngleDelta(transform):
    '''
    For each triangle in a triangulation transform measure the angles of the control triangle 
    and subtract the equivalent angle found in the mapped triangle.  Report the differences as a structured 
    arrray.

    :param triangulation transform: The transform to be analyzed
    :return: A nx3 array of the angle for each triangle
    '''

    fixed_tri = transform.FixedTriangles

    FixedTriAngles = TriangleAngles(fixed_tri, transform.TargetPoints)
    WarpedTriAngles = TriangleAngles(fixed_tri, transform.SourcePoints)

    return numpy.abs(FixedTriAngles - WarpedTriAngles)


def TriangleAngles(triangle_indicies, points):
    '''
    For each triangle in a triangulation transform measure the angles of the control triangle 
    and compare to the angles of the mapped triangle.  Report the differences as a structured 
    array.

    :param numpy.ndarray triangle_indicies: Triangle Indicies, an Nx3 array
    :param numpy.ndarray points: Vertex positions, an Nx2 array
    :return: A nx3 array of the angle for each triangle
    '''

    triangles = points[triangle_indicies]

    A = triangles[:, 0]
    B = triangles[:, 1]
    C = triangles[:, 2]

    OriginAAngles = numpy.abs(ArcAngle(A, B, C))
    OriginBAngles = numpy.abs(ArcAngle(B, C, A))
    #anglesC = numpy.abs(ArcAngle(C, A, B))
    calcAnglesC = numpy.pi - (OriginAAngles + OriginBAngles)

    #numpy.array_equal(anglesC, calcAnglesC)

    return numpy.swapaxes(numpy.vstack((OriginAAngles, OriginBAngles, calcAnglesC)),0,1)

def TriangleVertexAngleDelta(transform):
    '''
    For each vertex in a triangulation transform measure the angle differences of the control triangles 
    and subtract the equivalent angle found in the mapped triangles that the vertex is a part of.
    Report the differences as a structured array.

    :param triangulation transform: The transform to be analyzed
    :return: A nx3 array of the angle for each triangle
    '''

    fixed_tri = transform.FixedTriangles

    FixedTriAngles = TriangleAngles(fixed_tri, transform.TargetPoints)
    WarpedTriAngles = TriangleAngles(fixed_tri, transform.SourcePoints)

    delta = numpy.abs(FixedTriAngles - WarpedTriAngles)

    VertexValues = []
    for iVertex in range(0, transform.NumControlPoints):
        index = fixed_tri == iVertex
        values = delta[index]
        VertexValues.append(values)

    return VertexValues


if __name__ == '__main__':
    
    pass
