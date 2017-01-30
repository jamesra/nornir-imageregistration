import numpy

from .indicies import *


def PointBoundingBox(points):
    '''
    :param ndarray points: (Y,X) NxM array of points
    :return: (minY, minX, maxY, maxX)'''

    raise DeprecationWarning("Deprecated.  Use BoundingBox.BoundsArrayFromPoints instead")

    min_point = numpy.min(points, 0)
    max_point = numpy.max(points, 0)

    if(points.shape[1] == 2):
        return numpy.array((min_point[iPoint.Y], min_point[iPoint.X], max_point[iPoint.Y], max_point[iPoint.X]))
    elif(points.shape[1] == 3):
        return  numpy.array((min_point[iPoint3.Z], min_point[iPoint3.Y], min_point[iPoint3.X], max_point[iPoint3.Z], max_point[iPoint3.Y], max_point[iPoint3.X]))
    else:
        raise Exception("PointBoundingBox: Unexpected number of dimensions in point array" + str(points.shape))

if __name__ == '__main__':
    pass
