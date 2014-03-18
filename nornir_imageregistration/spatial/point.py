'''
Created on Mar 12, 2014

@author: u0490822
'''

import numpy

def PointBoundingBox(points):
    '''
    :param ndarray points: (Y,X) Nx2 array of points
    :return: (minY, minX, maxY, maxX)'''


    (minY, minX) = numpy.min(points, 0)
    (maxY, maxX) = numpy.max(points, 0)

    return (minY, minX, maxY, maxX)

if __name__ == '__main__':
    pass