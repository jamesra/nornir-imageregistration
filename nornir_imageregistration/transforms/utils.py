'''
Created on Apr 4, 2013

@author: u0490822
'''

import numpy as np



def InvalidIndicies(points):
    '''Removes rows with a NAN value and returns a list of indicies'''

    numPoints = points.shape[0]

    nanArray = np.isnan(points)

    nan1D = nanArray.any(axis = 1)

    invalidIndicies = np.nonzero(nan1D)[0]
    points = np.delete(points, invalidIndicies, axis = 0)

    assert(points.shape[0] + invalidIndicies.shape[0] == numPoints)

    return (points, invalidIndicies);

def RotationMatrix(rangle):
    return np.matrix([[np.cos(rangle), -np.sin(rangle), 0], [np.sin(rangle), np.cos(rangle), 0], [0, 0, 1]])

if __name__ == '__main__':
    pass


def FixedBoundingBox(transforms):
    '''Calculate the bounding box of the warped position for a set of transforms'''

    mbb = None
    for t in transforms:
        if mbb is None:
            mbb = t.ControlPointBoundingBox
        else:
            mbb = np.vstack((mbb, t.ControlPointBoundingBox))

    minX = np.min(mbb[:, 0])
    minY = np.min(mbb[:, 1])
    maxX = np.max(mbb[:, 2])
    maxY = np.max(mbb[:, 3])

    return (minX, minY, maxX, maxY)

def MappedBoundingBox(transforms):
    '''Calculate the bounding box of the warped position for a set of transforms'''

    mbb = None
    for t in transforms:
        if mbb is None:
            mbb = t.MappedPointBoundingBox
        else:
            mbb = np.vstack((mbb, t.MappedPointBoundingBox))

    minX = np.min(mbb[:, 0])
    minY = np.min(mbb[:, 1])
    maxX = np.max(mbb[:, 2])
    maxY = np.max(mbb[:, 3])

    return (minX, minY, maxX, maxY)