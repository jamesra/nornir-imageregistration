'''
Created on Apr 4, 2013

@author: u0490822
'''

import numpy as np


def InvalidIndicies(points):
    '''Removes rows with a NAN value and returns a list of indicies'''

    numPoints = points.shape[0]

    nan1D = np.isnan(points).any(axis=1)

    invalidIndicies = np.nonzero(nan1D)[0]
    points = np.delete(points, invalidIndicies, axis=0)

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
            mbb = np.array([t.FixedBoundingBox])
        else:
            mbb = np.vstack((mbb, t.FixedBoundingBox))

    minX = np.min(mbb[:, 1])
    minY = np.min(mbb[:, 0])
    maxX = np.max(mbb[:, 3])
    maxY = np.max(mbb[:, 2])

    return  (minY, minX, maxY, maxX)

def MappedBoundingBox(transforms):
    '''Calculate the bounding box of the warped position for a set of transforms'''

    mbb = None
    for t in transforms:
        if mbb is None:
            mbb = np.array([t.MappedBoundingBox])
        else:
            mbb = np.vstack((mbb, t.MappedBoundingBox))

    minX = np.min(mbb[:, 1])
    minY = np.min(mbb[:, 0])
    maxX = np.max(mbb[:, 3])
    maxY = np.max(mbb[:, 2])

    return  (minY, minX, maxY, maxX)

def TranslateToZeroOrigin(transforms):
    '''Translate the fixed space off all passed transforms such that that no point maps to a negative number.  Useful for image coordinates'''

    (minY, minX, maxY, maxX) = FixedBoundingBox(transforms)

    for t in transforms:
        t.TranslateFixed((-minY, -minX))


def FixedBoundingBoxWidth(transforms):
    (minY, minX, maxY, maxX) = FixedBoundingBox(transforms)
    return np.ceil(maxX) - np.floor(minX)


def FixedBoundingBoxHeight(transforms):
    (minY, minX, maxY, maxX) = FixedBoundingBox(transforms)
    return np.ceil(maxY) - np.floor(minY)


def MappedBoundingBoxWidth(transforms):
    (minY, minX, maxY, maxX) = MappedBoundingBox(transforms)
    return np.ceil(maxX) - np.floor(minX)


def MappedBoundingBoxHeight(transforms):
    (minY, minX, maxY, maxX) = MappedBoundingBox(transforms)
    return np.ceil(maxY) - np.floor(minY)