'''
Created on Apr 4, 2013

@author: u0490822
'''

import nornir_imageregistration
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
    '''
    :param float rangle: Angle in radians
    '''
    if rangle is None:
        raise ValueError("Angle must not be none")
    return np.matrix([[np.cos(rangle), -np.sin(rangle), 0], [np.sin(rangle), np.cos(rangle), 0], [0, 0, 1]])


if __name__ == '__main__':
    pass


def PointBoundingRect(points):
    raise DeprecationWarning("Use spatial.BoundsArrayFromPoints")

    (minY, minX) = np.min(points, 0)
    (maxY, maxX) = np.max(points, 0)
    return (minY, minX, maxY, maxX)


def PointBoundingBox(points):
    raise DeprecationWarning("Use spatial.BoundsArrayFromPoints")

    (minZ, minY, minX) = np.min(points, 0)
    (maxZ, maxY, maxX) = np.max(points, 0)
    return (minZ, minY, minX, maxZ, maxY, maxX)


def FixedBoundingBox(transforms):
    '''Calculate the bounding box of the warped position for a set of transforms'''
    
    if len(transforms) == 1:
        # Copy the data instead of passing the transforms object
        return nornir_imageregistration.Rectangle(transforms[0].FixedBoundingBox.ToTuple())

    mbb = None
    for t in transforms:
        if mbb is None:
            mbb = t.FixedBoundingBox.ToArray()
        else:
            mbb = np.vstack((mbb, t.FixedBoundingBox.ToArray()))

    minX = np.min(mbb[:, 1])
    minY = np.min(mbb[:, 0])
    maxX = np.max(mbb[:, 3])
    maxY = np.max(mbb[:, 2])

    return  nornir_imageregistration.Rectangle((float(minY), float(minX), float(maxY), float(maxX)))


def MappedBoundingBox(transforms):
    '''Calculate the bounding box of the warped position for a set of transforms'''
    
    if len(transforms) == 1:
        # Copy the data instead of passing the transforms object
        return nornir_imageregistration.Rectangle(transforms[0].MappedBoundingBox.ToTuple())

    mbb = None
    for t in transforms:
        if mbb is None:
            mbb = t.MappedBoundingBox.ToArray()
        else:
            mbb = np.vstack((mbb, t.MappedBoundingBox.ToArray()))

    minX = np.min(mbb[:, 1])
    minY = np.min(mbb[:, 0])
    maxX = np.max(mbb[:, 3])
    maxY = np.max(mbb[:, 2])

    return  nornir_imageregistration.Rectangle((float(minY), float(minX), float(maxY), float(maxX)))

 
def IsOriginAtZero(transforms):
    ''':return: True if transform bounding box has origin at 0,0 otherise false'''
    (minY, minX, maxY, maxX) = FixedBoundingBox(transforms).ToTuple()
    return minY == 0 and minX == 0


def TranslateToZeroOrigin(transforms):
    '''
    Translate the fixed space off all passed transforms such that that no point maps to a negative number.  Useful for image coordinates.
    :return: A Rectangle object describing the new fixed space bounding box
    :rtype: Rectangle
    '''
    bbox = FixedBoundingBox(transforms)

    for t in transforms:
        t.TranslateFixed(-bbox.BottomLeft)
        
    translated_bbox = nornir_imageregistration.Rectangle.translate(bbox, -bbox.BottomLeft)
    assert(np.array_equal(translated_bbox.BottomLeft, np.asarray((0,0)))) 

    return translated_bbox


def FixedBoundingBoxWidth(transforms):
    (minY, minX, maxY, maxX) = FixedBoundingBox(transforms).ToTuple()
    return np.ceil(maxX) - np.floor(minX)


def FixedBoundingBoxHeight(transforms):
    (minY, minX, maxY, maxX) = FixedBoundingBox(transforms).ToTuple()
    return np.ceil(maxY) - np.floor(minY)


def MappedBoundingBoxWidth(transforms):
    (minY, minX, maxY, maxX) = MappedBoundingBox(transforms).ToTuple()
    return np.ceil(maxX) - np.floor(minX)


def MappedBoundingBoxHeight(transforms):
    (minY, minX, maxY, maxX) = MappedBoundingBox(transforms).ToTuple()
    return np.ceil(maxY) - np.floor(minY)
