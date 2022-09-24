'''
Created on Apr 4, 2013

@author: u0490822
'''

import nornir_imageregistration
import numpy as np
from numpy.typing import NDArray
from nornir_shared import prettyoutput
from collections.abc import Iterable

 
def InvalidIndicies(points):
    '''Removes rows with a NAN value and returns a list of indicies'''

    if points is None:
        raise ValueError("points must not be None")
    
    numPoints = points.shape[0]

    nan1D = np.isnan(points).any(axis=1)

    invalidIndicies = np.nonzero(nan1D)[0]
    points = np.delete(points, invalidIndicies, axis=0)

    assert(points.shape[0] + invalidIndicies.shape[0] == numPoints)

    return (points, invalidIndicies)


def RotationMatrix(rangle):
    '''
    :param float rangle: Angle in radians
    '''
    if rangle is None:
        raise ValueError("Angle must not be none")
    return np.matrix([[np.cos(rangle), -np.sin(rangle), 0], [np.sin(rangle), np.cos(rangle), 0], [0, 0, 1]])


def IdentityMatrix():
    '''
    '''
    return np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def TranslateMatrixXY(offset: tuple[float, float] | NDArray):
    '''
    :param offset: An offset to translate by, either tuple of (Y,X) or an array
    '''

    if offset is None:
        raise ValueError("Angle must not be none") 
    elif hasattr(offset, "__iter__"):
        return np.matrix([[1, 0, 0], [0, 1, 0], [offset[0], offset[1], 1]])
    
    raise NotImplementedError("Unexpected argument")


def ScaleMatrixXY(scale: float):
    '''
    :param float scale: scale in radians, either a single value for all dimensions or a tuple of (Y,X) scale values
    '''
    if scale is None:
        raise ValueError("Angle must not be none")
    elif isinstance(scale, float):
        return np.matrix([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    elif hasattr(scale, "__iter__"):
        return np.matrix([[scale[1], 0, 0], [0, scale[0], 0], [0, 0, 1]])
    
    raise NotImplementedError("Unexpected argument")


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


def FixedOriginOffset(transforms):
    '''
    This is a fairly specific function to move a mosaic to have an origin at 0,0
    It handles both discrete and continuous functions the best it can.
    :return: tuple containing smallest origin offset
    '''
    
    if len(transforms) == 1:
        # Copy the data instead of passing the transforms object
        return nornir_imageregistration.Rectangle(transforms[0].FixedBoundingBox.ToTuple())

    mins = np.zeros((len(transforms), 2)) 
    for (i, t) in enumerate(transforms):
        if isinstance(t, nornir_imageregistration.IDiscreteTransform):
            mins[i,:] = t.FixedBoundingBox.BottomLeft
        elif isinstance(t, nornir_imageregistration.transforms.RigidNoRotation):
            mins[i,:] = t.target_offset
        elif hasattr(t, 'FixedBoundingBox'):
            mins[i,:] = t.FixedBoundingBox.BottomLeft
        else:
            raise ValueError(f"Unexpected transform type {t} at index {i}")

    return np.min(mins, 0)

  
def FixedBoundingBox(transforms, images=None):
    '''Calculate the bounding box of the warped position for a set of transforms
    :param list transforms: A list of transforms
    :param list images: A list of image parameters (strings, ndarrays, or 1x2 
                        ndarray.shape arrays, of the size of the image.  Only
                        required for continuous transforms so a bounding box can
                        be calculated
    :return: A rectangle describing the bounding box
    '''
    
    if len(transforms) == 1:
        # Copy the data instead of passing the transforms object
        return nornir_imageregistration.Rectangle(transforms[0].FixedBoundingBox.ToTuple())
    
    is_images_param_single_size = False
    if images is not None:
        if isinstance(images, np.ndarray):
            if(images.flat.shape != 2):
                raise ValueError("Must use a 1x2 array to specify a universal image size")
            is_images_param_single_size = True
        elif isinstance(images, Iterable):
            if len(images) != len(transforms):
                raise ValueError(f"images list not of equal length as transforms list. Transforms: {len(transforms)} Images: {len(images)}")
        elif not isinstance(images, Iterable): 
            raise ValueError("If not none or a single 1x2 array the images parameter must be an iterable of equal length as transform array.")
    
    mbb = np.zeros((len(transforms), 4))
    for (i, t) in enumerate(transforms):
        if isinstance(t, nornir_imageregistration.IDiscreteTransform):
            mbb[i,:] = t.FixedBoundingBox.ToArray()
        elif isinstance(t, nornir_imageregistration.transforms.RigidNoRotation):
            # Figure out if images is an iterable or just a single size for all tiles
            size = None
            if is_images_param_single_size:
                size = images
            else:
                size = nornir_imageregistration.GetImageSize(images[i])
                
            mbb[i,:2] = t.target_offset
            mbb[i, 2:] = t.target_offset + size
        elif hasattr(t, 'FixedBoundingBox'):
            mbb[i,:] = t.FixedBoundingBox.ToArray()
        else:
            raise ValueError(f"Unexpected type passed to FixedBoundingBox {t.__class__}")

    minX = np.min(mbb[:, 1])
    minY = np.min(mbb[:, 0])
    maxX = np.max(mbb[:, 3])
    maxY = np.max(mbb[:, 2])

    return  nornir_imageregistration.Rectangle((float(minY), float(minX), float(maxY), float(maxX)))


def MappedBoundingBox(transforms):
    '''Calculate the bounding box of the original source space positions for a set of transforms'''
    
    if len(transforms) == 1:
        # Copy the data instead of passing the transforms object
        return nornir_imageregistration.Rectangle(transforms[0].MappedBoundingBox.ToTuple())

    discrete_found = False
    mbb = np.zeros((len(transforms), 4))
    for (i, t) in enumerate(transforms):
        if not isinstance(t, nornir_imageregistration.IDiscreteTransform):
            continue
        
        mbb[i,:] = t.MappedBoundingBox.ToArray()
        discrete_found = True  

    if discrete_found is False:
        raise ValueError("No discrete transforms found in transforms list")

    minX = np.min(mbb[:, 1])
    minY = np.min(mbb[:, 0])
    maxX = np.max(mbb[:, 3])
    maxY = np.max(mbb[:, 2])

    return  nornir_imageregistration.Rectangle((float(minY), float(minX), float(maxY), float(maxX)))

 
def IsOriginAtZero(transforms):
    ''':return: True if transform bounding box has origin at 0,0 otherise false'''
    try:
        origin = FixedOriginOffset(transforms)
        (minY, minX) = (origin[0], origin[1])
        return minY == 0 and minX == 0
    except ValueError:
        prettyoutput.LogErr("Could not determine origin of transforms, continuing")
        return True


def TranslateToZeroOrigin(transforms):
    '''
    Translate the fixed space off all passed transforms such that that no point maps to a negative number.  Useful for image coordinates.
    :return: The offset the mosaic was translated by
    '''
    
    origin = None
    try:
        origin = FixedOriginOffset(transforms)
    except ValueError:
        prettyoutput.LogErr("Could not determine origin of transforms, continuing")
        return np.zeros((2,))
    
    if origin is None:
        return
    
    if np.array_equal(origin, np.zeros(2,)):
        return
    
    for t in transforms:
        t.TranslateFixed(-origin)
        
    # translated_bbox = nornir_imageregistration.Rectangle.translate(bbox, -bbox.BottomLeft)
    # assert(np.array_equal(translated_bbox.BottomLeft, np.asarray((0,0)))) 
    # return translated_bbox
    
    return -origin


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
