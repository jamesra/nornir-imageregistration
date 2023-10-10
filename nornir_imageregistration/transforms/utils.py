'''
Created on Apr 4, 2013

@author: u0490822
'''

from collections.abc import Iterable

import numpy as np
import cupy as cp
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.transforms.base import ITransform, IControlPoints
from nornir_shared import prettyoutput


def InvalidIndicies(points: NDArray[float]) -> tuple[NDArray[float], NDArray[int]]:
    '''Removes rows with a NAN value.
     :return: A flat array with NaN containing rows removed and set of row indicies that were removed
    '''

    if points is None:
        raise ValueError("points must not be None")

    numPoints = points.shape[0]

    nan1D = np.isnan(points).any(axis=1)

    invalidIndicies = np.flatnonzero(nan1D)
    points = np.delete(points, invalidIndicies, axis=0)

    assert (points.shape[0] + invalidIndicies.shape[0] == numPoints)

    return points, invalidIndicies


def InvalidIndicies_GPU(points: NDArray[float]) -> tuple[NDArray[float], NDArray[int]]:
    '''Removes rows with a NAN value.
     :return: A flat array with NaN containing rows removed and set of row indicies that were removed
    '''

    if points is None:
        raise ValueError("points must not be None")

    points = cp.asarray(points) if not isinstance(points,cp.ndarray) else points

    numPoints = points.shape[0]
    nan1D = cp.isnan(points).any(axis=1)
    invalidIndicies = cp.flatnonzero(nan1D)
    validIndicies = cp.flatnonzero(~nan1D)
    points = points[validIndicies, :]

    assert (points.shape[0] + invalidIndicies.shape[0] == numPoints)

    return points, invalidIndicies, validIndicies


def RotationMatrix(rangle: float) -> NDArray[float]:
    '''
    :param float rangle: Angle in radians
    '''
    if rangle is None:
        raise ValueError("Angle must not be none")

    rot_mat = np.array([[np.cos(rangle), np.sin(rangle), 0],
                        [-np.sin(rangle), np.cos(rangle), 0],
                        [0, 0, 1]])

    return rot_mat

    # interchange = np.array([[ 0,  1,  0],
    #                        [-1,  0,  0],
    #                        [ 0,  0,  1]])
    #
    # result = interchange @ rot_mat
    #
    # return result


def RotationMatrix_GPU(rangle: float) -> NDArray[float]:
    '''
    :param float rangle: Angle in radians
    '''
    if rangle is None:
        raise ValueError("Angle must not be none")

    rot_mat = cp.array([[np.cos(rangle), np.sin(rangle), 0],
                        [-np.sin(rangle), np.cos(rangle), 0],
                        [0, 0, 1]])

    return rot_mat


def IdentityMatrix() -> NDArray[float]:
    '''
    '''
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def TranslateMatrixXY(offset: tuple[float, float] | NDArray) -> NDArray[float]:
    '''
    :param offset: An offset to translate by, either tuple of (Y,X) or an array
    '''

    if offset is None:
        raise ValueError("Angle must not be none")
    elif hasattr(offset, "__iter__"):
        return np.array([[1, 0, 0], [0, 1, 0], [offset[0], offset[1], 1]])

    raise NotImplementedError("Unexpected argument")


def ScaleMatrixXY(scale: float) -> NDArray[float]:
    '''
    :param float scale: scale in radians, either a single value for all dimensions or a tuple of (Y,X) scale values
    '''
    if scale is None:
        raise ValueError("Angle must not be none")
    elif isinstance(scale, float):
        return np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    elif hasattr(scale, "__iter__"):
        return np.array([[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]])

    raise NotImplementedError("Unexpected argument")


def FlipMatrixY() -> NDArray[float]:
    '''
    Flip the Y axis
    '''
    return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

def FlipMatrixY_GPU() -> NDArray[float]:
    '''
    Flip the Y axis
    '''
    return cp.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])


def FlipMatrixX() -> NDArray[float]:
    '''
    Flip the Y axis
    '''
    return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])


def BlendWithLinear(transform: IControlPoints,
                    linear_factor: float | None = None,
                    travel_limit: float | None = None,
                    ignore_rotation: bool = False) -> ITransform:
    """
    Blends a transform with the estimate linear transform of its control points.  The goal is to "flatten" a transform to gradually reduce folds and other high distortion areas.
    :param transform:
    :param linear_factor:  The weight the linearized transform should have in calculating the new points
    :param ignore_rotation: This was added for SEM data which is known to not have rotation between slices.  Defaults to false.
    :return:  Either a mesh triangulation, a grid triangulation, or a linear transformation.  Grid and Triangulation
    match the input transform.  Linear transforms are only returned if linear_factor is 1.0.
    """

    # This check is here to help the IDE with autocompletion
    if not isinstance(transform, nornir_imageregistration.ITransform):
        raise ValueError("transform")

    linear_transform = nornir_imageregistration.transforms.converters.ConvertTransformToRigidTransform(transform,
                                                                                                       ignore_rotation=ignore_rotation)
    if linear_factor == 1.0:
        return linear_transform

    return BlendTransforms(transform, linear_transform=linear_transform, linear_factor=linear_factor, travel_limit=travel_limit)

def BlendTransforms(transform: IControlPoints,
                    linear_transform: ITransform,
                    linear_factor: float | None = None,
                    travel_limit: float | None = None):
    '''
    Transfrom the control points from transform through both transform and linear_transform.
    Blend the results according to parameters and return a new transform with the blended
    point positions as the target space control points.
    :param linear_factor:  The weight the linearized transform should have in calculating the new points
    :param ignore_rotation: This was added for SEM data which is known to not have rotation between slices.  Defaults to false.
    :return:  Either a mesh triangulation, a grid triangulation, or a linear transformation.  Grid and Triangulation
    match the input transform.  Linear transforms are only returned if linear_factor is 1.0.
    '''

    if linear_factor is not None and (linear_factor < 0 or linear_factor > 1.0):
        raise ValueError(f"linear_factor must be between 0 and 1.0, got {linear_factor}")

    if linear_factor is None and travel_limit is None:
        raise ValueError(f"Either travel_limit or linear_factor must have a value")

    if travel_limit is not None and travel_limit < 0:
        raise ValueError(f"travel_limit must be positive {travel_limit}")

    if linear_factor == 0 and travel_limit is None:
        #Why are we calling this?  Should I throw?
        return transform


    source_points = transform.SourcePoints
    target_points = transform.TargetPoints

    linear_points = linear_transform.Transform(source_points)

    if travel_limit is not None:
        delta = linear_points - target_points
        dist_squared = delta * delta
        hyp = np.sum(dist_squared, axis=1)
        distances = np.sqrt(hyp)

        #Arbitrary, but for a first pass points less than half of the travel distance use the transform
        #points more than halfway to the travel_limit have progressively more rigid tranfsorm blended in
        travel_blend_start_distance = travel_limit / 2
        travel_blend_range = travel_limit - travel_blend_start_distance
        linear_factors = (distances - travel_blend_start_distance) / travel_blend_range
        linear_factors.clip(0, 1.0, out=linear_factors)
        linear_factors = linear_factors.squeeze()
        blended_target_points = (target_points.swapaxes(0,1) * (1.0 - linear_factors)).swapaxes(0,1)
        blended_linear_points = (linear_points.swapaxes(0,1) *  linear_factors).swapaxes(0,1)
        output_target_points = blended_target_points + blended_linear_points
    else:
        blended_target_points = target_points * (1.0 - linear_factor)
        blended_linear_points = linear_points * linear_factor
        output_target_points = blended_target_points + blended_linear_points

    if isinstance(transform, nornir_imageregistration.transforms.IGridTransform):
        output_grid = nornir_imageregistration.ITKGridDivision(source_shape=transform.grid.source_shape,
                                                               cell_size=transform.grid.cell_size,
                                                               grid_dims=transform.grid.grid_dims,
                                                               transform=None)
        output_grid.TargetPoints = output_target_points
        output = nornir_imageregistration.transforms.GridWithRBFFallback(output_grid)
        return output
    else:
        output_points = np.append(output_target_points, source_points, 1)
        output = nornir_imageregistration.transforms.MeshWithRBFFallback(output_points)
        return output


def PointBoundingRect(points):
    raise DeprecationWarning("Use spatial.BoundsArrayFromPoints")

    (minY, minX) = np.min(points, 0)
    (maxY, maxX) = np.max(points, 0)
    return minY, minX, maxY, maxX


def PointBoundingBox(points):
    raise DeprecationWarning("Use spatial.BoundsArrayFromPoints")

    (minZ, minY, minX) = np.min(points, 0)
    (maxZ, maxY, maxX) = np.max(points, 0)
    return minZ, minY, minX, maxZ, maxY, maxX


def FixedOriginOffset(transforms):
    '''
    This is a fairly specific function to move a mosaic to have an origin at 0,0
    It handles both discrete and continuous functions the best it can.
    :return: tuple containing smallest origin offset
    '''

    mins = np.zeros((len(transforms), 2))
    for (i, t) in enumerate(transforms):
        if isinstance(t, nornir_imageregistration.IDiscreteTransform):
            mins[i, :] = t.FixedBoundingBox.BottomLeft
        elif isinstance(t, nornir_imageregistration.transforms.RigidNoRotation):
            mins[i, :] = t.target_offset
        elif hasattr(t, 'FixedBoundingBox'):
            mins[i, :] = t.FixedBoundingBox.BottomLeft
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
            if images.flat.shape != 2:
                raise ValueError("Must use a 1x2 array to specify a universal image size")
            is_images_param_single_size = True
        elif isinstance(images, Iterable):
            if len(images) != len(transforms):
                raise ValueError(
                    f"images list not of equal length as transforms list. Transforms: {len(transforms)} Images: {len(images)}")
        elif not isinstance(images, Iterable):
            raise ValueError(
                "If not none or a single 1x2 array the images parameter must be an iterable of equal length as transform array.")

    mbb = np.zeros((len(transforms), 4))
    for (i, t) in enumerate(transforms):
        if isinstance(t, nornir_imageregistration.IDiscreteTransform):
            mbb[i, :] = t.FixedBoundingBox.ToArray()
        elif isinstance(t, nornir_imageregistration.transforms.RigidNoRotation):
            # Figure out if images is an iterable or just a single size for all tiles
            if is_images_param_single_size:
                size = images
            else:
                size = nornir_imageregistration.GetImageSize(images[i])

            mbb[i, :2] = t.target_offset
            mbb[i, 2:] = t.target_offset + size
        elif hasattr(t, 'FixedBoundingBox'):
            mbb[i, :] = t.FixedBoundingBox.ToArray()
        else:
            raise ValueError(f"Unexpected type passed to FixedBoundingBox {t.__class__}")

    minX = np.min(mbb[:, 1])
    minY = np.min(mbb[:, 0])
    maxX = np.max(mbb[:, 3])
    maxY = np.max(mbb[:, 2])

    return nornir_imageregistration.Rectangle((float(minY), float(minX), float(maxY), float(maxX)))


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

        mbb[i, :] = t.MappedBoundingBox.ToArray()
        discrete_found = True

    if discrete_found is False:
        raise ValueError("No discrete transforms found in transforms list")

    minX = np.min(mbb[:, 1])
    minY = np.min(mbb[:, 0])
    maxX = np.max(mbb[:, 3])
    maxY = np.max(mbb[:, 2])

    return nornir_imageregistration.Rectangle((float(minY), float(minX), float(maxY), float(maxX)))


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

    try:
        origin = FixedOriginOffset(transforms)
    except ValueError:
        prettyoutput.LogErr("Could not determine origin of transforms, continuing")
        return np.zeros((2,))

    if origin is None:
        return

    if np.array_equal(origin, np.zeros(2, )):
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


if __name__ == '__main__':
    pass
