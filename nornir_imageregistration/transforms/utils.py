"""
Created on Apr 4, 2013

@author: u0490822
"""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from typing import Sequence

try:
    import cupy as cp
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
    # import nornir_imageregistration.cupyx_thunk as cupyx
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
    # import nornir_imageregistration.cupyx_thunk as cupyx

import nornir_imageregistration
from nornir_imageregistration.transforms.base import ITransform, IControlPoints, IDiscreteTransform, ITransformTranslation
from nornir_imageregistration.spatial.rectangle import Rectangle
from nornir_shared import prettyoutput
from nornir_imageregistration.type_info import ShapeLike, RectLike


def _xp_for_transform_geometry(transforms: Sequence[ITransform]):
    """NumPy vs CuPy for aggregate ops on transform geometry — follow stored points, not global backend."""
    for t in transforms:
        pts = getattr(t, "_points", None)
        if pts is not None:
            return cp.get_array_module(pts)
        off = getattr(t, "_target_offset", None)
        if off is not None and hasattr(off, "shape"):
            return cp.get_array_module(off)
    return np


def InvalidIndices(points: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.integer], NDArray[np.integer]]:
    """Remove rows containing NaN.
    :param points: NxM array of points (e.g. Nx2 or Nx4).
    :return: Tuple of (points_with_nan_rows_removed, invalid_indices, valid_indices).
    """
    if points is None:
        raise ValueError("points must not be None")

    xp = cp.get_array_module(points)

    numPoints = points.shape[0]

    nan1D = xp.isnan(points).any(axis=1)

    invalid_indices = xp.flatnonzero(nan1D)
    valid_indices = xp.flatnonzero(~nan1D)

    # Use indexing for both backends: xp.delete is not in older CuPy
    points = points[valid_indices, :].copy()

    assert (points.shape[0] + invalid_indices.shape[0] == numPoints)

    return points, invalid_indices, valid_indices


# Deprecated: use InvalidIndices (correct spelling).
InvalidIndicies = InvalidIndices

# Deprecated: use InvalidIndices; implementation is backend-agnostic via get_array_module.
InvalidIndices_GPU = InvalidIndices


def RotationMatrix(rangle: float) -> NDArray[np.floating]:
    """
    :param float rangle: Angle in radians
    """
    if rangle is None:
        raise ValueError("Angle must not be none")
    xp = nornir_imageregistration.GetComputationModule()
    a = float(rangle)
    # Use Python floats so CuPy's array() does not reject numpy scalars
    c, s = float(np.cos(a)), float(np.sin(a))
    rot_mat = xp.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return rot_mat

    # Legacy: alternative interchange matrix; unused.


def IdentityMatrix() -> NDArray[np.floating]:
    """Return a 3x3 identity matrix (numpy or cupy depending on active backend).

    :return: 3x3 identity matrix.
    """
    xp = nornir_imageregistration.GetComputationModule()
    return xp.identity(3)


def TranslateMatrixXY(offset: tuple[float, float] | NDArray) -> NDArray[np.floating]:
    """Build a 3x3 translation matrix for (Y, X) offset.

    :param offset: Translation offset as (Y, X) tuple or 2-element array.
    :return: 3x3 translation matrix.
    :raises ValueError: If offset is None.
    """
    xp = nornir_imageregistration.GetComputationModule()
    if offset is None:
        raise ValueError("offset must not be none")
    if hasattr(offset, "__iter__"):
        # Coerce to Python floats so CuPy's array() does not reject numpy scalars
        x0, x1 = float(offset[0]), float(offset[1])
        return xp.array([[1, 0, x0], [0, 1, x1], [0, 0, 1]])
    raise NotImplementedError("Unexpected argument")


def ScaleMatrixXY(scale: float | Sequence[float]) -> NDArray[np.floating]:
    """Build a 3x3 scale matrix for Y and X.

    :param scale: Scale factor (single value for uniform, or (Y, X) for per-axis).
    :return: 3x3 scale matrix.
    :raises ValueError: If scale is None.
    """
    xp = nornir_imageregistration.GetComputationModule()
    if scale is None:
        raise ValueError("scale must not be none")
    if isinstance(scale, (float, np.floating, int)):
        s = float(scale)
        return xp.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
    if hasattr(scale, "__iter__"):
        s0, s1 = float(scale[0]), float(scale[1])
        return xp.array([[s0, 0, 0], [0, s1, 0], [0, 0, 1]])
    raise NotImplementedError(f"Unexpected argument: {scale} is a {type(scale)}")


def FlipMatrixY() -> NDArray[np.floating]:
    """
    :return: 3x3 matrix that flips the Y axis
    """
    xp = nornir_imageregistration.GetComputationModule()
    return xp.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])


def FlipMatrixX() -> NDArray[np.floating]:
    """
    :return: 3x3 matrix that flips the X axis
    """
    xp = nornir_imageregistration.GetComputationModule()
    return xp.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])


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

    return BlendTransforms(transform, linear_transform=linear_transform, linear_factor=linear_factor,
                           travel_limit=travel_limit)  # type: ignore[return-value]


def BlendTransforms(transform: IControlPoints,
                    linear_transform: ITransform,
                    linear_factor: float | None = None,
                    travel_limit: float | None = None):
    """Blend control-point transform with a linear transform and return a new transform.

    Transforms control points through both transform and linear_transform, blends the
    results by linear_factor, and returns a new transform (mesh/grid/linear) with the
    blended target-space control points.

    :param transform: Control-point transform (mesh or grid) to blend.
    :param linear_transform: Linear transform used in the blend.
    :param linear_factor: Weight of the linear transform (0–1). None uses travel_limit.
    :param travel_limit: Max distance for full blend; beyond this, linear blend is reduced.
    :return: Mesh triangulation, grid triangulation, or linear transform matching input type.
    """

    if linear_factor is not None and (linear_factor < 0 or linear_factor > 1.0):
        raise ValueError(f"linear_factor must be between 0 and 1.0, got {linear_factor}")

    if linear_factor is None and travel_limit is None:
        raise ValueError(f"Either travel_limit or linear_factor must have a value")

    if travel_limit is not None and travel_limit < 0:
        raise ValueError(f"travel_limit must be positive {travel_limit}")

    if linear_factor == 0 and travel_limit is None:
        # Why are we calling this?  Should I throw?
        return transform

    source_points = transform.SourcePoints
    target_points = transform.TargetPoints

    linear_points = linear_transform.Transform(source_points)

    if travel_limit is not None:
        delta = linear_points - target_points
        dist_squared = delta * delta
        hyp = np.sum(dist_squared, axis=1)
        distances = np.sqrt(hyp)

        # Arbitrary, but for a first pass points less than half of the travel distance use the transform
        # points more than halfway to the travel_limit have progressively more rigid tranfsorm blended in
        travel_blend_start_distance = travel_limit / 2  # type: ignore[operator]
        travel_blend_range = travel_limit - travel_blend_start_distance
        linear_factors = (distances - travel_blend_start_distance) / travel_blend_range
        linear_factors.clip(0, 1.0, out=linear_factors)
        linear_factors = linear_factors.squeeze()
        blended_target_points = (target_points.swapaxes(0, 1) * (1.0 - linear_factors)).swapaxes(0, 1)
        blended_linear_points = (linear_points.swapaxes(0, 1) * linear_factors).swapaxes(0, 1)
        output_target_points = blended_target_points + blended_linear_points
    else:
        blended_target_points = target_points * (1.0 - linear_factor)  # type: ignore[operator]
        blended_linear_points = linear_points * linear_factor  # type: ignore[operator]
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


def FixedOriginOffset(transforms: Sequence[ITransform]) -> NDArray[np.floating]:
    """Compute the smallest fixed-space origin (min Y, min X) across transforms.

    Used to shift a mosaic so its origin is at (0, 0). Supports discrete and
    continuous transforms.

    :param transforms: Sequence of transforms (discrete or continuous).
    :return: 2-element array (minY, minX) — smallest origin offset in fixed space.
    :raises ValueError: If a transform type is not supported.
    """

    xp = _xp_for_transform_geometry(transforms)

    mins = xp.zeros((len(transforms), 2))
    for (i, t) in enumerate(transforms):
        if isinstance(t, nornir_imageregistration.IDiscreteTransform):
            mins[i, :] = xp.asarray(t.FixedBoundingBox.BottomLeft, dtype=xp.float64)
        elif isinstance(t, nornir_imageregistration.transforms.RigidTranslation):
            mins[i, :] = xp.asarray(t._target_offset, dtype=xp.float64)
        elif hasattr(t, 'FixedBoundingBox'):
            mins[i, :] = xp.asarray(t.FixedBoundingBox.BottomLeft, dtype=xp.float64)  # type: ignore[union-attr]
        else:
            raise ValueError(f"Unexpected transform type {t} at index {i}")

    return xp.min(mins, 0)


def FixedBoundingBox(transforms: Sequence[ITransform], images: list[nornir_imageregistration.ShapeLike] | None = None) -> Rectangle:
    """Calculate the bounding box of the warped position for a set of transforms
    :param list transforms: A list of transforms
    :param list images: A list of image parameters (strings, ndarrays, or 1x2
                        ndarray.shape arrays, of the size of the image.  Only
                        required for continuous transforms so a bounding box can
                        be calculated
    :return: A rectangle describing the bounding box
    """
    # Single-transform path omitted; multi-transform path used for consistency.

    is_images_param_single_size = False
    if images is not None:
        if isinstance(images, np.ndarray):
            if images.flat.shape != 2:  # type: ignore[union-attr]
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
            mbb[i, :] = t.TargetBoundingBox.ToArray()
        elif isinstance(t, nornir_imageregistration.transforms.RigidTranslation):
            # If there are no images we cannot calculate the bounding box
            if images is None or len(images) == 0:
                raise ValueError("Cannot calculate bounding box of rigid transform without images")

            # Figure out if images is an iterable or just a single size for all tiles
            t_rigid = t  # type: nornir_imageregistration.transforms.RigidTranslation
            if is_images_param_single_size:
                size = images
            else:
                size = nornir_imageregistration.GetImageSize(images[i])

            mbb[i, :2] = t_rigid.target_offset
            mbb[i, 2:] = t_rigid.target_offset + size  # type: ignore[operator]
        elif isinstance(t, IDiscreteTransform):
            mbb[i, :] = t.TargetBoundingBox.ToArray()
        elif hasattr(t, 'FixedBoundingBox'):
            mbb[i, :] = t.FixedBoundingBox.ToArray()  # type: ignore[union-attr]
        else:
            raise ValueError(f"Unexpected type passed to FixedBoundingBox {t.__class__}")

    minX = np.min(mbb[:, 1])
    minY = np.min(mbb[:, 0])
    maxX = np.max(mbb[:, 3])
    maxY = np.max(mbb[:, 2])

    return nornir_imageregistration.Rectangle((float(minY), float(minX), float(maxY), float(maxX)))


def MappedBoundingBox(transforms: Sequence[ITransform]) -> Rectangle:
    """Calculate the bounding box of the original source space positions for a set of transforms."""

    if len(transforms) == 1:
        # Copy the data instead of passing the transforms object
        return nornir_imageregistration.Rectangle(transforms[0].MappedBoundingBox.ToTuple())  # type: ignore[union-attr]

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
    """:return: True if transform bounding box has origin at 0,0 otherise false"""
    try:
        origin = FixedOriginOffset(transforms)
        origin_np = nornir_imageregistration.EnsureNumpyArray(origin).ravel()
        (minY, minX) = (float(origin_np[0]), float(origin_np[1]))
        return minY == 0 and minX == 0
    except ValueError:
        prettyoutput.LogErr("Could not determine origin of transforms, continuing")
        return True


def TranslateToZeroOrigin(transforms: Sequence[ITransform]) -> NDArray | None:
    """
    Translate the fixed space of all passed transforms so no point maps to a negative number. Useful for image coordinates.
    :return: The offset the mosaic was translated by, or None if no translation was needed.
    """

    try:
        origin = FixedOriginOffset(transforms)
    except ValueError:
        prettyoutput.LogErr("Could not determine origin of transforms, continuing")
        xp = _xp_for_transform_geometry(transforms)
        return xp.zeros((2,))

    if origin is None:
        return

    xp = cp.get_array_module(origin)
    if xp.array_equal(origin, xp.zeros(2)):
        return

    for t in transforms:
        if isinstance(t, ITransformTranslation):
            t.TranslateFixed(-origin)

    return -origin


def FixedBoundingBoxWidth(transforms: Sequence[ITransform]) -> float:
    """Return the width in fixed (target) space of the bounding box of the given transforms."""
    (minY, minX, maxY, maxX) = FixedBoundingBox(transforms).ToTuple()
    return np.ceil(maxX) - np.floor(minX)


def FixedBoundingBoxHeight(transforms: Sequence[ITransform]) -> float:
    """Return the height in fixed (target) space of the bounding box of the given transforms."""
    (minY, minX, maxY, maxX) = FixedBoundingBox(transforms).ToTuple()
    return np.ceil(maxY) - np.floor(minY)


def MappedBoundingBoxWidth(transforms: Sequence[ITransform]) -> float:
    """Return the width in mapped (source) space of the bounding box of the given transforms.

    :param transforms: Sequence of transforms (must include at least one discrete transform).
    :return: Width (maxX - minX) in mapped space, in pixels (ceiling/floor).
    """
    (minY, minX, maxY, maxX) = MappedBoundingBox(transforms).ToTuple()
    return np.ceil(maxX) - np.floor(minX)


def MappedBoundingBoxHeight(transforms: Sequence[ITransform]) -> float:
    """Return the height in mapped (source) space of the bounding box of the given transforms.

    :param transforms: Sequence of transforms (must include at least one discrete transform).
    :return: Height (maxY - minY) in mapped space, in pixels (ceiling/floor).
    """
    (minY, minX, maxY, maxX) = MappedBoundingBox(transforms).ToTuple()
    return np.ceil(maxY) - np.floor(minY)


def GetRotatedBoundaries(shape: ShapeLike,
                         angle: float) -> nornir_imageregistration.Rectangle:
    """Return the axis-aligned bounding box after rotating a shape by the given angle.

    :param shape: Image shape (H, W) or Rectangle; non-Rectangle is interpreted as (H, W).
    :param angle: Rotation angle in radians.
    :return: Rectangle enclosing the rotated shape.
    """

    if not isinstance(shape, nornir_imageregistration.Rectangle):
        rect = nornir_imageregistration.Rectangle.CreateFromBounds((0, 0, shape[0], shape[1]))
    else:
        rect = shape

    if angle % np.pi * 2 == 0:
        return rect

    corners = rect.Corners

    # If there are "off by one" errors, check that the center shouldn't have 0.5 pixels subtracted
    rigid_transform = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
        angle=angle,
        source_rotation_center=rect.Center,
        target_offset=(0, 0),
        scalar=1.0,
        flip_ud=False,
    )

    rotated_corners = rigid_transform.Transform(corners)
    return Rectangle.CreateBoundingRectangleForPoints(rotated_corners)


if __name__ == '__main__':
    pass

