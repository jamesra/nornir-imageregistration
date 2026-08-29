"""
Created on Apr 4, 2013

@author: u0490822
"""

from collections.abc import Iterable
import warnings

import numpy as np
from numpy.typing import NDArray
from typing import Any, Sequence

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


def host_copy_points(points: Any) -> NDArray[np.floating]:
    """Copy control points to host memory so a background build cannot race the UI."""
    if hasattr(points, "get"):
        return np.asarray(points.get(), dtype=np.float64, copy=True)
    return np.asarray(points, dtype=np.float64, copy=True)


def InvalidIndices(points: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Remove rows that are not finite, i.e. containing NaN or +-Inf.

    :param points: NxM array of points (e.g. Nx2 or Nx4).
    :return: Tuple of (points_with_non_finite_rows_removed, invalid_row_mask).
        Callers that need the valid side can invert with ``~invalid_mask``.
        Returning a bool mask avoids CuPy ``flatnonzero`` index materialization
        and the host syncs that come with integer index arrays on the GPU path.

    Inf counts as invalid, not just NaN. An infinite coordinate is no more usable
    than NaN: callers either route these rows to a continuous fallback transform
    or drop them before scattering, and an Inf that reads as valid becomes a
    garbage sample index instead. On the host ``np.seterr(invalid='raise',
    divide='raise')`` makes most ways of producing Inf raise first, but CuPy
    ignores ``seterr``, so on the GPU path an overflowing float64->float32
    downcast or a divide by zero yields Inf silently.
    """
    if points is None:
        raise ValueError("points must not be None")

    xp = cp.get_array_module(points)
    invalid_mask = (~xp.isfinite(points)).any(axis=1)
    filtered = points[~invalid_mask, :].copy()
    return filtered, invalid_mask


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


DEFAULT_MAX_BLEND_WEIGHT: float = 0.9
DEFAULT_REBLEND_ITERATIONS: int = 8
DEFAULT_REBLEND_TOLERANCE: float = 0.5
DEFAULT_REBLEND_WEIGHT_TOLERANCE: float = 0.01
INVERSE_MAP_Y_CORRELATION_THRESHOLD: float = 0.9


def _coalesce_min_blend(min_blend: float | None,
                        linear_factor: float | None) -> float | None:
    """Return min_blend, accepting deprecated linear_factor with a warning."""
    if linear_factor is not None:
        if min_blend is not None and min_blend != linear_factor:
            raise ValueError(
                f"min_blend ({min_blend}) and linear_factor ({linear_factor}) disagree")
        if min_blend is None:
            warnings.warn(
                "linear_factor is deprecated; use min_blend instead",
                DeprecationWarning,
                stacklevel=3)
            return linear_factor
    return min_blend


def resolve_effective_max_blend(min_blend: float | None,
                                travel_limit: float | None,
                                max_blend: float | None) -> float | None:
    """Choose per-point max rigid blend weight from explicit cap or subtle defaults."""
    if max_blend is not None:
        return max_blend
    if travel_limit is None and min_blend is not None:
        return min_blend
    return DEFAULT_MAX_BLEND_WEIGHT


def _as_numpy_points(points: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return control points as a NumPy array."""
    xp_mod = cp.get_array_module(points)
    if xp_mod is cp:
        return cp.asnumpy(points)
    return np.asarray(points)


def estimate_inverse_map_y_correlation(transform: ITransform,
                                       source_points: NDArray[np.floating] | None = None) -> float:
    """Pearson correlation of volume-Y vs section-Y along an inverse-map vertical midline."""
    if source_points is None:
        if not isinstance(transform, IControlPoints):
            return 0.0
        source_points = transform.SourcePoints

    source = _as_numpy_points(source_points)
    if source.ndim != 2 or source.shape[0] < 2 or source.shape[1] < 2:
        return 0.0

    y_min, x_min = source.min(axis=0)
    y_max, x_max = source.max(axis=0)
    height = float(y_max - y_min)
    width = float(x_max - x_min)
    if height <= 1.0 or width <= 1.0:
        return 0.0

    if not hasattr(transform, 'InverseTransform'):
        return 0.0

    volume_y = np.linspace(y_min + height * 0.25, y_min + height * 0.75, 20)
    volume_x = np.full(20, x_min + width * 0.5)
    section_y: list[float] = []
    for y, x in zip(volume_y, volume_x):
        mapped = transform.InverseTransform(np.array([y, x], dtype=float))
        mapped = _as_numpy_points(mapped).reshape(-1)
        section_y.append(float(mapped[0]))

    corr_matrix = np.corrcoef(volume_y, section_y)
    corr = float(corr_matrix[0, 1])
    if np.isnan(corr):
        return 0.0
    return corr


def _orientation_sign_preserved(original_corr: float, result_corr: float) -> bool:
    """Return True when inverse-map Y correlation sign is unchanged."""
    if abs(original_corr) < INVERSE_MAP_Y_CORRELATION_THRESHOLD:
        return True
    if result_corr == 0.0:
        return False
    return np.sign(original_corr) == np.sign(result_corr)


def _travel_blend_weights(distances: NDArray[np.floating],
                          travel_limit: float,
                          min_blend: float | None = None,
                          max_blend: float | None = DEFAULT_MAX_BLEND_WEIGHT,
                          *,
                          linear_factor: float | None = None) -> NDArray[np.floating]:
    """Return per-point linear blend weights from deviation distance and travel_limit."""
    min_blend = _coalesce_min_blend(min_blend, linear_factor)
    xp = cp.get_array_module(distances)
    base = min_blend if min_blend is not None else 0.0
    normalized = xp.clip(distances / travel_limit, 0.0, 1.0)
    travel_weight = normalized * normalized * (3.0 - 2.0 * normalized)
    weights = base + (1.0 - base) * travel_weight
    if max_blend is not None:
        weights = xp.minimum(weights, max_blend)
    return weights


def _blend_target_points(target_points: NDArray[np.floating],
                         linear_points: NDArray[np.floating],
                         linear_factors: NDArray[np.floating]) -> NDArray[np.floating]:
    """Blend target-space control points toward linear predictions."""
    xp = cp.get_array_module(target_points, linear_points)
    weights = xp.asarray(linear_factors, dtype=float).reshape(-1, 1)
    return target_points * (1.0 - weights) + linear_points * weights


def _control_points_from_blended_targets(transform: IControlPoints,
                                         output_target_points: NDArray[np.floating]) -> ITransform:
    """Build a mesh or grid transform with blended target control points."""
    source_points = transform.SourcePoints
    xp = cp.get_array_module(output_target_points, source_points)
    output_target_points = xp.asarray(output_target_points)
    source_points = xp.asarray(source_points)
    use_gpu = xp is cp or nornir_imageregistration.UsingCupy()
    if isinstance(transform, nornir_imageregistration.transforms.IGridTransform):
        output_grid = nornir_imageregistration.ITKGridDivision(source_shape=transform.grid.source_shape,
                                                               cell_size=transform.grid.cell_size,
                                                               grid_dims=transform.grid.grid_dims,
                                                               transform=None)
        output_grid.TargetPoints = output_target_points
        if use_gpu:
            return nornir_imageregistration.transforms.GridWithRBFFallback_GPUComponent(output_grid)
        return nornir_imageregistration.transforms.GridWithRBFFallback(output_grid)
    output_points = xp.hstack((output_target_points, source_points))
    if use_gpu:
        return nornir_imageregistration.transforms.MeshWithRBFFallback_GPUComponent(output_points)
    return nornir_imageregistration.transforms.MeshWithRBFFallback(output_points)


def BlendWithLinear(transform: IControlPoints,
                    min_blend: float | None = None,
                    travel_limit: float | None = None,
                    ignore_rotation: bool = False,
                    reblend_iterations: int = 1,
                    reblend_tolerance: float = DEFAULT_REBLEND_TOLERANCE,
                    reblend_weight_tolerance: float = DEFAULT_REBLEND_WEIGHT_TOLERANCE,
                    max_blend: float | None = None,
                    *,
                    linear_factor: float | None = None) -> ITransform:
    """
    Blends a transform with the estimate linear transform of its control points.  The goal is to "flatten" a transform to gradually reduce folds and other high distortion areas.
    :param transform:
    :param min_blend:  Floor weight toward the rigid linear approximation (uniform when travel_limit is omitted).
    :param ignore_rotation: This was added for SEM data which is known to not have rotation between slices.  Defaults to false.
    :return:  Either a mesh triangulation, a grid triangulation, or a linear transformation.  Grid and Triangulation
    match the input transform.  Linear transforms are only returned if min_blend is 1.0.
    """
    min_blend = _coalesce_min_blend(min_blend, linear_factor)
    effective_max_blend = resolve_effective_max_blend(min_blend, travel_limit, max_blend)

    # This check is here to help the IDE with autocompletion
    if not isinstance(transform, nornir_imageregistration.ITransform):
        raise ValueError("transform")

    mesh_corr = estimate_inverse_map_y_correlation(transform)
    linear_transform = nornir_imageregistration.transforms.converters.ConvertControlPointsToRigidTransformForBlend(
        transform,
        ignore_rotation=ignore_rotation)
    if min_blend == 1.0:
        return linear_transform

    if reblend_iterations > 1:
        blended = BlendTransformsIteratively(transform,
                                             linear_transform=linear_transform,
                                             min_blend=min_blend,
                                             travel_limit=travel_limit,
                                             reblend_iterations=reblend_iterations,
                                             reblend_tolerance=reblend_tolerance,
                                             reblend_weight_tolerance=reblend_weight_tolerance,
                                             max_blend=effective_max_blend)
    else:
        blended = BlendTransforms(transform, linear_transform=linear_transform, min_blend=min_blend,
                                  travel_limit=travel_limit, max_blend=effective_max_blend)

    if _orientation_sign_preserved(mesh_corr, estimate_inverse_map_y_correlation(blended)):
        return blended
    return transform


def BlendTransforms(transform: IControlPoints,
                    linear_transform: ITransform,
                    min_blend: float | None = None,
                    travel_limit: float | None = None,
                    max_blend: float | None = None,
                    *,
                    linear_factor: float | None = None):
    """Blend control-point transform with a linear transform and return a new transform.

    Transforms control points through both transform and linear_transform, blends the
    results by min_blend, and returns a new transform (mesh/grid/linear) with the
    blended target-space control points.

    :param transform: Control-point transform (mesh or grid) to blend.
    :param linear_transform: Linear transform used in the blend.
    :param min_blend: Floor weight toward rigid (0–1); uniform when travel_limit is omitted.
    :param travel_limit: Distance scale for smooth per-point blend toward linear_transform.
    :param max_blend: Cap per-point linear weight; defaults to min_blend or 0.9 by mode.
    :return: Mesh triangulation, grid triangulation, or linear transform matching input type.
    """
    min_blend = _coalesce_min_blend(min_blend, linear_factor)
    effective_max_blend = resolve_effective_max_blend(min_blend, travel_limit, max_blend)

    if min_blend is not None and (min_blend < 0 or min_blend > 1.0):
        raise ValueError(f"min_blend must be between 0 and 1.0, got {min_blend}")

    if min_blend is None and travel_limit is None:
        raise ValueError("Either travel_limit or min_blend must have a value")

    if travel_limit is not None and travel_limit < 0:
        raise ValueError(f"travel_limit must be positive {travel_limit}")

    if min_blend == 0 and travel_limit is None:
        return transform

    source_points = transform.SourcePoints
    target_points = transform.TargetPoints
    linear_points = linear_transform.Transform(source_points)
    xp = cp.get_array_module(target_points, linear_points, source_points)
    target_points = xp.asarray(target_points)
    linear_points = xp.asarray(linear_points)

    if travel_limit is not None:
        distances = xp.sqrt(xp.sum((linear_points - target_points) ** 2, axis=1))
        blend_weights = _travel_blend_weights(distances, travel_limit, min_blend, effective_max_blend)
    else:
        blend_weights = xp.full(target_points.shape[0], min_blend, dtype=float)

    output_target_points = _blend_target_points(target_points, linear_points, blend_weights)
    return _control_points_from_blended_targets(transform, output_target_points)


def BlendTransformsIteratively(transform: IControlPoints,
                               linear_transform: ITransform,
                               min_blend: float | None = None,
                               travel_limit: float | None = None,
                               reblend_iterations: int = DEFAULT_REBLEND_ITERATIONS,
                               reblend_tolerance: float = DEFAULT_REBLEND_TOLERANCE,
                               reblend_weight_tolerance: float = DEFAULT_REBLEND_WEIGHT_TOLERANCE,
                               max_blend: float | None = None,
                               *,
                               linear_factor: float | None = None) -> ITransform:
    """Iteratively blend toward linear_transform until points stabilize or iteration cap is reached."""
    min_blend = _coalesce_min_blend(min_blend, linear_factor)
    effective_max_blend = resolve_effective_max_blend(min_blend, travel_limit, max_blend)
    if reblend_iterations <= 1:
        return BlendTransforms(transform,
                               linear_transform=linear_transform,
                               min_blend=min_blend,
                               travel_limit=travel_limit,
                               max_blend=effective_max_blend)

    xp = cp.get_array_module(transform.TargetPoints)
    current_target_points = xp.asarray(transform.TargetPoints, dtype=float)
    source_points = transform.SourcePoints
    for _ in range(reblend_iterations):
        target_points = current_target_points
        linear_points = xp.asarray(linear_transform.Transform(source_points))

        if travel_limit is not None:
            distances = xp.sqrt(xp.sum((linear_points - target_points) ** 2, axis=1))
            weights = _travel_blend_weights(distances, travel_limit, min_blend, effective_max_blend)
        else:
            weights = xp.full(target_points.shape[0], min_blend, dtype=float)

        new_target_points = _blend_target_points(target_points, linear_points, weights)
        movement = float(xp.max(xp.sqrt(xp.sum((new_target_points - target_points) ** 2, axis=1))))
        current_target_points = new_target_points

        if movement < reblend_tolerance or float(xp.max(weights)) < reblend_weight_tolerance:
            break

    return _control_points_from_blended_targets(transform, current_target_points)


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

