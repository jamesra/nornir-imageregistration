"""
Created on Apr 22, 2013


"""
from __future__ import annotations

import collections
import contextlib
import os
import threading
import warnings
import logging

from nornir_imageregistration import IgnoreUnderflow

try:
    import cupy as cp
    import cupyx
    import cupyx.scipy
    import cupyx.scipy.ndimage
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx

import numpy as np
from numpy.typing import DTypeLike, NDArray
import scipy

import nornir_pools
import nornir_imageregistration
from nornir_imageregistration.transforms import ITransform, IRigidTransform, factory, triangulation
from nornir_imageregistration.transforms.utils import InvalidIndices

# Serializes GPU warp dispatches across threads. CuPy CUDA kernels on the
# default stream are ordered by the driver, but Python-level CuPy state
# (memory pool, error-checking) is not fully thread-safe. Holding this lock
# during _TransformImageUsingCoords lets threads overlap disk I/O and CPU
# coordinate transforms while GPU warps run one at a time.
_gpu_warp_lock: threading.Lock = threading.Lock()

# Ceiling on a single assemble output canvas. An exploded target-space transform asks
# for a canvas orders of magnitude larger than any real section, and without a ceiling
# the allocation either raises a bare MemoryError naming a number with no context or,
# on a machine with enough swap, thrashes for a long time before failing.
_DEFAULT_MAX_ASSEMBLE_BUFFER_BYTES = 16 * 1024 * 1024 * 1024


def _max_assemble_buffer_bytes() -> int:
    """Return the maximum allowed assemble output buffer size in bytes."""
    raw = os.environ.get('NORNIR_MAX_ASSEMBLE_BUFFER_BYTES')
    if raw is not None and raw.strip() != '':
        return int(raw)
    return _DEFAULT_MAX_ASSEMBLE_BUFFER_BYTES


def _raise_if_assemble_buffer_too_large(height: int, width: int, dtype: DTypeLike,
                                        include_zbuffer: bool = True) -> None:
    """Fail fast before allocating an unreasonably large assemble canvas.

    ``include_zbuffer`` accounts for the companion float16 distance buffer that the
    tile-compositing path allocates alongside the image. Callers that only allocate an
    image, such as ``TransformImage``, pass False so the limit means what it says.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Assemble output dimensions must be positive, got {height}x{width}")

    image_bytes = int(height) * int(width) * int(np.dtype(dtype).itemsize)
    total_bytes = image_bytes
    if include_zbuffer:
        total_bytes += int(height) * int(width) * int(np.dtype(np.float16).itemsize)
    limit_bytes = _max_assemble_buffer_bytes()
    if total_bytes > limit_bytes:
        raise ValueError(
            f"Refusing to allocate {total_bytes:,} bytes for assemble output "
            f"({width}x{height}, image dtype={np.dtype(dtype)}, limit={limit_bytes:,} from "
            "NORNIR_MAX_ASSEMBLE_BUFFER_BYTES). This usually indicates invalid mosaic transforms "
            "with exploded target-space control points; regenerate the grid transform or inspect "
            "per-tile target bounding boxes before assembling.")


def _ensure_on_array_module(array: NDArray, xp) -> NDArray:
    """Return *array* on *xp* without copying when already resident there.

    Accepts NumPy or CuPy; ``np.asarray`` is never used on a CuPy input because
    CuPy refuses the implicit host conversion.
    """
    if cp.get_array_module(array) is xp:
        return array
    if xp is np:
        return nornir_imageregistration.EnsureNumpyArray(array)
    return xp.asarray(array)


def _assemble_distance_warp_order() -> int | None:
    """Spline order for the distance z-buffer warp (image warp unchanged).

    Production default (unset ``NORNIR_ASSEMBLE_DISTANCE_WARP_ORDER``): **cubic** — same as the
    image warp (``interpolation_order=None`` → order 3 for float images). Set ``=0`` or ``=1`` for
    cheaper nearest/linear distance warps (opt-in; can change z-buffer seam winners by ±1 DN).

    **Pixel effect (order 0/1 vs cubic):** only affects the distance plane used for compositing,
    not image ``map_coordinates`` interpolation.
    """
    raw = os.environ.get('NORNIR_ASSEMBLE_DISTANCE_WARP_ORDER', '').strip()
    if raw == '':
        return None
    if raw.lower() in ('default', 'image', 'none', 'cubic'):
        return None
    return int(raw)


def GetROICoords(botleft: tuple[float, float] | NDArray, area: tuple[float, float] | NDArray,
                 *, xp=None) -> NDArray[np.floating]:
    """Integer YX meshgrid for a rectangle origin and area.

    Accepts NumPy or CuPy via the *xp* argument (defaults to ``GetComputationModule``).
    """
    if xp is None:
        xp = nornir_imageregistration.GetComputationModule()
    use_cp = xp is not np

    # Truncate the origin before calling arange instead of passing a float start with an
    # integer dtype: the two backends disagree there. np.arange(-0.5, 8.5, dtype=int32)
    # returns nine zeros (the step truncates to 0), collapsing every write coordinate to
    # the same pixel, while cp.arange returns 0..8. Truncating first makes both backends
    # agree and leaves whole-number origins, the common case, untouched.
    start_y = int(botleft[0])
    start_x = int(botleft[1])

    x_range = xp.arange(start_x, start_x + int(area[1]), dtype=np.int32)
    y_range = xp.arange(start_y, start_y + int(area[0]), dtype=np.int32)

    # Numpy arange sometimes accidentally adds an extra value to the array due to rounding error, remove the extra element if needed
    if len(x_range) > area[1]:
        x_range = x_range[:int(area[1])]

    if len(y_range) > area[0]:
        y_range = y_range[:int(area[0])]

    i_y, i_x = xp.meshgrid(y_range, x_range, sparse=False, indexing='ij')  # type: ignore[misc]

    if use_cp:
        coordArray = xp.vstack((i_y.ravel(), i_x.ravel())).transpose()
    else:
        coordArray = xp.vstack((i_y.flat, i_x.flat)).transpose()

    del i_y
    del i_x
    del x_range
    del y_range

    return coordArray  # type: ignore[return-value]


def write_to_source_roi_coords(transform: ITransform,
                               botleft: tuple[float, float] | NDArray,
                               area: tuple[float, float] | NDArray,
                               extrapolate: bool = False) -> tuple[NDArray, NDArray]:
    """
    This function is used to generate coordinates to transform image data in target space backwards into source space.


    Given a transform and a region in source space, create uniform integer coordinates over the region of interest in source space
    for each pixel.  Then run an forward transform to determine those coordinates in target space.  The target space
    coordinates will be used later to interpolate pixel values for each integer pixel valued destination space coordinates.

    :param extrapolate:
    :param transform transform: The transform used to map points between fixed and mapped space
    :param botleft: The (Y,X) coordinates of the bottom left corner in source space
    :param area: The (Height, Width) of the region of interest coordinates.
    :return: (read_space_coords, write_space_coords)
    """

    write_space_coords = GetROICoords(botleft, area)

    read_space_coords = transform.Transform(write_space_coords, extrapolate=extrapolate).astype(np.float32, copy=False)
    (valid_read_space_coords, invalid_coords_mask) = InvalidIndices(read_space_coords)

    del read_space_coords

    valid_write_space_coords = write_space_coords[~invalid_coords_mask, :]
    # valid_write_space_coords = valid_write_space_coords  # - botleft

    return valid_read_space_coords, valid_write_space_coords


def write_to_target_roi_coords(transform: ITransform,
                               botleft: tuple[float, float] | NDArray,
                               area: tuple[float, float] | NDArray,
                               extrapolate: bool = False) -> tuple[NDArray, NDArray]:
    """
    This function is used to generate coordinates to transform image data in source space forward into target space.

    Given a transform and a region in target space, create uniform integer coordinates over the region of interest in source
    space for each pixel.  Then run a inverse transform to map target coordinates back in source space.  The source
    space coordinates will be used later to interpolate pixel values for each integer pixel valued destination target
    coordinates.

    :param extrapolate:
    :param transform transform: The transform used to map points between fixed and mapped space
    :param botleft: The (Y,X) coordinates of the bottom left corner in target space
    :param area: The (Height, Width) of the region of interest
e coordinates.
    :return: (read_space_coords, write_space_coords)
    """
    from nornir_imageregistration.transforms.gridtransform import _assemble_inverse_use_scipy

    use_gpu_assemble = (
            nornir_imageregistration.GetActiveComputationLib()
            == nornir_imageregistration.ComputationLib.cupy
    )
    use_host_roi_inverse = use_gpu_assemble and _assemble_inverse_use_scipy()

    if use_host_roi_inverse:
        write_space_coords = GetROICoords(botleft, area, xp=np)
    else:
        write_space_coords = GetROICoords(botleft, area)

    read_space_coords = transform.InverseTransform(
        write_space_coords, extrapolate=extrapolate,
    ).astype(np.float32, copy=False)

    # IRigidTransform.InverseTransform (translation/rigid/similarity) is a closed-form
    # affine op on finite input and can never produce NaN rows. Skip InvalidIndices'
    # xp.flatnonzero calls entirely for this transform class: on CuPy those calls force a
    # device sync whose result is always "nothing to remove" here, so the sync is pure
    # overhead in hot per-cell loops (e.g. STOS grid-refine ROI extraction).
    #
    # NOTE: IRigidTransform is not the only NaN-safe (globally-defined/"continuous")
    # transform -- RBF-based transforms (OneWayRBFWithLinearCorrection,
    # TwoWayRBFWithLinearCorrection) are also defined everywhere and should never emit
    # NaN from finite input, but there is no shared "continuous transform" marker
    # interface in transforms/base.py to check against (IDiscreteTransform captures the
    # opposite idea -- bounding boxes for control-point/triangulation transforms that can
    # be undefined outside their hull). This fast path is scoped to IRigidTransform
    # because that is what the STOS grid-refine per-cell path actually constructs
    # (ApproximateRigidTransformBySourcePoints). If RBF transforms are ever routed
    # through this function's hot path, they will correctly but more slowly fall through
    # to the InvalidIndices path below -- revisit then.
    if isinstance(transform, IRigidTransform):
        valid_read_space_coords = read_space_coords
        valid_write_space_coords = write_space_coords
        if use_host_roi_inverse and use_gpu_assemble:
            valid_read_space_coords = cp.asarray(valid_read_space_coords)
            valid_write_space_coords = cp.asarray(valid_write_space_coords)
        return valid_read_space_coords, valid_write_space_coords

    (valid_read_space_coords, invalid_coords_mask) = InvalidIndices(read_space_coords)

    del read_space_coords

    # use_host_roi_inverse leaves write_space_coords on the host while the transform may
    # still return device coordinates, so the mask has to follow the array it indexes.
    invalid_coords_mask = _ensure_on_array_module(
        invalid_coords_mask, cp.get_array_module(write_space_coords))

    valid_write_space_coords = write_space_coords[~invalid_coords_mask, :]
    if use_host_roi_inverse and use_gpu_assemble:
        valid_read_space_coords = cp.asarray(valid_read_space_coords)
        valid_write_space_coords = cp.asarray(valid_write_space_coords)

    return valid_read_space_coords, valid_write_space_coords


def assembly_source_sample_mask(
        transform: ITransform,
        fixed_image_shape: NDArray | tuple[int, int] | list[int],
        source_image_shape: NDArray | tuple[int, int] | list[int],
        *,
        extrapolate: bool = False) -> NDArray[np.bool_]:
    """Return True at fixed pixels whose inverse transform lands inside the source image."""
    transform = transform_for_host_assembly(transform)
    fixed_shape = np.asarray(fixed_image_shape, dtype=np.int64).ravel()
    source_shape = np.asarray(source_image_shape, dtype=np.int64).ravel()
    fixed_h, fixed_w = int(fixed_shape[0]), int(fixed_shape[1])
    source_h, source_w = int(source_shape[0]), int(source_shape[1])

    read_coords, write_coords = write_to_target_roi_coords(
        transform,
        (0, 0),
        (fixed_h, fixed_w),
        extrapolate=extrapolate,
    )
    read_coords = nornir_imageregistration.EnsureNumpyArray(read_coords)
    write_coords = nornir_imageregistration.EnsureNumpyArray(write_coords)

    mask = np.zeros((fixed_h, fixed_w), dtype=bool)
    if write_coords.shape[0] == 0:
        return mask

    in_source = (
            (read_coords[:, 0] >= 0) & (read_coords[:, 0] < source_h)
            & (read_coords[:, 1] >= 0) & (read_coords[:, 1] < source_w)
    )
    if not np.any(in_source):
        return mask

    valid_write = write_coords[in_source]
    flat = nornir_imageregistration.ravel_index(valid_write, mask.shape).astype(np.int64, copy=False)
    mask.ravel()[flat] = True
    return mask


def get_valid_coords(coords: NDArray, image_shape, origin=(0, 0), area=None) -> tuple[NDArray, NDArray]:
    """Given an Nx2 array off image coordinates, remove the coordinates that
    fall outside the image_shape boundaries.
    :param coords: Nx2 array of image coordinates
    :param image_shape: 1x2 array of image dimensions
    :param origin: 1x2 array with minimum valid coordinate
    :parm area: 1x2 array of expected area, which may exceed image_shape.  coords will be cropped to whatever is less
    :return: The coordinates greater than or equal to origin and less than origin + area and a mask indicating (== True) which coordinates met the criteria
    """

    xp = cp.get_array_module(coords)
    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy

    if isinstance(origin, int):
        adjusted_origin = np.array((origin, origin), dtype=np.int32)
    elif isinstance(origin, tuple):
        adjusted_origin = np.array(origin)
    elif isinstance(origin, np.ndarray) or isinstance(origin, cp.ndarray):
        adjusted_origin = origin.copy()
    else:
        raise ValueError("Unexpected type passed to origin")

    if isinstance(area, int):
        adjusted_area = np.array((area, area), dtype=np.int32)
    elif isinstance(area, tuple):
        adjusted_area = np.array(area)
    elif isinstance(area, np.ndarray) or isinstance(area, cp.ndarray):
        adjusted_area = area.copy()  # type: ignore[union-attr]
    elif area is None:
        adjusted_area = xp.copy(image_shape)
    else:
        raise ValueError("Unexpected type passed to area")

    origin = cp.asarray(origin) if use_cp and not isinstance(origin, cp.ndarray) else origin
    image_shape = cp.asarray(image_shape) if use_cp and not isinstance(image_shape, cp.ndarray) else image_shape

    adjusted_origin = cp.asarray(adjusted_origin) if use_cp and not isinstance(adjusted_origin,
                                                                               cp.ndarray) else adjusted_origin
    adjusted_area = cp.asarray(adjusted_area) if use_cp and not isinstance(adjusted_area,
                                                                           cp.ndarray) else adjusted_area

    adjust_area_mask = adjusted_origin < 0
    adjusted_area[adjust_area_mask] = adjusted_area[adjust_area_mask] + adjusted_origin[adjust_area_mask]
    adjusted_origin[adjust_area_mask] = 0
    # Warning - Clement: adjusted_area and adjusted_origin are not being used below...

    valid_coords_mask = xp.logical_and(xp.min(coords >= origin, 1), xp.min(coords < image_shape, 1))
    if use_cp:
        valid_adjusted_coords = coords[valid_coords_mask, :]
    else:
        valid_adjusted_coords = xp.delete(coords, xp.logical_not(valid_coords_mask), 0)  # coords[valid_coords_mask, :]

    return valid_adjusted_coords, valid_coords_mask


def _CropImageToFitCoords(input_image: NDArray, coordinates: NDArray, padding: int, cval=0) -> tuple[
    NDArray[np.floating], NDArray[np.integer]]:
    """For large images we only need a specific range of coordinates from the image.  However Scipy calls such as map_coordinates will
       send the entire image through a spline_filter first.  To avoid this we crop the image with a padding of one and adjust the
       coordinates appropriately
       :param ndarray input_image: image we will be extracting data from at the specified coordinates
       :param ndarray coordinates: Nx2 array of points indexing into the image
       :param float cval: Value to use for regions outside the existing image when padding
       :return: (cropped_image, translated_coordinates, coordinate_mask) Returns the cropped image, the coordinates translated into the cropped image, and a mask set to False for any coordinates that did not fit within the image boundaries
       """

    xp = cp.get_array_module(input_image)
    coordinates = _ensure_on_array_module(coordinates, xp)

    bottom_left = xp.floor(xp.min(coordinates, 0))
    # bottom_left[bottom_left < 0] = 0
    top_right = xp.ceil(xp.max(coordinates, 0))
    # top_right_out_of_bounds = top_right >= input_image.shape
    # top_right[top_right >= input_image.shape] = np.min(top_right, input_image.shape)

    filtered_coordinates, coord_mask = get_valid_coords(coordinates, input_image.shape, origin=(0, 0),
                                                        area=(top_right - bottom_left) + 1)

    # get_valid_coords already compacted filtered_coordinates via boolean-mask fancy
    # indexing, which forces CuPy to resolve the output size (a device sync) to know
    # .shape[0]. Reuse that already-resolved host-side shape instead of re-syncing with
    # xp.all(coord_mask == False) -- same result, zero extra syncs.
    if filtered_coordinates.shape[0] == 0:
        # No mappable coords, just return an empty image
        return xp.empty((0, 0)), xp.empty((0, 2)), coord_mask  # type: ignore[return-value]

    # Recalculate boundaries to account for filtered coords
    filtered_bottom_left = xp.floor(xp.min(filtered_coordinates, 0))
    filtered_top_right = xp.ceil(xp.max(filtered_coordinates, 0))

    # Read both bounds back to host together instead of the four separate int(...)
    # conversions below (each of which is its own CuPy device sync): one sync here
    # instead of four.
    bounds_host = nornir_imageregistration.EnsureNumpyArray(
        xp.stack((filtered_bottom_left, filtered_top_right))).astype(np.int64, copy=False)
    filtered_bottom_left_host, filtered_top_right_host = bounds_host[0], bounds_host[1]

    padded_bottom_left_host = filtered_bottom_left_host - padding
    # padded_top_right = filtered_top_right + padding

    Width = int(filtered_top_right_host[1] - filtered_bottom_left_host[1]) + 1 + (padding * 2)
    Height = int(filtered_top_right_host[0] - filtered_bottom_left_host[0]) + 1 + (padding * 2)

    cropped_image = nornir_imageregistration.CropImage(input_image, Xo=int(padded_bottom_left_host[1]),
                                                       Yo=int(padded_bottom_left_host[0]),
                                                       Width=Width, Height=Height, cval=cval)

    translated_coordinates = (filtered_coordinates - filtered_bottom_left) + padding

    return cropped_image, translated_coordinates, coord_mask  # type: ignore[return-value]


def my_cheesy_map_coordinates(image, coords):
    """Sample image at integer floor of coords; returns image values at those indices."""
    floor_coords = np.floor(coords).astype(int, copy=False)
    return image[floor_coords]


def _TransformImageUsingCoords(target_coords: NDArray,
                               source_coords: NDArray,
                               source_image: NDArray,
                               output_origin: NDArray[np.integer] | tuple[int, int] | None,
                               output_area: NDArray[np.integer] | tuple[float, float],
                               cval=0,
                               return_shared_memory: bool = False,
                               interpolation_order: int | None = None,
                               return_valid_mask: bool = False,
                               clamp_source_coords: bool = False):
    """Use the passed coordinates to create a warped image
    :Param fixed_coords: 2D coordinates in fixed space
    :Param warped_coords: 2D coordinates in warped space
    :Param FixedImageArea: Dimensions of fixed space
    :Param WarpedImage: Image to read pixel values from while creating fixed space images
    :Param output_origin: Origin, in target coordinate space, of the output image.  Use this to translate the target_coords to the desired location in the output image.  If None, the origin is the minimum target_coord.
    :Param output_area: Expected dimensions of output
    :Param cval: Value to place in unmappable regions, defaults to zero.
    :param use_shared_memory: If true, create and write output to a shared memory array
    :param interpolation_order: ``map_coordinates`` spline order (0=nearest). When None, use
        order 1 for NaN/bool inputs and cubic (3) otherwise.
    :param return_valid_mask: If True, return ``(image, valid_mask)`` where
        ``valid_mask`` is a boolean array of ``output_area`` that is True exactly
        at the output pixels that received a mapped source sample (the scatter
        targets). This is the true coverage of the warp, derived for free from
        the scatter indices, avoiding a second warp of a ones-image. Not
        supported together with ``return_shared_memory``.
    """

    if return_valid_mask and return_shared_memory:
        raise ValueError("return_valid_mask is not supported with return_shared_memory")

    use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy
    xp = cp if use_cp else np
    sp = cupyx.scipy if use_cp else scipy

    # cupyx.scipy.ndimage.map_coordinates (and xp.full) must not receive Python bool: NVRTC sees "False"/"True".
    if isinstance(cval, (bool, np.bool_)):
        cval = int(cval)

    if output_origin is None:
        output_origin = target_coords.min(0)

    output_area = nornir_imageregistration.EnsurePointsAre1DNumpyArray(output_area, dtype=np.int32)
    output_origin = nornir_imageregistration.EnsurePointsAre1DArray(output_origin, dtype=np.int32)  # type: ignore[arg-type]
    output_area_shape = tuple(int(x) for x in output_area.ravel())

    if use_cp:
        target_coords = nornir_imageregistration.EnsurePointsAre2DArray(target_coords)
        source_coords = nornir_imageregistration.EnsurePointsAre2DArray(source_coords)
        if not isinstance(source_image, cp.ndarray):
            source_image = cp.asarray(source_image)
        if cp.get_array_module(source_coords) is not cp:
            source_coords = cp.asarray(source_coords)
        if cp.get_array_module(target_coords) is not cp:
            target_coords = cp.asarray(target_coords)
        # Tuple/list origins are NumPy by Ensure* policy; promote for GPU warp math.
        output_origin = cp.asarray(output_origin, dtype=np.int32)
        output_area = cp.asarray(output_area, dtype=np.int32)
    else:
        if output_origin.dtype != np.int32:
            output_origin = np.asarray(output_origin, dtype=np.int32)
        if output_area.dtype != np.int32:
            output_area = np.asarray(output_area, dtype=np.int32)

    if source_coords.shape[0] == 0:
        # No points transformed into the requested area, return empty image
        transformedImage = xp.full(output_area_shape, cval, dtype=source_image.dtype)
        if return_valid_mask:
            return transformedImage, xp.zeros(transformedImage.shape, dtype=bool)
        return transformedImage

    # Convert to a type the interpolation.map_coordinates supports
    original_dtype = source_image.dtype

    inbounds_target_coords = target_coords - output_origin  # If tempted to use target_coords -= output_origin, don't.  It modifies input parameter and if called in a loop the next iteration will fail
    del target_coords
    # Remove coordinates that fall outside the output region
    # inbounds_target_coords, inbounds_target_coord_mask = get_valid_coords(target_coords, output_area)
    # inbounds_source_coords = source_coords[inbounds_target_coord_mask]

    subroi_warpedImage = None
    # For large images we only need a specific range of the image, but the entire image is passed through a spline filter by map_coordinates
    # In this case use only a subset of the warpedimage
    if np.prod(source_image.shape) > source_coords.shape[0]:
        # if not area[0] == FixedImageArea[0] and area[1] == FixedImageArea[1]:
        # if area[0] <= FixedImageArea[0] or area[1] <= FixedImageArea[1]:
        (subroi_warpedImage, filtered_source_coords, source_coord_mask) = _CropImageToFitCoords(source_image,  # type: ignore[misc]
                                                                                                source_coords,
                                                                                                padding=0, cval=cval)
        # subroi_warpedImage[] #Replace NaN entries with random values
        # Remove target coords that match removed source_coords
        inbounds_target_coords = inbounds_target_coords[source_coord_mask]

        # source_coords, source_coord_mask = get_valid_coords(coords=source_coords, image_shape=subroi_warpedImage.shape, origin=(1,1), area=(subroi_warpedImage.shape) - 1) #Remove one for padding
        if subroi_warpedImage.shape[0] == 0 or subroi_warpedImage.shape[1] == 0:
            # No points transformed into the requested area, return empty area
            if return_shared_memory:
                # output_area_shape, not output_area: under CuPy the latter is a device array,
                # which cannot be used as a host buffer shape. (#102)
                output_shared_mem_meta, outputImage = nornir_imageregistration.create_shared_memory_array(
                    output_area_shape,
                    dtype=original_dtype)
                outputImage.fill(cval)
                return output_shared_mem_meta
            else:
                empty_image = xp.full(output_area_shape, cval, dtype=original_dtype)
                if return_valid_mask:
                    return empty_image, xp.zeros(empty_image.shape, dtype=bool)
                return empty_image

        del source_image
    else:
        filtered_source_coords = source_coords
        # inbounds_target_coords = inbounds_target_coords
        subroi_warpedImage = source_image

    # del inbounds_target_coords

    # Use a dtype interpolation.map_coordinates supports
    if subroi_warpedImage.dtype == np.float16:
        subroi_warpedImage = subroi_warpedImage.astype(np.float32, copy=False)

    # Rounding helped solve a problem with image shift when using the CloughTocher interpolator with an identity function
    # filtered_source_coords = np.around(filtered_source_coords, 3)

    # Deferred: this is a full pass over the source plus a host sync, and several callers never
    # need the answer -- a caller that names an interpolation_order and warps with a sentinel cval
    # (the distance plane does both) used to pay for it and then discard it. Same lazy-stats
    # treatment as _underflow_assemble_log_msg below. (#110)
    _nan_answer: list[bool] = []

    def _source_has_nan() -> bool:
        if not _nan_answer:
            _nan_answer.append(bool(xp.any(xp.isnan(subroi_warpedImage))))
        return _nan_answer[0]

    if interpolation_order is None:
        # Any interpolation of NaN returns NaN so ensure we use order=1 when using NaN as a fill value.
        # dtype first so a bool source short-circuits before the isnan pass.
        order = 1 if subroi_warpedImage.dtype == bool or _source_has_nan() else 3
    else:
        order = int(interpolation_order)
    prefilter = order > 1
    _h, _w = subroi_warpedImage.shape[:2]
    scatter_in_bounds = (
            (filtered_source_coords[:, 0] >= 0) & (filtered_source_coords[:, 0] < _h) &
            (filtered_source_coords[:, 1] >= 0) & (filtered_source_coords[:, 1] < _w)
    )
    if clamp_source_coords:
        _last_y = max(0, _h - 1)
        _last_x = max(0, _w - 1)
        filtered_source_coords = xp.stack([
            xp.clip(filtered_source_coords[:, 0], 0, _last_y),
            xp.clip(filtered_source_coords[:, 1], 0, _last_x),
        ], axis=1)
        scatter_in_bounds = xp.ones(scatter_in_bounds.shape[0], dtype=bool)
    elif return_valid_mask:
        # Do not clamp out-of-bounds source coords to the edge (that smears border pixels).
        # Only sample pixels whose inverse map lands inside the source image.
        filtered_source_coords = filtered_source_coords[scatter_in_bounds]
        inbounds_target_coords = inbounds_target_coords[scatter_in_bounds]
        scatter_in_bounds = scatter_in_bounds[scatter_in_bounds]

    # Defer image stats until an underflow is actually logged in debug mode.
    # Eager min/max/mean/std here forced host sync on every CuPy warp.
    def _underflow_assemble_log_msg() -> str:
        min_val = float(subroi_warpedImage.min())
        max_val = float(subroi_warpedImage.max())
        mean_val = float(subroi_warpedImage.mean())
        std_val = float(xp.std(subroi_warpedImage))
        return (
            f"Underflow error assembling image.  min_val={min_val} max_val={max_val} "
            f"mean={mean_val} standardDev={std_val}"
        )

    with IgnoreUnderflow(_underflow_assemble_log_msg):
        outputValues = sp.ndimage.map_coordinates(subroi_warpedImage,  # type: ignore[union-attr]
                                                  filtered_source_coords.transpose(),
                                                  mode='constant',
                                                  order=order,
                                                  cval=cval,
                                                  prefilter=prefilter).astype(original_dtype, copy=False)

    del filtered_source_coords

    # Scipy's interpolation can infer values slightly outside the source data's range.
    cval_float = float(cval) if cval is not None else 0.0
    preserve_cval_sentinel = cval_float > 1.0
    # Do not be tempted to skip this for order <= 1 on the grounds that linear and nearest
    # interpolation cannot overshoot the source range. They cannot, but that is not all this
    # clip does: with mode='constant' the samples near the source border blend toward cval, so
    # they land outside the source range even at order 1. Gating on order changed the border
    # pixels of every fractional-offset warp. (#110)
    if not preserve_cval_sentinel:
        if _source_has_nan():
            nan_mask = xp.logical_not(xp.isnan(subroi_warpedImage))
            finite_source = subroi_warpedImage[nan_mask]
            min_val = finite_source.min()
            max_val = finite_source.max()
        else:
            min_val = subroi_warpedImage.min()
            max_val = subroi_warpedImage.max()
        cval_outside_source_range = cval_float < float(min_val) or cval_float > float(max_val)
        if cval_outside_source_range:
            in_bounds_values = outputValues[scatter_in_bounds]
            xp.clip(in_bounds_values, a_min=min_val, a_max=max_val, out=in_bounds_values)
            outputValues[scatter_in_bounds] = in_bounds_values
        else:
            xp.clip(outputValues, a_min=min_val, a_max=max_val, out=outputValues)

    # outputvalaues = my_cheesy_map_coordinates(subroi_warpedImage, filtered_source_coords.transpose())

    # outputImage = np.full(output_area, cval, dtype=original_dtype) #Use same DType as source_image for output, we are past the call to map_coordinates that cannot handle float16
    output_shared_mem_meta = None
    if return_shared_memory:
        output_shared_mem_meta, outputImage = nornir_imageregistration.create_shared_memory_array(
            output_area_shape,
            dtype=original_dtype)
        outputImage.fill(cval)
    else:
        outputImage = xp.full(output_area_shape, cval,
                              dtype=original_dtype)  # Use same DType as source_image for output, we are past the call to map_coordinates that cannot handle float16

    target_coords_flat = nornir_imageregistration.ravel_index(inbounds_target_coords, outputImage.shape).astype(  # type: ignore[arg-type]
        np.int32, copy=False)
    # del filtered_target_coords

    # Note - Clement: flat assignment doesn't work with cupy
    if cp.get_array_module(outputImage) == cp:  # type: ignore[comparison-overlap]
        start_shape = outputImage.shape
        outputImage = outputImage.ravel()
        outputImage[target_coords_flat] = outputValues
        outputImage = outputImage.reshape(start_shape)
    else:
        outputImage.flat[target_coords_flat] = outputValues
    # outputImage[fixed_coords] = outputValues

    # Coverage of the warp is exactly the set of output pixels that received a
    # mapped sample (the scatter targets); derive it from the same flat indices
    # instead of warping a separate ones-image. The canvas is already filled with
    # cval before scatter, so unmapped pixels need no second pass (#111).
    valid_mask = None
    if return_valid_mask:
        valid_mask = xp.zeros(int(np.prod(outputImage.shape)), dtype=bool)
        if target_coords_flat.shape[0] > 0:
            valid_mask[target_coords_flat] = True
        valid_mask = valid_mask.reshape(outputImage.shape)

    # outputImage = outputImage.reshape(area)

    #     if fixed_coords.shape[0] == np.prod(area):
    #         # All coordinates mapped, so we can return the output warped image as is.
    #         outputImage = outputImage.reshape(area)
    #         return outputImage
    #     else:
    #         # Not all coordinates mapped, create an image of the correct size and place the warped image inside it.
    #         transformedImage = np.full((area), cval, dtype=outputImage.dtype)
    #         fixed_coords_rounded = np.round(fixed_coords).astype(dtype=np.int64)
    #         transformedImage[fixed_coords_rounded[:, 0], fixed_coords_rounded[:, 1]] = outputImage
    #         return transformedImage
    if return_shared_memory:
        return output_shared_mem_meta
    elif return_valid_mask:
        return outputImage, valid_mask
    else:
        return outputImage


def _ReplaceFilesWithImages(listImages: list[str] | list[NDArray] | NDArray | str):
    """Replace any filepath strings in the passed parameter with loaded images."""

    if isinstance(listImages, list):
        for i, value in enumerate(listImages):
            listImages[i] = nornir_imageregistration.ImageParamToImageArray(value)  # type: ignore[index]
    else:
        return nornir_imageregistration.ImageParamToImageArray(listImages)

    return listImages


def FixedImageToWarpedSpace(transform: ITransform, DataToTransform, botleft=None, area=None, cval=None,
                            extrapolate=False):
    warnings.warn("FixedImageToWarpedSpace should be replaced with TargetImageToSourceSpace", DeprecationWarning)
    return TargetImageToSourceSpace(transform, DataToTransform, output_botleft=botleft, output_area=area, cval=cval,
                                    extrapolate=extrapolate)


def TargetImageToSourceSpace(transform: ITransform,
                             DataToTransform,
                             output_botleft: NDArray | tuple[float, float] | None = None,
                             output_area: NDArray | tuple[float, float] | None = None,
                             cval=None, extrapolate: bool = False, return_shared_memory: bool = False):
    """Warps every image in the DataToTransform list using the provided transform.
    :param transform: transform to pass fixed space coordinates through to obtain warped space coordinates
    :param DataToTransform: Images to read pixel values from while creating fixed space images.  A list of images can be passed to map multiple images using the same coordinates.  A list may contain filename strings or numpy.ndarrays
    :param output_botleft: Origin of region to map data into, in source Space coordinates
    :param output_area: Area of region to map data into, in source space coordinates
    :param cval: Value to place in unmappable regions, defaults to zero.
    :param bool extrapolate: If true map points that fall outside the bounding box of the transform
    """

    ImagesToTransform = _ReplaceFilesWithImages(DataToTransform)

    if output_botleft is None:
        output_botleft = (0, 0)

    if output_area is None:
        firstImage = ImagesToTransform
        if isinstance(firstImage, list):
            firstImage = firstImage[0]
            raise ValueError("Area calculation is not implemented for lists of transforms, but could be")

        bounds = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0, 0), firstImage.shape)
        source_corners = transform.InverseTransform(bounds.Corners)
        output_area = np.ravel((np.max(source_corners, 0) - np.min(source_corners, 0)) + 1)

    if cval is None:
        cval = [0] * len(DataToTransform)

    if not isinstance(cval, list):
        cval = [cval] * len(DataToTransform)

    # This sometimes appears backwards, but what we are doing is defining the region in source space we want to obtain
    # values for, then determining the target space coordinates for each pixel to fill the source image region.  Then we map values
    # to each pixel using the target space coordinates
    (roi_read_coords, roi_write_coords) = write_to_source_roi_coords(transform, output_botleft, output_area,
                                                                     extrapolate=extrapolate)

    if isinstance(ImagesToTransform, list):
        if not isinstance(cval, list):
            cval = [cval] * len(DataToTransform)

        output_list = []
        for i, wi in enumerate(ImagesToTransform):
            fi = _TransformImageUsingCoords(roi_write_coords, roi_read_coords, wi, output_origin=output_botleft,  # type: ignore[arg-type]
                                            output_area=output_area,
                                            cval=cval[i], return_shared_memory=return_shared_memory)
            output_list.append(fi)

        return output_list
    else:
        return _TransformImageUsingCoords(roi_write_coords, roi_read_coords, ImagesToTransform,
                                          output_origin=output_botleft, output_area=output_area,  # type: ignore[arg-type]
                                          cval=cval[0], return_shared_memory=return_shared_memory)


def WarpedImageToFixedSpace(transform: ITransform, DataToTransform, botleft=None, area=None, cval=None,
                            extrapolate=False):
    warnings.warn("WarpedImageToFixedSpace should be replaced with SourceImageToTargetSpace", DeprecationWarning)
    return SourceImageToTargetSpace(transform, DataToTransform, output_botleft=botleft, output_area=area, cval=cval,
                                    extrapolate=extrapolate)


def SourceImageToTargetSpace(transform: ITransform,
                             DataToTransform,
                             output_botleft: NDArray | tuple[float, float] | None = None,
                             output_area: NDArray | tuple[float, float] | None = None,
                             cval=None, extrapolate=False, return_shared_memory: bool = False,
                             return_valid_mask: bool = False, clamp_source_coords: bool = False,
                             interpolation_order: int | None = None):
    """Warps every image in the DataToTransform list using the provided transform.
    :param transform: transform to pass warped space coordinates through to obtain fixed space coordinates
    :param output_shape: shape of the output image
    :param DataToTransform: Images to read pixel values from while creating fixed space images.  A list of images can be passed to map multiple images using the same coordinates.  A list may contain filename strings or numpy.ndarrays
    :param output_botleft: Origin of region to map data into, in target space coordinates
    :param output_area: Area of region to map data into, in target space coordinates
    :param cval: Value to place in unmappable regions, defaults to zero.
    :Param transform: transform to pass warped space coordinates through to obtain fixed space coordinates
    :Param FixedImageArea: Size of fixed space region to map pixels into
    :Param DataToTransform: Images to read pixel values from while creating fixed space images.  A list of images can be passed to map multiple images using the same coordinates.  A list may contain filename strings or numpy.ndarrays
    :Param botleft: Origin of region to map
    :Param area: Expected dimensions of output
    :Param cval: Value to place in unmappable regions, defaults to zero.
    :param bool extrapolate: If true map points that fall outside the bounding box of the transform
    """

    ImagesToTransform = _ReplaceFilesWithImages(DataToTransform)

    if output_botleft is None:
        output_botleft = (0, 0)

    if output_area is None:
        firstImage = ImagesToTransform
        if isinstance(firstImage, list):
            firstImage = firstImage[0]
            raise ValueError("Area calculation is not implemented for lists of transforms, but could be")

        bounds = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0, 0), firstImage.shape)
        target_corners = transform.Transform(bounds.Corners)
        target_bounds = nornir_imageregistration.Rectangle.CreateBoundingRectangleForPoints(target_corners)
        rounded_target_bounds = nornir_imageregistration.Rectangle.SafeRound(target_bounds)
        # output_area = np.ravel((np.max(target_corners, 0) - np.min(target_corners, 0)))
        # output_area = np.ceil(output_area)
        output_area = rounded_target_bounds.Dimensions

    if cval is None:
        cval = 0

    # output_botleft = cp.asarray(output_botleft) if use_cp and not isinstance(output_botleft,
    #                                                                          cp.ndarray) else output_botleft
    # output_area = cp.asarray(output_area) if use_cp and not isinstance(output_area, cp.ndarray) else output_area

    # This sometimes appears backwards, but what we are doing is defining the region in target space we want to obtain
    # values for, then determining the Source space coordinates for each pixel in the target image.  Then we map values
    # to each pixel using the source space coordinates

    # timer.Start('write_to_target_roi_coords')

    (roi_read_coords, roi_write_coords) = write_to_target_roi_coords(transform, output_botleft, output_area,
                                                                     extrapolate=extrapolate)

    # Serialise GPU map_coordinates only when this warp's coordinates are on-device.
    # Do not key off GetActiveComputationLib() — coords follow GetROICoords / transform output type.
    _use_gpu_warp = cp.get_array_module(roi_write_coords) is cp
    _lock = _gpu_warp_lock if _use_gpu_warp else contextlib.nullcontext()
    if isinstance(ImagesToTransform, list):
        if not isinstance(cval, list):
            cval = [cval] * len(DataToTransform)

        output_list = []
        distance_warp_order = _assemble_distance_warp_order()
        with _lock:
            for i, wi in enumerate(ImagesToTransform):
                warp_order = None if i == 0 else distance_warp_order
                fi = _TransformImageUsingCoords(roi_write_coords, roi_read_coords, wi, output_origin=output_botleft,  # type: ignore[arg-type]
                                                output_area=output_area, cval=cval[i],
                                                return_shared_memory=return_shared_memory,
                                                return_valid_mask=return_valid_mask and i == 0,
                                                clamp_source_coords=clamp_source_coords,
                                                interpolation_order=warp_order if warp_order is not None else interpolation_order)
                output_list.append(fi)
                # nornir_imageregistration.close_shared_memory(DataToTransform[i])

        return output_list
    else:
        with _lock:
            result = _TransformImageUsingCoords(roi_write_coords, roi_read_coords, ImagesToTransform,
                                                output_origin=output_botleft, output_area=output_area, cval=cval,  # type: ignore[arg-type]
                                                return_shared_memory=return_shared_memory,
                                                return_valid_mask=return_valid_mask,
                                                clamp_source_coords=clamp_source_coords,
                                                interpolation_order=interpolation_order)
        # nornir_imageregistration.close_shared_memory(DataToTransform)
        return result


def _ParameterToStosTransformAndFile(
        transformData: str | NDArray | nornir_imageregistration.StosFile | nornir_imageregistration.ITransform
) -> tuple[ITransform | None, nornir_imageregistration.StosFile | None]:
    """Resolve *transformData* to a transform and, when one exists, its stos file.

    Callers that need the stos file's image paths cannot recover them from the
    transform alone, so both are returned together.
    """
    stos: nornir_imageregistration.StosFile | None = None
    stostransform: ITransform | None = None

    if isinstance(transformData, str):
        if not os.path.exists(transformData):
            raise ValueError("transformData is not a valid path to a .stos file %s" % transformData)
        stos = nornir_imageregistration.StosFile.Load(transformData)
        stostransform = factory.LoadTransform(stos.Transform)  # type: ignore[arg-type]
    elif isinstance(transformData, nornir_imageregistration.StosFile):
        stos = transformData
        # StosFile.Transform is already the IRTools transform string.
        stostransform = factory.LoadTransform(stos.Transform)  # type: ignore[arg-type]
    elif isinstance(transformData, ITransform):
        stostransform = transformData

    return stostransform, stos


def ParameterToStosTransform(
        transformData: str | NDArray | nornir_imageregistration.StosFile | nornir_imageregistration.ITransform):
    """
    :param object transformData: Either a full path to a .stos file, a stosfile, or a transform object
    :return: A transform
    """
    stostransform, _stos = _ParameterToStosTransformAndFile(transformData)
    return stostransform


def TransformStos(transformData, OutputFilename: str | None = None, fixedImage=None, warpedImage=None,
                  scalar: float = 1.0, CropUndefined: bool = False):
    """Assembles an image based on the passed transform.
    :param transformData:
    :param OutputFilename:
    :param str fixedImage: Image describing the size we want the warped image to fill, either a string or ndarray
    :param str warpedImage: Image we will warp into fixed space, either a string or ndarray
    :param float scalar: Amount to scale the transform before passing the image through
    :param bool CropUndefined: If true exclude areas outside the convex hull of the transform, if it exists
    """

    stostransform, stos = _ParameterToStosTransformAndFile(transformData)

    if fixedImage is None:
        if stos is None:
            return None

        fixedImage = stos.ControlImageFullPath

    if warpedImage is None:
        if stos is None:
            return None

        warpedImage = stos.MappedImageFullPath

    fixedImageSize = nornir_imageregistration.GetImageSize(fixedImage)
    fixedImageShape = np.array(fixedImageSize) * scalar
    warpedImage = nornir_imageregistration.ImageParamToImageArray(warpedImage)

    if isinstance(stostransform, nornir_imageregistration.transforms.ITransformScaling) is False:
        raise NotImplementedError(f"Cannot scale transform that does not implement ITransformScaling {transformData}")

    stostransform.Scale(scalar)  # type: ignore[attr-defined]

    warpedImage = TransformImage(stostransform, fixedImageShape, warpedImage, CropUndefined)

    if not OutputFilename is None:
        nornir_imageregistration.SaveImage(OutputFilename,
                                           warpedImage.get() if cp.get_array_module(warpedImage) == cp else warpedImage,  # type: ignore[union-attr, comparison-overlap]
                                           cmap='gray', bpp=8)

    return warpedImage


def _host_grid_division_from_grid(grid: nornir_imageregistration.ITKGridDivision,
                                  target_points: NDArray) -> nornir_imageregistration.ITKGridDivision:
    """Clone a grid division with host NumPy target points for CPU assembly."""
    host_grid = nornir_imageregistration.ITKGridDivision(
        source_shape=nornir_imageregistration.EnsureNumpyArray(grid.source_shape),
        cell_size=nornir_imageregistration.EnsureNumpyArray(grid.cell_size),
        grid_dims=nornir_imageregistration.EnsureNumpyArray(grid.grid_dims),
    )
    host_grid.TargetPoints = nornir_imageregistration.EnsureNumpyArray(target_points)
    return host_grid


def transform_for_host_assembly(transform: ITransform) -> ITransform:
    """Return a CPU/pickle-safe transform for tiled host assembly and image export."""
    from nornir_imageregistration.transforms.base import IControlPoints
    from nornir_imageregistration.transforms.gridtransform import GridTransform, GridTransform_GPUComponent
    from nornir_imageregistration.transforms.gridwithrbffallback import (
        GridWithRBFFallback,
        GridWithRBFFallback_GPUComponent,
    )
    from nornir_imageregistration.transforms.meshwithrbffallback import (
        MeshWithRBFFallback,
        MeshWithRBFFallback_GPUComponent,
    )
    from nornir_imageregistration.transforms.triangulation import Triangulation, Triangulation_GPUComponent

    if isinstance(transform, GridWithRBFFallback_GPUComponent):
        host_grid = _host_grid_division_from_grid(transform.grid, transform.TargetPoints)
        return GridWithRBFFallback(host_grid)

    if isinstance(transform, GridTransform_GPUComponent):
        host_grid = _host_grid_division_from_grid(transform.grid, transform.TargetPoints)
        return GridTransform(host_grid)

    if isinstance(transform, MeshWithRBFFallback_GPUComponent):
        return MeshWithRBFFallback(nornir_imageregistration.EnsureNumpyArray(transform.points))

    if isinstance(transform, Triangulation_GPUComponent):
        return Triangulation(nornir_imageregistration.EnsureNumpyArray(transform.points))

    if type(transform).__name__.endswith('_GPUComponent') and isinstance(transform, IControlPoints):
        points = nornir_imageregistration.EnsureNumpyArray(transform.points)
        return Triangulation(points)

    return transform


def TransformImage(transform: ITransform,
                   fixedImageShape: tuple[float, float] | NDArray,
                   warpedImage: NDArray, CropUndefined: bool,
                   interpolation_order: int | None = None,
                   extrapolate: bool | None = None,
                   enforce_background_cval: float | int | None = None) -> NDArray:
    """
    Cut image into tiles, assemble small chunks
    :param transform: Transform to apply to point to map from warped image to fixed space
    :param fixedImageShape: Width and Height of the image to create
    :param warpedImage: Image to transform to fixed space
    :param CropUndefined: If true exclude areas outside the convex hull of the transform, if it exists
    :param extrapolate: When set, controls whether transforms extrapolate outside their hull during assembly.
        Defaults to ``not CropUndefined`` when omitted.
    :param enforce_background_cval: When set (typically ``0`` for export), only scatter samples whose
        inverse map lands inside the source image; all other output pixels are set to this value.
    :return: An ndimage array of the transformed image
    """

    if CropUndefined:
        transform = triangulation.Triangulation(pointpairs=transform.points)  # type: ignore[attr-defined]

    transform = transform_for_host_assembly(transform)
    warpedImage = nornir_imageregistration.EnsureNumpyArray(warpedImage)
    working_dtype = _assembly_working_dtype(warpedImage.dtype)
    if warpedImage.dtype != working_dtype:
        warpedImage = warpedImage.astype(working_dtype, copy=False)

    tilesize = [2048, 2048]
    extrapolate_flag = extrapolate if extrapolate is not None else not CropUndefined

    fixedImageShape = fixedImageShape.astype(dtype=np.int64, copy=False)  # type: ignore[union-attr]
    height = int(fixedImageShape[0])
    width = int(fixedImageShape[1])

    # Both branches below allocate the whole canvas, so check it once here. Without
    # this an exploded transform reached np.zeros and surfaced as a bare MemoryError,
    # while the tile-compositing path refused the same canvas with a diagnosis.
    _raise_if_assemble_buffer_too_large(
        height, width, _assembly_output_dtype(warpedImage.dtype) or warpedImage.dtype,
        include_zbuffer=False)

    # print('\nConverting image to ' + str(self.NumCols) + "x" + str(self.NumRows) + ' grid of OpenGL textures')

    grid_shape = nornir_imageregistration.TileGridShape(warpedImage.shape, tilesize)  # type: ignore[arg-type]
    warp_kwargs: dict = {
        'extrapolate': extrapolate_flag,
        'interpolation_order': interpolation_order,
    }
    if enforce_background_cval is not None:
        warp_kwargs['cval'] = enforce_background_cval
        warp_kwargs['return_valid_mask'] = True

    if np.all(grid_shape == np.array([1, 1])):
        # Single threaded
        result = SourceImageToTargetSpace(
            transform,
            warpedImage,
            output_botleft=np.array([0, 0]),
            output_area=fixedImageShape,
            **warp_kwargs,
        )
        # The warp already reports its own coverage when a cval is enforced, so
        # take that mask rather than recomputing one. Rebuilding it via
        # assembly_source_sample_mask ran a second whole-canvas inverse transform
        # for a mask that was pixel-identical to this one, costing 27-51% of the
        # warp itself and tens of coordinate-array MiB per call.
        sample_mask = None
        if isinstance(result, tuple):
            sample_mask = result[1]
            result = result[0]
        output = nornir_imageregistration.EnsureNumpyArray(
            result,
            dtype=_assembly_output_dtype(warpedImage.dtype),
        )
        if enforce_background_cval is not None:
            if sample_mask is None:
                sample_mask = assembly_source_sample_mask(
                    transform,
                    fixedImageShape,
                    warpedImage.shape[:2],
                    extrapolate=extrapolate_flag,
                )
            sample_mask = nornir_imageregistration.EnsureNumpyArray(sample_mask)
            # No copy: SourceImageToTargetSpace allocates its own warp output, so
            # this buffer is never the caller's warpedImage. Copying here doubled
            # peak memory for a full section.
            output[~sample_mask] = enforce_background_cval
        return output  # type: ignore[return-value]
    else:
        output_dtype = _assembly_output_dtype(warpedImage.dtype) or warpedImage.dtype
        outputImage = np.zeros(fixedImageShape, dtype=output_dtype)
        sharedwarpedimage_metadata, sharedWarpedImage = nornir_imageregistration.npArrayToSharedArray(warpedImage)
        mpool = nornir_pools.GetGlobalLocalMachinePool()

        try:
            # Accumulate the coverage each tile already reports, instead of
            # recomputing it for the whole canvas after the warp has finished.
            # A canvas of bool costs 1 byte per pixel; the recompute built two
            # float64 Nx2 coordinate arrays, 32 bytes per pixel, to arrive at the
            # same mask.
            tiled_sample_mask = None
            tiles_reporting_coverage = 0
            tiles_submitted = 0
            if enforce_background_cval is not None:
                tiled_sample_mask = np.zeros(tuple(int(v) for v in fixedImageShape), dtype=bool)

            def _tile_regions():
                for region_iY in range(0, height, int(tilesize[0])):
                    region_end_iY = min(region_iY + int(tilesize[0]), height)
                    for region_iX in range(0, width, int(tilesize[1])):
                        region_end_iX = min(region_iX + int(tilesize[1]), width)
                        yield region_iY, region_end_iY, region_iX, region_end_iX

            def _submit_next() -> bool:
                nonlocal tiles_submitted
                region = next(tile_regions, None)
                if region is None:
                    return False

                iY, end_iY, iX, end_iX = region
                # return_shared_memory must stay False here. Shared memory
                # works parent->worker (sharedwarpedimage_metadata above) but
                # not worker->parent: the segment is registered in the
                # creating process, so on Windows it is destroyed when the
                # worker task returns and the parent's attach fails with
                # FileNotFoundError. unlink_shared_memory would also no-op,
                # since it only unlinks names this process allocated.
                # Pickling the tile back costs ~1% of the tile's own warp
                # (5.5 ms transfer vs 569 ms warp for 2048x2048 float32).
                task = mpool.add_task(str(iX) + "x_" + str(iY) + "y", SourceImageToTargetSpace, transform,
                                      sharedwarpedimage_metadata, output_botleft=[iY, iX],
                                      output_area=[end_iY - iY, end_iX - iX],
                                      return_shared_memory=False,
                                      **warp_kwargs)
                task.iY = iY  # type: ignore[attr-defined]
                task.end_iY = end_iY  # type: ignore[attr-defined]
                task.iX = iX  # type: ignore[attr-defined]
                task.end_iX = end_iX  # type: ignore[attr-defined]
                pending.append(task)
                tiles_submitted += 1
                return True

            # Submit a bounded window rather than every tile up front. Waiting for
            # the whole pool before reading the first result left every warped tile
            # resident in the parent at once, so peak grew with the tile count
            # instead of the worker count: 16 of 16 tiles alive on a 4x4 canvas, at
            # ~16 MiB per 2048x2048 float32 tile. Each tile is pickled back as a
            # plain array (shared memory cannot travel worker->parent, see above),
            # so the parent's copy is the cost being bounded. Roughly one queued
            # tile per worker still keeps the pool saturated, and collection stays
            # in submission order, so the output is unchanged.
            tile_regions = _tile_regions()
            pool_workers = getattr(mpool, 'max_workers', None) or os.cpu_count() or 4
            max_in_flight = max(2, int(pool_workers) * 2)
            pending: collections.deque = collections.deque()

            while len(pending) < max_in_flight and _submit_next():
                pass

            while pending:
                task = pending.popleft()
                # Refill before blocking, so a worker is never left idle while the
                # parent waits on the oldest tile.
                _submit_next()
                result = task.wait_return()
                if result is None:
                    raise RuntimeError(f"Multiprocess tile assembly failed for task {task.name}")
                tile_mask = None
                if isinstance(result, tuple):
                    tile_mask = result[1]
                    result = result[0]
                registered_tile = nornir_imageregistration.EnsureNumpyArray(
                    nornir_imageregistration.ImageParamToImageArray(result))
                outputImage[task.iY:task.end_iY, task.iX:task.end_iX] = registered_tile
                if tiled_sample_mask is not None and tile_mask is not None:
                    tiled_sample_mask[task.iY:task.end_iY, task.iX:task.end_iX] = \
                        nornir_imageregistration.EnsureNumpyArray(tile_mask)
                    tiles_reporting_coverage += 1
                # No unlink_shared_memory here: tasks return plain ndarrays, so the
                # call was a silent no-op left over from an earlier shared-memory
                # return path. Dropping the tile reference is the actual release.
                del registered_tile, result
        finally:
            nornir_imageregistration.unlink_shared_memory(sharedwarpedimage_metadata)
            del sharedWarpedImage

    outputImage = nornir_imageregistration.EnsureNumpyArray(outputImage, dtype=_assembly_output_dtype(warpedImage.dtype))
    if enforce_background_cval is not None:
        sample_mask = tiled_sample_mask
        if sample_mask is not None and tiles_reporting_coverage != tiles_submitted:
            # Partial coverage reports would leave un-reported tiles looking
            # unmapped, so only trust the accumulated mask when every tile
            # contributed one.
            sample_mask = None
        if sample_mask is None:
            # Tiles did not report coverage; fall back to deriving it.
            sample_mask = assembly_source_sample_mask(
                transform,
                fixedImageShape,
                warpedImage.shape[:2],
                extrapolate=extrapolate_flag,
            )
        # No copy: outputImage is the locally allocated tile-assembly buffer.
        outputImage[~sample_mask] = enforce_background_cval
    return outputImage


def _assembly_working_dtype(source_dtype: np.dtype) -> np.dtype:
    """Use float32 scratch buffers when source tiles are float16."""
    if source_dtype == np.float16:
        return np.float32
    return source_dtype


def _assembly_output_dtype(source_dtype: np.dtype) -> np.dtype | None:
    """Pick a host dtype for assembled export; float16 is promoted to float32 for save/interop."""
    if source_dtype == np.float16:
        return np.float32
    return None
