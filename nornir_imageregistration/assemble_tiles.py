"""
Created on Oct 28, 2013

Deals with assembling images composed of mosaics or dividing images into tiles
"""

import copy
from collections import deque
import contextlib
import logging
import multiprocessing
import os
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Deque, Iterable, List, Optional, Tuple
import weakref

import numpy as np
from numpy.typing import DTypeLike, NDArray

import nornir_imageregistration
import nornir_imageregistration.assemble as assemble
import nornir_imageregistration.transformed_image_data
import nornir_imageregistration.transformed_image_data_temp_files
import nornir_pools
import nornir_shared.prettyoutput as prettyoutput
import nornir_shared.tasktimer

from nornir_imageregistration.type_info import ShapeLike
from nornir_imageregistration.image_filter_cache import WindowFilterCache
from nornir_imageregistration.distance import CreateDistanceImage

# from nornir_imageregistration.files.mosaicfile import MosaicFile
# from nornir_imageregistration.mosaic import Mosaic
# import nornir_imageregistration.transforms.meshwithrbffallback as meshwithrbffallback
# import nornir_imageregistration.transforms.triangulation as triangulation


distance_image_cache = WindowFilterCache('distance', CreateDistanceImage)

_DEFAULT_MAX_ASSEMBLE_BUFFER_BYTES = 16 * 1024 * 1024 * 1024

# Tile prefetch tuning (network/NAS-optimised defaults for 2 GB/core systems).
# _PREFETCH_WORKERS: threads that read tiles from disk/network concurrently.
#   8 simultaneous reads saturates a typical 10GbE NAS; lower to 4 if the
#   NAS shows contention, raise toward 16 if the GPU still stalls.
# _PREFETCH_DEPTH: tiles kept in-flight ahead of the current GPU position.
#   2× worker count so the queue is never empty even under high latency.
#   Worst-case host RAM cost = _PREFETCH_DEPTH × tile_size
#   (e.g. 16 × 32 MB = 512 MB for 4096×4096 uint16 tiles).
_PREFETCH_WORKERS: int = 8
_PREFETCH_DEPTH: int = 16
# GPU tile warps: overlap per-tile CPU prep (inverse transform, deepcopy) while
# map_coordinates stays serialised via _gpu_warp_lock in assemble.SourceImageToTargetSpace.
_TRANSFORM_WORKERS: int = max(2, min(os.cpu_count() or 4, 8))

_distance_cache_lock = threading.Lock()
_composite_lock = threading.Lock()

# Per-assemble-pass cache of scaled transforms keyed by (tile_id, source_scale, target_scale).
# Cleared at the end of TilesToImage so InverseInterpolator built on first use is reused
# if TransformTile is invoked again for the same tile and scales within one assemble.
_scaled_transform_assemble_cache: dict[tuple[int, float, float], nornir_imageregistration.ITransform] | None = None


def _max_assemble_buffer_bytes() -> int:
    """Return the maximum allowed assemble output buffer size in bytes."""
    raw = os.environ.get('NORNIR_MAX_ASSEMBLE_BUFFER_BYTES')
    if raw is not None and raw.strip() != '':
        return int(raw)
    return _DEFAULT_MAX_ASSEMBLE_BUFFER_BYTES


def _raise_if_assemble_buffer_too_large(height: int, width: int, dtype: DTypeLike) -> None:
    """Fail fast before allocating an unreasonably large assemble canvas."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Assemble output dimensions must be positive, got {height}x{width}")

    image_bytes = int(height) * int(width) * int(np.dtype(dtype).itemsize)
    zbuffer_bytes = int(height) * int(width) * int(np.dtype(np.float16).itemsize)
    total_bytes = image_bytes + zbuffer_bytes
    limit_bytes = _max_assemble_buffer_bytes()
    if total_bytes > limit_bytes:
        raise ValueError(
            f"Refusing to allocate {total_bytes:,} bytes for assemble output "
            f"({width}x{height}, image dtype={np.dtype(dtype)}, limit={limit_bytes:,} from "
            "NORNIR_MAX_ASSEMBLE_BUFFER_BYTES). This usually indicates invalid mosaic transforms "
            "with exploded target-space control points; regenerate the grid transform or inspect "
            "per-tile target bounding boxes before assembling.")


# TODO: Use atexit to delete the temporary files
# TODO: use_memmap does not work when assembling tiles on a cluster, disable for now.  Specific test is IDOCTests.test_AssembleTilesIDoc
def _use_memmap() -> bool:
    return False


nextNumpyMemMapFilenameIndex = 0


def GetProcessAndThreadUniqueString():
    """We use the index because if the same thread makes a new tile of the same size and the original has not been garbage collected yet we get errors"""
    global nextNumpyMemMapFilenameIndex
    nextNumpyMemMapFilenameIndex += 1
    return "%d_%d_%d" % (os.getpid(), threading.get_ident(), nextNumpyMemMapFilenameIndex)


def CompositeImage(FullImage, SubImage, offset):
    minX = offset[1]
    minY = offset[0]
    maxX = minX + SubImage.shape[1]
    maxY = minY + SubImage.shape[0]

    iNonZero = SubImage > 0.0

    temp = FullImage[minY:maxY, minX:maxX]

    temp[iNonZero] = SubImage[iNonZero]

    # FullImage[minY:maxY, minX:maxX] += SubImage
    FullImage[minY:maxY, minX:maxX] = temp

    return FullImage


def CompositeImageWithZBuffer(FullImage, FullZBuffer, SubImage, SubZBuffer, offset):
    canvas_h, canvas_w = FullImage.shape[:2]
    minX = int(offset[1])
    minY = int(offset[0])
    maxX = minX + SubImage.shape[1]
    maxY = minY + SubImage.shape[0]

    src_y0 = 0
    src_x0 = 0
    if minY < 0:
        src_y0 = -minY
        minY = 0
    if minX < 0:
        src_x0 = -minX
        minX = 0
    maxY = min(canvas_h, maxY)
    maxX = min(canvas_w, maxX)
    if minY >= maxY or minX >= maxX:
        return

    sub_image = SubImage[src_y0:src_y0 + (maxY - minY), src_x0:src_x0 + (maxX - minX)]
    sub_zbuffer = SubZBuffer[src_y0:src_y0 + (maxY - minY), src_x0:src_x0 + (maxX - minX)]

    if (np.array([maxY - minY, maxX - minX]) != sub_zbuffer.shape).any():
        raise ValueError("Buffers do not have the same dimensions")

    full_slice = FullZBuffer[minY:maxY, minX:maxX]
    # Strict > so uninitialized/sentinel z-buffer (== max) is not replaced by invalid tile margins.
    # Skip zero-valued samples so padding holes do not block overlapping neighbors.
    iUpdate = (full_slice > sub_zbuffer) & (sub_image != 0)
    FullImage[minY:maxY, minX:maxX][iUpdate] = sub_image[iUpdate]
    full_slice[iUpdate] = sub_zbuffer[iUpdate]

    return


def __MaxZBufferValue(dtype):
    return np.finfo(dtype).max


def EmptyDistanceBuffer(shape: ShapeLike, dtype: DTypeLike | None = None):
    dtype = np.float16 if dtype is None else dtype

    xp = nornir_imageregistration.GetComputationModule()

    if _use_memmap():  # use_memmap:
        full_distance_image_array_path = os.path.join(nornir_imageregistration.gettempdir(),
                                                      'distance_image_%dx%d_%s.npy' % (
                                                          shape[0], shape[1], GetProcessAndThreadUniqueString()))
        fullImageZbuffer = np.memmap(full_distance_image_array_path, dtype=dtype, mode='w+', shape=shape)  # type: ignore[call-overload]
        fullImageZbuffer.fill(__MaxZBufferValue(dtype))
        return fullImageZbuffer
        # fullImageZbuffer = np.memmap(full_distance_image_array_path, dtype=np.float16, mode='r+', shape=shape)
    else:
        return xp.full(shape, __MaxZBufferValue(dtype), dtype=dtype)


#
# def __CreateOutputBufferForTransforms(transforms, target_space_scale=None):
#     '''Create output images using the passed rectangle
#     :param tuple rectangle: (minY, minX, maxY, maxX)
#     :return: (fullImage, ZBuffer)
#     '''
#     fullImage = None
#     fixed_bounding_box = tutils.FixedBoundingBox(transforms)
#     (maxY, maxX) = fixed_bounding_box.shape
#     fullImage_shape = (int(np.ceil(target_space_scale * maxY)), int(np.ceil(target_space_scale * maxX)))
# 
#     if use_memmap:
#         try:
#             fullimage_array_path = os.path.join(tempfile.gettempdir(), 'image_%dx%d_%s.npy' % (fullImage_shape[0], fullImage_shape[1], GetProcessAndThreadUniqueString()))
#             fullImage = np.memmap(fullimage_array_path, dtype=np.float16, mode='w+', shape=fullImage_shape)
#             fullImage[:] = 0
#             fullImage.flush()
#             del fullImage
#             fullImage = np.memmap(fullimage_array_path, dtype=np.float16, mode='r+', shape=fullImage_shape)
#         except: 
#             prettyoutput.LogErr("Unable to open memory mapped file %s." % (fullimage_array_path))
#             raise 
#     else:
#         fullImage = np.zeros(fullImage_shape, dtype=np.float16)
# 
#     fullImageZbuffer = EmptyDistanceBuffer(fullImage.shape, dtype=fullImage.dtype)
#     return (fullImage, fullImageZbuffer)


def __CreateOutputBufferForArea(Height: int, Width: int, dtype: DTypeLike):
    """Create output images using the passed width and height."""
    _raise_if_assemble_buffer_too_large(int(Height), int(Width), dtype)

    fullImage = None
    fullImage_shape = (int(Height), int(Width))

    if _use_memmap():  # use_memmap:
        try:
            fullimage_array_path = os.path.join(nornir_imageregistration.gettempdir(), 'image_%dx%d_%s.npy' % (
                fullImage_shape[0], fullImage_shape[1], GetProcessAndThreadUniqueString()))
            fullImage = np.memmap(fullimage_array_path, dtype=dtype, mode='w+', shape=fullImage_shape)
            fullImage.fill(0)
            finalizer = weakref.finalize(fullImage, os.remove, fullimage_array_path)
        except:
            prettyoutput.LogErr("Unable to open memory mapped file %s." % fullimage_array_path)
            raise
        fullImageZbuffer = EmptyDistanceBuffer(fullImage.shape)
    else:
        xp = nornir_imageregistration.GetComputationModule()
        fullImage = xp.zeros(fullImage_shape, dtype=dtype)
        fullImageZbuffer = EmptyDistanceBuffer(fullImage.shape)

    return fullImage, fullImageZbuffer


def _prefetch_tile_image(tile: nornir_imageregistration.tile.Tile) -> None:
    """Access tile.Image so it is loaded and cached before the main thread needs it."""
    _ = tile.Image


def _assemble_prefetch_enabled() -> bool:
    """Return True when TilesToImage should prefetch upcoming tile PNGs on a thread pool."""
    raw = os.environ.get('NORNIR_ASSEMBLE_PREFETCH', '').strip().lower()
    if raw in ('0', 'false', 'no'):
        return False
    if raw in ('1', 'true', 'yes'):
        return True
    return (nornir_imageregistration.GetActiveComputationLib()
            == nornir_imageregistration.ComputationLib.cupy)


def _assemble_grid_extrapolate() -> bool:
    """Whether tile warps request RBF/grid extrapolation outside the discrete mesh."""
    raw = os.environ.get('NORNIR_ASSEMBLE_GRID_EXTRAPOLATE', '').strip().lower()
    if raw in ('0', 'false', 'no'):
        return False
    if raw in ('1', 'true', 'yes'):
        return True
    # Production default: skip RBF extrapolation for assemble; edge tiles are rare and costly.
    return False


@contextlib.contextmanager
def _scaled_transform_cache_scope():
    """Enable scaled-transform reuse for the duration of one TilesToImage pass."""
    global _scaled_transform_assemble_cache
    prior = _scaled_transform_assemble_cache
    _scaled_transform_assemble_cache = {}
    try:
        yield
    finally:
        _scaled_transform_assemble_cache = prior


def _scale_key(source_space_scale: float, target_space_scale: float) -> tuple[float, float]:
    return (float(source_space_scale), float(target_space_scale))


def _build_scaled_transform(
        base_transform: nornir_imageregistration.ITransform,
        source_space_scale: float,
        target_space_scale: float) -> nornir_imageregistration.ITransform:
    """Return a transform scaled for the requested source/target pyramid levels."""
    transform = base_transform
    if source_space_scale == target_space_scale:
        if source_space_scale != 1.0:
            scaled_transform = __CreateScalableTransformCopy(base_transform)
            scaled_transform.Scale(source_space_scale)
            transform = scaled_transform
    else:
        if source_space_scale != 1.0:
            scaled_transform = __CreateScalableTransformCopy(base_transform)
            scaled_transform.ScaleWarped(source_space_scale)  # type: ignore[attr-defined]
            transform = scaled_transform

        if target_space_scale != 1.0:
            scaled_transform = __CreateScalableTransformCopy(base_transform)
            scaled_transform.ScaleFixed(target_space_scale)  # type: ignore[attr-defined]
            transform = scaled_transform
    return transform


def _get_scaled_transform_for_tile(
        tile: nornir_imageregistration.tile.Tile,
        source_space_scale: float,
        target_space_scale: float) -> nornir_imageregistration.ITransform:
    """Return a scaled transform, reusing the per-assemble cache when active."""
    if source_space_scale == 1.0 and target_space_scale == 1.0:
        return tile.Transform

    cache = _scaled_transform_assemble_cache
    if cache is not None:
        key = (tile.ID, *_scale_key(source_space_scale, target_space_scale))
        cached = cache.get(key)
        if cached is not None:
            return cached
        scaled = _build_scaled_transform(tile.Transform, source_space_scale, target_space_scale)
        cache[key] = scaled
        return scaled

    return _build_scaled_transform(tile.Transform, source_space_scale, target_space_scale)


def TilesToImage(mosaic_tileset: nornir_imageregistration.MosaicTileset,
                 TargetRegion: nornir_imageregistration.Rectangle | List[float] | None = None,
                 target_space_scale: float | None = None,
                 use_cp: bool = False) -> Tuple[NDArray | None, NDArray | None]:
    """
    Generate an image of the TargetRegion.
    :param MosaicTileset mosaic_tileset: Tileset to assemble
    :param tuple TargetRegion: (MinX, MinY, Width, Height) or Rectangle class.  Specifies the SourceSpace to render from
    :param float target_space_scale: Scalar for the target space coordinates.  Used to downsample or upsample the output image.  Changes the coordinates of the target space control points of the transform. 
    :param use_cp: use CuPy library for GPU processing
    """

    if target_space_scale is not None and target_space_scale > 1.0:
        raise ValueError(
            "It isn't impossible this is what the caller requests, but this value expands the resulting image beyond full resolution of the transform.")

    # logger = logging.getLogger(__name__ + '.TilesToImage')
    source_space_scale = 1.0 / mosaic_tileset.image_to_source_space_scale
    if target_space_scale is None:
        target_space_scale = source_space_scale

    distanceImage = None
    original_fixed_rect_floats = None

    if TargetRegion is not None:
        if isinstance(TargetRegion, nornir_imageregistration.Rectangle):
            original_fixed_rect_floats = TargetRegion
        else:
            original_fixed_rect_floats = nornir_imageregistration.Rectangle.CreateFromPointAndArea(
                (TargetRegion[0], TargetRegion[1]),
                (TargetRegion[2] - TargetRegion[0], TargetRegion[3] - TargetRegion[1]))
    else:
        # We could use mosaic_tileset.TargetBoundingBox, but for mosaic-to-volume
        # transforms the non-zero origin is important, so we always use an origin
        # of 0, 0 and the max coordinates of the target bounding box
        # original_fixed_rect_floats = mosaic_tileset.TargetBoundingBox #Breaks mosaic-to-volume image assembly
        original_fixed_rect_floats = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0, 0),
                                                                                               mosaic_tileset.TargetBoundingBox.TopRight)

    targetRect = nornir_imageregistration.Rectangle.SafeRound(original_fixed_rect_floats)
    scaled_targetRect = nornir_imageregistration.Rectangle.scale_on_origin(original_fixed_rect_floats,
                                                                           target_space_scale)
    scaled_targetRect = nornir_imageregistration.Rectangle.SafeRound(scaled_targetRect)

    tiles_list = list(mosaic_tileset.values())
    if not tiles_list:
        raise ValueError("Mosaic Tileset has no tiles.")

    output_dtype = tiles_list[0].Image.dtype  # loads and caches tile[0].Image

    (fullImage, fullImageZbuffer) = __CreateOutputBufferForArea(int(scaled_targetRect.Height), int(scaled_targetRect.Width),
                                                                dtype=output_dtype)

    work_items: List[Tuple[nornir_imageregistration.tile.Tile, nornir_imageregistration.Rectangle]] = []
    for tile in tiles_list:
        region_to_render = nornir_imageregistration.Rectangle.Intersect(targetRect, tile.TargetSpaceBoundingBox)
        if region_to_render is not None and region_to_render.Area > 0:
            work_items.append((tile, region_to_render))

    prefetch_enabled = _assemble_prefetch_enabled()
    prefetch_executor: ThreadPoolExecutor | None = None
    prefetch_futures: dict[int, Future[None]] = {}
    if prefetch_enabled and work_items:
        prefetch_executor = ThreadPoolExecutor(max_workers=_PREFETCH_WORKERS)

        def _schedule_prefetch(index: int) -> None:
            if index >= len(work_items) or index in prefetch_futures:
                return
            tile_to_load = work_items[index][0]
            prefetch_futures[index] = prefetch_executor.submit(_prefetch_tile_image, tile_to_load)

        for prefetch_index in range(min(_PREFETCH_DEPTH, len(work_items))):
            _schedule_prefetch(prefetch_index)

    try:
        with _scaled_transform_cache_scope():
            for work_index, (tile, regionToRender) in enumerate(work_items):
                if prefetch_enabled and prefetch_executor is not None:
                    pending = prefetch_futures.pop(work_index, None)
                    if pending is not None:
                        pending.result()
                    _schedule_prefetch(work_index + _PREFETCH_DEPTH)

                global distance_image_cache
                distanceImage = distance_image_cache.KeepGetOrCreate(distanceImage, tile.ImageSize)  # type: ignore[arg-type]

                transformedImageData = TransformTile(tile, distanceImage, target_space_scale=target_space_scale,
                                                     TargetRegion=regionToRender, SingleThreadedInvoke=True)
                try:
                    transformed_image = transformedImageData.image
                    transformed_distance = transformedImageData.centerDistanceImage
                except ValueError:
                    prettyoutput.LogErr('Convert task failed: ' + str(transformedImageData))
                    if transformedImageData.errormsg is not None:
                        prettyoutput.LogErr(transformedImageData.errormsg)
                    continue

                CompositeOffset = (
                    transformedImageData.rendered_target_space_origin * transformedImageData.target_space_scale
                ) - scaled_targetRect.BottomLeft  # type: ignore[operator]
                CompositeOffset = CompositeOffset.astype(np.int64)

                CompositeImageWithZBuffer(fullImage, fullImageZbuffer,
                                          transformed_image, transformed_distance,
                                          CompositeOffset)

                del transformedImageData
    finally:
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True)

    if isinstance(fullImage, np.memmap):
        xp = np
    else:
        xp = nornir_imageregistration.GetComputationModule()
    mask = xp.less(fullImageZbuffer, __MaxZBufferValue(fullImageZbuffer.dtype))
    del fullImageZbuffer

    fullImage = xp.maximum(fullImage, 0, out=fullImage)
    # Checking for > 1.0 makes sense for floating point images.  During the DM4 migration
    # I was getting images which used 0-255 values, and the 1.0 check set them to entirely black
    # fullImage[fullImage > 1.0] = 1.0

    if isinstance(fullImage, np.memmap):
        fullImage.flush()
    elif hasattr(fullImage, 'get'):
        fullImage = fullImage.get()
        if hasattr(mask, 'get'):
            mask = mask.get()

    return fullImage, mask


def _composite_transformed_tile_onto_canvas(
        transformedImageData: nornir_imageregistration.transformed_image_data.ITransformedImageData,
        fullImage: NDArray,
        fullImageZbuffer: NDArray,
        scaled_targetRect: nornir_imageregistration.Rectangle) -> None:
    """Composite one warped tile into the output accumulation buffers."""
    if transformedImageData.errormsg is not None:
        prettyoutput.LogErr('Convert task failed: ' + str(transformedImageData))
        prettyoutput.LogErr(transformedImageData.errormsg)
        return
    try:
        transformed_image = transformedImageData.image
        transformed_distance = transformedImageData.centerDistanceImage
    except ValueError:
        prettyoutput.LogErr('Convert task failed: ' + str(transformedImageData))
        if transformedImageData.errormsg is not None:
            prettyoutput.LogErr(transformedImageData.errormsg)
        return

    composite_offset = (
        transformedImageData.rendered_target_space_origin * transformedImageData.target_space_scale
    ) - scaled_targetRect.BottomLeft  # type: ignore[operator]
    composite_offset = composite_offset.astype(np.int64)

    CompositeImageWithZBuffer(fullImage, fullImageZbuffer,
                              transformed_image, transformed_distance,
                              composite_offset)


def _transform_tile_worker(
        tile: nornir_imageregistration.tile.Tile,
        region_to_render: nornir_imageregistration.Rectangle,
        target_space_scale: float,
) -> nornir_imageregistration.transformed_image_data.ITransformedImageData:
    """Run TransformTile in a worker thread (I/O + inverse + GPU warp under _gpu_warp_lock)."""
    with _distance_cache_lock:
        distance_image = distance_image_cache.KeepGetOrCreate(None, tile.ImageSize)  # type: ignore[arg-type]
    return TransformTile(tile, distance_image, target_space_scale=target_space_scale,
                         TargetRegion=region_to_render, SingleThreadedInvoke=True)


def TilesToImageThreaded(mosaic_tileset: nornir_imageregistration.MosaicTileset,
                         TargetRegion: nornir_imageregistration.Rectangle | List[float] | None = None,
                         target_space_scale: float | None = None,
                         use_cp: bool = False) -> Tuple[NDArray | None, NDArray | None]:
    """GPU-oriented assemble: thread-parallel per-tile pipeline with serialised GPU warps.

    Workers overlap disk I/O and CPU-side inverse-transform work while
    ``assemble._gpu_warp_lock`` serialises ``map_coordinates`` dispatches.
    Compositing into the shared output canvas is serialised by ``_composite_lock``.
    """
    if target_space_scale is not None and target_space_scale > 1.0:
        raise ValueError(
            "It isn't impossible this is what the caller requests, but this value expands the resulting image beyond full resolution of the transform.")

    source_space_scale = 1.0 / mosaic_tileset.image_to_source_space_scale
    if target_space_scale is None:
        target_space_scale = source_space_scale

    if TargetRegion is not None:
        if isinstance(TargetRegion, nornir_imageregistration.Rectangle):
            original_fixed_rect_floats = TargetRegion
        else:
            original_fixed_rect_floats = nornir_imageregistration.Rectangle.CreateFromPointAndArea(
                (TargetRegion[0], TargetRegion[1]),
                (TargetRegion[2] - TargetRegion[0], TargetRegion[3] - TargetRegion[1]))
    else:
        original_fixed_rect_floats = nornir_imageregistration.Rectangle.CreateFromPointAndArea(
            (0, 0), mosaic_tileset.TargetBoundingBox.TopRight)

    target_rect = nornir_imageregistration.Rectangle.SafeRound(original_fixed_rect_floats)
    scaled_target_rect = nornir_imageregistration.Rectangle.scale_on_origin(original_fixed_rect_floats,
                                                                            target_space_scale)
    scaled_target_rect = nornir_imageregistration.Rectangle.SafeRound(scaled_target_rect)

    tiles_list = list(mosaic_tileset.values())
    if not tiles_list:
        raise ValueError("Mosaic Tileset has no tiles.")

    output_dtype = tiles_list[0].Image.dtype

    full_image, full_image_zbuffer = __CreateOutputBufferForArea(
        int(scaled_target_rect.Height), int(scaled_target_rect.Width), dtype=output_dtype)

    work_items: List[Tuple[nornir_imageregistration.tile.Tile, nornir_imageregistration.Rectangle]] = []
    for tile in tiles_list:
        region_to_render = nornir_imageregistration.Rectangle.Intersect(target_rect, tile.TargetSpaceBoundingBox)
        if region_to_render is not None and region_to_render.Area > 0:
            work_items.append((tile, region_to_render))

    with ThreadPoolExecutor(max_workers=_TRANSFORM_WORKERS) as transform_executor:
        futures = [
            transform_executor.submit(_transform_tile_worker, tile, region, target_space_scale)
            for tile, region in work_items
        ]
        for future in as_completed(futures):
            transformed_image_data = future.result()
            with _composite_lock:
                _composite_transformed_tile_onto_canvas(
                    transformed_image_data, full_image, full_image_zbuffer, scaled_target_rect)
            transformed_image_data.Clear()
            del transformed_image_data

    mask = np.less(full_image_zbuffer, __MaxZBufferValue(full_image_zbuffer.dtype))
    del full_image_zbuffer

    full_image = np.maximum(full_image, 0, out=full_image)

    if isinstance(full_image, np.memmap):
        full_image.flush()

    return full_image, mask


def TilesToImageParallel(mosaic_tileset: nornir_imageregistration.MosaicTileset,
                         TargetRegion: nornir_imageregistration.Rectangle | List[float] | None = None,
                         target_space_scale: float | None = None,
                         pool=None) -> Tuple[NDArray | None, NDArray | None]:
    """Assembles a set of transforms and imagepaths to a single image using parallel techniques.
    :param pool:
    :param MosaicTileset mosaic_tileset: Tileset to assemble
    :param tuple TargetRegion: (MinX, MinY, Width, Height) or Rectangle class.  Specifies the SourceSpace to render from
    :param float target_space_scale: Scalar for the target space coordinates.  Used to downsample or upsample the output image.  Changes the coordinates of the target space control points of the transform. 
    :param float target_space_scale: Scalar for the source space coordinates.  Must match the change in scale of input images relative to the transform source space coordinates.  So if downsampled by
    4 images are used, this value should be 0.25.  Calculated to be correct if None.  Specifying is an optimization to reduce I/O of reading image files to calculate.
    """
    timer = nornir_shared.tasktimer.TaskTimer()

    timer.Start('Prep')

    logger = logging.getLogger('TilesToImageParallel')
    if pool is None:
        pool = nornir_pools.GetGlobalMultithreadingPool()

    if target_space_scale is not None and target_space_scale > 1.0:
        raise ValueError(
            "It isn't impossible this is what the caller requests, but a target_space_scale value > 1 expands the resulting image beyond full resolution of the transform.")

    source_space_scale = 1.0 / mosaic_tileset.image_to_source_space_scale
    if target_space_scale is None:
        target_space_scale = source_space_scale

    original_fixed_rect_floats = None

    if TargetRegion is not None:
        if isinstance(TargetRegion, nornir_imageregistration.Rectangle):
            original_fixed_rect_floats = TargetRegion
        else:
            original_fixed_rect_floats = nornir_imageregistration.Rectangle.CreateFromPointAndArea(
                (TargetRegion[0], TargetRegion[1]),
                (TargetRegion[2] - TargetRegion[0], TargetRegion[3] - TargetRegion[1]))
    else:
        # We could use mosaic_tileset.TargetBoundingBox, but for mosaic-to-volume
        # transforms the non-zero origin is important, so we always use an origin
        # of 0, 0 and the max coordinates of the target bounding box
        # original_fixed_rect_floats = mosaic_tileset.TargetBoundingBox #Breaks mosaic-to-volume image assembly
        original_fixed_rect_floats = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0, 0),
                                                                                               mosaic_tileset.TargetBoundingBox.TopRight)

    targetRect = nornir_imageregistration.Rectangle.SafeRound(original_fixed_rect_floats)
    scaled_targetRect = nornir_imageregistration.Rectangle.scale_on_origin(original_fixed_rect_floats,
                                                                           target_space_scale)
    scaled_targetRect = nornir_imageregistration.Rectangle.SafeRound(scaled_targetRect)
    #    targetRect = original_fixed_rect_floats#nornir_imageregistration.Rectangle.scale_on_origin(scaled_targetRect, 1.0 / target_space_scale)

    first_tile = next(iter(mosaic_tileset.values()))
    if first_tile is None:
        raise ValueError("Mosaic Tileset has no tiles.")

    output_dtype = nornir_imageregistration.default_image_dtype()
    (fullImage, fullImageZbuffer) = __CreateOutputBufferForArea(int(scaled_targetRect.Height), int(scaled_targetRect.Width),
                                                                dtype=output_dtype)

    timer.End('Prep')
    timer.Start('Task Queuing')
    timer.Start('Task Execution')
    CheckTaskInterval = multiprocessing.cpu_count() * 2
    tasks = []  # type: List[nornir_pools.Task]
    # Ensure the shared memory manager has been created so child processes can
    # access it
    # shared_memory_manager = nornir_pools.get_or_create_shared_memory_manager()
    for i, tile in enumerate(mosaic_tileset.values()):
        # original_transform_target_rect = nornir_imageregistration.Rectangle(transform.FixedBoundingBox)
        original_transform_target_rect = tile.TargetSpaceBoundingBox
        transform_target_rect = nornir_imageregistration.Rectangle.SafeRound(original_transform_target_rect)

        regionToRender = nornir_imageregistration.Rectangle.Intersect(targetRect, tile.TargetSpaceBoundingBox)
        if regionToRender is None:
            continue

        if regionToRender.Area == 0:
            continue

        # Replaced by rendered_target_space_origin on TransformedImageData
        # scaled_region_rendered = nornir_imageregistration.Rectangle.scale_on_origin(regionToRender, target_space_scale)
        # scaled_region_rendered = nornir_imageregistration.Rectangle.SafeRound(scaled_region_rendered)

        task = pool.add_task(f"TransformTile {tile.ImagePath}",
                             TransformTile, tile=tile,
                             distanceImage=None,
                             target_space_scale=target_space_scale, TargetRegion=regionToRender,
                             SingleThreadedInvoke=False)
        task.transform = tile.Transform  # type: ignore[attr-defined]
        task.regionToRender = regionToRender  # type: ignore[attr-defined]
        # task.scaled_region_rendered = scaled_region_rendered
        task.transform_fixed_rect = transform_target_rect  # type: ignore[attr-defined]
        tasks.append(task)

        if not i % CheckTaskInterval == 0:
            continue

        while len(tasks) > CheckTaskInterval:  # Don't bother cleaning completed tasks if we can still add to the queue
            iTask = len(tasks) - 1
            while iTask >= 0:
                t = tasks[iTask]
                if t.iscompleted:
                    transformed_image_data = t.wait_return()
                    __AddTransformedTileTaskToComposite(t, transformed_image_data, fullImage, fullImageZbuffer,
                                                        scaled_targetRect)
                    transformed_image_data.Clear()
                    del transformed_image_data
                    del tasks[iTask]

                iTask -= 1

            if len(tasks) > CheckTaskInterval:  # Sleep a while if we are still over the limit
                time.sleep(0.1)
    timer.End('Task Queuing')
    logger.info('All warps queued, integrating results into final image')

    while len(tasks) > 0:
        # Pass through the entire loop and eliminate completed tasks in case any finished out of order
        iTask = len(tasks) - 1
        while iTask >= 0:
            t = tasks[iTask]
            if t.iscompleted:
                transformed_image_data = t.wait_return()
                __AddTransformedTileTaskToComposite(t, transformed_image_data, fullImage, fullImageZbuffer,
                                                    scaled_targetRect)
                transformed_image_data.Clear()
                del transformed_image_data
                del tasks[iTask]

            iTask -= 1

        if len(tasks) > 0:
            time.sleep(0.1)  # Give tasks some time to complete before we interrogate again

    timer.End('Task Execution')
    logger.info('Final image complete, building mask')

    mask = np.less(fullImageZbuffer, __MaxZBufferValue(fullImageZbuffer.dtype))
    del fullImageZbuffer

    # fullImage = np.clip(fullImage, 0, 1.0, out=fullImage)
    fullImage = np.maximum(fullImage, 0, out=fullImage)
    # Checking for > 1.0 makes sense for floating point images.  During the DM4 migration
    # I was getting images which used 0-255 values, and the 1.0 check set them to entirely black
    # fullImage[fullImage > 1.0] = 1.0

    logger.info('Assemble complete')

    if isinstance(fullImage, np.memmap):
        fullImage.flush()

    return fullImage, mask


def __AddTransformedTileTaskToComposite(task,
                                        transformedImageData: nornir_imageregistration.transformed_image_data_temp_files.TransformedImageDataViaTempFile,
                                        fullImage: NDArray,
                                        fullImageZBuffer: NDArray,
                                        scaled_target_rect: nornir_imageregistration.Rectangle | None = None):
    if transformedImageData is None:
        # logger = logging.getLogger('TilesToImageParallel')
        prettyoutput.LogErr('Convert task failed: ' + str(transformedImageData))
        return

    try:
        transformed_image = transformedImageData.image
        transformed_distance = transformedImageData.centerDistanceImage
    except ValueError:
        prettyoutput.LogErr('Convert task failed: ' + str(transformedImageData))
        if transformedImageData.errormsg is not None:
            prettyoutput.LogErr(transformedImageData.errormsg)
        return fullImage, fullImageZBuffer

    # The output buffer (fullImage/fullImageZBuffer) is always numpy. CuPy tile
    # results must be moved back to host before compositing.
    if hasattr(transformed_image, 'get'):
        transformed_image = transformed_image.get()
    if hasattr(transformed_distance, 'get'):
        transformed_distance = transformed_distance.get()

    CompositeOffset = (
                              transformedImageData.rendered_target_space_origin * transformedImageData.target_space_scale) - scaled_target_rect.BottomLeft  # type: ignore[union-attr]
    CompositeOffset = CompositeOffset.astype(np.int32)

    try:
        CompositeImageWithZBuffer(fullImage, fullImageZBuffer,
                                  transformed_image, transformed_distance,
                                  CompositeOffset)
    except ValueError as e:
        # This is frustrating and usually indicates the input transform passed to assemble mapped to negative coordinates.
        # logger = logging.getLogger('TilesToImageParallel')
        prettyoutput.LogErr(f'Could not add tile to composite: {transformedImageData}\n{e}')
        pass

    return


def __CreateScalableTransformCopy(transform):
    if not isinstance(transform, nornir_imageregistration.transforms.ITransform):
        raise ValueError("Expected transform to be an ITransform type")

    if isinstance(transform, nornir_imageregistration.transforms.ITransformScaling):
        return copy.deepcopy(transform)

    raise ValueError("Transform does not support ITransformScaling and does not have a hand-coded mapping here")


def TransformTile(tile: nornir_imageregistration.Tile,
                  distanceImage: NDArray | None = None,
                  target_space_scale: float | None = None,
                  TargetRegion: nornir_imageregistration.Rectangle | Tuple[float] | NDArray | None = None,
                  SingleThreadedInvoke: bool = False) -> nornir_imageregistration.transformed_image_data.ITransformedImageData:
    """
       Transform the passed image.  DistanceImage is an existing image recording the distance to the center of the
       image for each pixel.  target_space_scale is used when the image size does not match the image size encoded in the
       transform.  A scale will be calculated in this case and if it does not match the required scale the tile will 
       not be transformed.
get_space_scale: Optional pre-calculated scalar to apply to the transforms target space control points.  If None the scale is calculated based on the difference
                                   between input image size and the image size of the transform. i.e.  If the source_space is downsampled by 4 then the target_space will be downsampled to match
       :param tile:
       :param SingleThreadedInvoke:
       :param use_cp: use CuPy library for GPU processing
       :param TargetRegion: [MinY MinX MaxY MaxX] If specified only the specified region is populated.  Otherwise transform the entire image.'''
    """

    TargetRegionRect = None
    if TargetRegion is not None:
        if isinstance(TargetRegion, nornir_imageregistration.Rectangle):
            TargetRegionRect = TargetRegion.copy()
        elif isinstance(TargetRegion, Iterable):
            TargetRegionRect = nornir_imageregistration.Rectangle.CreateFromBounds(TargetRegion)  # type: ignore[arg-type]
    else:
        TargetRegion = tile.TargetSpaceBoundingBox
        TargetRegionRect = TargetRegion

    del TargetRegion

    nornir_imageregistration.spatial.RaiseValueErrorOnInvalidBounds(TargetRegionRect)

    # if not os.path.exists(imagefullpath):
    #    return nornir_imageregistration.transformed_image_data.TransformedImageData(errorMsg='Tile does not exist ' + imagefullpath)

    # if isinstance(transform, meshwithrbffallback.MeshWithRBFFallback):
    # Don't bother mapping points falling outside the defined boundaries because we won't have image data for it
    #   transform = triangulation.Triangulation(transform.points)
    source_image = None
    try:
        # warpedImage = nornir_imageregistration.ImageParamToImageArray(tile.Image, dtype=np.float32)
        source_image = tile.Image
    except IOError:
        return nornir_imageregistration.transformed_image_data.TransformedImageDataError(
            error_msg=f'Tile does not exist {tile.ImagePath}')
    except ValueError as ve:
        return nornir_imageregistration.transformed_image_data.TransformedImageDataError(error_msg=f'{ve}')

    source_image = nornir_imageregistration.ForceGrayscale(source_image)

    # Automatically scale the transform if the input image shape does not match the transform bounds
    source_space_scale = 1.0 / tile.image_to_source_space_scale  # Pass tile to this function and use the image_to_source_space attribute  #tiles.__DetermineTransformScale(transform, warpedImage.shape)

    if target_space_scale is None:
        target_space_scale = source_space_scale

    ########## Scale the transform output to fit the input image coordspace ####
    transform = _get_scaled_transform_for_tile(tile, source_space_scale, target_space_scale)

    ############################################################################

    if target_space_scale != 1.0:
        # TargetRegion = np.array(TargetRegion) * target_space_scale
        Scaled_TargetRegionRect = nornir_imageregistration.Rectangle.scale_on_origin(TargetRegionRect,
                                                                                     target_space_scale)
        Scaled_rounded_TargetRegionRect = nornir_imageregistration.Rectangle.SnapRound(Scaled_TargetRegionRect)
    else:
        Scaled_TargetRegionRect = TargetRegionRect
        Scaled_rounded_TargetRegionRect = nornir_imageregistration.Rectangle.SnapRound(TargetRegionRect)

    (target_width, target_height, target_minX, target_minY) = (Scaled_rounded_TargetRegionRect.Width,
                                                               Scaled_rounded_TargetRegionRect.Height,
                                                               Scaled_rounded_TargetRegionRect.MinX,
                                                               Scaled_rounded_TargetRegionRect.MinY)

    target_width = int(target_width)
    target_height = int(target_height)
    target_minX = int(target_minX)
    target_minY = int(target_minY)

    # if TargetRegion is None:
    #     TargetRegion = nornir_imageregistration.Rectangle.SafeRound(tile.TargetSpaceBoundingBox)
    #     Scaled_TargetRegionRect = nornir_imageregistration.Rectangle.scale_on_origin(tile.TargetSpaceBoundingBox, target_space_scale)
    #     Scaled_TargetRegionRect = nornir_imageregistration.Rectangle.SafeRound(Scaled_TargetRegionRect)
    #
    #     width = Scaled_TargetRegionRect.Width
    #     height = Scaled_TargetRegionRect.Height

    # if TargetRegion is None:
    #     if hasattr(transform, 'FixedBoundingBox'):
    #         width = transform.FixedBoundingBox.Width
    #         height = transform.FixedBoundingBox.Height
    #
    #         TargetRegionRect = transform.FixedBoundingBox
    #         TargetRegionRect = nornir_imageregistration.Rectangle.SafeRound(TargetRegionRect)
    #         Scaled_TargetRegionRect = TargetRegionRect
    #
    #         (minY, minX, maxY, maxX) = TargetRegionRect.ToTuple()
    #     else:
    #         width = warpedImage.shape[1]
    #         height = warpedImage.shape[0]
    #         TargetRegionRect = nornir_imageregistration.Rectangle.CreateFromPointAndArea((0,0), warpedImage.shape)
    #         Scaled_TargetRegionRect = TargetRegionRect
    # else:
    #     assert(len(TargetRegion) == 4)
    #     (minY, minX, maxY, maxX) = Scaled_TargetRegionRect.ToTuple()
    #     height = maxY - minY
    #     width = maxX - minX

    # Round up to the nearest integer value
    # height = np.ceil(height)
    # width = np.ceil(width)
    global distance_image_cache
    distanceImage = distance_image_cache.KeepGetOrCreate(distanceImage, source_image.shape[0:2])

    _distance_cval = float(__MaxZBufferValue(np.float16))

    (fixedImage, centerDistanceImage) = assemble.SourceImageToTargetSpace(  # type: ignore[assignment]
        transform,
        [source_image, distanceImage],
        output_botleft=(target_minY, target_minX),
        output_area=(target_height, target_width),
        cval=[0, _distance_cval],
        extrapolate=_assemble_grid_extrapolate(),
        return_shared_memory=False,
    )

    source_image_dtype = source_image.dtype
    del source_image
    del distanceImage

    return nornir_imageregistration.transformed_image_data_temp_files.TransformedImageDataViaTempFile.Create(fixedImage,  # type: ignore[arg-type]
                                                                                                             centerDistanceImage,  # type: ignore[arg-type]
                                                                                                             transform,
                                                                                                             source_space_scale,
                                                                                                             target_space_scale,
                                                                                                             rendered_target_space_origin=(
                                                                                                                 target_minY * (
                                                                                                                         1.0 / target_space_scale),
                                                                                                                 target_minX * (
                                                                                                                         1.0 / target_space_scale)),
                                                                                                             SingleThreadedInvoke=SingleThreadedInvoke)


if __name__ == '__main__':
    pass
