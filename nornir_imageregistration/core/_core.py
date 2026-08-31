"""
scipy image arrays are indexed [y,x]
"""

from collections import deque
from collections.abc import Iterable, Sequence
import bisect
import math
import multiprocessing
from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory
import multiprocessing.sharedctypes

import os
import queue
import tempfile
import threading
import time
import typing
import warnings
import weakref
from typing import Literal, cast

import numpy as np

from PIL import Image

import nornir_shared.prettyoutput as prettyoutput

# Only show cupy/mkl missing messages once and on the main process
is_main_process = multiprocessing.current_process().name == 'MainProcess'
__cupy_missing_message_shown = not is_main_process
__mkl_fft_missing_message_shown = not is_main_process
__mkl_random_missing_message_shown = not is_main_process
# Check if cupy is available, and if it is not import thunks that refer to scipy/numpy
try:
    import cupy as cp
    import cupyx
except (ModuleNotFoundError, ImportError) as e:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx

    if not __cupy_missing_message_shown:
        prettyoutput.Log('cupy not installed on system - cupy is an optional package that may increase performance')
        __cupy_missing_message_shown = True

try:
    import cupy.random as random
except (ModuleNotFoundError, ImportError):
    # try:
    #     import mkl_random.mklrand as random
    # except (ModuleNotFoundError, ImportError):

    if not __mkl_random_missing_message_shown:
        prettyoutput.Log(
            'mkl_fft not installed on system - mkl_random is an optional packages that may increase performance')
        __mkl_random_missing_message_shown = True

    import numpy.random as random

from numpy.typing import DTypeLike, NDArray

import nornir_imageregistration
import nornir_imageregistration.image_stats

import nornir_pools
import nornir_shared.images
import nornir_shared.prettyoutput as prettyoutput
from nornir_imageregistration import ImageLike
from nornir_imageregistration.mmap_metadata import memmap_metadata

# Disable decompression bomb protection since we are dealing with huge images on purpose
Image.MAX_IMAGE_PIXELS = None

# A dictionary of finalizers and shared memory blocks that is used to close shared memory when it goes out of scope
__known_shared_memory_allocations: dict[str, tuple[shared_memory.SharedMemory, typing.Callable]] = {}


# Legacy: atexit cleanup for shared memory was disabled; allocations are tracked in __known_shared_memory_allocations.

# from memory_profiler import profile

def ravel_index(idx: NDArray[np.integer], shp: NDArray) -> NDArray[np.integer]:
    """Convert an NxD array of coordinates into flat indices for an array of shape shp.

    :param idx: Nx2 (or NxD) array of coordinates [[x1,y1], [x2,y2], ...].
    :param shp: Shape of the target array (e.g. image shape).
    :return: 1D array of flat indices (numpy or cupy depending on idx).
    """
    xp = cp.get_array_module(idx)

    if shp[0] == 1:
        return idx[:, 1]  # type: ignore[return-value]

    if idx.shape[1] == len(shp):
        idx = xp.transpose(idx)  # type: ignore[assignment]
    else:
        pass

    result = xp.ravel_multi_index(idx, shp)
    return result  # type: ignore[return-value]
    # return np.transpose(np.concatenate((np.asarray(shp[1:])[::-1].cumprod()[::-1], [1])).dot(idx))


def index_with_array(image: NDArray, indices: NDArray) -> NDArray:
    """Return image values at the given pixel coordinates.

    :param image: 2D (or ND) array to index into.
    :param indices: Nx2 array of (x, y) or (col, row) pixel coordinates.
    :return: 1D array of values at those indices (same backend as image).
    """
    xp = cp.get_array_module(image)

    return xp.take(image, ravel_index(indices, xp.asarray(image.shape)))


def array_distance(array: NDArray) -> NDArray:
    """Compute Euclidean norm for each row of an Mx2 (or MxD) array.

    :param array: Mx2 (or MxD) array of vectors.
    :return: 1D array of length M (euclidean distance per row); scalar if array is 1D.
    """
    xp = cp.get_array_module(array)

    if array.ndim == 1:
        return xp.sqrt(xp.sum(array ** 2))

    return xp.sqrt(xp.sum(array ** 2, 1))

def ApproxEqual(a: float, b: float, epsilon=None) -> bool:
    """Return True if |a - b| < epsilon (default 0.01)."""
    if epsilon is None:
        epsilon = 0.01

    return np.abs(a - b) < epsilon


def ImageParamToNumpyImageArray(imageparam: ImageLike, dtype=None) -> NDArray | np.memmap:
    image = None
    if isinstance(imageparam, cp.ndarray):
        imageparam = nornir_imageregistration.EnsureNumpyArray(imageparam, dtype)  # type: ignore[arg-type]

    if isinstance(imageparam, np.ndarray):
        if dtype is None:
            image = imageparam
        elif np.issubdtype(imageparam.dtype, np.integer) and np.issubdtype(dtype, np.floating):
            # Scale image to 0.0 to 1.0
            image = imageparam.astype(dtype, copy=False) / np.iinfo(imageparam.dtype).max
        else:
            image = imageparam.astype(dtype=dtype, copy=False)
    elif isinstance(imageparam, str):
        image = LoadImage(imageparam, dtype=dtype)
        image = nornir_imageregistration.EnsureNumpyArray(image, dtype)
    elif isinstance(imageparam, nornir_imageregistration.Shared_Mem_Metadata):
        shared_mem = shared_memory.SharedMemory(name=imageparam.name, create=False)
        image = np.ndarray(imageparam.shape, dtype=imageparam.dtype, buffer=shared_mem.buf)
        image.setflags(write=not imageparam.readonly)
        finalizer = weakref.finalize(image, nornir_imageregistration.close_shared_memory, shared_mem)
        __known_shared_memory_allocations[shared_mem.name] = shared_mem, finalizer
    elif isinstance(imageparam, memmap_metadata):
        if dtype is None:
            dtype = imageparam.dtype

        image = np.memmap(imageparam.path, dtype=imageparam.dtype, mode=imageparam.mode, shape=imageparam.shape)  # type: ignore[call-overload]
        if dtype != imageparam.dtype:
            image = image.astype(dtype=dtype, copy=False)

    if image is None:
        raise ValueError("Image param %s is not a numpy array or image file" % (str(imageparam)))

    return image


def ImageParamToImageArray(imageparam: ImageLike, dtype=None) -> NDArray:
    image = None
    xp = nornir_imageregistration.GetComputationModule()

    if isinstance(imageparam, np.ndarray) or isinstance(imageparam, cp.ndarray):
        xp = cp.get_array_module(imageparam)  # type: ignore[arg-type]
        if dtype is None:
            image = imageparam
        elif xp.issubdtype(imageparam.dtype, np.integer) and xp.issubdtype(dtype, np.floating):  # type: ignore[union-attr]
            # Scale image to 0.0 to 1.0
            image = imageparam.astype(dtype, copy=False) / xp.iinfo(imageparam.dtype).max  # type: ignore[union-attr]
        else:
            image = imageparam.astype(dtype=dtype, copy=False)  # type: ignore[union-attr]
    elif isinstance(imageparam, str):
        image = LoadImage(imageparam, dtype=dtype)
    elif isinstance(imageparam, nornir_imageregistration.Shared_Mem_Metadata):
        # POSIX shared memory is host memory; always use NumPy views (explicit CPU boundary).
        shared_mem = shared_memory.SharedMemory(name=imageparam.name, create=False)
        image = np.ndarray(imageparam.shape, dtype=imageparam.dtype, buffer=shared_mem.buf)
        image.setflags(write=not imageparam.readonly)
        finalizer = weakref.finalize(image, nornir_imageregistration.close_shared_memory, shared_mem)
        __known_shared_memory_allocations[shared_mem.name] = shared_mem, finalizer
    elif isinstance(imageparam, memmap_metadata):
        if dtype is None:
            dtype = imageparam.dtype

        image = np.memmap(imageparam.path, dtype=imageparam.dtype, mode=imageparam.mode, shape=imageparam.shape)  # type: ignore[call-overload]
        if dtype != imageparam.dtype:
            image = image.astype(dtype=dtype, copy=False)

    if image is None:
        raise ValueError("Image param %s is not a numpy array or image file" % (str(imageparam)))

    return image  # type: ignore[return-value]


def ScalarForMaxDimension(max_dim: float, shapes):
    """Returns the scalar value to use so the largest dimensions in a list of shapes has the maximum value"""
    shapearray = None
    if not isinstance(shapes, list):
        shapearray = np.array(shapes)
    else:
        shapeArrays = list(map(np.array, shapes))
        shapearray = np.hstack(shapeArrays)

    maxVal = float(np.max(shapearray))

    return max_dim / maxVal


def remove_duplicate_points(points: NDArray, columns: Iterable[int] = ()) -> NDArray:
    """Remove rows who have equal values in the specified columns.  Result will be sorted
       using the column order provided.  Lexsort is used, so the last column entry is the primary sort key."""
    xp = cp.get_array_module(points)
    columns_list = list(columns)
    if len(columns_list) == 0:
        return points.copy()

    sort_values = tuple(points[:, i] for i in columns_list)
    # NumPy's lexsort(keys) takes a tuple of 1-D keys; CuPy's lexsort expects shape (K, N).
    if xp is np:
        sorted_indices = np.lexsort(sort_values)
    else:
        sorted_indices = xp.lexsort(xp.vstack(sort_values))
    sorted_point_pairs = points[sorted_indices, :]
    i = 0

    c = xp.asarray(columns_list, dtype=xp.int32)
    # Remove duplicates
    while i < sorted_point_pairs.shape[0] - 1:
        close = xp.all(xp.isclose(sorted_point_pairs[i, c], sorted_point_pairs[i + 1, c]))
        if xp is np:
            is_dup = bool(close)
        else:
            is_dup = bool(close.item())
        if is_dup:
            if xp is np:
                sorted_point_pairs = np.delete(sorted_point_pairs, i, axis=0)
            else:
                sorted_point_pairs = xp.concatenate(
                    (sorted_point_pairs[:i], sorted_point_pairs[i + 1 :]), axis=0
                )
        else:
            i += 1

    return sorted_point_pairs


def ScaleImage(image: NDArray, scalar: float) -> NDArray:
    """
    Returns a scaled array using spline interpolation (CPU/GPU agnostic function)
    """
    sp = cupyx.scipy.get_array_module(image)
    arr_xp = cp.get_array_module(image)
    if scalar == 1.0:
        return image.copy()

    order = 1 if scalar < 1.0 else 3
    order = 0 if scalar < 0.5 else order
    if arr_xp is np:
        return sp.ndimage.zoom(image.astype(np.float32, copy=False), zoom=scalar, order=order)
    return sp.ndimage.zoom(image, zoom=scalar, order=order)


def ExtractROI(image: NDArray, center, area) -> NDArray:
    """Returns an ROI around a center point with the area, if the area passes a boundary the ROI
       maintains the same area, but is shifted so the entire area remains in the image.
       USES NUMPY (Y,X) INDEXING"""

    half_area = area / 2.0
    x_range = SafeROIRange(center - half_area[1], area[1], maxVal=image.shape[1])
    y_range = SafeROIRange(center - half_area[0], area[0], maxVal=image.shape[0])

    ROI = image[y_range, x_range]

    return ROI


def SafeROIRange(start: int, count: int, maxVal: int, minVal: int = 0) -> list[int]:
    """
    Returns a range cropped within min and max values, but always attempts to have count entries in the ROI.
    If minVal or maxVal would crop the list then start is shifted to ensure the resulting value has the correct number of entries.
    :param int start: Starting value
    :param int count: Number of items in the list, incremented by 1, to return.
    :param int maxVal: Maximum value allowed to be returned.  Output list will be cropped if it equals or exceeds this value.
    :param int minVal: Minimum value allowed to be returned.  Output list will be cropped below this value.
    :return:  [start start+1, start+2, ..., start+count]
    :raises ValueError: If maxVal < minVal or maxVal - minVal < count
    """

    if count == 0:
        return list()

    if maxVal < minVal:
        raise ValueError(f"maxVal must be greater than minVal. {maxVal} > {minVal}")

    if maxVal - minVal < count:
        raise ValueError(
            f"Not enough room to return a ROI of requested size.  maxVal - minVal must be >= count. {maxVal} - {minVal} >= {count}")

    r = None

    if start < minVal:
        r = list(range(minVal, minVal + count))
    elif start + count >= maxVal:
        r = list(range(maxVal - count, maxVal))
    else:
        r = list(range(start, start + count))

    return r


def ConstrainedRange(start: int, count: int, maxVal: int, minVal: int = 0) -> list[int]:
    """Return a range of count integers starting at start, clamped to [minVal, maxVal)."""

    end = start + count
    r = None
    if maxVal - minVal < count:
        return list(range(minVal, maxVal))

    if start < minVal:
        r = list(range(minVal, end))
    elif end >= maxVal:
        r = list(range(start, maxVal))
    else:
        r = list(range(start, end))

    return r


def _ShrinkNumpyImageFile(InFile: str, OutFile: str, Scalar: float):
    image = LoadImage(InFile)
    resized_image = ResizeImage(image, Scalar)
    SaveImage(OutFile, resized_image)


def _ShrinkPillowImageFile(InFile: str, OutFile: str, Scalar: float, **kwargs):
    resample = kwargs.pop('resample', None)

    if resample is None:
        resample = Image.Resampling.BILINEAR
        if Scalar < 1.0:
            resample = Image.Resampling.LANCZOS

    with Image.open(InFile, mode='r') as img:

        dims = np.asarray(img.size).astype(dtype=np.float32, copy=False)
        desired_dims = dims * Scalar
        desired_dims = np.around(desired_dims).astype(dtype=np.int64)

        shrunk_img = img.resize(size=desired_dims, resample=resample)
        del img

        shrunk_img.save(OutFile, **kwargs)
        shrunk_img.close()
        del shrunk_img

    return None


# Shrinks the passed image file, return procedure handle of invoked command
def Shrink(InFile: str, OutFile: str, Scalar: float, **kwargs):
    """Shrinks the passed image file.  If Pool is not None the
       task is returned. kwargs are passed on to Pillow's image save function
       :param Scalar:
       :param str InFile: Path to input file
       :param str OutFile: Path to output file
    """

    (root, ext) = os.path.splitext(InFile)
    if ext == '.npy':
        _ShrinkNumpyImageFile(InFile, OutFile, Scalar)
    else:
        _ShrinkPillowImageFile(InFile, OutFile, Scalar, **kwargs)


def ResizeImage(image: NDArray, scalar: float | Iterable[float] | NDArray[np.floating]) -> NDArray:
    """Change image size by scalar"""

    xp = cp.get_array_module(image)
    sp = cupyx.scipy.get_array_module(image)
    original_min = image.min()
    original_max = image.max()

    zoom_value: float | tuple[float, ...]
    order = 2
    if isinstance(scalar, (int, float, np.integer, np.floating)):
        zoom_value = float(scalar)
        if zoom_value < 1.0:
            order = 3
    else:
        scalar_arr = nornir_imageregistration.EnsurePointsAre1DNumpyArray(cast(Sequence[float], scalar))
        zoom_values = tuple(float(s) for s in scalar_arr.tolist())
        zoom_value = zoom_values
        order = 3 if any(s < 1.0 for s in zoom_values) else 2

    result = sp.ndimage.zoom(image, zoom=zoom_value, order=order)
    xp.clip(result, original_min, original_max, out=result)
    return result


def _ConvertSingleImage(input_image_param, Flip: bool = False, Flop: bool = False,
                        Bpp: int | None = None, Invert: bool = False,
                        MinMax: tuple[float, float] | None = None,
                        Gamma: float | None = None):
    """
    Converts a single image according to the passed parameters (NumPy or CuPy, matching input backend).
    Image returned will match the dtype of the loaded image

    File loads always stay on NumPy so process-pool convert workers never upload
    per-tile arrays to the GPU under a global CuPy setting.
    """

    # Explicit host boundary for path loads: process-pool ConvertImagesInDict must
    # not H→D each tile when UsingCupy() (VRAM thrash across workers).
    if isinstance(input_image_param, str):
        image = LoadImage(input_image_param, backend="numpy")
    else:
        image = ImageParamToImageArray(input_image_param)
        image = nornir_imageregistration.EnsureNumpyArray(image)
    xp = cp.get_array_module(image)
    original_dtype = image.dtype
    max_possible_int_val = None

    # max_possible_float_val = 1.0

    NeedsClip = False

    # After lots of pain it is simplest to ensure all images are represented by floats before operating on them
    if nornir_imageregistration.IsIntArray(original_dtype):
        max_possible_int_val = nornir_imageregistration.ImageMaxPixelValue(image)
        probable_bpp = nornir_imageregistration.ImageBpp(image)
        working_dtype = np.float16
        if probable_bpp < 16:
            pass
        elif probable_bpp < 32:
            working_dtype = np.float32
        else:
            working_dtype = np.float64

        image = image.astype(
            working_dtype) / max_possible_int_val  # Always use float32 to prevent overflow errors.  We can downconvert later

    if Flip is not None and Flip:
        image = xp.flip(image, 0)

    if Flop is not None and Flop:
        image = xp.flip(image, 1)

    if MinMax is not None:
        (min_val, max_val) = MinMax

        if nornir_imageregistration.IsIntArray(original_dtype) is True:
            min_val /= max_possible_int_val  # type: ignore[operator]
            max_val /= max_possible_int_val  # type: ignore[operator]

        if min_val is None:
            min_val = 0

        if max_val is None:
            max_val = 1.0

        max_minus_min = max_val - min_val
        image -= min_val
        image /= max_minus_min

        NeedsClip = True

    if Gamma is None:
        Gamma = 1.0

    if Gamma != 1.0:
        exp = 1.0 / Gamma
        pos = image >= 0
        # xp.where evaluates both branches, which can trigger invalid-power errors
        # on negative values even when the mask excludes them.
        image[pos] = xp.power(image[pos], exp)
        NeedsClip = True

    if NeedsClip:
        xp.clip(image, 0, 1.0, out=image)

    if Invert is not None and Invert:
        image = 1.0 - image

    if nornir_imageregistration.IsIntArray(original_dtype) is True:
        image *= max_possible_int_val  # type: ignore[operator]

    image = image.astype(original_dtype, copy=False)

    return image


def _ConvertSingleImageToFile(input_image_param, output_filename: str, Flip: bool = False, Flop: bool = False,
                              InputBpp: int | None = None, OutputBpp: int | None = None,
                              Invert=False, MinMax=None, Gamma=None):
    image = _ConvertSingleImage(input_image_param,
                                Flip=Flip,
                                Flop=Flop,
                                Bpp=InputBpp,
                                Invert=Invert,
                                MinMax=MinMax,
                                Gamma=Gamma)

    if OutputBpp is None:
        OutputBpp = InputBpp

    (_, ext) = os.path.splitext(output_filename)
    if ext.lower() == '.png':
        SaveImage(output_filename, image, bpp=OutputBpp, optimize=True)
    else:
        SaveImage(output_filename, image, bpp=OutputBpp)
    return


# Shared throttled reporter; alias keeps call sites stable.
_TaskProgressReporter = prettyoutput.TaskProgressReporter


def ConvertImagesInDict(ImagesToConvertDict, Flip: bool = False, Flop: bool = False, InputBpp: int | None = None,
                        OutputBpp: int | None = None, Invert: bool = False,
                        bDeleteOriginal: bool = False, RightLeftShift: int | None = None,
                        AndValue: int | None = None, MinMax: tuple[float, float] | None = None,
                        Gamma: float | None = None, progress_name: str | None = None,
                        progress_task_key: str | None = None):
    """
    The key and value in the dictionary have the full path of an image to convert.
    MinMax is a tuple [Min,Max] passed to the -level parameter if it is not None
    RightLeftShift is a tuple containing a right then left then return to center shift which should be done to remove useless bits from the data
    I do not use an and because I do not calculate ImageMagick's quantum size yet.
    Every image must share the same colorspace

    :return: True if images were converted
    :rtype: bool
    """

    if len(ImagesToConvertDict) == 0:
        return False

    if InputBpp is None:
        for k in ImagesToConvertDict.keys():
            if os.path.exists(k):
                InputBpp = nornir_shared.images.GetImageBpp(k)
                break

    prettyoutput.CurseString('Stage', "ConvertImagesInDict")

    if MinMax is not None:
        if MinMax[0] > MinMax[1]:
            raise ValueError("Invalid MinMax parameter passed to ConvertImagesInDict")

    num_threads = multiprocessing.cpu_count() * 2
    if num_threads > len(ImagesToConvertDict):
        num_threads = len(ImagesToConvertDict) + 1

    pool = nornir_pools.GetMultithreadingPool("ConvertImagesInDict", num_threads=num_threads)
    tasks = []

    for (input_image, output_image) in ImagesToConvertDict.items():
        task = pool.add_task("{0} -> {1}".format(input_image, output_image),
                             _ConvertSingleImageToFile,
                             input_image_param=input_image,
                             output_filename=output_image,
                             Flip=Flip,
                             Flop=Flop,
                             InputBpp=InputBpp,
                             OutputBpp=OutputBpp,
                             Invert=Invert,
                             MinMax=MinMax,
                             Gamma=Gamma)
        tasks.append((task, input_image))

    task_key = progress_task_key or 'ConvertImagesInDict'
    reporter = _TaskProgressReporter(task_key, len(tasks), name=progress_name)
    completed = 0
    try:
        reporter.start()
        while len(tasks) > 0:
            t, input_image = tasks.pop(0)
            try:
                t.wait()
            except Exception as e:
                if __debug__:
                    raise

                prettyoutput.LogErr(f"Failed to convert {t.name}\n{e}")
            completed += 1
            reporter.update(
                completed,
                element=os.path.basename(input_image),
                path=input_image)
    finally:
        reporter.complete()

    if bDeleteOriginal:
        for (input_image, output_image) in ImagesToConvertDict.items():
            if input_image != output_image:
                pool.add_task("Delete {0}".format(input_image), os.remove, input_image)

        while len(tasks) > 0:
            t = tasks.pop(0)
            try:
                t.wait()
            except OSError as e:
                prettyoutput.LogErr("Unable to delete {0}\n{1}".format(t.name, e))
                pass

    if pool is not None:
        pool.wait_completion()
        pool.shutdown()
        pool = None

    del tasks


# ---------------------------------------------------------------------------
# GPU contrast conversion helpers
# ---------------------------------------------------------------------------

# Fraction of *free* GPU VRAM to budget for one chunk's float32 data.
# The gamma computation creates ~4 temporary device arrays (two float32 copies,
# one bool mask, one output), so peak device usage is ~4× the chunk data size.
# Using 40% / 4 = 10% of free VRAM as the effective data ceiling leaves plenty
# of room for the CUDA driver (~400–500 MB), CuPy memory pool, and other GPU
# processes sharing the card.
_GPU_MEMORY_FRACTION: float = 0.40
_GPU_SAFETY_FACTOR: int = 4
_GPU_MAX_CHUNK: int = 64   # pipeline-balance cap: no chunk larger than this


def _gpu_chunk_size(tile_shape: tuple) -> int:
    """Return the number of tiles per GPU chunk, auto-tuned from free VRAM.

    Uses :data:`_GPU_MEMORY_FRACTION` of currently-free VRAM divided by
    :data:`_GPU_SAFETY_FACTOR` (for gamma temporaries), capped at
    :data:`_GPU_MAX_CHUNK`.  All three module-level constants can be patched by
    tests or profiling scripts without changing this signature.
    """
    try:
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
    except Exception:
        return 1  # CuPy CUDA unavailable; fall through to numpy path anyway
    tile_float32_bytes = int(np.prod(tile_shape)) * 4
    chunk = max(1, int(free_bytes * _GPU_MEMORY_FRACTION / (_GPU_SAFETY_FACTOR * tile_float32_bytes)))
    return min(chunk, _GPU_MAX_CHUNK)


def _apply_contrast_gpu(batch: 'cp.ndarray',
                         min_val: float, max_val: float,
                         gamma: float,
                         max_int_val: float | None) -> 'cp.ndarray':
    """Apply level / gamma / clip to a device batch in-place where possible.

    :param batch: CuPy float32 array of shape ``(N, H, W)``, values already
        normalised to [0, 1] if the source dtype was integer.
    :param min_val: Level minimum (already normalised to [0, 1] for int sources).
    :param max_val: Level maximum (already normalised to [0, 1] for int sources).
    :param gamma: Gamma exponent (1.0 = no correction).
    :param max_int_val: If the source was integer, the max representable value
        (e.g. 255 for uint8); used to scale back before returning.  None for
        float sources.
    :return: The (possibly new) device array after all transforms.
    """
    needs_clip = False

    if min_val != 0.0 or max_val != 1.0:
        batch -= min_val
        batch /= (max_val - min_val)
        needs_clip = True

    if gamma != 1.0:
        exp = 1.0 / gamma
        # cp.where evaluates both branches — use cp.maximum to guard negatives
        # before power so we never compute NaN, then restore negatives via where.
        batch = cp.where(batch >= 0.0,
                         cp.power(cp.maximum(batch, 0.0), exp),
                         batch)
        needs_clip = True

    if needs_clip:
        cp.clip(batch, 0.0, 1.0, out=batch)

    if max_int_val is not None:
        batch *= max_int_val

    return batch


# ---------------------------------------------------------------------------
# Downsample helpers (GPU and CPU)
# ---------------------------------------------------------------------------

def _downsample2x_gpu(arr: 'cp.ndarray') -> 'cp.ndarray':
    """2× area-average downsample using a 2×2 box filter on a CuPy array.

    Works on any trailing two spatial dimensions: ``(..., H, W)`` → ``(..., H//2, W//2)``.
    Odd trailing heights/widths are trimmed by one pixel so the four 2×2 slices align.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if (h % 2) or (w % 2):
        arr = arr[..., :h - (h % 2), :w - (w % 2)]
    return (arr[..., 0::2, 0::2] + arr[..., 1::2, 0::2] +
            arr[..., 0::2, 1::2] + arr[..., 1::2, 1::2]) * cp.float32(0.25)


def _downsample2x_cpu(arr: np.ndarray) -> np.ndarray:
    """CPU (NumPy) 2× area-average downsample — same 2×2 box filter as :func:`_downsample2x_gpu`.

    Odd trailing heights/widths are trimmed by one pixel so the four 2×2 slices align.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if (h % 2) or (w % 2):
        arr = arr[..., :h - (h % 2), :w - (w % 2)]
    f32 = arr.astype(np.float32, copy=False)
    averaged = (f32[0::2, 0::2] + f32[1::2, 0::2] +
                f32[0::2, 1::2] + f32[1::2, 1::2]) * np.float32(0.25)
    return averaged.astype(arr.dtype)


# ---------------------------------------------------------------------------
# Pipelined GPU contrast conversion
# ---------------------------------------------------------------------------

# Batch size used by ConvertImagesInDictGpu when the caller does not specify
# one explicitly.  Override at runtime via the environment variable
# NORNIR_GPU_CONTRAST_BATCH_MB (integer megabytes, e.g. "128").
#
# Benchmark: AMD Ryzen 9 9950X (16-core) · NVIDIA RTX 4500 Ada (24 GB VRAM)
#            NFS tile storage · 3 iterations · float32 in-flight
#
#   4× tiles  2048²  (4 MB/tile, 128 tiles):
#   ┌──────────┬────────────┬─────────┬──────────┬─────────┬─────────┐
#   │ Batch MB │ chunk tiles│ Min (s) │ Mean (s) │ tiles/s │ Speedup │
#   ├──────────┼────────────┼─────────┼──────────┼─────────┼─────────┤
#   │ CPU      │     —      │  0.908  │  1.637   │  78.2   │  1.00×  │
#   │  64 MB   │     16     │  0.614  │  0.654   │ 195.7   │  2.50×  │ ← default
#   │ 128 MB   │     32     │  0.678  │  0.683   │ 187.4   │  2.40×  │
#   │ 256 MB   │     64     │  0.712  │  0.730   │ 175.4   │  2.24×  │
#   │ 512 MB   │    128     │  0.804  │  0.858   │ 149.1   │  1.91×  │
#   └──────────┴────────────┴─────────┴──────────┴─────────┴─────────┘
#
#   1× tiles  4096²  (64 MB/tile, 32 tiles) — tiles actually contrast-adjusted:
#   ┌──────────┬────────────┬─────────┬──────────┬─────────┬─────────┐
#   │ Batch MB │ chunk tiles│ Min (s) │ Mean (s) │ tiles/s │ Speedup │
#   ├──────────┼────────────┼─────────┼──────────┼─────────┼─────────┤
#   │ CPU      │     —      │  2.590  │  2.757   │  11.6   │  1.00×  │
#   │  64 MB   │      1     │  1.682  │  1.772   │  18.1   │  1.56×  │ ← default
#   │ 128 MB   │      2     │  1.784  │  1.804   │  17.7   │  1.53×  │
#   │ 256 MB   │      4     │  1.994  │  2.004   │  16.0   │  1.38×  │
#   │ 512 MB   │      8     │  1.905  │  2.074   │  15.4   │  1.33×  │
#   └──────────┴────────────┴─────────┴──────────┴─────────┴─────────┘
#
# 64 MB wins for both tile sizes.  Smaller chunks start the load/compute/save
# pipeline earlier, outweighing the H→D latency amortisation of larger slabs.
_DEFAULT_GPU_BATCH_MB: int = int(
    os.environ.get("NORNIR_GPU_CONTRAST_BATCH_MB", "64")
)
CONVERT_IMAGES_GPU_BATCH_BYTES: int = _DEFAULT_GPU_BATCH_MB * 1024 * 1024

# Host-memory budget for decoded tiles held ahead of the GPU in
# :func:`ConvertImagesInDictGpu`.  Loads were previously submitted for every tile
# at once, so a completed task retained its decoded host array until its chunk
# was consumed and resident memory tracked the *tile count* rather than any
# budget.  A 419 MB section measured a 415 MB peak; an 8 GB section would have
# held 8 GB.
#
# Dispatch mirrors :func:`ConvertImagesInDictGpuPyramid`, whose hybrid rule is
# already benchmark-tuned:
#   - section_bytes <= budget: all loads submitted upfront (max NFS concurrency,
#     byte-for-byte the previous behaviour, so tuned sections do not regress).
#   - otherwise: a sliding window of ``budget // tile_host_bytes`` decoded tiles.
#
# Override via ``NORNIR_GPU_CONTRAST_LOAD_BUDGET_MB`` (integer megabytes).
_DEFAULT_GPU_CONTRAST_LOAD_BUDGET_MB: int = int(
    os.environ.get("NORNIR_GPU_CONTRAST_LOAD_BUDGET_MB", "2048")
)
CONVERT_IMAGES_GPU_LOAD_BUDGET_BYTES: int = (
    _DEFAULT_GPU_CONTRAST_LOAD_BUDGET_MB * 1024 * 1024)

# Chunks of saves allowed to remain outstanding in ConvertImagesInDictGpu.
# Each queued save holds a view into its chunk's D->H result buffer, so an
# unbounded save queue pins one host buffer per chunk for the whole section.
#
# 2 preserves the intended overlap -- saves for chunk N run while the GPU works
# on chunk N+1 -- while capping resident result buffers at two.
CONVERT_IMAGES_GPU_SAVE_LOOKAHEAD_CHUNKS: int = max(1, int(
    os.environ.get("NORNIR_GPU_CONTRAST_SAVE_LOOKAHEAD_CHUNKS", "2")
))

# Host-memory budget for :func:`ConvertImagesInDictGpuPyramid` (decoded source tiles
# resident ahead of the GPU).  Separate from contrast-only chunk sizing because 1× 4K
# tiles (~64 MB float32 each) need many tiles prefetched for sustained NFS overlap.
# Override via ``NORNIR_GPU_PYRAMID_BATCH_MB`` (integer megabytes).
#
# Dispatch is hybrid (see ConvertImagesInDictGpuPyramid):
#   - section_bytes <= budget → all loads submitted upfront (max NFS concurrency).
#   - otherwise → cpu_count() loader threads with a result queue capped at
#     prefetch_count = budget // tile_float32_bytes (bounded host memory).
#
# RPC3 benchmark (128×4096², ~8 GB/section float32, 9 pyramid levels, RTX 4500 Ada):
#    64 MB → prefetch  1 → ~40 s/section (GPU waits on NFS)
#   256 MB → prefetch  4 → ~19 s/section
#     2 GB → prefetch 32 → cpu_count() parallel loaders (default; closes most of
#            the gap to the all-upfront chunk path at bounded memory).
_DEFAULT_GPU_PYRAMID_BATCH_MB: int = int(
    os.environ.get("NORNIR_GPU_PYRAMID_BATCH_MB", "2048")
)
CONVERT_IMAGES_GPU_PYRAMID_BATCH_BYTES: int = _DEFAULT_GPU_PYRAMID_BATCH_MB * 1024 * 1024


def _gpu_load_window_size(n_tiles: int,
                          tile_host_bytes: int,
                          chunk_size: int,
                          num_io_workers: int,
                          budget_bytes: int) -> int:
    """How many tile loads :func:`ConvertImagesInDictGpu` may have outstanding.

    Mirrors the hybrid dispatch of :func:`ConvertImagesInDictGpuPyramid`: submit
    the whole section when it fits the host-memory budget (maximum NFS
    concurrency, and byte-for-byte the behaviour before the window existed),
    otherwise slide a window sized to the budget.

    The window is floored at one chunk and one task per I/O worker; below that
    the GPU stalls waiting on loads. That floor makes resident host memory scale
    with worker count rather than with section size, which is the point -- an
    8 GB section no longer means 8 GB of decoded tiles.

    :param n_tiles: Tiles in the section.
    :param tile_host_bytes: Decoded size of one tile in its original dtype.
    :param chunk_size: Tiles per GPU chunk.
    :param num_io_workers: Threads in the load pool.
    :param budget_bytes: Host-memory budget for decoded tiles held ahead of the GPU.
    :return: Maximum outstanding load tasks, never more than ``n_tiles``.
    """
    if n_tiles <= 0:
        return 0

    if n_tiles * tile_host_bytes <= budget_bytes:
        return n_tiles

    window = max(int(budget_bytes // max(tile_host_bytes, 1)),
                 chunk_size,
                 num_io_workers)
    return min(window, n_tiles)


def _clear_gpu_convert_load_chunk(load_tasks: Sequence, start: int, end: int) -> None:
    """Drop decoded tile arrays retained on completed load tasks for ``[start, end)``."""
    for i in range(start, end):
        task = load_tasks[i]
        if hasattr(task, 'returned_value'):
            task.returned_value = None


def _free_cupy_convert_pools() -> None:
    """Return CuPy device and pinned blocks to the driver after a convert call."""
    if not nornir_imageregistration.HasCupy():
        return
    cp.get_default_memory_pool().free_all_blocks()
    pinned_pool = getattr(cp, 'get_default_pinned_memory_pool', None)
    if pinned_pool is not None:
        pinned_pool().free_all_blocks()


def ConvertImagesInDictGpu(ImagesToConvertDict: dict[str, str],
                            Flip: bool = False,
                            Flop: bool = False,
                            InputBpp: int | None = None,
                            OutputBpp: int | None = None,
                            MinMax: tuple[float, float] | None = None,
                            Gamma: float | None = None,
                            batch_bytes: int | None = None,
                            progress_name: str | None = None,
                            progress_task_key: str | None = None) -> bool:
    """GPU-accelerated contrast conversion using a chunked pipeline.

    Submits load tasks to a thread pool through a window sized by the
    ``CONVERT_IMAGES_GPU_LOAD_BUDGET_BYTES`` host-memory budget -- the whole
    section upfront when it fits (maximum NFS concurrency), otherwise a sliding
    window -- then processes tiles in chunks sized by *batch_bytes*.  Each chunk:

    1. Collects loaded arrays from the pool (nearly zero-wait — tasks are
       already running in the background).
    2. Copies them into a reused pinned-memory buffer with a **single-pass**
       ``np.multiply(src, scale, out=pinned_buf[i], casting='unsafe')`` —
       this combines dtype conversion and normalisation with no intermediate
       allocations.
    3. H→D transfers the entire pinned slab in one DMA operation.
    4. Optionally flips/flops on device, then applies level / gamma / clip
       vectorised over the batch axis on the GPU.
    5. D→H downloads the result, then dispatches per-tile saves to a second
       thread pool so saving chunk *N* overlaps with GPU work on chunk *N+1*.
       Saves hold views into the chunk's D→H buffer, so at most
       ``CONVERT_IMAGES_GPU_SAVE_LOOKAHEAD_CHUNKS`` chunks stay outstanding.

    **Host memory** — decoded tiles resident ahead of the GPU are capped by
    ``CONVERT_IMAGES_GPU_LOAD_BUDGET_BYTES``
    (``NORNIR_GPU_CONTRAST_LOAD_BUDGET_MB``), so in-flight memory scales with
    worker count rather than with section size.

    **Chunk sizing** — ``batch_bytes`` controls how many tiles fit in one GPU
    round-trip.  Smaller batches allow load/compute/save overlap to start
    sooner; larger batches amortise H→D/D→H latency but stall the pipeline
    waiting for the full chunk to load.  See the ``CONVERT_IMAGES_GPU_BATCH_BYTES``
    module constant and the ``NORNIR_GPU_CONTRAST_BATCH_MB`` environment variable
    for process-wide override without touching call sites.

    Falls back to :func:`ConvertImagesInDict` when:
    - CuPy is unavailable or not the active backend.
    - The tile set is empty.
    - Mixed tile shapes are detected after the first chunk.

    :param ImagesToConvertDict: Mapping of input path → output path.
    :param Flip: If True, flip each tile vertically (axis 0), matching
        :func:`_ConvertSingleImage`.
    :param Flop: If True, flip each tile horizontally (axis 1).
    :param InputBpp: Bits-per-pixel of input tiles (auto-detected when None).
    :param OutputBpp: Bits-per-pixel for output tiles (matches InputBpp when None).
    :param MinMax: ``(min, max)`` intensity cutoff tuple for contrast stretch.
    :param Gamma: Gamma correction value (None or 1.0 means no correction).
    :param batch_bytes: Target float32 byte budget per GPU chunk.  Defaults to
        ``CONVERT_IMAGES_GPU_BATCH_BYTES`` (64 MB unless overridden by the
        ``NORNIR_GPU_CONTRAST_BATCH_MB`` environment variable).
        ``chunk_size = max(1, batch_bytes // tile_float32_bytes)``.
    :return: True if any images were converted.
    """
    if batch_bytes is None:
        batch_bytes = CONVERT_IMAGES_GPU_BATCH_BYTES
    if not nornir_imageregistration.HasCupy() or not nornir_imageregistration.UsingCupy():
        return ConvertImagesInDict(ImagesToConvertDict,
                                   Flip=Flip,
                                   Flop=Flop,
                                   InputBpp=InputBpp,
                                   OutputBpp=OutputBpp,
                                   MinMax=MinMax,
                                   Gamma=Gamma,
                                   progress_name=progress_name,
                                   progress_task_key=progress_task_key or 'ConvertImagesInDictGpu')

    if len(ImagesToConvertDict) == 0:
        return False

    if MinMax is not None and MinMax[0] > MinMax[1]:
        raise ValueError("Invalid MinMax parameter passed to ConvertImagesInDictGpu")

    if InputBpp is None:
        for k in ImagesToConvertDict.keys():
            if os.path.exists(k):
                InputBpp = nornir_shared.images.GetImageBpp(k)
                break

    if OutputBpp is None:
        OutputBpp = InputBpp

    gamma_val = float(Gamma) if Gamma is not None else 1.0
    do_flip = bool(Flip)
    do_flop = bool(Flop)

    prettyoutput.CurseString('Stage', "ConvertImagesInDictGpu")

    input_paths = list(ImagesToConvertDict.keys())
    output_paths = [ImagesToConvertDict[p] for p in input_paths]
    n_tiles = len(input_paths)

    # ------------------------------------------------------------------
    # I/O pools — thread pools, NOT process pools.
    #
    # Both Pillow PNG decode (load) and PNG encode (save) call into C
    # extensions that release the GIL, so Python threads are truly
    # parallel for these operations.
    #
    # Process pools require fork()ing worker processes and joining them
    # on every call (~420 ms to spawn 32 workers + 3–4 s to join them
    # via multiprocessing.Pool's 0.1 s-per-worker exit-polling).  For a
    # pipeline that processes 100 sections this adds 350+ seconds of
    # pure process-management overhead.
    #
    # Thread pools are created in < 1 ms (Python Thread objects), cost
    # nothing to reuse across calls (workers idle-expire on their own
    # after 5 s), and carry zero IPC serialisation overhead (shared
    # memory, no pickle).  The named pools are cached in nornir_pools
    # _known_pools and returned as-is on subsequent calls.
    #
    # wait_completion() is called at the end of each call; shutdown() is
    # intentionally NOT called so the pools survive for the next call.
    # ------------------------------------------------------------------
    num_io_workers = min(multiprocessing.cpu_count() * 2, n_tiles + 1)
    load_pool = nornir_pools.GetThreadPool("ConvertImagesInDictGpu_load", num_io_workers)

    # Loads are submitted through a sliding window rather than all at once.
    # Submitting every tile up front kept a decoded host array alive in each
    # completed task until its chunk was consumed, so in-flight host memory grew
    # with the tile count instead of the chunk budget: a 200-tile 1024x1024
    # uint16 set peaked at 466 MB, essentially the whole set, against a 64 MB
    # chunk budget.
    #
    # The window still has to outrun the GPU, or the loop stalls waiting on I/O
    # and the original throughput rationale (saturate NFS concurrency) is lost.
    # _load_window_size keeps every worker fed plus whole chunks of look-ahead.
    all_load_tasks: list = [None] * n_tiles
    _load_cursor = 0

    def _submit_loads_through(target_exclusive: int) -> None:
        """Ensure load tasks are queued for every index below target_exclusive."""
        nonlocal _load_cursor
        limit = min(target_exclusive, n_tiles)
        while _load_cursor < limit:
            path = input_paths[_load_cursor]
            all_load_tasks[_load_cursor] = load_pool.add_task(
                path, _LoadImageByExtension, path, None)
            _load_cursor += 1

    # Only the first tile is queued here. Tile size is unknown until it lands,
    # and priming the workers instead would commit num_io_workers tiles blind --
    # 64 4096x4096 uint16 tiles is 2 GB before any budget is known. One serial
    # read costs a fraction of a section.
    _submit_loads_through(1)

    save_pool = nornir_pools.GetThreadPool("ConvertImagesInDictGpu_save", num_io_workers)

    # ------------------------------------------------------------------
    # Bootstrap: wait for the first tile to determine shape and dtype.
    # ------------------------------------------------------------------
    first_array: np.ndarray | None = None
    try:
        first_array = all_load_tasks[0].wait_return()
    except Exception as exc:
        prettyoutput.LogErr(f"ConvertImagesInDictGpu: failed to load first tile {input_paths[0]}\n{exc}")

    if first_array is None:
        load_pool.wait_completion()
        return False

    tile_shape = first_array.shape
    original_dtype = first_array.dtype
    is_int = nornir_imageregistration.IsIntArray(original_dtype)
    max_int_val = float(nornir_imageregistration.ImageMaxPixelValue(first_array)) if is_int else None
    scale = (1.0 / max_int_val) if max_int_val else 1.0

    # Normalise MinMax to [0, 1] float space once.  For int sources divide by max_int_val.
    if MinMax is not None:
        min_val = float(MinMax[0])
        max_val = float(MinMax[1])
        if is_int and max_int_val is not None:
            min_val /= max_int_val
            max_val /= max_int_val
    else:
        min_val, max_val = 0.0, 1.0

    # Chunk size from batch_bytes budget.
    tile_float32_elems = int(np.prod(tile_shape))
    tile_float32_bytes = tile_float32_elems * 4
    chunk_size = max(1, batch_bytes // tile_float32_bytes)
    n_chunks = (n_tiles + chunk_size - 1) // chunk_size

    # Budget against the decoded host size (original dtype), which is what the
    # load tasks actually retain.
    _load_window_size = _gpu_load_window_size(
        n_tiles=n_tiles,
        tile_host_bytes=tile_float32_elems * original_dtype.itemsize,
        chunk_size=chunk_size,
        num_io_workers=num_io_workers,
        budget_bytes=CONVERT_IMAGES_GPU_LOAD_BUDGET_BYTES)

    _submit_loads_through(_load_window_size)

    # Outstanding save tasks, grouped by chunk. Saves receive views into the
    # chunk's result_np, so an unbounded save queue pins every chunk's D->H
    # buffer: 13 chunks x 33.5 MB measured as a 435 MB peak for a 419 MB tile
    # set. Retiring older chunks caps that at
    # CONVERT_IMAGES_GPU_SAVE_LOOKAHEAD_CHUNKS buffers while still overlapping
    # saves for chunk N with GPU work on chunk N+1.
    _pending_save_chunks: deque[list] = deque()

    def _retire_save_chunks(max_outstanding: int) -> None:
        while len(_pending_save_chunks) > max_outstanding:
            for save_task in _pending_save_chunks.popleft():
                try:
                    save_task.wait()
                except Exception as exc:
                    prettyoutput.LogErr(
                        f"ConvertImagesInDictGpu: save failed {save_task.name}\n{exc}")

    # Pinned host buffer sized for one chunk.
    # np.frombuffer requires explicit count= when wrapping a PinnedMemoryPointer;
    # without it the buffer protocol may expose only the pointer's metadata size.
    pinned = None
    pinned_buf: np.ndarray | None = None
    fell_back = False

    task_key = progress_task_key or 'ConvertImagesInDictGpu'
    reporter = _TaskProgressReporter(task_key, n_tiles, name=progress_name)
    tiles_completed = 0

    # ------------------------------------------------------------------
    # Stage 2: GPU chunk loop.
    #
    # For each chunk we:
    #   a) Collect loaded arrays (near-zero wait — pool runs ahead).
    #   b) Copy into pinned buffer via np.multiply with casting='unsafe':
    #      a single-pass cast+scale with zero intermediate allocations,
    #      replacing the old chain of astype() + * scale + copyto() that
    #      created ~2 × chunk_size × tile_bytes of ephemeral heap traffic.
    #   c) H→D the entire pinned slab in one DMA.
    #   d) GPU: level / gamma / clip vectorised over the batch axis.
    #   e) D→H the results.
    #   f) Dispatch each tile's save to the save pool (async).
    #   g) Drop load-task returned arrays so host tiles are not retained
    #      for the rest of the section.
    #
    # Step (f) overlaps save-N with GPU processing of chunk N+1.
    # ------------------------------------------------------------------
    try:
        pinned = cp.cuda.alloc_pinned_memory(chunk_size * tile_float32_bytes)
        pinned_buf = np.frombuffer(pinned, dtype=np.float32,
                                   count=chunk_size * tile_float32_elems).reshape(
                                       chunk_size, *tile_shape)
        try:
            reporter.start()
            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, n_tiles)
                chunk_out = output_paths[start:end]

                # Top up before consuming, so the loads for the chunks after
                # this one are already in flight while the GPU works.
                _submit_loads_through(end + _load_window_size)

                arrays_chunk: list[np.ndarray | None] = []
                for i in range(start, end):
                    if chunk_idx == 0 and i == 0:
                        arrays_chunk.append(first_array)
                        continue
                    arr_i: np.ndarray | None = None
                    try:
                        arr_i = all_load_tasks[i].wait_return()
                    except Exception as exc:
                        prettyoutput.LogErr(
                            f"ConvertImagesInDictGpu: load failed {input_paths[i]}\n{exc}")
                    arrays_chunk.append(arr_i)

                valid = [a for a in arrays_chunk if a is not None]
                if not valid:
                    _clear_gpu_convert_load_chunk(all_load_tasks, start, end)
                    if chunk_idx == 0:
                        first_array = None
                    continue

                if not fell_back and any(a.shape != tile_shape for a in valid):
                    prettyoutput.Log(
                        "ConvertImagesInDictGpu: mixed tile shapes — falling back remaining chunks to CPU")
                    fell_back = True

                if fell_back:
                    fallback_saves: list = []
                    for i, (out_path, arr) in enumerate(zip(chunk_out, arrays_chunk)):
                        if arr is None:
                            continue
                        result = _ConvertSingleImage(arr, Flip=do_flip, Flop=do_flop,
                                                     MinMax=MinMax, Gamma=gamma_val, Bpp=InputBpp)
                        (_, ext) = os.path.splitext(out_path)
                        kw: dict = {'optimize': True} if ext.lower() == '.png' else {}
                        fallback_saves.append(
                            save_pool.add_task(f"save {out_path}", SaveImage,
                                               out_path, result, bpp=OutputBpp, **kw))
                        tiles_completed += 1
                        in_path = input_paths[start + i]
                        reporter.update(
                            tiles_completed,
                            element=os.path.basename(in_path),
                            path=in_path)
                    _pending_save_chunks.append(fallback_saves)
                    _retire_save_chunks(CONVERT_IMAGES_GPU_SAVE_LOOKAHEAD_CHUNKS)
                    _clear_gpu_convert_load_chunk(all_load_tasks, start, end)
                    if chunk_idx == 0:
                        first_array = None
                    continue

                n = len(arrays_chunk)

                for i, arr in enumerate(arrays_chunk):
                    if arr is None:
                        pinned_buf[i] = 0.0
                    else:
                        np.multiply(arr, scale, out=pinned_buf[i], casting='unsafe')

                batch = cp.asarray(pinned_buf[:n])
                # Match _ConvertSingleImage order: Flip/Flop before level/gamma.
                # Batch axes are (N, H, W) → Flip = axis 1, Flop = axis 2.
                if do_flip:
                    batch = cp.flip(batch, axis=1)
                if do_flop:
                    batch = cp.flip(batch, axis=2)
                batch = _apply_contrast_gpu(batch, min_val, max_val, gamma_val,
                                             max_int_val if is_int else None)
                batch = batch.astype(original_dtype)
                result_np: np.ndarray = cp.asnumpy(batch)
                del batch

                last_input_path: str | None = None
                chunk_saves: list = []
                for i, (out_path, arr) in enumerate(zip(chunk_out, arrays_chunk)):
                    if arr is None:
                        continue
                    (_, ext) = os.path.splitext(out_path)
                    kw = {'optimize': True} if ext.lower() == '.png' else {}
                    chunk_saves.append(
                        save_pool.add_task(f"save {out_path}", SaveImage,
                                           out_path, result_np[i], bpp=OutputBpp, **kw))
                    last_input_path = input_paths[start + i]

                tiles_completed += sum(1 for a in arrays_chunk if a is not None)
                if last_input_path is not None:
                    reporter.update(
                        tiles_completed,
                        element=os.path.basename(last_input_path),
                        path=last_input_path)
                else:
                    reporter.update(tiles_completed)
                # Host tiles are copied into pinned / result_np; drop load retention.
                _clear_gpu_convert_load_chunk(all_load_tasks, start, end)
                if chunk_idx == 0:
                    first_array = None
                del arrays_chunk

                # Saves hold views into result_np, so dropping the local name is
                # not enough; retire older chunks to actually release them.
                _pending_save_chunks.append(chunk_saves)
                del chunk_saves
                del result_np
                _retire_save_chunks(CONVERT_IMAGES_GPU_SAVE_LOOKAHEAD_CHUNKS)
        finally:
            # Drain async I/O before removing the dashboard track so the bar
            # stays visible while saves are still in flight.
            try:
                load_pool.wait_completion()
                save_pool.wait_completion()
            finally:
                reporter.complete()
        # Intentionally not calling shutdown(): pools are cached by name in
        # nornir_pools._known_pools and reused on the next call.  Workers
        # idle-expire after their WorkerCheckInterval (default 5 s).
    finally:
        # Drop pinned views before freeing pools so Task Manager dedicated VRAM
        # does not ratchet across section converts via CuPy's retained blocks.
        pinned_buf = None
        pinned = None
        _free_cupy_convert_pools()

    return n_tiles > 0


# ---------------------------------------------------------------------------
# Per-tile pyramid helper (CPU path, shared by both CPU and GPU fallback)
# ---------------------------------------------------------------------------

def _convert_and_build_pyramid_tile(
    input_path: str,
    all_output_paths: list[str],
    MinMax: tuple[float, float] | None,
    gamma_val: float,
    InputBpp: int | None,
    OutputBpp: int | None,
) -> None:
    """Load one tile, contrast-adjust it, and build all pyramid levels (CPU, single-tile).

    :param input_path: Source image path.
    :param all_output_paths: ``[level-1 path, level-2 path, level-4 path, …]``.
        The first entry is the contrast-adjusted full-resolution output; subsequent
        entries receive 2× area-average downsampled versions, chained from the
        previous level.
    :param MinMax: Intensity cutoff tuple, or ``None``.
    :param gamma_val: Gamma exponent (1.0 = no correction).
    :param InputBpp: Bits-per-pixel of the source image (used by
        :func:`_ConvertSingleImage`).
    :param OutputBpp: Bits-per-pixel for all output images.
    """
    arr = _LoadImageByExtension(input_path, None)
    if arr is None:
        return

    result = _ConvertSingleImage(arr, MinMax=MinMax, Gamma=gamma_val, Bpp=InputBpp)

    (_, ext) = os.path.splitext(all_output_paths[0])
    kw: dict = {'optimize': True} if ext.lower() == '.png' else {}
    SaveImage(all_output_paths[0], result, bpp=OutputBpp, **kw)

    prev = result
    for out_path in all_output_paths[1:]:
        prev = _downsample2x_cpu(prev)
        (_, ext) = os.path.splitext(out_path)
        kw = {'optimize': True} if ext.lower() == '.png' else {}
        SaveImage(out_path, prev, bpp=OutputBpp, **kw)


def _save_image_ext_defaults(out_path: str) -> dict:
    (_, ext) = os.path.splitext(out_path)
    return {'optimize': True} if ext.lower() == '.png' else {}


def _downsample_step(image: NDArray, factor: float) -> NDArray:
    """One pyramid downsample step on *image* (NumPy or CuPy), factor typically 0.5."""
    if abs(factor - 0.5) < 1e-9:
        xp = cp.get_array_module(image)
        if xp is cp:
            return _downsample2x_gpu(image)
        return _downsample2x_cpu(image)
    sp = cupyx.scipy.get_array_module(image)
    xp = cp.get_array_module(image)
    if xp is cp:
        return sp.ndimage.zoom(image, float(factor))
    return ResizeImage(image, factor)


def _build_pyramid_tile_cpu(input_path: str,
                            output_paths: list[str | None],
                            shrink_factors: list[float]) -> None:
    """Load once, chain CPU downsamples, save selected levels without re-reading.

    *output_paths* has one entry per step; ``None`` means downsample in memory only
    (intermediate level already valid on disk).
    """
    if len(output_paths) != len(shrink_factors):
        raise ValueError("output_paths and shrink_factors must have the same length")

    arr = _LoadImageByExtension(input_path, None)
    if arr is None:
        return

    prev = arr
    for out_path, factor in zip(output_paths, shrink_factors):
        prev = _downsample_step(prev, factor)
        if out_path is not None:
            SaveImage(out_path, prev, **_save_image_ext_defaults(out_path))


def _build_pyramid_tile_gpu(input_path: str,
                            output_paths: list[str | None],
                            shrink_factors: list[float]) -> None:
    """Load once, H→D, chain GPU downsamples, D→H + save selected levels."""
    if len(output_paths) != len(shrink_factors):
        raise ValueError("output_paths and shrink_factors must have the same length")

    if not nornir_imageregistration.UsingCupy():
        _build_pyramid_tile_cpu(input_path, output_paths, shrink_factors)
        return

    arr = _LoadImageByExtension(input_path, None)
    if arr is None:
        return

    prev = cp.asarray(arr)
    for out_path, factor in zip(output_paths, shrink_factors):
        prev = _downsample_step(prev, factor)
        if out_path is not None:
            SaveImage(out_path, cp.asnumpy(prev), **_save_image_ext_defaults(out_path))


def BuildTilePyramidsMemoryCpu(
    tiles: dict[str, list[str | None]],
    shrink_factors: list[float],
    num_threads: int | None = None,
) -> None:
    """Build pyramid levels in memory for many tiles (CPU, threaded).

    :param tiles: Maps each finest-level input path to a list of output paths, one
        per downsample step in order. Use ``None`` for a step that should run in
        memory but not be written (downstream level still updated in the chain).
    :param shrink_factors: Downsample factor for each step (typically ``0.5``).
    :param num_threads: Worker count; defaults to ``cpu_count * 2``.
    """
    if not tiles:
        return
    if num_threads is None:
        num_threads = min(multiprocessing.cpu_count() * 2, len(tiles) + 1)

    pool = nornir_pools.GetThreadPool("BuildTilePyramidsMemoryCpu", num_threads)
    tasks = []
    for input_path, output_paths in tiles.items():
        t = pool.add_task(
            f"pyramid-cpu {input_path}",
            _build_pyramid_tile_cpu,
            input_path,
            output_paths,
            shrink_factors,
        )
        tasks.append(t)

    for t in tasks:
        try:
            t.wait()
        except Exception as exc:
            if __debug__:
                raise
            prettyoutput.LogErr(f"BuildTilePyramidsMemoryCpu: {t.name}\n{exc}")

    pool.wait_completion()


def BuildTilePyramidsMemoryGpu(
    tiles: dict[str, list[str | None]],
    shrink_factors: list[float],
) -> None:
    """Build pyramid levels in memory for many tiles (GPU, one tile at a time).

    Falls back to :func:`BuildTilePyramidsMemoryCpu` when CuPy is unavailable.
    """
    if not tiles:
        return

    if not nornir_imageregistration.UsingCupy():
        BuildTilePyramidsMemoryCpu(tiles, shrink_factors)
        return

    for input_path, output_paths in tiles.items():
        try:
            _build_pyramid_tile_gpu(input_path, output_paths, shrink_factors)
        except Exception as exc:
            if __debug__:
                raise
            prettyoutput.LogErr(f"BuildTilePyramidsMemoryGpu: {input_path}\n{exc}")


def ConvertImagesInDictPyramid(ImagesToConvertDict: dict[str, str],
                                PyramidOutputDicts: list[dict[str, str]],
                                InputBpp: int | None = None,
                                OutputBpp: int | None = None,
                                MinMax: tuple[float, float] | None = None,
                                Gamma: float | None = None,
                                progress_name: str | None = None,
                                progress_task_key: str | None = None) -> bool:
    """CPU contrast + all pyramid levels in one tile pass.

    For each tile, loads it once from NFS, applies contrast via
    :func:`_ConvertSingleImage`, then chains :func:`_downsample2x_cpu` for each
    entry in *PyramidOutputDicts*.  All tiles are processed concurrently by a
    thread pool (Pillow PNG encode/decode release the GIL).

    This eliminates the per-level NFS reads that :func:`ConvertImagesInDict` +
    :func:`BuildTilePyramids` would require: *N* reads instead of *N × num_levels*.

    :param ImagesToConvertDict: ``{input_path → output_path}`` for the finest level.
    :param PyramidOutputDicts: List of ``{input_path → output_path}`` dicts, one per
        coarser pyramid level in ascending order (level 2, 4, 8, …).
    :param InputBpp: Bits-per-pixel of input images (auto-detected when ``None``).
    :param OutputBpp: Bits-per-pixel for outputs (matches InputBpp when ``None``).
    :param MinMax: ``(min, max)`` intensity cutoff tuple.
    :param Gamma: Gamma correction value (``None`` or 1.0 = no correction).
    :return: ``True`` if any images were converted.
    """
    if len(ImagesToConvertDict) == 0:
        return False

    if MinMax is not None and MinMax[0] > MinMax[1]:
        raise ValueError("Invalid MinMax parameter passed to ConvertImagesInDictPyramid")

    if InputBpp is None:
        for k in ImagesToConvertDict.keys():
            if os.path.exists(k):
                InputBpp = nornir_shared.images.GetImageBpp(k)
                break

    if OutputBpp is None:
        OutputBpp = InputBpp

    gamma_val = float(Gamma) if Gamma is not None else 1.0

    prettyoutput.CurseString('Stage', "ConvertImagesInDictPyramid")

    n_tiles = len(ImagesToConvertDict)
    num_io_workers = min(multiprocessing.cpu_count() * 2, n_tiles + 1)
    pool = nornir_pools.GetThreadPool("ConvertImagesInDictPyramid", num_io_workers)
    tasks = []

    for input_path, out_l1 in ImagesToConvertDict.items():
        all_output_paths = [out_l1] + [
            pyr_dict[input_path]
            for pyr_dict in PyramidOutputDicts
            if input_path in pyr_dict
        ]
        t = pool.add_task(
            f"convert+pyramid {input_path}",
            _convert_and_build_pyramid_tile,
            input_path, all_output_paths, MinMax, gamma_val, InputBpp, OutputBpp,
        )
        tasks.append((t, input_path))

    task_key = progress_task_key or 'ConvertImagesInDictPyramid'
    reporter = _TaskProgressReporter(task_key, len(tasks), name=progress_name)
    completed = 0
    try:
        reporter.start()
        for t, input_path in tasks:
            try:
                t.wait()
            except Exception as exc:
                if __debug__:
                    raise
                prettyoutput.LogErr(f"ConvertImagesInDictPyramid: {t.name}\n{exc}")
            completed += 1
            reporter.update(
                completed,
                element=os.path.basename(input_path),
                path=input_path)
    finally:
        reporter.complete()

    pool.wait_completion()
    return n_tiles > 0


def ConvertImagesInDictGpuPyramid(ImagesToConvertDict: dict[str, str],
                                   PyramidOutputDicts: list[dict[str, str]],
                                   InputBpp: int | None = None,
                                   OutputBpp: int | None = None,
                                   MinMax: tuple[float, float] | None = None,
                                   Gamma: float | None = None,
                                   batch_bytes: int | None = None,
                                   progress_name: str | None = None,
                                   progress_task_key: str | None = None) -> bool:
    """GPU contrast + all pyramid levels in a single per-tile pass.

    Each source tile is loaded once, transferred to the GPU, contrast-adjusted,
    then downsampled in a 2× area-average chain via :func:`_downsample2x_gpu` for
    every requested pyramid level.  Saves are dispatched per level so NFS writes
    overlap with GPU work on the next tile.

    Load-pool width (``cpu×2`` workers) is independent of *batch_bytes*, which
    caps how many decoded source tiles may be prefetched ahead of the GPU tile.

    Compared to running :func:`ConvertImagesInDictGpu` followed by
    :func:`BuildTilePyramids`, this approach:

    - Reads each source tile **once** regardless of the number of pyramid levels.
    - Keeps contrast-adjusted float32 data on the GPU for free downsampling.
    - Eliminates all intermediate NFS read-write cycles between pyramid levels.

    Falls back to :func:`ConvertImagesInDictPyramid` (CPU) when CuPy is unavailable
    or tiles have mixed shapes.

    :param ImagesToConvertDict: ``{input_path → output_path}`` for the finest level.
    :param PyramidOutputDicts: List of ``{input_path → output_path}`` dicts, one per
        coarser pyramid level in ascending order (level 2, 4, 8, …).  Keys must be a
        subset of ``ImagesToConvertDict`` keys.
    :param InputBpp: Bits-per-pixel of input images (auto-detected when ``None``).
    :param OutputBpp: Bits-per-pixel for outputs (matches InputBpp when ``None``).
    :param MinMax: ``(min, max)`` intensity cutoff tuple.
    :param Gamma: Gamma correction value (``None`` or 1.0 = no correction).
    :param batch_bytes: Host-memory budget for decoded source tiles resident ahead of
        the GPU.  When the whole section fits (``section_bytes <= batch_bytes``) all
        loads are submitted upfront; otherwise ``cpu_count()`` loader threads feed a
        result queue capped at ``prefetch_count = batch_bytes // tile_float32_bytes``.
        Defaults to ``CONVERT_IMAGES_GPU_PYRAMID_BATCH_BYTES`` (2 GB unless overridden
        by the ``NORNIR_GPU_PYRAMID_BATCH_MB`` environment variable).
    :return: ``True`` if any images were converted.
    """
    if batch_bytes is None:
        batch_bytes = CONVERT_IMAGES_GPU_PYRAMID_BATCH_BYTES
    if not nornir_imageregistration.HasCupy() or not nornir_imageregistration.UsingCupy():
        return ConvertImagesInDictPyramid(ImagesToConvertDict, PyramidOutputDicts,
                                          InputBpp=InputBpp, OutputBpp=OutputBpp,
                                          MinMax=MinMax, Gamma=Gamma,
                                          progress_name=progress_name,
                                          progress_task_key=progress_task_key or 'ConvertImagesInDictGpuPyramid')

    if len(ImagesToConvertDict) == 0:
        return False

    if MinMax is not None and MinMax[0] > MinMax[1]:
        raise ValueError("Invalid MinMax parameter passed to ConvertImagesInDictGpuPyramid")

    if InputBpp is None:
        for k in ImagesToConvertDict.keys():
            if os.path.exists(k):
                InputBpp = nornir_shared.images.GetImageBpp(k)
                break

    if OutputBpp is None:
        OutputBpp = InputBpp

    gamma_val = float(Gamma) if Gamma is not None else 1.0

    prettyoutput.CurseString('Stage', "ConvertImagesInDictGpuPyramid")

    input_paths = list(ImagesToConvertDict.keys())
    output_paths = [ImagesToConvertDict[p] for p in input_paths]
    n_tiles = len(input_paths)
    n_pyramid_levels = len(PyramidOutputDicts)

    # ------------------------------------------------------------------
    # Bootstrap: load the first tile synchronously to learn shape / dtype
    # before sizing the loader pool.  Getting a tile onto the GPU is the
    # expensive step, so once it is resident we produce its entire pyramid
    # in one pass rather than re-reading the source per level.
    # ------------------------------------------------------------------
    first_array: np.ndarray | None = None
    try:
        first_array = _LoadImageByExtension(input_paths[0], None)
    except Exception as exc:
        prettyoutput.LogErr(
            f"ConvertImagesInDictGpuPyramid: failed to load first tile {input_paths[0]}\n{exc}")

    if first_array is None:
        return False

    tile_shape = first_array.shape
    original_dtype = first_array.dtype
    is_int = nornir_imageregistration.IsIntArray(original_dtype)
    max_int_val = float(nornir_imageregistration.ImageMaxPixelValue(first_array)) if is_int else None
    scale = (1.0 / max_int_val) if max_int_val else 1.0

    if MinMax is not None:
        min_val = float(MinMax[0])
        max_val = float(MinMax[1])
        if is_int and max_int_val is not None:
            min_val /= max_int_val
            max_val /= max_int_val
    else:
        min_val, max_val = 0.0, 1.0

    tile_float32_elems = int(np.prod(tile_shape))
    tile_float32_bytes = tile_float32_elems * 4

    # ------------------------------------------------------------------
    # I/O pools — thread pools (Pillow GIL-releasing, no IPC overhead).
    #
    # Load dispatch is hybrid, sized by the *batch_bytes* host-memory budget:
    #
    #   - all-upfront: when the whole section fits in the budget
    #     (section_bytes <= batch_bytes) every load is submitted at once
    #     (matches ConvertImagesInDictGpu) so the GPU never waits on NFS.
    #
    #   - bounded queue: otherwise cpu_count() loader threads read in
    #     parallel but a result queue capped at prefetch_count applies
    #     backpressure, keeping at most ~batch_bytes of decoded tiles
    #     resident while still feeding the GPU continuously.
    #
    # save pool: cpu×2 workers; saves overlap GPU work on the next tile.
    #
    # wait_completion() is called at the end; shutdown() is intentionally
    # NOT called so the named pools survive (cached in nornir_pools) and
    # are reused on subsequent calls.
    # ------------------------------------------------------------------
    prefetch_count = max(1, batch_bytes // tile_float32_bytes)
    section_bytes = n_tiles * tile_float32_bytes
    all_upfront = section_bytes <= batch_bytes

    num_load_workers = min(multiprocessing.cpu_count(), n_tiles)
    num_save_workers = min(multiprocessing.cpu_count() * 2,
                           (n_tiles * (n_pyramid_levels + 1)) + 1)
    save_pool = nornir_pools.GetThreadPool("ConvertImagesInDictGpuPyramid_save", num_save_workers)

    task_key = progress_task_key or 'ConvertImagesInDictGpuPyramid'
    reporter = _TaskProgressReporter(task_key, n_tiles, name=progress_name)
    tiles_completed = 0

    # Single-tile pinned host buffer reused for every H→D transfer.
    # np.frombuffer needs explicit count= when wrapping a PinnedMemoryPointer.
    pinned = cp.cuda.alloc_pinned_memory(tile_float32_bytes)
    pinned_buf = np.frombuffer(pinned, dtype=np.float32,
                               count=tile_float32_elems).reshape(tile_shape)

    def _png_kw(path: str) -> dict:
        return {'optimize': True} if os.path.splitext(path)[1].lower() == '.png' else {}

    def _process_tile(in_path: str, out_path: str, arr: np.ndarray | None) -> None:
        """Contrast-adjust one tile and write its full pyramid (GPU, CPU fallback)."""
        if arr is None:
            return

        if arr.shape != tile_shape:
            # Mixed shape — handle this tile on the CPU so the GPU buffer
            # (sized for tile_shape) stays valid.  Other tiles keep the GPU path.
            prettyoutput.Log(
                f"ConvertImagesInDictGpuPyramid: tile shape {arr.shape} != {tile_shape}; "
                f"processing {in_path} on CPU")
            result = _ConvertSingleImage(arr, MinMax=MinMax, Gamma=gamma_val, Bpp=InputBpp)
            save_pool.add_task(f"save {out_path}", SaveImage,
                               out_path, result, bpp=OutputBpp, **_png_kw(out_path))
            prev_cpu = result
            for pyr_output_dict in PyramidOutputDicts:
                prev_cpu = _downsample2x_cpu(prev_cpu)
                pyr_out = pyr_output_dict.get(in_path)
                if pyr_out is None:
                    continue
                save_pool.add_task(f"save {pyr_out}", SaveImage,
                                   pyr_out, prev_cpu, bpp=OutputBpp, **_png_kw(pyr_out))
            return

        # Single-pass cast + normalise into the reused pinned buffer.
        np.multiply(arr, scale, out=pinned_buf, casting='unsafe')

        # H→D: single-tile DMA.
        gpu_tile = cp.asarray(pinned_buf)

        # GPU: level / gamma / clip (operates on (H, W) via ... indexing).
        gpu_tile = _apply_contrast_gpu(gpu_tile, min_val, max_val, gamma_val,
                                       max_int_val if is_int else None)

        # Level 1: D→H (fresh host array, independent of pinned_buf) and save.
        result_l1 = cp.asnumpy(gpu_tile.astype(original_dtype))
        save_pool.add_task(f"save {out_path}", SaveImage,
                           out_path, result_l1, bpp=OutputBpp, **_png_kw(out_path))

        # Pyramid levels: chain 2× downsamples in float32 on the GPU; for each
        # requested level convert to target dtype, D→H, and dispatch the save.
        prev = gpu_tile
        for pyr_output_dict in PyramidOutputDicts:
            prev = _downsample2x_gpu(prev)
            pyr_out = pyr_output_dict.get(in_path)
            if pyr_out is None:
                continue
            result_pyr = cp.asnumpy(prev.astype(original_dtype))
            save_pool.add_task(f"save {pyr_out}", SaveImage,
                               pyr_out, result_pyr, bpp=OutputBpp, **_png_kw(pyr_out))

        del gpu_tile, prev

    def _after_tile_processed(in_path: str | None = None) -> None:
        nonlocal tiles_completed
        tiles_completed += 1
        if in_path is not None:
            reporter.update(
                tiles_completed,
                element=os.path.basename(in_path),
                path=in_path)
        else:
            reporter.update(tiles_completed)

    try:
        reporter.start()
        if all_upfront:
            load_pool = nornir_pools.GetThreadPool(
                "ConvertImagesInDictGpuPyramid_parallel_load", num_load_workers)
            load_tasks: list = [None] * n_tiles
            for idx in range(1, n_tiles):
                load_tasks[idx] = load_pool.add_task(
                    input_paths[idx], _LoadImageByExtension, input_paths[idx], None)

            for i, (in_path, out_path) in enumerate(zip(input_paths, output_paths)):
                if i == 0:
                    arr = first_array
                else:
                    arr = None
                    try:
                        arr = load_tasks[i].wait_return()
                    except Exception as exc:
                        prettyoutput.LogErr(
                            f"ConvertImagesInDictGpuPyramid: load failed {in_path}\n{exc}")
                _process_tile(in_path, out_path, arr)
                arr = None
                _after_tile_processed(in_path)

            load_pool.wait_completion()
        else:
            work_queue: queue.Queue = queue.Queue()
            load_queue: queue.Queue = queue.Queue(maxsize=prefetch_count)
            _stop = object()

            def _loader_worker() -> None:
                while True:
                    item = work_queue.get()
                    if item is _stop:
                        return
                    idx = item
                    arr_i: np.ndarray | None = None
                    try:
                        arr_i = _LoadImageByExtension(input_paths[idx], None)
                    except Exception as exc:
                        prettyoutput.LogErr(
                            f"ConvertImagesInDictGpuPyramid: load failed {input_paths[idx]}\n{exc}")
                    load_queue.put((idx, arr_i))

            for idx in range(1, n_tiles):
                work_queue.put(idx)
            for _ in range(num_load_workers):
                work_queue.put(_stop)

            loader_threads = [
                threading.Thread(target=_loader_worker,
                                 name=f"GpuPyramidLoader-{w}", daemon=True)
                for w in range(num_load_workers)
            ]
            for t in loader_threads:
                t.start()

            _process_tile(input_paths[0], output_paths[0], first_array)
            first_array = None
            _after_tile_processed(input_paths[0])

            for _ in range(n_tiles - 1):
                idx, arr = load_queue.get()
                _process_tile(input_paths[idx], output_paths[idx], arr)
                arr = None
                _after_tile_processed(input_paths[idx])

            for t in loader_threads:
                t.join()
    finally:
        try:
            save_pool.wait_completion()
        finally:
            reporter.complete()
    # Intentionally not calling shutdown() — pools are reused across calls.

    return n_tiles > 0


def CropImageRect(imageparam, bounding_rect, cval=None):
    return CropImage(imageparam, Xo=int(bounding_rect[1]), Yo=int(bounding_rect[0]), Width=int(bounding_rect.Width),
                     Height=int(bounding_rect.Height), cval=cval)


def CropImage(imageparam: NDArray | str, Xo: int, Yo: int, Width: int, Height: int,
              cval: float | int | str | None = None,
              image_stats: nornir_imageregistration.ImageStats | None = None):
    """
       Crop the image at the passed bounds and returns the cropped ndarray.
       If the requested area is outside the bounds of the array then the correct region is returned
       with a background color set

       :param ndarray imageparam: An ndarray image to crop.  A string containing a path to an image is also acceptable.e
       :param int Xo: X origin for crop
       :param int Yo: Y origin for crop
       :param int Width: New width of image
       :param int Height: New height of image
       :param int cval: default value for regions outside the original image boundaries.  Defaults to 0.  Use 'random' to fill with random noise matching images statistical profile

       :return: Cropped image
       :rtype: ndarray
       """
    xp = cp.get_array_module(imageparam)  # type: ignore[arg-type]

    image = ImageParamToImageArray(imageparam)

    if image is None:
        return None

    #     if not isinstance(Width, int):
    #         Width = int(Width)
    #
    #     if not isinstance(Height, int):
    #         Height = int(Height)

    assert (isinstance(Width, int) or isinstance(Width, np.integer))
    assert (isinstance(Height, int) or isinstance(Height, np.integer))

    if isinstance(cval, str) and cval != 'random':
        raise ValueError("'random' is the only supported string argument for cval")

    # CuPy elementwise kernels do not accept Python's "False"/"True" identifiers in generated CUDA C.
    # Normalize bool (and numpy bool scalars) to 0/1 early so downstream fill paths are numeric.
    if isinstance(cval, (bool, np.bool_)):
        cval = int(cval)

    if Width < 0:
        raise ValueError("Negative dimensions are not allowed")

    if Height < 0:
        raise ValueError("Negative dimensions are not allowed")

    image_rectangle = nornir_imageregistration.Rectangle([0, 0, image.shape[0], image.shape[1]])
    crop_rectangle = nornir_imageregistration.Rectangle.CreateFromPointAndArea((Yo, Xo), (Height, Width))

    overlap_rectangle = nornir_imageregistration.Rectangle.overlap_rect(image_rectangle, crop_rectangle)

    in_startY = Yo
    in_startX = Xo
    in_endX = Xo + Width
    in_endY = Yo + Height

    out_startY = 0
    out_startX = 0
    out_endX = Width
    out_endY = Height

    if overlap_rectangle is None:
        out_startY = 0
        out_startX = 0
        out_endX = 0
        out_endY = 0

        in_startY = Yo
        in_startX = Xo
        in_endX = Xo
        in_endY = Yo
    else:
        (in_startY, in_startX) = overlap_rectangle.BottomLeft
        (in_endY, in_endX) = overlap_rectangle.TopRight

        (out_startY, out_startX) = overlap_rectangle.BottomLeft - crop_rectangle.BottomLeft
        (out_endY, out_endX) = np.array([out_startY, out_startX]) + overlap_rectangle.Size

    # To correct a numpy warning, convert values to int
    in_startX = int(in_startX)
    in_startY = int(in_startY)
    in_endX = int(in_endX)
    in_endY = int(in_endY)

    out_startX = int(out_startX)
    out_startY = int(out_startY)
    out_endX = int(out_endX)
    out_endY = int(out_endY)

    # Create mask
    rMask = None
    if cval == 'random':
        rMask = xp.zeros((Height, Width), dtype=bool)
        rMask[out_startY:out_endY, out_startX:out_endX] = True

    # Create output image
    cropped = None
    if cval is None:
        cropped = xp.zeros((Height, Width), dtype=image.dtype)
    elif cval == 'random':
        cropped = xp.ones((Height, Width), dtype=image.dtype)
    else:
        cropped = xp.ones((Height, Width), dtype=image.dtype) * cval
        cropped = cropped.astype(image.dtype, copy=False)
        if not (cropped.dtype == image.dtype):  # For some reason != operator returns an incorrect answer, but == works
            raise ValueError(f"cval (={cval}) changed the dtype of the input")

    cropped[out_startY:out_endY, out_startX:out_endX] = image[in_startY:in_endY, in_startX:in_endX]

    if rMask is not None:
        return RandomNoiseMask(cropped, rMask, Copy=False, imagestats=image_stats)  # type: ignore[arg-type]

    return cropped


def close_shared_memory(input: nornir_imageregistration.Shared_Mem_Metadata | SharedMemory):
    """
    Checks if the input is shared memory, if it is, closes it to indicate
    this process is done using it, but others may still be using it.
    Note that once this function executes the dictionary entry is removed and
    the memory cannot be unlinked.  So make sure the array does not go out of
    scope if you are responsible for unlinking it.
    """
    if isinstance(input, nornir_imageregistration.Shared_Mem_Metadata):
        if input.shared_memory is not None:
            input.shared_memory.close()
    elif isinstance(input, SharedMemory):
        try:
            input.close()
        except Exception as e:
            prettyoutput.LogErr(f"Error closing shared memory {input.name}\n{e}")
            return

        # Legacy: per-allocation close was inlined elsewhere; dict cleanup handled on unlink.


def _posix_dev_shm_avail_bytes() -> int | None:
    """Best-effort free space on /dev/shm (POSIX shared memory). None if unknown or non-POSIX."""
    if os.name != "posix":
        return None
    shm_path = "/dev/shm"
    if not os.path.isdir(shm_path):
        return None
    try:
        st = os.statvfs(shm_path)
    except OSError:
        return None
    return int(st.f_bavail) * int(st.f_frsize)


_NP_POOL_SHM_HEADROOM = 256 * 1024


def _unlink_memmap_path_quiet(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def unlink_shared_memory(input: nornir_imageregistration.Shared_Mem_Metadata | memmap_metadata) -> None:
    """
    Checks if the input is shared memory, if it is, closes it to indicate
    this process is done using it and unlinks it to free the underlying
    memory block.  This renders it unusable for all other processes as well.
    Make sure the array does not go out of
    scope if you are responsible for unlinking it.

    For :class:`memmap_metadata` (file-backed pool buffers), removes the backing file.
    """
    if isinstance(input, memmap_metadata):
        _unlink_memmap_path_quiet(input.path)
        return
    if isinstance(input, nornir_imageregistration.Shared_Mem_Metadata):
        if input.name in __known_shared_memory_allocations:
            shared_mem, finalizer = __known_shared_memory_allocations[input.name]
            shared_mem.unlink()
            try:
                del __known_shared_memory_allocations[input.name]
            except KeyError:
                pass

            finalizer()
        else:
            prettyoutput.LogErr(f"Missing memory block, could not unlink {input.name}")


def _np_array_to_memmap_pool_file(host_arr: NDArray, read_only: bool) -> tuple[memmap_metadata, NDArray]:
    """Copy *host_arr* to a temp file and open it as memmap for multiprocess pool handoff."""
    fallback_root = (
        os.environ.get("NORNIR_MEMMAP_POOL_DIR")
        or os.environ.get("TESTOUTPUTPATH")
        or tempfile.gettempdir()
    )
    os.makedirs(fallback_root, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="nir-pool-", suffix=".shm-fallback", dir=fallback_root)
    os.close(fd)
    shape_tuple = host_arr.shape
    mmw = np.memmap(path, dtype=host_arr.dtype, shape=shape_tuple, mode="w+")
    np.copyto(mmw, host_arr)
    mmw.flush()
    del mmw
    mmap_mode = "r" if read_only else "r+"
    mm = np.memmap(path, dtype=host_arr.dtype, shape=shape_tuple, mode=mmap_mode)
    meta = memmap_metadata(
        path,
        shape=np.asarray(shape_tuple, dtype=np.int64),
        dtype=host_arr.dtype,
        mode=mmap_mode,
    )
    weakref.finalize(mm, _unlink_memmap_path_quiet, path)
    return meta, mm


def npArrayToSharedArray(input: NDArray, read_only: bool = True) -> tuple[
    nornir_imageregistration.Shared_Mem_Metadata | memmap_metadata, NDArray]:
    """Creates a shared memory block (or a file-backed memmap if /dev/shm is too small) and copies
    the input array into it.  This block must be released with :func:`unlink_shared_memory` when no
    longer needed.

    :return: Metadata (:class:`Shared_Mem_Metadata` or :class:`memmap_metadata`) and a NumPy array
        view of the backing storage for use in the current process.
    """
    if cp.get_array_module(input) is cp:
        host_arr = np.ascontiguousarray(cp.asnumpy(input))
    else:
        host_arr = np.ascontiguousarray(np.asarray(input))
    nbytes = int(host_arr.nbytes)
    avail = _posix_dev_shm_avail_bytes()
    # If nbytes exceeds free /dev/shm (common default 64 MiB in Docker/WSL), filling POSIX shm
    # faults sparse pages beyond the filesystem capacity and can raise SIGBUS instead of ENOSPC.
    if avail is not None and nbytes + _NP_POOL_SHM_HEADROOM > avail:
        return _np_array_to_memmap_pool_file(host_arr, read_only)

    shared_mem = SharedMemory(size=nbytes, create=True)
    shared_array = np.ndarray(host_arr.shape, dtype=host_arr.dtype, buffer=shared_mem.buf, order="C")
    np.copyto(shared_array, host_arr)
    output = nornir_imageregistration.Shared_Mem_Metadata(name=shared_mem.name, dtype=shared_array.dtype,
                                                          shape=shared_array.shape, readonly=read_only,
                                                          shared_memory=None)

    # Create a finalizer to close the shared memory when the array is garbage collected
    finalizer = weakref.finalize(shared_array, close_shared_memory, shared_mem)
    __known_shared_memory_allocations[shared_mem.name] = (shared_mem, finalizer)
    return output, shared_array


def create_shared_memory_array(shape: nornir_imageregistration.ShapeLike, dtype: DTypeLike,
                               read_only: bool = True) -> tuple[
    nornir_imageregistration.Shared_Mem_Metadata, NDArray]:
    """Creates a shared memory block and copies the input array to shared memory.  This memory block must be unlinked
    when it is no longer in use.

    :param shape: Output shape.  Any shape-like is accepted: tuple, list, NumPy array or CuPy
        array.  The backing buffer is always host memory, so a device-resident shape is brought
        across here rather than at every call site.
    :return: The name of the shared memory and a shared memory array.  Used to reduce memory footprint when passing parameters to multiprocess pools
    """
    # This used to require an ndarray, because it read shape.prod() and handed *shape* straight
    # to np.ndarray. Both callers got it wrong in different ways -- one passed a tuple, which has
    # no .prod(), and one passed a CuPy array, which np.ndarray will not interpret as a shape --
    # so every branch of the return_shared_memory path raised. Normalising here fixes both and
    # matches what the parameter name already promises. (#102)
    shape = tuple(int(extent) for extent in np.ravel(nornir_imageregistration.EnsureNumpyArray(shape)))

    # shared_memory_manager = nornir_pools.get_or_create_shared_memory_manager()
    byte_size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    # shared_mem = shared_memory_manager.SharedMemory(size=int(byte_size))
    shared_mem = SharedMemory(size=int(byte_size), create=True)
    shared_array = np.ndarray(shape, dtype=dtype, buffer=shared_mem.buf)
    output = nornir_imageregistration.Shared_Mem_Metadata(name=shared_mem.name, dtype=shared_array.dtype,
                                                          shape=shared_array.shape, readonly=read_only,
                                                          shared_memory=None)

    # Create a finalizer to close the shared memory when the array is garbage collected
    finalizer = weakref.finalize(shared_array, close_shared_memory, shared_mem)
    # finalizer = None
    __known_shared_memory_allocations[shared_mem.name] = (shared_mem, finalizer)
    return output, shared_array


def _value_range_fits_dtype(min_val: float, max_val: float, dtype: DTypeLike) -> bool:
    """Return True if [min_val, max_val] lies within the finite range of *dtype*."""
    dt = np.dtype(dtype)
    if min_val > max_val:
        return False
    if np.issubdtype(dt, np.floating):
        finfo = np.finfo(np.dtype(dt).name)
        # Use Python floats so comparisons never promote operands to float16 (overflow).
        lo = float(finfo.min)
        hi = float(finfo.max)
        return float(min_val) >= lo and float(max_val) <= hi
    if np.issubdtype(dt, np.integer):
        iinfo = np.iinfo(np.dtype(dt).name)
        return float(min_val) >= float(iinfo.min) and float(max_val) <= float(iinfo.max)
    return True


def promote_dtype_for_value_range(
    preferred_dtype: DTypeLike,
    min_val: float,
    max_val: float,
) -> np.dtype:
    """
    Choose a dtype that can represent *min_val* and *max_val*, preferring types
    at least as wide as *preferred_dtype* when it is floating.

    For floating preferences, tries float16 → float32 → float64. For integer
    preferences, tries wider integers then falls back to float promotion.
    """
    if min_val > max_val:
        raise ValueError(f"min_val ({min_val}) must be <= max_val ({max_val})")

    pref = np.dtype(preferred_dtype)

    if np.issubdtype(pref, np.floating):
        candidates: list = []
        for name in ("float16", "float32", "float64"):
            dt = np.dtype(name)
            if dt.itemsize >= pref.itemsize:
                candidates.append(dt)
        if not candidates:
            candidates = [np.dtype("float64")]
        for dt in candidates:
            if _value_range_fits_dtype(min_val, max_val, dt):
                return dt
        raise ValueError(
            f"min_val={min_val} max_val={max_val} cannot be represented in float64"
        )

    if np.issubdtype(pref, np.integer):
        if _value_range_fits_dtype(min_val, max_val, pref):
            return pref
        for wider_name in ("int32", "int64"):
            wdt = np.dtype(wider_name)
            if _value_range_fits_dtype(min_val, max_val, wdt):
                return wdt
        return promote_dtype_for_value_range(np.dtype("float32"), min_val, max_val)

    return promote_dtype_for_value_range(np.dtype("float32"), min_val, max_val)


DEFAULT_RANDOM_DATA_SEED = 0x6E6F726E  # 'norn'

# One generator per array module, so a run is reproducible from its first call.
# A fixed seed *per call* would be wrong: phase correlation pads both the target and
# the source, and giving them the same noise would correlate the padding regions and
# hand the correlation a peak that is not in the data. The generator therefore
# advances between calls, and only its starting state is pinned.
_random_generators: dict[str, typing.Any] = {}


def seed_random_data(seed: int | None = DEFAULT_RANDOM_DATA_SEED) -> None:
    """Reset the generators backing :func:`GenRandomData`.

    Pass ``None`` to seed from entropy, restoring the old non-reproducible
    behaviour for callers that genuinely want a fresh draw each run.
    """
    _random_generators.clear()
    _random_generators['__seed__'] = seed


def random_generator(xp: typing.Any | None = None) -> typing.Any:
    """The generator backing every noise fill, so all of them seed together."""
    if xp is None:
        xp = nornir_imageregistration.GetComputationModule()
    return _default_random_generator(xp)


def _default_random_generator(xp: typing.Any) -> typing.Any:
    key = 'cupy' if xp is not np else 'numpy'
    generator = _random_generators.get(key)
    if generator is not None:
        return generator

    seed = _random_generators.get('__seed__', DEFAULT_RANDOM_DATA_SEED)
    generator = xp.random.default_rng(seed)
    _random_generators[key] = generator
    return generator


def GenRandomData(height: int, width: int, mean: float, standardDev: float, min_val: float, max_val: float,
                  dtype: DTypeLike | None = None,
                  xp: typing.Any | None = None,
                  rng: typing.Any | None = None) -> NDArray[np.floating]:
    """
    Generate random data of shape with the specified mean and standard deviation.
    If *xp* is None, uses ``GetComputationModule()``; otherwise uses that array module so
    output matches a caller-provided array (numpy vs cupy).

    :param rng: Generator to draw from. Defaults to a module-level generator with a
        fixed starting seed, so a given run reproduces. See :func:`seed_random_data`.
    """
    if xp is None:
        xp = nornir_imageregistration.GetComputationModule()
    resolved = nornir_imageregistration.default_image_dtype() if dtype is None else dtype
    dtype_out = promote_dtype_for_value_range(resolved, float(min_val), float(max_val))

    if not math.isfinite(mean) or not math.isfinite(standardDev):
        raise ValueError(f"mean and standardDev must be finite; got mean={mean!r} standardDev={standardDev!r}")

    if rng is None:
        rng = _default_random_generator(xp)

    image = (rng.standard_normal((int(height), int(width))) * standardDev) + mean
    xp.clip(image, a_min=min_val, a_max=max_val, out=image)
    # Benign underflow when casting float64 buffer to float16/float32; range already validated above.
    with np.errstate(under="ignore", invalid="ignore"):
        image = image.astype(dtype_out, copy=False)

    return image


def GetImageSize(image_param: str | np.ndarray | Iterable) -> NDArray[np.integer]:
    """
    :param image_param: Either a path to an image file, an ndarray, or a list
    of paths/ndimages
    :returns: The image's (height, width) or [(height,width),...] for a list
    :rtype: tuple
    """

    if isinstance(image_param, str):
        return nornir_shared.images.GetImageSize(image_param)
    elif isinstance(image_param, np.ndarray):
        return np.asarray(image_param.shape, dtype=int)
    elif isinstance(image_param, cp.ndarray):
        return np.asarray(image_param.shape, dtype=int)
    elif isinstance(image_param, Iterable):
        return np.asarray([GetImageSize(i) for i in image_param], dtype=int)

    raise ValueError(f'Unexpected image argument {image_param}')


def ForceGrayscale(image: np.ndarray):
    """
    Ensure that the image is a 2d array.  This function does not do any intelligent
    conversion to grayscale, it simple eliminates extra dimensions if they exist.
    :param: ndarray with 3 dimensions
    :returns: grayscale data
    :rtype: ndarray with 2 dimensions"""

    if len(image.shape) > 2:
        xp = cp.get_array_module(image)
        image = image[:, :, 0]
        return xp.squeeze(image)

    return image


def RgbLikeToGrayscaleLuminance(image: NDArray) -> tuple[NDArray, bool]:
    """Convert RGB/RGBA (or HxWx2 grayscale+alpha) stacks to a single 2D plane.

    Uses Rec. 601 luma coefficients on the first three channels when ``image`` is HxWx3 or HxWx4.
    HxWx1 is squeezed to 2D (returns ``False`` — not treated as an RGB file). Already-2D arrays
    are returned unchanged with ``False``.

    Works with NumPy or CuPy arrays (``cp.get_array_module``).
    """
    if image.ndim < 3:
        return image, False
    c = int(image.shape[2])
    xp = cp.get_array_module(image)
    if c == 1:
        return xp.squeeze(image, axis=2), False
    if c == 2:
        return image[..., 0], True
    if c == 3:
        r = image[..., 0].astype(xp.float64, copy=False)
        g = image[..., 1].astype(xp.float64, copy=False)
        b = image[..., 2].astype(xp.float64, copy=False)
        out = r * 0.299 + g * 0.587 + b * 0.114
        return out.astype(image.dtype, copy=False), True
    if c == 4:
        r = image[..., 0].astype(xp.float64, copy=False)
        g = image[..., 1].astype(xp.float64, copy=False)
        b = image[..., 2].astype(xp.float64, copy=False)
        out = r * 0.299 + g * 0.587 + b * 0.114
        return out.astype(image.dtype, copy=False), True
    return image[..., 0], True


def image_to_uint8(image):
    """Convert image to uint8. If input is float, scale to 0-255; if int and max > 255, scale down."""
    if image.dtype == np.uint8:
        return image

    elif image.dtype == bool:
        image = image.astype(np.uint8) * 255

    elif nornir_imageregistration.IsFloatArray(image.dtype):
        iMax = image.max()
        if iMax <= 1.0:
            image = image * 255.0  # Copy, because input may be read-only
        else:
            pass
            # image = #(255.0 / iMax)
    elif nornir_imageregistration.IsIntArray(image.dtype):
        iMax = image.max()
        if iMax > 255:
            image = image / (iMax / 255.0)

    try:
        image = image.astype(np.uint8)
    except FloatingPointError as fe:
        raise ValueError(
            "Unable to cast image to uint8 due to floating point error.  This can be caused by NaN or infinite values") from fe

    return image


def OneBit_img_from_bool_array(data):
    """Convert a boolean numpy array to a Pillow 1-bit image (workaround for Pillow bit-image handling)."""
    size = data.shape[::-1]

    if data.dtype == bool:
        return Image.frombytes(mode='1', size=size, data=np.packbits(data, axis=1))
    else:
        return Image.frombytes(mode='1', size=size, data=np.packbits(data > 0, axis=1))


def uint16_img_from_uint16_array(data):
    """Convert a uint16 numpy array to a Pillow 16-bit image (workaround for Pillow I;16 handling)."""
    assert (nornir_imageregistration.IsIntArray(data))

    size = data.shape[::-1]
    img = Image.new("I", size=data.T.shape)
    img.frombytes(data.tobytes(), 'raw', 'I;16')
    return img


def uint16_img_from_float_array(image):
    """Convert a float image (0-1 or 0-max) to a Pillow 16-bit image."""
    assert (nornir_imageregistration.IsFloatArray(image))
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    iMax = float(image.max())
    if iMax <= 1.0:
        image = image * ((1 << 16) - 1)
    image = np.clip(image, 0.0, None)
    return image.astype(np.uint16)


def SaveImage(ImageFullPath: str, image: NDArray, bpp: int | None = None, **kwargs):
    """Saves the image as greyscale with no contrast-stretching
    :param str ImageFullPath: The filename to save
    :param ndarray image: The image data to save
    :param int bpp: The bit depth to save, if the image data bpp is higher than this value it will be reduced.  Otherwise only the bpp required to preserve the image data will be used. (8-bit data will not be upsampled to 16-bit)
    """
    dirname = os.path.dirname(ImageFullPath)
    may_need_to_create_dir = dirname is not None and len(dirname) > 0

    image = nornir_imageregistration.EnsureNumpyArray(image)

    if bpp is None:
        if nornir_imageregistration.IsFloatArray(image):
            # Float storage width (e.g. float32 itemsize) is not export bit depth.
            bpp = 16
        else:
            bpp = nornir_imageregistration.ImageBpp(image)
            if bpp > 16:
                prettyoutput.LogErr(
                    "Saving image at 32 bits-per-pixel, check SaveImageParameters for efficiency:\n{0}".format(
                        ImageFullPath))
    elif nornir_imageregistration.IsFloatArray(image) and bpp > 16:
        bpp = 16

    if bpp > 8 and not nornir_imageregistration.IsFloatArray(image):
        # Ensure we even have the data to bother saving a higher bit depth
        detected_bpp = nornir_imageregistration.ImageBpp(image)
        if detected_bpp < bpp:
            bpp = detected_bpp

    (root, ext) = os.path.splitext(ImageFullPath)
    if ext == '.jp2':
        try:
            SaveImage_JPeg2000(ImageFullPath, image, **kwargs)
        except FileNotFoundError as e:
            if may_need_to_create_dir:
                os.makedirs(dirname, exist_ok=True)
                SaveImage_JPeg2000(ImageFullPath, image, **kwargs)
            else:
                raise e

    elif ext == '.npy':
        try:
            np.save(ImageFullPath, image)
        except FileNotFoundError as e:
            if may_need_to_create_dir:
                os.makedirs(dirname, exist_ok=True)
                np.save(ImageFullPath, image)
            else:
                raise e
    else:
        if np.issubdtype(image.dtype, bool) or bpp == 1:
            # Covers for pillow bug with bit images
            # https://stackoverflow.com/questions/50134468/convert-boolean-numpy-array-to-pillow-image
            # im = Image.fromarray(image.astype(np.uint8) * 255, mode='L').convert('1')
            im = OneBit_img_from_bool_array(image)
        elif bpp == 8:
            Uint8_image = image_to_uint8(image)
            del image
            im = Image.fromarray(Uint8_image, mode="L")
        elif nornir_imageregistration.IsFloatArray(image):
            # Pillow cannot construct images directly from float16 arrays.
            if image.dtype == np.float16:
                image = image.astype(np.float32, copy=False)

            if bpp <= 8:
                Uint8_image = image_to_uint8(image)
                im = Image.fromarray(Uint8_image, mode="L")
            else:
                uint16_image = uint16_img_from_float_array(image)
                if ext.lower() == '.png':
                    im = uint16_img_from_uint16_array(uint16_image)
                else:
                    im = Image.fromarray(uint16_image, mode="I;16")
        else:
            if bpp < 32:
                if ext.lower() == '.png':
                    im = uint16_img_from_uint16_array(image)
                else:
                    im = Image.fromarray(image, mode=f"I;{bpp}")
            else:
                im = Image.fromarray(image, mode=f"I;{bpp}")

        try:
            im.save(ImageFullPath, **kwargs)
        except FileNotFoundError as e:
            if may_need_to_create_dir:
                os.makedirs(dirname, exist_ok=True)
                im.save(ImageFullPath, **kwargs)
            else:
                raise e
        finally:
            im.close()
            del im 
    return


def SaveImage_JPeg2000(ImageFullPath, image, tile_dim=None):
    """Saves the image as greyscale with no contrast-stretching"""

    if tile_dim is None:
        tile_dim = (512, 512)

    Uint8_image = image_to_uint8(image)
    del image

    with Image.fromarray(Uint8_image) as im:
        im.save(ImageFullPath, tile_size=tile_dim)


# Legacy: SaveImage_JPeg2000_Tile (PIL tile save) not used; current save path uses other methods.
#

def _LoadImageByExtension(ImageFullPath: str, dtype: DTypeLike | None):
    """
    Loads an image file and returns an ndarray of dtype
    :param dtype dtype: Numpy datatype of returned array. If the type is a float then the returned array is in the range 0 to 1.  Otherwise it is whatever pillow and numpy decide.
    """
    (root, ext) = os.path.splitext(ImageFullPath)

    image = None
    try:
        if ext == '.npy':
            image = np.load(ImageFullPath, 'c')
            if dtype is not None:
                image = image.astype(dtype, copy=False)
        else:
            # image = plt.imread(ImageFullPath)
            with Image.open(ImageFullPath, "r") as im:

                expected_dtype = nornir_imageregistration.pillow_helpers.dtype_for_pillow_image(im)  # type: ignore[arg-type]
                image = np.array(im, dtype=expected_dtype)
                max_pixel_val = nornir_imageregistration.ImageMaxPixelValue(image)

                if dtype is not None:
                    if nornir_imageregistration.IsIntArray(image.dtype) and nornir_imageregistration.IsFloatArray(
                            dtype):
                        # Ensure we remap values to the range of 0 to 1 without loss before converting to desired floating type
                        # if image.dtype.itemsize == dtype.itemsize: #Check if we need to bump up the item size
                        if np.dtype(dtype).itemsize <= image.dtype.itemsize:
                            # Converting to float with the same number of bytes as the integer type can produce infinite output.
                            # To handle this, increase precision of image during conversion. 
                            temp_dtype = np.dtype(f'f{image.dtype.itemsize * 2}')
                            image = image.astype(temp_dtype)
                        else:
                            image = image.astype(dtype)

                        max_val = image.max()
                        if max_val != 0:
                            image /= max_val
                    elif nornir_imageregistration.IsFloatArray(dtype):
                        # Ensure data is in the range 0 to 1 for floating types
                        if im.mode[0] == 'F':
                            (_, im_max_val) = im.getextrema()
                            if im_max_val <= 1.0:  # type: ignore[operator]
                                return image

                        max_val = max_pixel_val
                        if max_val > 0:
                            image /= max_val

                    image = image.astype(dtype, copy=False)
                #                 else:
                #                     #Reduce to smallest integer type that can hold the data
                #                     if im.mode[0] == 'I' and (np.issubdtype(image.dtype, np.int32) or np.issubdtype(image.dtype, np.uint32)):
                #                         (min_val, max_val) = im.getextrema()
                #                         smallest_dtype = np.uint32
                #                         if max_val <= 65535:
                #                             smallest_dtype = np.uint16
                #                         if max_val <= 255:
                #                             smallest_dtype = np.uint8
                #
                #                         image = image.astype(smallest_dtype)
                #
                #                     dtype = image.dtype

                

    except IOError as E:
        prettyoutput.LogErr("IO error loading image {0}\n{1}".format(ImageFullPath, str(E)))
        raise
    # except Exception as E:
    #     prettyoutput.LogErr("Unexpected exception loading image {0}\n{1}".format(ImageFullPath, str(E)))
    #     import traceback
    #     traceback.print_exc()
    #     raise

    return image


# @profile
def LoadImage(ImageFullPath: str,
              ImageMaskFullPath: str | None = None,
              MaxDimension: float | None = None,
              dtype: DTypeLike | None = None,
              backend: Literal["numpy", "cupy"] | None = None):
    """
    Loads an image converts to greyscale, masks it, and removes extrema pixels.

    :param dtype:
    :param str ImageFullPath: Path to image
    :param str ImageMaskFullPath: Path to mask, dimension should match input image
    :param MaxDimension: Limit the largest dimension of the returned image to this size.  Downsample if necessary.
    :param backend: If "numpy", return a NumPy array (error if conversion fails). If "cupy", return a CuPy array (error if CuPy not available). If None, use the active computation backend (current behaviour).
    :returns: Loaded image.  Masked areas and extrema pixel values are replaced with gaussian noise matching the median and std. dev. of the unmasked image.
    :rtype: ndimage
    """
    if not os.path.isfile(ImageFullPath):
        # logger = logging.getLogger(__name__)
        prettyoutput.LogErr(f'File does not exist: {ImageFullPath}')
        raise FileNotFoundError(f"Unable to load image: {ImageFullPath}")

    (root, ext) = os.path.splitext(ImageFullPath)

    image = _LoadImageByExtension(ImageFullPath, dtype)

    if not MaxDimension is None:
        scalar = ScalarForMaxDimension(MaxDimension, image.shape)
        if scalar < 1.0:
            image = ScaleImage(image, scalar)

    image_mask = None

    if ImageMaskFullPath is not None:
        if not os.path.isfile(ImageMaskFullPath):
            # logger = logging.getLogger(__name__)
            prettyoutput.LogErr('Fixed image mask file does not exist: ' + ImageMaskFullPath)
        else:
            image_mask = _LoadImageByExtension(ImageMaskFullPath, bool)
            if nornir_imageregistration.UsingCupy():
                image_mask = cp.array(image_mask)

            if MaxDimension is not None:
                scalar = ScalarForMaxDimension(MaxDimension, image_mask.shape)
                if scalar < 1.0:
                    image_mask = ScaleImage(image_mask, scalar)

            assert (image.shape == image_mask.shape)
            image = RandomNoiseMask(image, image_mask)
    elif backend is None and nornir_imageregistration.UsingCupy():
        image = cp.asarray(image)

    if backend == "numpy":
        image = image.get() if hasattr(image, "get") else np.asarray(image)  # type: ignore[union-attr]
    elif backend == "cupy":
        if not nornir_imageregistration.HasCupy():
            raise RuntimeError("CuPy is not available; cannot return cupy array from LoadImage(..., backend='cupy')")
        if not isinstance(image, cp.ndarray):
            image = cp.asarray(image)

    return image


def NormalizeImage(image: NDArray):
    """Adjusts the image to have a range of 0 to 1.0"""

    xp = cp.get_array_module(image)
    miniszeroimage = image - image.min()
    denom = miniszeroimage.max()
    scalar_dtype = np.result_type(np.dtype(miniszeroimage.dtype), np.float32)
    scalar = xp.asarray(1.0, dtype=scalar_dtype) / denom
    if xp.any(xp.isinf(scalar)):
        scalar = xp.asarray(1.0, dtype=scalar_dtype)

    typecode = 'f%d' % image.dtype.itemsize
    return (miniszeroimage * scalar).astype(typecode, copy=False)


def TileGridShape(source_image_shape: nornir_imageregistration.Rectangle | tuple[float, float] | NDArray,
                  tile_size: tuple[float, float] | tuple[int, int] | NDArray):
    """Given an image and tile size, return the dimensions of the grid"""

    if isinstance(source_image_shape, nornir_imageregistration.Rectangle):
        source_image_shape = source_image_shape.shape
    elif isinstance(source_image_shape, np.ndarray):
        pass
    else:
        source_image_shape = np.asarray(source_image_shape)

    if not isinstance(tile_size, np.ndarray):
        tile_shape = np.asarray(tile_size)
    else:
        tile_shape = tile_size

    return np.ceil(source_image_shape / tile_shape).astype(np.int32, copy=False)


def ImageToTiles(source_image: NDArray,
                 tile_size: nornir_imageregistration.ShapeLike,
                 grid_shape: nornir_imageregistration.ShapeLike | None = None,
                 cval: int | None = 0):
    """
    :param ndarray source_image: Image to cut into tiles
    :param array tile_size: Shape of each tile
    :param array grid_shape: Dimensions of grid, if None the grid is large enough to reproduce the source_image with zero padding if needed
    :param object cval: Fill value for images that are padded.  Default is zero.  Use 'random' to generate random noise
    :return: Dictionary of images indexed by tuples
    """
    # Build the output dictionary
    grid = {}
    for (iRow, iCol, tile) in ImageToTilesGenerator(source_image, tile_size):  # type: ignore[arg-type]
        grid[iRow, iCol] = tile

    return grid


def ImageToTilesGenerator(source_image: NDArray,
                          tile_size: NDArray,
                          grid_shape: NDArray | None = None,
                          coord_offset: NDArray | None = None,
                          cval: float | int | str | None = 0,
                          coverage_mask: NDArray | None = None):
    """An iterator generating that divides a large image into a collection of smaller non-overlapping tiles.
    :param source_image: The image to divide
    :param tile_size: Shape of each tile
    :param grid_shape: Dimensions of grid, if None the grid is large enough to reproduce the source_image with zero padding if needed
    :param tuple coord_offset: Add this amount to coordinates returned by this function, used if the image passed is part of a larger image
    :param object cval: Fill value for images that are padded.  Default is zero.  Use 'random' to generate random noise
    :param coverage_mask: When set, only yield tiles where this boolean mask has any True pixels in the tile ROI
    :return: (iRow, iCol, tile_image)
    """
    source_image = ImageParamToImageArray(source_image)

    if grid_shape is None:
        grid_shape = TileGridShape(source_image.shape, tile_size)

    if coord_offset is None:
        coord_offset = np.array([0, 0])

    (required_shape) = grid_shape * tile_size
    req_h = int(math.ceil(float(required_shape[0])))
    req_w = int(math.ceil(float(required_shape[1])))
    src_h, src_w = int(source_image.shape[0]), int(source_image.shape[1])
    if (src_h, src_w) != (req_h, req_w):
        source_image_padded = CropImage(source_image,
                                        Xo=0, Yo=0,
                                        Width=int(math.ceil(required_shape[1])),
                                        Height=int(math.ceil(required_shape[0])),
                                        cval=0)
    else:
        source_image_padded = source_image

    coverage_mask_padded = None
    if coverage_mask is not None:
        coverage_mask = ImageParamToImageArray(coverage_mask)
        mask_h, mask_w = int(coverage_mask.shape[0]), int(coverage_mask.shape[1])
        if (mask_h, mask_w) != (req_h, req_w):
            coverage_mask_padded = CropImage(coverage_mask,
                                             Xo=0, Yo=0,
                                             Width=int(math.ceil(required_shape[1])),
                                             Height=int(math.ceil(required_shape[0])),
                                             cval=False)
        else:
            coverage_mask_padded = coverage_mask

    # nornir_imageregistration.ShowGrayscale(source_image_padded)

    # Build the output dictionary
    StartY = 0
    EndY = tile_size[0]

    for iRow in range(grid_shape[0]):

        StartX = 0
        EndX = tile_size[1]

        for iCol in range(grid_shape[1]):
            tile_image = source_image_padded[StartY:EndY, StartX:EndX]  # type: ignore[index]
            if coverage_mask_padded is not None:
                mask_tile = coverage_mask_padded[StartY:EndY, StartX:EndX]
                if not bool(np.any(np.asarray(
                        nornir_imageregistration.EnsureNumpyArray(mask_tile)))):
                    StartX += tile_size[1]
                    EndX += tile_size[1]
                    continue
            t = (iRow + coord_offset[0], iCol + coord_offset[1], tile_image)
            # nornir_imageregistration.ShowGrayscale(tile)
            (yield t)

            StartX += tile_size[1]
            EndX += tile_size[1]

        StartY += tile_size[0]
        EndY += tile_size[0]

    return


def GetImageTile(source_image, iRow, iCol, tile_size):
    StartY = tile_size[0] * iRow
    EndY = StartY + tile_size[0]
    StartX = tile_size[1] * iCol
    EndX = StartX + tile_size[1]

    return source_image[StartY:EndY, StartX:EndX]


def RandomNoiseMask(image: NDArray, Mask: NDArray[np.bool_],
                    imagestats: nornir_imageregistration.image_stats.ImageStats | None = None, Copy=False) -> NDArray:
    """
    Fill the masked area with random noise with gaussian distribution about the image
    mean and with standard deviation matching the image's standard deviation.  Mask
    pixels that are False will be replaced with random noise

    :param ndimage image: Input image
    :param ndimage Mask: Mask, zeros are replaced with noise.  Ones pull values from input image
    :param ImageStats imagestats: Image stats.  Calculated from image if none
    :param bool Copy: Returns a copy of input image if true, otherwise write noise to the input image
    :rtype: ndimage
    """

    image = ImageParamToImageArray(image)
    Mask = ImageParamToImageArray(Mask)
    xp = cp.get_array_module(image)
    if cp.get_array_module(Mask) is not xp:
        Mask = xp.asarray(Mask, dtype=xp.bool_)

    assert (image.shape == Mask.shape)

    MaskedImage = image.copy() if Copy else image

    # iPixelsToReplace = Mask.flat == 0
    iPixelsToReplace = xp.logical_not(Mask.ravel())

    numInvalidPixels = xp.sum(iPixelsToReplace)

    if numInvalidPixels == 0:
        # Entire image is masked, there is no noise to create
        return MaskedImage

    Image1D = MaskedImage.ravel()

    if imagestats is None:
        numValidPixels = int(image.size) - int(numInvalidPixels)
        # Create masked array for accurate stats
        if numValidPixels == 0:
            raise ValueError("Entire image is masked, cannot calculate median or standard deviation")
            # return MaskedImage
        elif numValidPixels <= 2:
            raise ValueError(f"All but {numValidPixels} pixels are masked, cannot calculate statistics")

        if xp is not np:  # Cupy did not support masked arrays when this was written
            pixels_for_stats = Image1D[~iPixelsToReplace]
            imagestats = nornir_imageregistration.ImageStats.Create(pixels_for_stats)
            del pixels_for_stats
        else:  # The original numpy code
            UnmaskedImage1D = xp.ma.masked_array(Image1D, iPixelsToReplace).compressed()
            imagestats = nornir_imageregistration.ImageStats.Create(UnmaskedImage1D)
            del UnmaskedImage1D

    n_noise = int(numInvalidPixels)
    NoiseData = imagestats.GenerateNoise(n_noise, dtype=image.dtype, xp=xp)
    if cp.get_array_module(NoiseData) is not xp:
        if xp is np:
            NoiseData = nornir_imageregistration.EnsureNumpyArray(NoiseData, dtype=image.dtype)
        else:
            NoiseData = xp.asarray(NoiseData)
    Image1D[iPixelsToReplace] = NoiseData

    # iPixelsToReplace = transpose(nonzero(iPixelsToReplace))
    if xp is not np:  # If we used ravel() we may have copied the underlying data, so reshape Image1D and return that to ensure we get the mask
        output_image = Image1D.reshape(MaskedImage.shape)  # type: ignore[union-attr]
        return output_image
    else:  # NumPy: ravel is a view; writes through Image1D update MaskedImage.
        return MaskedImage


def EnsureMatchingImageMaskShape(image: NDArray, mask: NDArray) -> tuple[NDArray, NDArray]:
    """Crop *image* and *mask* to their overlapping top-left region when shapes differ.

    Pyramid / downsample rounding can leave image and mask off by one pixel; boolean
    indexing then fails. Prefer keeping the shared content over aborting registration.
    """
    if image.shape == mask.shape:
        return image, mask

    common_h = int(min(image.shape[0], mask.shape[0]))
    common_w = int(min(image.shape[1], mask.shape[1]))
    warnings.warn(
        f"Image shape {tuple(image.shape)} and mask shape {tuple(mask.shape)} differ; "
        f"cropping both to ({common_h}, {common_w}).",
        RuntimeWarning,
        stacklevel=2,
    )
    return image[:common_h, :common_w], mask[:common_h, :common_w]


def CreateExtremaMask(image: np.ndarray, mask: np.ndarray | None = None, size_cutoff=0.001, minima=None, maxima=None):
    """
    Returns a mask for features above a set size that are at max or min pixel value
    :param image:
    :param mask: Valid-pixel mask (True = include in analysis). Invalid regions are
        excluded from min/max and treated as extrema candidates for size filtering.
    :param minima:
    :param maxima:
    :param size_cutoff: Determines how large a continuous region must be before it is masked. If 0 to 1 this is a fraction of total area.  If > 1 it is an absolute count of pixels. If None all min/max are masked regardless of size
    :returns: Mask of extrema pixels, pixels that are FALSE are extrema to be excluded
    """
    # (minima, maxima, iMin, iMax) = scipy.ndimage.measurements.extrema(image) 

    xp = cp.get_array_module(image)
    sp = cupyx.scipy.get_array_module(image)

    if mask is not None:
        if cp.get_array_module(mask) is not xp:
            mask = xp.asarray(mask)
        mask = xp.asarray(mask, dtype=xp.bool_)
        image, mask = EnsureMatchingImageMaskShape(image, mask)
        # Exclude invalid pixels from min/max via NaN (True in mask = valid).
        image = xp.asarray(image, dtype=xp.float64).copy()
        image[~mask] = xp.nan

    if minima is None:
        minima = xp.nanmin(image) if mask is not None else image.min()

    if maxima is None:
        maxima = xp.nanmax(image) if mask is not None else image.max()

    # Pixels that are TRUE will be excluded, exclude pixels equal to the min or max.
    # However, the ndimage.label function finds features that are TRUE.  So we start with an
    # inverted mask
    extrema_mask = xp.logical_or(image == maxima, image == minima)

    if mask is not None:
        extrema_mask = xp.logical_or(extrema_mask, xp.logical_not(mask))

    if size_cutoff is None:
        return extrema_mask
    else:
        (extrema_mask_label, nLabels) = sp.ndimage.label(extrema_mask)
        if nLabels == 0:  # If there are no labels, do not mask anything
            return xp.ones(image.shape, extrema_mask.dtype)

        # Identify the label of non-extrema pixels

        label_sums = sp.ndimage.sum_labels(
            extrema_mask.astype(xp.int32), extrema_mask_label,
            xp.arange(0, nLabels, dtype=xp.int32))

        cutoff_value = None
        # if cutoff value is less than one treat it as a fraction of total area
        if size_cutoff <= 1.0:
            cutoff_value = xp.prod(xp.array(image.shape, np.int64)) * size_cutoff
        elif not isinstance(size_cutoff, int):
            warnings.warn(
                f"Expecting an integer to specify min area of labels to mask in CreateExtremaMask.  Got {size_cutoff}.")
            cutoff_value = size_cutoff
        else:
            cutoff_value = size_cutoff

        small_regions = label_sums < cutoff_value
        if xp.any(small_regions):
            cutoff_labels = xp.flatnonzero(small_regions)
            extrema_mask_minus_small_features = xp.isin(extrema_mask_label, cutoff_labels)

            # nornir_imageregistration.ShowGrayscale((image, extrema_mask, extrema_mask_minus_small_features))

            return extrema_mask_minus_small_features
        else:
            return xp.ones(image.shape, dtype=bool)


def ReplaceImageExtremaWithNoise(image: np.ndarray, imagemask: np.ndarray | None = None,
                                 imagestats: nornir_imageregistration.image_stats.ImageStats | None = None,
                                 size_cutoff: float = 0.001, Copy=True):
    """
    Replaced the min/max values in the image with random noise.  This is useful when aligning images composed mostly of dark or bright regions. 
    It is usually best to pass None for statistical parameters since the function will calculate the statistics with the extrema removed.
    :param image:
    :param Copy:
    :param numpy.ndarray imagemask: Additional pixels we wish to be included in the extrema mask
    :param nornir_imageregistration.ImageStats imagestats: Image statistics. Will be calculated if not passed.
    :param size_cutoff: 0 to 1.0, determines how large a continuos min or max region must be before it is masked. If None all min/max are masked regardless of size.  Defaults to 0.001, None will mask all min/max
    """

    # If profiling shows this is slow there are older implementations in git
    mask = CreateExtremaMask(image, imagemask, size_cutoff=size_cutoff)

    noised_image = nornir_imageregistration.RandomNoiseMask(image, mask, imagestats, Copy=Copy)
    return noised_image


def _clamped_overlap(overlap: float | None) -> float:
    """Clamp an overlap fraction into [0, 1], treating None as 0."""
    if overlap is None:
        return 0.0

    if overlap > 1.0:
        return 1.0

    if overlap < 0.0:
        return 0.0

    return overlap


def NearestPowerOfTwoWithOverlap(val: float, overlap: float = 1.0) -> int:
    """
    :param val:
    :param float overlap: Minimum amount of overlap possible between images, from 0 to 1.  Values greater than 0.5 require no increase to image size.
    :return: Same as DimensionWithOverlap, but output dimension is increased to the next power of two for faster FFT operations
    """

    overlap = _clamped_overlap(overlap)

    # Figure out the minimum dimension to accomodate the requested overlap
    min_dimension = DimensionWithOverlap(val, overlap)

    # Figure out the power of two dimension
    # return int(math.pow(2, int(math.ceil(math.log(min_dimension, 2)))))
    return 1 << int(math.ceil(math.log(min_dimension, 2)))


def DimensionWithOverlap(val, overlap=1.0):
    """
    :param float val: Original dimension
    :param float overlap: Amount of overlap possible between images, from 0 to 1
    :returns: Required dimension size to unambiguously determine the offset in an fft image
    """

    # An overlap of 50% is half of the image, so we don't need to expand the image to find the peak in the correct quadrant
    if overlap >= 0.5:
        return val

    overlap += 0.5

    return val + (val * (1.0 - overlap) * 2.0)


_EVEN_SMOOTH_FFT_SIZES: tuple[int, ...] | None = None
# Generous ceiling: a correlation frame is at most a few times the largest image dimension,
# and the table costs a few hundred ints.
_SMOOTH_FFT_SIZE_LIMIT: int = 1 << 26


def _even_smooth_fft_sizes() -> tuple[int, ...]:
    """Ascending even 5-smooth (2**a * 3**b * 5**c) sizes, built once and cached."""
    global _EVEN_SMOOTH_FFT_SIZES
    if _EVEN_SMOOTH_FFT_SIZES is None:
        sizes = []
        p2 = 2  # start at 2, never 1, so every entry keeps a factor of two -- see NextSmoothFFTSize
        while p2 <= _SMOOTH_FFT_SIZE_LIMIT:
            p3 = p2
            while p3 <= _SMOOTH_FFT_SIZE_LIMIT:
                p5 = p3
                while p5 <= _SMOOTH_FFT_SIZE_LIMIT:
                    sizes.append(p5)
                    p5 *= 5
                p3 *= 3
            p2 *= 2
        _EVEN_SMOOTH_FFT_SIZES = tuple(sorted(sizes))

    return _EVEN_SMOOTH_FFT_SIZES


def NextSmoothFFTSize(val: float) -> int:
    """Smallest **even** 5-smooth integer >= *val*.

    A cheaper alternative to :func:`NearestPowerOfTwo` for sizing an FFT frame. Both
    pocketfft and cuFFT are fast for any size factorable into small primes, so rounding a
    6000px requirement up to 8192 pays 1.86x the area for nothing. Since powers of two are
    themselves even and 5-smooth, the result is never *larger* than
    :func:`NearestPowerOfTwo`, so frame memory cannot regress.

    Measured on float32 ``fft2`` across eleven required sizes spanning 4100..8000, the
    smooth frame is 1.21x to 4.62x faster on numpy and 1.06x to 2.95x on CuPy -- with one
    exception: a requirement near 7300 selects 7500, which is 5.7% *slower* than 8192 on
    CuPy while still 1.39x faster on numpy. Frame area still falls, so that band trades a
    little GPU throughput for less memory. It was not worth a special case: the crossover
    was measured on a single card, and hardcoding a size exception is exactly the mistake
    #228 recorded for batch budgets.

    Even sizes only, and that is a correctness constraint rather than a preference:
    ``find_peak`` derives the shift as ``shape / 2.0 - peak_center_of_mass`` using true-half,
    while ``fftshift`` places the zero-shift sample at ``(n - 1) / 2`` for odd *n*. Measured,
    an odd frame biases every offset by exactly +0.5px, at 65, 129, 255 and 6075 alike.
    Power-of-two sizes are always even, which is why nothing has tripped over this; a smooth
    rule has to exclude odd candidates explicitly. This costs a little -- 6075 is 1.5x faster
    than 6144 on numpy -- and is not worth a half-pixel bias.

    See review #234.
    """
    target = int(math.ceil(val))
    if target <= 2:
        return 2

    sizes = _even_smooth_fft_sizes()
    index = bisect.bisect_left(sizes, target)
    if index >= len(sizes):
        raise ValueError(
            f"No even 5-smooth FFT size >= {target}; raise _SMOOTH_FFT_SIZE_LIMIT")

    return sizes[index]


def SmoothFFTSizeWithOverlap(val: float, overlap: float = 1.0) -> int:
    """
    :param val: Original dimension
    :param float overlap: Minimum amount of overlap possible between images, from 0 to 1.  Values greater than 0.5 require no increase to image size.
    :return: Same as :func:`NearestPowerOfTwoWithOverlap`, but rounded up to the next even
        5-smooth size rather than the next power of two. See :func:`NextSmoothFFTSize`.
    """
    return NextSmoothFFTSize(DimensionWithOverlap(val, _clamped_overlap(overlap)))


def ImageIntensityAtPercent(image, Percent=0.995):
    """Return the intensity at the given percentile (default 99.5%) of pixel values in the image."""
    NumPixels = image.size

    #   Sorting the list is a more correct and straightforward implementation, but using numpy.histogram is about 1 second faster
    #   image1D = numpy.sort(image, axis=None)
    #   targetIndex = math.floor(float(NumPixels) * Percent)
    #
    #   val = image1D[targetIndex]
    #
    #   del image1D
    #   return val

    NumBins = 1024
    [histogram, binEdge] = np.histogram(image, bins=NumBins)

    PixelNum = float(NumPixels) * Percent
    CumulativePixelsInBins = 0
    CutOffHistogramValue = None
    for iBin in range(0, len(histogram)):
        if CumulativePixelsInBins > PixelNum:
            CutOffHistogramValue = binEdge[iBin]
            break

        CumulativePixelsInBins += histogram[iBin]

    if CutOffHistogramValue is None:
        CutOffHistogramValue = binEdge[-1]

    return CutOffHistogramValue
