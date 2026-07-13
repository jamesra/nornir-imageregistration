"""

alignment_record
----------------

.. autoclass:: nornir_imageregistration.alignment_record.AlignmentRecord

core
----

.. automodule:: nornir_imageregistration.core
   :members:

assemble
--------

.. automodule:: nornir_imageregistration.assemble
   :members:

assemble_tiles
--------------

.. automodule:: nornir_imageregistration.assemble_tiles

"""
import os
import types
from typing import Iterable, Sequence, Any

from PIL import Image
import numpy as np
import numpy.typing
from numpy.typing import NDArray, DTypeLike
from nornir_shared.mathhelper import RoundingPrecision

# CUDA pathfinder canary subprocess fails under debugpy; bootstrap before any cupy import.
import nornir_imageregistration.computational_lib as _computational_lib
from nornir_imageregistration.computational_lib import (
    ComputationLib,
    GetActiveComputationLib,
    HasCupy,
    HasCuVS,
    SetActiveComputationLib,
    TryInitCupyContext,
    UsingCupy,
)

if _computational_lib._has_cupy and _computational_lib.cp is not None:
    cp = _computational_lib.cp
else:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx_thunk

# Disable decompression bomb protection since we are dealing with huge images on purpose
Image.MAX_IMAGE_PIXELS = None

import collections.abc
import matplotlib

from nornir_imageregistration.headless import is_headless

matplotlib.use("Agg" if is_headless() else "qtAgg")

import matplotlib.pyplot as plt

plt.ioff()


def default_image_dtype() -> DTypeLike:
    """
    :return: The default dtype for image data
    """
    return np.float16


def default_depth_image_dtype() -> DTypeLike:
    """
    :return: The default dtype for image data
    """
    return np.float32


import nornir_imageregistration.debugging as debugging
from nornir_imageregistration.debugging import in_debug_mode

import nornir_imageregistration.temporaryfiles as temporaryfiles
from nornir_imageregistration.temporaryfiles import gettempdir

import nornir_imageregistration.mathfuncs as math

import nornir_imageregistration.type_info as typing
from nornir_imageregistration.type_info import *

import nornir_imageregistration.mmap_metadata as mmap_metadata
from nornir_imageregistration.mmap_metadata import *

import nornir_imageregistration.nornir_image_types as nornir_image_types
from nornir_imageregistration.nornir_image_types import *


def GetComputationModule() -> types.ModuleType:
    """Return the computational module in use (numpy or cupy)."""
    return cp if UsingCupy() else np


def ParamToDtype(param: NDArray | DTypeLike) -> DTypeLike:
    """Return the dtype of param (array or scalar); for arrays, returns array.dtype."""
    if param is None:
        raise ValueError("'None' cannot be converted to a dtype")

    dtype = param
    if isinstance(dtype, np.ndarray) or isinstance(dtype, np.nditer):
        return dtype.dtype

    return dtype


def __DetermineDType(array: Sequence[Any] | NDArray) -> DTypeLike:
    """Given a numpy/cupy array, or a sequence, determine an appropriate dtype"""
    try:
        dtype = getattr(array, 'dtype')
        if not isinstance(dtype, np.dtype):
            raise AttributeError("dtype property on points object is not a numpy dtype")
    except AttributeError:  # The points object doesn't have a dtype attribute or the type is incorrect
        if isinstance(array[0], int):
            dtype = np.int32
        else:
            dtype = np.float32

    return dtype


def IsFloatArray(param: NDArray | DTypeLike) -> bool:
    """Return True if param is or has a floating-point dtype."""
    if param is None:
        return False

    return np.issubdtype(ParamToDtype(param), np.floating)


def IsIntArray(param: NDArray | DTypeLike) -> bool:
    """Return True if param is or has an integer dtype."""
    if param is None:
        return False

    return np.issubdtype(ParamToDtype(param), np.integer)


def IsBoolArray(param: NDArray | DTypeLike) -> bool:
    """Return True if param is or has a boolean dtype."""
    if param is None:
        return False

    return np.issubdtype(ParamToDtype(param), bool)


def ImageMaxPixelValue(image: NDArray) -> int:
    """The maximum value that can be stored in an image represented by integers"""
    probable_bpp = int(image.itemsize * 8)
    if probable_bpp > 8:
        if 'i' == image.dtype.kind:  # Signed integers we use a smaller probable_bpp
            probable_bpp -= 1
    return (1 << probable_bpp) - 1


def ImageBpp(image: NDArray) -> int:
    """Return the bits-per-pixel (depth) of the image array from its dtype itemsize."""
    probable_bpp = int(image.itemsize * 8)
    # if probable_bpp > 8:
    #    if 'i' == dt.kind: #Signed integers we use a smaller probable_bpp
    #        probable_bpp = probable_bpp - 1
    return probable_bpp


def IndexOfValues(A, values) -> numpy.typing.NDArray:
    """
    :param array A: Array of length N that we want to return indices into
    :param array values: Array of length M containing values we need to find in A
    :returns: An array of length M containing the first index in A where the Values occur, or None
    """

    sorter = np.argsort(A)
    return np.searchsorted(A, values, side='left', sorter=sorter)


def EnsureArray(points: NDArray | Sequence, dtype=None) -> NDArray:
    """Convert points to an array, preserving the input array's backend (numpy in → numpy out, cupy in → cupy out).

    Sequences and scalars are treated as host data and become NumPy arrays.  Use
    :func:`EnsureCupyArray` or :func:`EnsureNumpyArray` to force a specific backend.
    """
    if cp.get_array_module(points) is cp:
        return EnsureCupyArray(points, dtype)
    return EnsureNumpyArray(points, dtype)


def EnsureNumpyArray(points: NDArray | Sequence, dtype=None) -> NDArray:
    """Convert points to a NumPy array; CuPy arrays are transferred to host."""
    if not isinstance(points, np.ndarray):
        if not isinstance(points, collections.abc.Iterable):
            raise ValueError("points must be Iterable")

        if dtype is None:
            dtype = __DetermineDType(points)

        if HasCupy() and isinstance(points, cp.ndarray):
            points = points.get()  # type: ignore[union-attr]
            return points.astype(dtype, copy=False)  # type: ignore[union-attr]

        points = np.asarray(points, dtype=dtype)
    elif dtype is not None:
        points = points.astype(dtype, copy=False)

    return points


def EnsureCupyArray(points: NDArray | Sequence, dtype=None) -> NDArray:
    """Convert points to a CuPy array (GPU); copies from host if needed."""
    if not isinstance(points, cp.ndarray):
        if not isinstance(points, collections.abc.Iterable):
            raise ValueError("points must be Iterable")

        if dtype is None:
            dtype = __DetermineDType(points)

        points = cp.asarray(points, dtype=dtype)
    elif dtype is not None:
        points = points.astype(dtype, copy=False)  # type: ignore[union-attr]

    return points  # type: ignore[return-value]


def EnsurePointsAre1DNumpyArray(points: NDArray | Sequence, dtype=None) -> NDArray:
    points = EnsureNumpyArray(points, dtype)

    if points.ndim > 1:
        points = np.ravel(points)

    return points


def EnsurePointsAre1DCuPyArray(points: NDArray | Sequence, dtype=None) -> NDArray:
    """Ensure points are a 1D CuPy array (raveled if needed)."""
    points = EnsureCupyArray(points, dtype)

    if points.ndim > 1:
        points = cp.ravel(points)

    return points


def EnsurePointsAre1DArray(points: NDArray | Sequence, dtype=None) -> NDArray:
    """Ensure points are a 1D array, preserving the input array's backend (numpy in → numpy out, cupy in → cupy out)."""
    if cp.get_array_module(points) is cp:
        return EnsurePointsAre1DCuPyArray(points, dtype)
    return EnsurePointsAre1DNumpyArray(points, dtype)


def EnsurePointsAre2DNumpyArray(points: NDArray | Sequence, dtype=None) -> NDArray:
    points = EnsureNumpyArray(points, dtype)

    if points.ndim == 1:
        points = np.resize(points, (1, 2))

    return points


def EnsurePointsAre2DCuPyArray(points: NDArray | Sequence, dtype=None) -> NDArray:
    """Ensure points are a 2D CuPy array (Nx2)."""
    points = EnsureCupyArray(points, dtype)

    if points.ndim == 1:
        points = cp.resize(points, (1, 2))

    return points


def EnsurePointsAre2DArray(points: NDArray | Sequence, dtype=None) -> NDArray:
    """Ensure points are a 2D array (Nx2), preserving the input array's backend (numpy in → numpy out, cupy in → cupy out)."""
    if cp.get_array_module(points) is cp:
        return EnsurePointsAre2DCuPyArray(points, dtype)
    return EnsurePointsAre2DNumpyArray(points, dtype)


def EnsurePointsAre4xN_NumpyArray(points: NDArray[np.floating] | Sequence[float], dtype=None) -> NDArray[np.floating]:
    points = EnsureNumpyArray(points, dtype)

    if points.ndim == 1:
        points = np.resize(points, (1, 4))

    if points.shape[1] != 4:
        raise ValueError("There are not 4 columns in the corrected array")

    return points


def EnsurePointsAre4xN_CuPyArray(points: NDArray[np.floating] | Sequence[float], dtype=None) -> NDArray[np.floating]:
    """Ensure points are a CuPy array with 4 columns."""
    points = EnsureCupyArray(points, dtype)

    if points.ndim == 1:
        points = cp.resize(points, (1, 4))

    if points.shape[1] != 4:
        raise ValueError("There are not 4 columns in the corrected array")

    return points


def EnsurePointsAre4xN_Array(points: NDArray[np.floating] | Sequence[float], dtype=None) -> NDArray[np.floating]:
    """Ensure points are a 4-column array, preserving the input array's backend (numpy in → numpy out, cupy in → cupy out)."""
    if cp.get_array_module(points) is cp:
        return EnsurePointsAre4xN_CuPyArray(points, dtype)
    return EnsurePointsAre4xN_NumpyArray(points, dtype)


import nornir_shared.mathhelper
from nornir_shared.mathhelper import NearestPowerOfTwo, RoundingPrecision

import nornir_imageregistration.runtime_warnings as runtime_warnings
from nornir_imageregistration.runtime_warnings import IgnoreRuntimeWarnings, IgnoreUnderflow, IgnoreOverflow, \
    IgnoreUnderAndOverflow, IgnoreLinAlgWarning

import nornir_imageregistration.shared_mem_metadata
from nornir_imageregistration.shared_mem_metadata import Shared_Mem_Metadata

import nornir_imageregistration.transformed_image_data
from nornir_imageregistration.transformed_image_data import ITransformedImageData

import nornir_imageregistration.igrid as igrid
from nornir_imageregistration.igrid import IGrid

import nornir_imageregistration.spatial as spatial
from nornir_imageregistration.spatial import *

import nornir_imageregistration.pillow_helpers as pillow_helpers

import nornir_imageregistration.image_stats as image_stats
from nornir_imageregistration.image_stats import ImageStats, Prune, Histogram

import nornir_imageregistration.image_permutation_helper as image_permutation_helper
from nornir_imageregistration.image_permutation_helper import ImagePermutationHelper

import nornir_imageregistration.core as core
from nornir_imageregistration.core import *

import nornir_imageregistration.alignment_record as alignment_record
from nornir_imageregistration.alignment_record import AlignmentRecord, EnhancedAlignmentRecord

import nornir_imageregistration.phasecorrelation as phasecorrelation
from nornir_imageregistration.phasecorrelation import *

import nornir_imageregistration.settings as settings

import nornir_imageregistration.transforms as transforms
from nornir_imageregistration.transforms import ITransform, ITransformChangeEvents, ITransformTranslation, \
    IDiscreteTransform, ITransformScaling, IControlPoints, ITransformTargetRotation, ITransformSourceRotation, \
    IGridTransform, ITriangulatedSourceSpace, ITriangulatedTargetSpace, ITransformRelativeScaling, \
    IRigidTransform, ITransfomFlip, ITransformFlip, IControlPointAddRemove, IControlPointEdit, ISourceSpaceControlPointEdit

import nornir_imageregistration.files as files
from nornir_imageregistration.files import MosaicFile, StosFile, AddStosTransforms

import nornir_imageregistration.tile as tile
from nornir_imageregistration.tile import Tile

import nornir_imageregistration.mosaic as mosaic
from nornir_imageregistration.mosaic import Mosaic

import nornir_imageregistration.mosaic_tileset as mosaic_tileset
from nornir_imageregistration.mosaic_tileset import MosaicTileset

import nornir_imageregistration.stos_brute as stos_brute

import nornir_imageregistration.tile_overlap as tile_overlap

import nornir_imageregistration.tileset as tileset
from nornir_imageregistration.tileset import ShadeCorrectionTypes

import nornir_imageregistration.assemble as assemble
import nornir_imageregistration.assemble_tiles as assemble_tiles
import nornir_imageregistration.layout as layout
import nornir_imageregistration.local_distortion_correction as local_distortion_correction
import nornir_imageregistration.refine_shared as refine_shared
import nornir_imageregistration.mosaic_refine as mosaic_refine
import nornir_imageregistration.stos_refine as stos_refine
from nornir_imageregistration.stos_refine import WeightMethod
import nornir_imageregistration.tileset_functions as tileset_functions
import nornir_imageregistration.views as views
import nornir_imageregistration.volume as volume

import nornir_imageregistration.arrange_mosaic as arrange_mosaic
from nornir_imageregistration.arrange_mosaic import TranslateTiles2

from nornir_imageregistration.volume import Volume
from nornir_imageregistration.overlapmasking import GetOverlapMask, GetOverlapMaskOnDevice
from nornir_imageregistration.mosaic_refine import RefineMosaic, RefineGridMosaic, MosaicRefinementDiagnostics
from nornir_imageregistration.stos_refine import RefineStosFile, RefineTransform

from nornir_imageregistration.spatial.indices import *
from nornir_imageregistration.views import ShowWithPassFail
from nornir_imageregistration.views.display_images import ShowGrayscale
from nornir_imageregistration.files.stosfile import StosFile, AddStosTransforms

from nornir_imageregistration.grid_subdivision import CenteredGridDivision, ITKGridDivision

import nornir_imageregistration.pillow_helpers
from nornir_imageregistration.pillow_helpers import get_image_file_dtype, dtype_for_pillow_image

import nornir_imageregistration.blur as blur

import nornir_imageregistration.distance as distance

import nornir_imageregistration.image_filter_cache as image_filter_cache

# In a remote process we need errors raised, otherwise we crash for the wrong reason and debugging is tougher. 
np.seterr(divide='raise', over='raise', under='warn', invalid='raise')

# Public API is defined by the imports in this file; __all__ is a subset for tooling.
__all__ = ['HasCupy', 'HasCuVS', 'image_stats', 'core', 'files', 'views', 'transforms', 'spatial', 'ITransform',
           'ITransformChangeEvents',
           'ITransformTranslation', 'IDiscreteTransform', 'ITransformScaling', 'IControlPoints']
