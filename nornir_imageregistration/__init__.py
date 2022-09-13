'''

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

'''

import numpy as np
from PIL import Image

# Disable decompression bomb protection since we are dealing with huge images on purpose
Image.MAX_IMAGE_PIXELS = None

import collections.abc

import matplotlib.pyplot as plt

plt.ioff()


def ParamToDtype(param):
    if param is None:
        raise ValueError("'None' cannot be converted to a dtype")

    dtype = param
    if isinstance(dtype, np.ndarray) or isinstance(dtype, np.nditer):
        return param.dtype

    return dtype


def IsFloatArray(param):
    if param is None:
        return False

    return np.issubdtype(ParamToDtype(param), np.floating)


def IsIntArray(param):
    if param is None:
        return False

    return np.issubdtype(ParamToDtype(param), np.integer)


def ImageMaxPixelValue(image):
    '''The maximum value that can be stored in an image represented by integers'''
    probable_bpp = int(image.itemsize * 8)
    if probable_bpp > 8:
        if 'i' == image.dtype.kind:  # Signed integers we use a smaller probable_bpp
            probable_bpp = probable_bpp - 1
    return (1 << probable_bpp) - 1


def ImageBpp(image):
    probable_bpp = int(image.itemsize * 8)
    # if probable_bpp > 8:
    #    if 'i' == dt.kind: #Signed integers we use a smaller probable_bpp
    #        probable_bpp = probable_bpp - 1
    return probable_bpp


def IndexOfValues(A, values):
    '''
    :param array A: Array of length N that we want to return indicies into
    :param array Values: Array of length M containing values we need to find in A
    :return: Array of length M containing the first index in A where the Values occur, or None
    '''

    sorter = np.argsort(A)
    indicies = np.searchsorted(A, values, side='left', sorter=sorter)
    return indicies


def EnsurePointsAre1DNumpyArray(points, dtype=None):
    if not isinstance(points, np.ndarray):
        if not isinstance(points, collections.abc.Iterable):
            raise ValueError("points must be Iterable")

        if dtype is None:
            if isinstance(points[0], int):
                dtype = np.int32
            else:
                dtype = np.float32

        points = np.asarray(points, dtype=dtype)
    elif not dtype is None:
        if points.dtype != dtype:
            Warning('EnsurePointsAre4xN_NumpyArray dtype not equal to passed dtype, input array unchanged')

    if points.ndim > 1:
        points = np.array(points.flat())

    return points


def EnsurePointsAre2DNumpyArray(points, dtype=None):
    if not isinstance(points, np.ndarray):
        if not isinstance(points, collections.abc.Iterable):
            raise ValueError("points must be Iterable")

        if dtype is None:
            if isinstance(points[0], int):
                dtype = np.int32
            else:
                dtype = np.float32

        points = np.asarray(points, dtype=dtype)
    elif not dtype is None:
        if points.dtype != dtype:
            Warning('EnsurePointsAre4xN_NumpyArray dtype not equal to passed dtype, input array unchanged')

    if points.ndim == 1:
        points = np.resize(points, (1, 2))

    return points


def EnsurePointsAre4xN_NumpyArray(points, dtype=None):
    if not isinstance(points, np.ndarray):
        if not isinstance(points, collections.abc.Iterable):
            raise ValueError("points must be Iterable")

        if dtype is None:
            if isinstance(points[0], int):
                dtype = np.int32
            else:
                dtype = np.float32

        points = np.asarray(points, dtype=np.float32)
    elif not dtype is None:
        if points.dtype != dtype:
            Warning('EnsurePointsAre4xN_NumpyArray dtype not equal to passed dtype, input array unchanged')

    if points.ndim == 1:
        points = np.resize(points, (1, 4))

    if points.shape[1] != 4:
        raise ValueError("There are not 4 columns in the corrected array")

    return points


import nornir_imageregistration.spatial as spatial
from nornir_imageregistration.spatial import *

import nornir_imageregistration.pillow_helpers as pillow_helpers

import nornir_imageregistration.core as core
from nornir_imageregistration.core import *

import nornir_imageregistration.alignment_record as alignment_record
from nornir_imageregistration.alignment_record import AlignmentRecord, EnhancedAlignmentRecord

import nornir_imageregistration.settings as settings
import nornir_imageregistration.transforms as transforms
from nornir_imageregistration.transforms import ITransform, ITransformChangeEvents, ITransformTranslation, \
    IDiscreteTransform, ITransformScaling, IControlPoints

import nornir_imageregistration.files as files
from nornir_imageregistration.files import MosaicFile, StosFile, AddStosTransforms

import nornir_imageregistration.mosaic as mosaic
from nornir_imageregistration.mosaic import Mosaic

import nornir_imageregistration.mosaic_tileset as mosaic_tileset
from nornir_imageregistration.mosaic_tileset import MosaicTileset

import nornir_imageregistration.stos_brute as stos_brute
import nornir_imageregistration.tile as tile
from nornir_imageregistration.tile import Tile

import nornir_imageregistration.tile_overlap as tile_overlap

import nornir_imageregistration.tileset as tileset
import nornir_imageregistration.image_stats as image_stats
import nornir_imageregistration.assemble as assemble
import nornir_imageregistration.assemble_tiles as assemble_tiles
import nornir_imageregistration.layout as layout
import nornir_imageregistration.local_distortion_correction as local_distortion_correction
import nornir_imageregistration.tileset_functions as tileset_functions
import nornir_imageregistration.views as views
import nornir_imageregistration.volume as volume

import nornir_imageregistration.arrange_mosaic as arrange_mosaic
from nornir_imageregistration.arrange_mosaic import TranslateTiles2

from nornir_imageregistration.volume import Volume
from nornir_imageregistration.overlapmasking import GetOverlapMask
from nornir_imageregistration.local_distortion_correction import RefineMosaic, RefineStosFile, RefineTransform

from nornir_imageregistration.spatial.indicies import *
from nornir_imageregistration.tileset import ShadeCorrectionTypes
from nornir_imageregistration.views import ShowWithPassFail
from nornir_imageregistration.views.display_images import ShowGrayscale
from nornir_imageregistration.files.stosfile import StosFile, AddStosTransforms

from nornir_imageregistration.overlapmasking import GetOverlapMask

from nornir_imageregistration.image_stats import ImageStats, Prune, Histogram
from nornir_imageregistration.grid_subdivision import CenteredGridDivision, ITKGridDivision

# In a remote process we need errors raised, otherwise we crash for the wrong reason and debugging is tougher. 
np.seterr(divide='raise', over='raise', under='warn', invalid='raise')

__all__ = ['image_stats', 'core', 'files', 'geometry', 'transforms', 'spatial']
