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

from nornir_imageregistration.alignment_record import AlignmentRecord, EnhancedAlignmentRecord
from nornir_imageregistration.core import *
from nornir_imageregistration.mosaic import Mosaic
from nornir_imageregistration.spatial import *
from nornir_imageregistration.volume import Volume
from nornir_imageregistration.overlapmasking import GetOverlapMask
from nornir_imageregistration.local_distortion_correction import RefineMosaic, RefineStosFile, RefineTransform

import nornir_imageregistration.files as files
import nornir_imageregistration.tile as tile
import nornir_imageregistration.tileset as tileset
import nornir_imageregistration.transforms as transforms
import nornir_imageregistration.spatial as spatial
import nornir_imageregistration.image_stats as image_stats

from nornir_imageregistration.spatial.indicies import *
from nornir_imageregistration.tileset import ShadeCorrectionTypes

from nornir_imageregistration.files.stosfile import StosFile
from nornir_imageregistration.files.mosaicfile import MosaicFile
from nornir_imageregistration.overlapmasking import GetOverlapMask
from nornir_imageregistration.mosaic import Mosaic
from nornir_imageregistration.image_stats import ImageStats, Prune, Histogram
from nornir_imageregistration.grid_subdivision import CenteredGridDivision, ITKGridDivision

import numpy as np
# In a remote process we need errors raised, otherwise we crash for the wrong reason and debugging is tougher. 
np.seterr(divide='raise', over='raise', under='warn', invalid='raise')


__all__ = ['image_stats', 'core', 'files', 'geometry', 'transforms', 'spatial']
