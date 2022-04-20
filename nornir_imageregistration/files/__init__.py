__all__ = ['mosaicfile', 'stosfile']


# StosNameTemplate = "%(mappedsection)04u-%(controlsection)04u_%(channels)s_%(mosaicfilters)s_%(stostype)s_%(downsample)u.stos"
# ImageNameTemplate = "%(section)04u_%(channel)s_%(filter)s_%(downsample)u.%(ext)s"
# MosaicNameTemplate = "%(channel)s_%(filter)s.mosaic"

from .mosaicfile import MosaicFile
from .stosfile import StosFile, AddStosTransforms