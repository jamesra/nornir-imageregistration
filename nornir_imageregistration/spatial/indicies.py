"""Deprecated: use nornir_imageregistration.spatial.indices instead (correct spelling)."""
import warnings

from nornir_imageregistration.spatial.indices import (
    iArea,
    iBox,
    iPoint,
    iPoint3,
    iRect,
    iVolume,
)

__all__ = ['iArea', 'iBox', 'iPoint', 'iPoint3', 'iRect', 'iVolume']

warnings.warn(
    "nornir_imageregistration.spatial.indicies is deprecated, use spatial.indices instead.",
    DeprecationWarning,
    stacklevel=2,
)
