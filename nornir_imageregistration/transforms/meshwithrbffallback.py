"""
Created on Oct 18, 2012

@author: Jamesan
"""

from dataclasses import dataclass
from typing import Any

import numpy
import numpy as np
from numpy.typing import NDArray
import scipy.spatial
from scipy.interpolate import LinearNDInterpolator

try:
    import cupy as cp
    # import cupyx
    from cupyx.scipy.interpolate import RBFInterpolator as cuRBFInterpolator
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
    # import nornir_imageregistration.cupyx_thunk as cupyx
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
    # import nornir_imageregistration.cupyx_thunk as cupyx

import nornir_imageregistration
import nornir_pools
from nornir_imageregistration.transforms.one_way_rbftransform import OneWayRBFWithLinearCorrection, \
    OneWayRBFWithLinearCorrection_GPUComponent
from nornir_imageregistration.transforms.transform_type import TransformType
from nornir_imageregistration.nearest_neighbor import build_nearest_neighbor_index
from . import utils
from .triangulation import (
    Triangulation,
    Triangulation_GPUComponent,
    _defer_structure_rebuild,
    barycentric_sample_delaunay,
)
from nornir_imageregistration.transforms.landmark import Landmark_GPU, Landmark_CPU


def _nan_mapped_points(points) -> NDArray[np.floating]:
    """Return an all-NaN Nx2 map without rebuilding interpolators or Qhull."""
    pts = nornir_imageregistration.EnsureNumpyArray(points)
    out = numpy.empty((pts.shape[0], 2), dtype=numpy.float32)
    out[:] = numpy.nan
    return out


def _ensure_float32_64(arr, xp):
    """Ensure array has float32 or float64 dtype for RBF; use xp (numpy or cupy) so CuPy arrays stay on device."""
    if arr.dtype == xp.float32 or arr.dtype == xp.float64:
        return arr
    return xp.asarray(arr, dtype=xp.float32)


def _build_cpu_rbf_with_weights(source_points, target_points) -> OneWayRBFWithLinearCorrection:
    """Construct a CPU RBF transform and force the weight solve."""
    instance = OneWayRBFWithLinearCorrection(source_points, target_points)
    instance.PrecomputeWeights()
    return instance


def GetTransformPrewarmPool():
    """Single sticky thread that drives transform RBF/mesh init off the UI.

    CPU meshes fan out Forward/Reverse weight solves to the global thread pool
    from this driver. GPU/CuPy weight solves run on this same thread for CUDA
    context safety.
    """
    return nornir_pools.GetThreadPool("Transform prewarm", 1)


@dataclass(frozen=True)
class MeshRefreshedStructures:
    """Delaunay interpolators, KD-trees, and RBF instances built from a point snapshot."""

    warpedtri: scipy.spatial.Delaunay
    fixedtri: scipy.spatial.Delaunay
    forward_interpolator: Any
    inverse_interpolator: Any
    warped_kdtree: Any
    fixed_kdtree: Any
    forward_rbf: Any
    reverse_rbf: Any
    scipy_forward_interp: bool = True
    scipy_inverse_interp: bool = True


def _coerce_to_reference_backend(arr, reference):
    """Coerce arr to the same array backend as reference (numpy/cupy)."""
    xp = cp.get_array_module(reference)
    if xp is cp:
        return cp.asarray(arr)
    if hasattr(arr, "get"):
        return numpy.asarray(arr.get())
    return numpy.asarray(arr)


def _coerce_indices_to_reference_backend(indices, reference):
    """Coerce indices to an indexing array compatible with the reference backend."""
    xp = cp.get_array_module(reference)
    if xp is cp:
        return cp.asarray(indices, dtype=cp.intp)
    if hasattr(indices, "get"):
        indices = indices.get()  # type: ignore[union-attr]
    return nornir_imageregistration.EnsureNumpyArray(indices).astype(numpy.intp, copy=False)


def _invalidate_host_control_point_cache(mesh: Any) -> None:
    """Drop cached host Source/Target copies after a control-point edit."""
    mesh._host_source_points = None
    mesh._host_target_points = None


def _cached_host_source_points(mesh: Any) -> NDArray[np.floating]:
    """Host SourcePoints snapshot, refreshed after source edits rather than per query."""
    cached = getattr(mesh, "_host_source_points", None)
    if cached is None:
        mesh._host_source_points = utils.host_copy_points(mesh.SourcePoints)
        mesh._host_source_copy_count = int(getattr(mesh, "_host_source_copy_count", 0)) + 1
    return mesh._host_source_points


def _cached_host_target_points(mesh: Any) -> NDArray[np.floating]:
    """Host TargetPoints snapshot, refreshed after target edits rather than per query."""
    cached = getattr(mesh, "_host_target_points", None)
    if cached is None:
        mesh._host_target_points = utils.host_copy_points(mesh.TargetPoints)
        mesh._host_target_copy_count = int(getattr(mesh, "_host_target_copy_count", 0)) + 1
    return mesh._host_target_points


class MeshWithRBFFallback(Triangulation):
    """Triangulation warp with an RBF fallback for queries outside the hull."""

    _continuous_stale: bool = False
    _ForwardRBFInstance: Any
    _ReverseRBFInstance: Any
    _host_source_points: NDArray[np.floating] | None
    _host_target_points: NDArray[np.floating] | None
    _host_source_copy_count: int
    _host_target_copy_count: int

    @property
    def type(self) -> TransformType:
        return TransformType.MESH

    def __getstate__(self):

        odict = super(MeshWithRBFFallback, self).__getstate__()
        odict['_ReverseRBFInstance'] = self._ReverseRBFInstance  # type: ignore[assignment]
        odict['_ForwardRBFInstance'] = self._ForwardRBFInstance  # type: ignore[assignment]
        odict['_continuous_stale'] = self._continuous_stale  # type: ignore[assignment]
        return odict

    def __setstate__(self, dictionary):
        super(MeshWithRBFFallback, self).__setstate__(dictionary)
        self._continuous_stale = dictionary.get('_continuous_stale', False)
        _invalidate_host_control_point_cache(self)
        self._host_source_copy_count = 0
        self._host_target_copy_count = 0

    @property
    def ReverseRBFInstance(self):
        if self._ReverseRBFInstance is None:
            self._ReverseRBFInstance = OneWayRBFWithLinearCorrection(self.TargetPoints, self.SourcePoints)

        return self._ReverseRBFInstance

    @property
    def ForwardRBFInstance(self):
        if self._ForwardRBFInstance is None:
            self._ForwardRBFInstance = OneWayRBFWithLinearCorrection(self.SourcePoints, self.TargetPoints)

        return self._ForwardRBFInstance

    def build_refreshed_continuous(self) -> MeshRefreshedStructures:
        """Build replacement Delaunay interpolators and RBF weights from a point snapshot.

        Does not mutate this instance. Intended to run on the transform-prewarm thread.
        """
        src = utils.host_copy_points(self.SourcePoints)
        tgt = utils.host_copy_points(self.TargetPoints)
        pool = nornir_pools.GetGlobalThreadPool()
        forward_task = pool.add_task(
            "Solve forward RBF transform",
            _build_cpu_rbf_with_weights,
            src,
            tgt,
        )
        reverse_task = pool.add_task(
            "Solve reverse RBF transform",
            _build_cpu_rbf_with_weights,
            tgt,
            src,
        )
        warpedtri = scipy.spatial.Delaunay(src, incremental=False)
        fixedtri = scipy.spatial.Delaunay(tgt, incremental=False)
        return MeshRefreshedStructures(
            warpedtri=warpedtri,
            fixedtri=fixedtri,
            forward_interpolator=LinearNDInterpolator(warpedtri, tgt),
            inverse_interpolator=LinearNDInterpolator(fixedtri, src),
            warped_kdtree=build_nearest_neighbor_index(src),
            fixed_kdtree=build_nearest_neighbor_index(tgt),
            forward_rbf=forward_task.wait_return(),
            reverse_rbf=reverse_task.wait_return(),
        )

    def apply_refreshed_continuous(self, bundle: MeshRefreshedStructures) -> None:
        """Install off-UI Delaunay/RBF structures onto this live mesh."""
        self._warpedtri = bundle.warpedtri
        self._fixedtri = bundle.fixedtri
        self._ForwardInterpolator = bundle.forward_interpolator
        self._InverseInterpolator = bundle.inverse_interpolator
        self._WarpedKDTree = bundle.warped_kdtree
        self._FixedKDTree = bundle.fixed_kdtree
        self._ForwardRBFInstance = bundle.forward_rbf
        self._ReverseRBFInstance = bundle.reverse_rbf
        self._continuous_stale = False

    def InitializeDataStructures(self):
        """Build triangulation and precompute Forward/Reverse RBF weights.

        Forward and Reverse weight solves run on the shared global thread pool.
        Call this from a non-pool driver thread (e.g. UI prewarm) so workers are
        not nested-waiting on the same pool.
        """
        self.apply_refreshed_continuous(self.build_refreshed_continuous())

    def OnFixedPointChanged(self):
        super(MeshWithRBFFallback, self).OnFixedPointChanged()
        self._continuous_stale = True
        self._host_target_points = None

    def OnWarpedPointChanged(self):
        super(MeshWithRBFFallback, self).OnWarpedPointChanged()
        self._continuous_stale = True
        self._host_source_points = None

    def ClearDataStructures(self):
        """Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data"""

        super(MeshWithRBFFallback, self).ClearDataStructures()
        self._continuous_stale = True
        _invalidate_host_control_point_cache(self)

    def _discrete_forward(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Map source→target with cached source Delaunay; do not rebuild interpolators."""
        if self._ForwardInterpolator is not None:
            return super(MeshWithRBFFallback, self).Transform(points)
        warpedtri = self._warpedtri
        if warpedtri is None:
            if _defer_structure_rebuild():
                return _nan_mapped_points(points)
            return super(MeshWithRBFFallback, self).Transform(points)
        mapped = barycentric_sample_delaunay(
            nornir_imageregistration.EnsureNumpyArray(points),
            warpedtri,
            _cached_host_target_points(self),
            nan_outside=True,
        )
        return mapped.astype(np.float32, copy=False)

    def _discrete_inverse(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Map target→source with cached target Delaunay; do not rebuild interpolators."""
        if self._InverseInterpolator is not None:
            return super(MeshWithRBFFallback, self).InverseTransform(points)
        fixedtri = self._fixedtri
        if fixedtri is None:
            if _defer_structure_rebuild():
                return _nan_mapped_points(points)
            return super(MeshWithRBFFallback, self).InverseTransform(points)
        mapped = barycentric_sample_delaunay(
            nornir_imageregistration.EnsureNumpyArray(points),
            fixedtri,
            _cached_host_source_points(self),
            nan_outside=True,
        )
        return mapped.astype(np.float32, copy=False)

    def Transform(self, points, **kwargs):
        """
        Transform from warped space to fixed space
        :param ndarray points: [[ControlY, ControlX, MappedY, MappedX],...]
        """

        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        if points.shape[0] == 0:
            return []

        TransformedPoints = self._discrete_forward(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        (_GoodPoints, invalid_mask) = utils.InvalidIndices(TransformedPoints)

        if not bool(invalid_mask.any()):
            return TransformedPoints
        else:
            if len(points) > 1:
                BadPoints = points[invalid_mask]
            else:
                BadPoints = points

        BadPoints = _ensure_float32_64(BadPoints, cp.get_array_module(points))

        rbf = self._ForwardRBFInstance
        if rbf is None:
            rbf = self.ForwardRBFInstance
        FixedPoints = rbf.Transform(BadPoints)
        FixedPoints = _coerce_to_reference_backend(FixedPoints, TransformedPoints)
        TransformedPoints[invalid_mask] = FixedPoints
        return TransformedPoints

    def InverseTransform(self, points, **kwargs):
        """
        Transform from fixed space to warped space
        :param points:
        """

        points = nornir_imageregistration.EnsurePointsAre2DArray(points)

        if points.shape[0] == 0:
            return []

        TransformedPoints = self._discrete_inverse(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        (_GoodPoints, invalid_mask) = utils.InvalidIndices(TransformedPoints)

        if not bool(invalid_mask.any()):
            return TransformedPoints
        else:
            if points.ndim > 1:
                BadPoints = points[invalid_mask]
            else:
                BadPoints = points  # This is likely no longer needed since this function always returns a 2D array now

        BadPoints = _ensure_float32_64(BadPoints, cp.get_array_module(points))

        rbf = self._ReverseRBFInstance
        if rbf is None:
            rbf = self.ReverseRBFInstance
        FixedPoints = rbf.Transform(BadPoints)
        FixedPoints = _coerce_to_reference_backend(FixedPoints, TransformedPoints)
        TransformedPoints[invalid_mask] = FixedPoints
        return TransformedPoints

    def __init__(self, pointpairs):
        """
        :param ndarray pointpairs: [TargetY, TargetX, SourceY, SourceX]
        """
        super(MeshWithRBFFallback, self).__init__(pointpairs)

        self._ReverseRBFInstance = None
        self._ForwardRBFInstance = None
        self._continuous_stale = False
        self._host_source_points = None
        self._host_target_points = None
        self._host_source_copy_count = 0
        self._host_target_copy_count = 0

    @staticmethod
    def Load(TransformString, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseMeshTransform(TransformString, pixelSpacing)


class MeshWithRBFFallback_GPUComponent(Triangulation_GPUComponent):
    """Triangulation warp with an RBF fallback for queries outside the hull (CuPy)."""

    _continuous_stale: bool = False
    _ForwardRBFInstance: Any
    _ReverseRBFInstance: Any
    _host_source_points: NDArray[np.floating] | None
    _host_target_points: NDArray[np.floating] | None
    _host_source_copy_count: int
    _host_target_copy_count: int

    @property
    def type(self) -> TransformType:
        return TransformType.MESH

    def __getstate__(self):

        odict = super(MeshWithRBFFallback_GPUComponent, self).__getstate__()
        odict['_ReverseRBFInstance'] = self._ReverseRBFInstance  # type: ignore[assignment]
        odict['_ForwardRBFInstance'] = self._ForwardRBFInstance  # type: ignore[assignment]
        odict['_continuous_stale'] = self._continuous_stale  # type: ignore[assignment]
        return odict

    def __setstate__(self, dictionary):
        super(MeshWithRBFFallback_GPUComponent, self).__setstate__(dictionary)
        self._continuous_stale = dictionary.get('_continuous_stale', False)
        _invalidate_host_control_point_cache(self)
        self._host_source_copy_count = 0
        self._host_target_copy_count = 0

    @property
    def ReverseRBFInstance(self):
        if self._ReverseRBFInstance is None:
            self._ReverseRBFInstance = OneWayRBFWithLinearCorrection_GPUComponent(self.TargetPoints, self.SourcePoints)

        return self._ReverseRBFInstance

    @property
    def ForwardRBFInstance(self):
        if self._ForwardRBFInstance is None:
            self._ForwardRBFInstance = OneWayRBFWithLinearCorrection_GPUComponent(self.SourcePoints, self.TargetPoints)

        return self._ForwardRBFInstance

    def build_refreshed_continuous(self) -> MeshRefreshedStructures:
        """Build replacement Delaunay interpolators and RBF weights from a point snapshot.

        Runs on the sticky CUDA transform-init thread. Does not mutate this instance.
        """
        from nornir_imageregistration.transforms.gridtransform import _build_linear_nd_interpolator

        src = utils.host_copy_points(self.SourcePoints)
        tgt = utils.host_copy_points(self.TargetPoints)
        warpedtri = scipy.spatial.Delaunay(src, incremental=False)
        fixedtri = scipy.spatial.Delaunay(tgt, incremental=False)
        forward_interp, scipy_fwd = _build_linear_nd_interpolator(src, tgt)
        inverse_interp, scipy_inv = _build_linear_nd_interpolator(tgt, src)
        forward_rbf = OneWayRBFWithLinearCorrection_GPUComponent(src, tgt)
        forward_rbf.PrecomputeWeights()
        reverse_rbf = OneWayRBFWithLinearCorrection_GPUComponent(tgt, src)
        reverse_rbf.PrecomputeWeights()
        return MeshRefreshedStructures(
            warpedtri=warpedtri,
            fixedtri=fixedtri,
            forward_interpolator=forward_interp,
            inverse_interpolator=inverse_interp,
            warped_kdtree=build_nearest_neighbor_index(src),
            fixed_kdtree=build_nearest_neighbor_index(tgt),
            forward_rbf=forward_rbf,
            reverse_rbf=reverse_rbf,
            scipy_forward_interp=scipy_fwd,
            scipy_inverse_interp=scipy_inv,
        )

    def apply_refreshed_continuous(self, bundle: MeshRefreshedStructures) -> None:
        """Install off-UI Delaunay/RBF structures onto this live mesh."""
        self._warpedtri = bundle.warpedtri
        self._fixedtri = bundle.fixedtri
        self._ForwardInterpolator = bundle.forward_interpolator
        self._InverseInterpolator = bundle.inverse_interpolator
        self._WarpedKDTree = bundle.warped_kdtree
        self._FixedKDTree = bundle.fixed_kdtree
        self._ForwardRBFInstance = bundle.forward_rbf
        self._ReverseRBFInstance = bundle.reverse_rbf
        self._scipy_forward_interp = bundle.scipy_forward_interp
        self._scipy_inverse_interp = bundle.scipy_inverse_interp
        self._continuous_stale = False

    def InitializeDataStructures(self):
        """Build triangulation and precompute Forward/Reverse RBF weights on this thread.

        Intended to run on the sticky CUDA transform-init thread, not a random
        multi-worker pool (CuPy context).
        """
        self.apply_refreshed_continuous(self.build_refreshed_continuous())

    def ClearDataStructures(self):
        """Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data"""

        super(MeshWithRBFFallback_GPUComponent, self).ClearDataStructures()
        self._continuous_stale = True
        _invalidate_host_control_point_cache(self)

    def OnFixedPointChanged(self):
        super(MeshWithRBFFallback_GPUComponent, self).OnFixedPointChanged()
        self._continuous_stale = True
        self._host_target_points = None

    def OnWarpedPointChanged(self):
        super(MeshWithRBFFallback_GPUComponent, self).OnWarpedPointChanged()
        self._continuous_stale = True
        self._host_source_points = None

    def _discrete_forward(self, points):
        """Map source→target with cached source Delaunay; do not rebuild interpolators."""
        if self._ForwardInterpolator is not None:
            return super(MeshWithRBFFallback_GPUComponent, self).Transform(points)
        warpedtri = self._warpedtri
        if warpedtri is None:
            if _defer_structure_rebuild():
                return cp.asarray(_nan_mapped_points(points), dtype=cp.float32)
            return super(MeshWithRBFFallback_GPUComponent, self).Transform(points)
        mapped = barycentric_sample_delaunay(
            nornir_imageregistration.EnsureNumpyArray(points),
            warpedtri,
            _cached_host_target_points(self),
            nan_outside=True,
        )
        return cp.asarray(mapped, dtype=cp.float32)

    def _discrete_inverse(self, points):
        """Map target→source with cached target Delaunay; do not rebuild interpolators."""
        if self._InverseInterpolator is not None:
            return super(MeshWithRBFFallback_GPUComponent, self).InverseTransform(points)
        fixedtri = self._fixedtri
        if fixedtri is None:
            if _defer_structure_rebuild():
                return cp.asarray(_nan_mapped_points(points), dtype=cp.float32)
            return super(MeshWithRBFFallback_GPUComponent, self).InverseTransform(points)
        mapped = barycentric_sample_delaunay(
            nornir_imageregistration.EnsureNumpyArray(points),
            fixedtri,
            _cached_host_source_points(self),
            nan_outside=True,
        )
        return cp.asarray(mapped, dtype=cp.float32)

    def Transform(self, points, **kwargs):
        """
        Transform from warped space to fixed space
        :param ndarray points: [[ControlY, ControlX, MappedY, MappedX],...]
        """

        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)

        if points.shape[0] == 0:
            return []

        TransformedPoints = self._discrete_forward(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        TransformedPoints = cp.asarray(TransformedPoints) if not isinstance(TransformedPoints,
                                                                            cp.ndarray) else TransformedPoints
        (_GoodPoints, invalid_mask) = utils.InvalidIndices(TransformedPoints)

        if not bool(invalid_mask.any()):
            return TransformedPoints
        else:
            if len(points) > 1:
                BadPoints = points[invalid_mask]
            else:
                BadPoints = points

        BadPoints = _ensure_float32_64(BadPoints, cp)

        rbf = self._ForwardRBFInstance
        if rbf is None:
            rbf = self.ForwardRBFInstance
        FixedPoints = rbf.Transform(BadPoints)
        FixedPoints = cp.asarray(FixedPoints) if not isinstance(FixedPoints,
                                                                cp.ndarray) else FixedPoints

        TransformedPoints[invalid_mask] = FixedPoints
        return TransformedPoints

    def InverseTransform(self, points, **kwargs):
        """
        Transform from fixed space to warped space
        :param points:
        """

        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)

        if points.shape[0] == 0:
            return []

        TransformedPoints = self._discrete_inverse(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        TransformedPoints = cp.asarray(TransformedPoints) if not isinstance(TransformedPoints,
                                                                            cp.ndarray) else TransformedPoints
        (_GoodPoints, invalid_mask) = utils.InvalidIndices(TransformedPoints)

        if not bool(invalid_mask.any()):
            return TransformedPoints
        else:
            if points.ndim > 1:
                BadPoints = points[invalid_mask]
            else:
                BadPoints = points  # This is likely no longer needed since this function always returns a 2D array now

        BadPoints = _ensure_float32_64(BadPoints, cp)

        rbf = self._ReverseRBFInstance
        if rbf is None:
            rbf = self.ReverseRBFInstance
        FixedPoints = rbf.Transform(BadPoints)
        FixedPoints = cp.asarray(FixedPoints) if not isinstance(FixedPoints,
                                                                cp.ndarray) else FixedPoints

        TransformedPoints[invalid_mask] = FixedPoints
        return TransformedPoints

    def __init__(self, pointpairs):
        """
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX]
        """
        super(MeshWithRBFFallback_GPUComponent, self).__init__(pointpairs)

        self._ReverseRBFInstance = None
        self._ForwardRBFInstance = None
        self._continuous_stale = False
        self._host_source_points = None
        self._host_target_points = None
        self._host_source_copy_count = 0
        self._host_target_copy_count = 0

    @staticmethod
    def Load(TransformString, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseMeshTransform(TransformString, pixelSpacing)


class MeshWithRBFInterpolator_GPU(Landmark_GPU):
    """
    classdocs
    """

    @property
    def type(self) -> TransformType:
        return TransformType.MESH

    def __getstate__(self):

        odict = super(MeshWithRBFInterpolator_GPU, self).__getstate__()
        odict['_ReverseRBFInstance'] = self._ReverseRBFInstance  # type: ignore[assignment]
        odict['_ForwardRBFInstance'] = self._ForwardRBFInstance  # type: ignore[assignment]
        return odict

    def __setstate__(self, dictionary):
        super(MeshWithRBFInterpolator_GPU, self).__setstate__(dictionary)

    @property
    def ReverseRBFInstance(self):
        if self._ReverseRBFInstance is None:
            self._ReverseRBFInstance = super(MeshWithRBFInterpolator_GPU, self).InverseInterpolator

        return self._ReverseRBFInstance

    @property
    def ForwardRBFInstance(self):
        if self._ForwardRBFInstance is None:
            self._ForwardRBFInstance = super(MeshWithRBFInterpolator_GPU, self).ForwardInterpolator

        return self._ForwardRBFInstance

    # def InitializeDataStructures(self):
    #
    #     self._ForwardRBFInstance = cuRBFInterpolator(self.SourcePoints, self.TargetPoints)
    #     self._ReverseRBFInstance = cuRBFInterpolator(self.TargetPoints, self.SourcePoints)
    #
    #
    # def ClearDataStructures(self):
    #     """Something about the transform has changed, for example the points.
    #        Clear out our data structures so we do not use bad data"""
    #
    #     super(MeshWithRBFInterpolator_GPU, self).ClearDataStructures()
    #
    #     self._ForwardRBFInstance = None
    #     self._ReverseRBFInstance = None

    def OnFixedPointChanged(self):
        super(MeshWithRBFInterpolator_GPU, self).OnFixedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def OnWarpedPointChanged(self):
        super(MeshWithRBFInterpolator_GPU, self).OnWarpedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def Transform(self, points, **kwargs):
        """
        Transform from warped space to fixed space
        :param ndarray points: [[ControlY, ControlX, MappedY, MappedX],...]
        """

        TransformedPoints = super(MeshWithRBFInterpolator_GPU, self).Transform(points)
        return TransformedPoints

    def InverseTransform(self, points, **kwargs):
        """
        Transform from fixed space to warped space
        :param points:
        """

        iTransformedPoints = super(MeshWithRBFInterpolator_GPU, self).InverseTransform(points)
        return iTransformedPoints

    def __init__(self, pointpairs):
        """
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX]
        """
        super(MeshWithRBFInterpolator_GPU, self).__init__(pointpairs)

        self._ReverseRBFInstance = None
        self._ForwardRBFInstance = None

    @staticmethod
    def Load(TransformString, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseMeshTransform(TransformString, pixelSpacing)


class MeshWithRBFInterpolator_CPU(Landmark_CPU):
    """
    classdocs
    """

    @property
    def type(self) -> TransformType:
        return TransformType.MESH

    def __getstate__(self):

        odict = super(MeshWithRBFInterpolator_CPU, self).__getstate__()
        odict['_ReverseRBFInstance'] = self._ReverseRBFInstance  # type: ignore[assignment]
        odict['_ForwardRBFInstance'] = self._ForwardRBFInstance  # type: ignore[assignment]
        return odict

    def __setstate__(self, dictionary):
        super(MeshWithRBFInterpolator_CPU, self).__setstate__(dictionary)

    @property
    def ReverseRBFInstance(self):
        if self._ReverseRBFInstance is None:
            self._ReverseRBFInstance = super(MeshWithRBFInterpolator_CPU, self).InverseInterpolator

        return self._ReverseRBFInstance

    @property
    def ForwardRBFInstance(self):
        if self._ForwardRBFInstance is None:
            self._ForwardRBFInstance = super(MeshWithRBFInterpolator_CPU, self).ForwardInterpolator

        return self._ForwardRBFInstance

    # def InitializeDataStructures(self):
    #
    #     self._ForwardRBFInstance = cuRBFInterpolator(self.SourcePoints, self.TargetPoints)
    #     self._ReverseRBFInstance = cuRBFInterpolator(self.TargetPoints, self.SourcePoints)
    #
    #
    # def ClearDataStructures(self):
    #     """Something about the transform has changed, for example the points.
    #        Clear out our data structures so we do not use bad data"""
    #
    #     super(MeshWithRBFInterpolator_CPU, self).ClearDataStructures()
    #
    #     self._ForwardRBFInstance = None
    #     self._ReverseRBFInstance = None

    def OnFixedPointChanged(self):
        super(MeshWithRBFInterpolator_CPU, self).OnFixedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def OnWarpedPointChanged(self):
        super(MeshWithRBFInterpolator_CPU, self).OnWarpedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    # def Transform(self, points, **kwargs):
    #     """
    #     Transform from warped space to fixed space
    #     :param ndarray points: [[ControlY, ControlX, MappedY, MappedX],...]
    #     """
    #
    #     super(MeshWithRBFInterpolator_CPU, self).Transform()
    #
    #     points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
    #
    #     if points.shape[0] == 0:
    #         return []
    #
    #     TransformedPoints = self.ForwardRBFInstance.Transform(points)
    #     return TransformedPoints
    #
    #
    # def InverseTransform(self, points, **kwargs):
    #     """
    #     Transform from fixed space to warped space
    #     :param points:
    #     """
    #
    #     points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
    #
    #     if points.shape[0] == 0:
    #         return []
    #
    #     iTransformedPoints = self.ReverseRBFInstance.Transform(points)
    #     return iTransformedPoints

    def __init__(self, pointpairs):
        """
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX]
        """
        super(MeshWithRBFInterpolator_CPU, self).__init__(pointpairs)

        self._ReverseRBFInstance = None
        self._ForwardRBFInstance = None

    @staticmethod
    def Load(TransformString, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseMeshTransform(TransformString, pixelSpacing)


if __name__ == '__main__':
    print("Test OneWayRBFWithLinearCorrection")
    p = numpy.array([[0, 0, 0, 0],
                     [0, 10, 0, -10],
                     [10, 0, -10, 0],
                     [10, 10, -10, -10]])

    (Fixed, Moving) = numpy.hsplit(p, 2)
    T: Any = OneWayRBFWithLinearCorrection(Fixed, Moving)

    warpedPoints = [[0, 0], [-5, -5]]
    fp = T.Transform(warpedPoints)
    print(("__Transform " + str(warpedPoints) + " to " + str(fp)))
    # wp = T.InverseTransform(fp)

    print("Fixed Verts")
    print(T.FixedTriangles)
    print("\nWarped Verts")
    print(T.WarpedTriangles)

    T.AddPoint([5, 5, -5, -5])
    print("\nPoint added")
    print("Fixed Verts")
    print(T.FixedTriangles)
    print("\nWarped Verts")
    print(T.WarpedTriangles)

    T.AddPoint([5, 5, 5, 5])
    print("\nDuplicate Point added")
    print("Fixed Verts")
    print(T.FixedTriangles)
    print("\nWarped Verts")
    print(T.WarpedTriangles)

    warpedPoint = [[-5, -5]]
    fp = T.Transform(warpedPoint)
    print(("__Transform " + str(warpedPoint) + " to " + str(fp)))
    # wp = T.InverseTransform(fp)

    # T.UpdatePoint(3, [10, 15, -10, -15])
    # print("\nPoint updated")
    # print("Fixed Verts")
    # print(T.FixedTriangles)
    # print("\nWarped Verts")
    # print(T.WarpedTriangles)

    warpedPoint = [[-9, -14]]
    fp = T.Transform(warpedPoint)
    print(("__Transform " + str(warpedPoint) + " to " + str(fp)))
    # wp = T.InverseTransform(fp)

    T.RemovePoint(1)
    print("\nPoint removed")
    print("Fixed Verts")
    print(T.FixedTriangles)
    print("\nWarped Verts")
    print(T.WarpedTriangles)

    print("\nFixedPointsInRect")
    print(T.GetFixedPointsInRect([-1, -1, 14, 4]))

    # GPU
    print("Test OneWayRBFWithLinearCorrection_GPUComponent")
    p_gpu = cp.array([[0, 0, 0, 0],
                      [0, 10, 0, -10],
                      [10, 0, -10, 0],
                      [10, 10, -10, -10]])

    (Fixed, Moving) = cp.hsplit(p_gpu, 2)
    T = OneWayRBFWithLinearCorrection_GPUComponent(Fixed, Moving)

    warpedPoints = [[0, 0], [-5, -5]]
    fp = T.Transform(warpedPoints)
    print(("__Transform " + str(warpedPoints) + " to " + str(fp)))
    # wp = T.InverseTransform(fp)

    print("Fixed Verts")
    print(T.FixedTriangles)
    print("\nWarped Verts")
    print(T.WarpedTriangles)
