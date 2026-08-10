"""
Created on Oct 18, 2012

@author: Jamesan
"""

import numpy
from typing import Any

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
from . import utils
from .triangulation import Triangulation, Triangulation_GPUComponent
from nornir_imageregistration.transforms.landmark import Landmark_GPU, Landmark_CPU


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


class MeshWithRBFFallback(Triangulation):
    """
    classdocs
    """

    @property
    def type(self) -> TransformType:
        return TransformType.MESH

    def __getstate__(self):

        odict = super(MeshWithRBFFallback, self).__getstate__()
        odict['_ReverseRBFInstance'] = self._ReverseRBFInstance  # type: ignore[assignment]
        odict['_ForwardRBFInstance'] = self._ForwardRBFInstance  # type: ignore[assignment]
        return odict

    def __setstate__(self, dictionary):
        super(MeshWithRBFFallback, self).__setstate__(dictionary)

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

    def InitializeDataStructures(self):
        """Build triangulation and precompute Forward/Reverse RBF weights.

        Forward and Reverse weight solves run on the shared global thread pool.
        Call this from a non-pool driver thread (e.g. UI prewarm) so workers are
        not nested-waiting on the same pool.
        """
        Pool = nornir_pools.GetGlobalThreadPool()

        ForwardTask = Pool.add_task(
            "Solve forward RBF transform",
            _build_cpu_rbf_with_weights,
            self.SourcePoints,
            self.TargetPoints,
        )
        ReverseTask = Pool.add_task(
            "Solve reverse RBF transform",
            _build_cpu_rbf_with_weights,
            self.TargetPoints,
            self.SourcePoints,
        )

        super(MeshWithRBFFallback, self).InitializeDataStructures()

        self._ForwardRBFInstance = ForwardTask.wait_return()
        self._ReverseRBFInstance = ReverseTask.wait_return()

    def ClearDataStructures(self):
        """Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data"""

        super(MeshWithRBFFallback, self).ClearDataStructures()

        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def OnFixedPointChanged(self):
        super(MeshWithRBFFallback, self).OnFixedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def OnWarpedPointChanged(self):
        super(MeshWithRBFFallback, self).OnWarpedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def Transform(self, points, **kwargs):
        """
        Transform from warped space to fixed space
        :param ndarray points: [[ControlY, ControlX, MappedY, MappedX],...]
        """

        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        if points.shape[0] == 0:
            return []

        TransformedPoints = super(MeshWithRBFFallback, self).Transform(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        (GoodPoints, invalid_indices, valid_indices) = utils.InvalidIndices(TransformedPoints)

        if len(invalid_indices) == 0:
            return TransformedPoints
        else:
            if len(points) > 1:
                invalid_indices = nornir_imageregistration.EnsureNumpyArray(invalid_indices)
                # print invalid_indices;
                BadPoints = points[invalid_indices]
            else:
                BadPoints = points

        BadPoints = _ensure_float32_64(BadPoints, cp.get_array_module(points))

        FixedPoints = self.ForwardRBFInstance.Transform(BadPoints)
        FixedPoints = _coerce_to_reference_backend(FixedPoints, TransformedPoints)
        invalid_indices = _coerce_indices_to_reference_backend(invalid_indices, TransformedPoints)

        TransformedPoints[invalid_indices] = FixedPoints
        return TransformedPoints

    def InverseTransform(self, points, **kwargs):
        """
        Transform from fixed space to warped space
        :param points:
        """

        points = nornir_imageregistration.EnsurePointsAre2DArray(points)

        if points.shape[0] == 0:
            return []

        TransformedPoints = super(MeshWithRBFFallback, self).InverseTransform(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        (GoodPoints, invalid_indices, valid_indices) = utils.InvalidIndices(TransformedPoints)

        if len(invalid_indices) == 0:
            return TransformedPoints
        else:
            if points.ndim > 1:
                invalid_indices = nornir_imageregistration.EnsureNumpyArray(invalid_indices)
                BadPoints = points[invalid_indices]
            else:
                BadPoints = points  # This is likely no longer needed since this function always returns a 2D array now

        BadPoints = _ensure_float32_64(BadPoints, cp.get_array_module(points))

        FixedPoints = self.ReverseRBFInstance.Transform(BadPoints)
        FixedPoints = _coerce_to_reference_backend(FixedPoints, TransformedPoints)
        invalid_indices = _coerce_indices_to_reference_backend(invalid_indices, TransformedPoints)

        TransformedPoints[invalid_indices] = FixedPoints
        return TransformedPoints

    def __init__(self, pointpairs):
        """
        :param ndarray pointpairs: [TargetY, TargetX, SourceY, SourceX]
        """
        super(MeshWithRBFFallback, self).__init__(pointpairs)

        self._ReverseRBFInstance = None
        self._ForwardRBFInstance = None

    @staticmethod
    def Load(TransformString, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseMeshTransform(TransformString, pixelSpacing)


class MeshWithRBFFallback_GPUComponent(Triangulation_GPUComponent):
    """
    classdocs
    """

    @property
    def type(self) -> TransformType:
        return TransformType.MESH

    def __getstate__(self):

        odict = super(MeshWithRBFFallback_GPUComponent, self).__getstate__()
        odict['_ReverseRBFInstance'] = self._ReverseRBFInstance  # type: ignore[assignment]
        odict['_ForwardRBFInstance'] = self._ForwardRBFInstance  # type: ignore[assignment]
        return odict

    def __setstate__(self, dictionary):
        super(MeshWithRBFFallback_GPUComponent, self).__setstate__(dictionary)

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

    def InitializeDataStructures(self):
        """Build triangulation and precompute Forward/Reverse RBF weights on this thread.

        Intended to run on the sticky CUDA transform-init thread, not a random
        multi-worker pool (CuPy context).
        """
        super(MeshWithRBFFallback_GPUComponent, self).InitializeDataStructures()

        self._ForwardRBFInstance = OneWayRBFWithLinearCorrection_GPUComponent(self.SourcePoints, self.TargetPoints)
        self._ForwardRBFInstance.PrecomputeWeights()
        self._ReverseRBFInstance = OneWayRBFWithLinearCorrection_GPUComponent(self.TargetPoints, self.SourcePoints)
        self._ReverseRBFInstance.PrecomputeWeights()

    def ClearDataStructures(self):
        """Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data"""

        super(MeshWithRBFFallback_GPUComponent, self).ClearDataStructures()

        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def OnFixedPointChanged(self):
        super(MeshWithRBFFallback_GPUComponent, self).OnFixedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def OnWarpedPointChanged(self):
        super(MeshWithRBFFallback_GPUComponent, self).OnWarpedPointChanged()
        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None

    def Transform(self, points, **kwargs):
        """
        Transform from warped space to fixed space
        :param ndarray points: [[ControlY, ControlX, MappedY, MappedX],...]
        """

        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)

        if points.shape[0] == 0:
            return []

        TransformedPoints = super(MeshWithRBFFallback_GPUComponent, self).Transform(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            #     return TransformedPoints
            return TransformedPoints

        TransformedPoints = cp.asarray(TransformedPoints) if not isinstance(TransformedPoints,
                                                                            cp.ndarray) else TransformedPoints
        (GoodPoints, invalid_indices, valid_indices) = utils.InvalidIndices(TransformedPoints)

        if len(invalid_indices) == 0:
            return TransformedPoints
        else:
            if len(points) > 1:
                # print invalid_indices;
                BadPoints = points[invalid_indices]
            else:
                BadPoints = points

        BadPoints = _ensure_float32_64(BadPoints, cp)

        FixedPoints = self.ForwardRBFInstance.Transform(BadPoints)
        FixedPoints = cp.asarray(FixedPoints) if not isinstance(FixedPoints,
                                                                cp.ndarray) else FixedPoints

        TransformedPoints[invalid_indices] = FixedPoints
        return TransformedPoints

    def InverseTransform(self, points, **kwargs):
        """
        Transform from fixed space to warped space
        :param points:
        """

        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)

        if points.shape[0] == 0:
            return []

        TransformedPoints = super(MeshWithRBFFallback_GPUComponent, self).InverseTransform(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        TransformedPoints = cp.asarray(TransformedPoints) if not isinstance(TransformedPoints,
                                                                            cp.ndarray) else TransformedPoints
        (GoodPoints, invalid_indices, valid_indices) = utils.InvalidIndices(TransformedPoints)

        if len(invalid_indices) == 0:
            return TransformedPoints
        else:
            if points.ndim > 1:
                BadPoints = points[invalid_indices]
            else:
                BadPoints = points  # This is likely no longer needed since this function always returns a 2D array now

        BadPoints = _ensure_float32_64(BadPoints, cp)

        FixedPoints = self.ReverseRBFInstance.Transform(BadPoints)
        FixedPoints = cp.asarray(FixedPoints) if not isinstance(FixedPoints,
                                                                cp.ndarray) else FixedPoints

        TransformedPoints[invalid_indices] = FixedPoints
        return TransformedPoints

    def __init__(self, pointpairs):
        """
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX]
        """
        super(MeshWithRBFFallback_GPUComponent, self).__init__(pointpairs)

        self._ReverseRBFInstance = None
        self._ForwardRBFInstance = None

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
