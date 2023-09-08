"""
Created on Oct 18, 2012

@author: Jamesan
"""

import nornir_imageregistration 
from nornir_imageregistration.transforms.one_way_rbftransform import OneWayRBFWithLinearCorrection, OneWayRBFWithLinearCorrection_GPUComponent
from nornir_imageregistration.transforms.transform_type import TransformType
import numpy
import cupy as cp

import nornir_pools 
# from .triangulation import Triangulation
from nornir_imageregistration.transforms.triangulation import Triangulation, Triangulation_GPUComponent

from . import utils, NumberOfControlPointsToTriggerMultiprocessing
# Added by Clem for direct execution (main function)
# import utils
# NumberOfControlPointsToTriggerMultiprocessing = 20


class MeshWithRBFFallback(Triangulation):
    """
    classdocs
    """

    @property
    def type(self) -> TransformType:
        return TransformType.MESH
    
    def __getstate__(self):
        
        odict = super(MeshWithRBFFallback, self).__getstate__()
        odict['_ReverseRBFInstance'] = self._ReverseRBFInstance
        odict['_ForwardRBFInstance'] = self._ForwardRBFInstance
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

        if self.NumControlPoints <= NumberOfControlPointsToTriggerMultiprocessing:
            Pool = nornir_pools.GetGlobalThreadPool()
        else:
            Pool = nornir_pools.GetGlobalMultithreadingPool()

        ForwardTask = Pool.add_task("Solve forward RBF transform", OneWayRBFWithLinearCorrection, self.SourcePoints, self.TargetPoints)
        ReverseTask = Pool.add_task("Solve reverse RBF transform", OneWayRBFWithLinearCorrection, self.TargetPoints, self.SourcePoints)

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

        (GoodPoints, InvalidIndicies) = utils.InvalidIndicies(TransformedPoints)

        if(len(InvalidIndicies) == 0):
            return TransformedPoints
        else:
            if len(points) > 1:
                # print InvalidIndicies;
                BadPoints = points[InvalidIndicies]
            else:
                BadPoints = points

        BadPoints = numpy.asarray(BadPoints, dtype=numpy.float32)
        if not (BadPoints.dtype == numpy.float32 or BadPoints.dtype == numpy.float64):
            BadPoints = numpy.asarray(BadPoints, dtype=numpy.float32)
            
        FixedPoints = self.ForwardRBFInstance.Transform(BadPoints)

        TransformedPoints[InvalidIndicies] = FixedPoints
        return TransformedPoints

    def InverseTransform(self, points, **kwargs):
        """
        Transform from fixed space to warped space
        :param points:
        """

        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        if points.shape[0] == 0:
            return []

        TransformedPoints = super(MeshWithRBFFallback, self).InverseTransform(points)
        extrapolate = kwargs.get('extrapolate', True)
        if not extrapolate:
            return TransformedPoints

        (GoodPoints, InvalidIndicies) = utils.InvalidIndicies(TransformedPoints)

        if(len(InvalidIndicies) == 0):
            return TransformedPoints
        else:
            if points.ndim > 1:
                BadPoints = points[InvalidIndicies]
            else:
                BadPoints = points  # This is likely no longer needed since this function always returns a 2D array now

        if not (BadPoints.dtype == numpy.float32 or BadPoints.dtype == numpy.float64):
            BadPoints = numpy.asarray(BadPoints, dtype=numpy.float32)

        FixedPoints = self.ReverseRBFInstance.Transform(BadPoints)

        TransformedPoints[InvalidIndicies] = FixedPoints
        return TransformedPoints

    def __init__(self, pointpairs):
        """
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX] 
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
        odict['_ReverseRBFInstance'] = self._ReverseRBFInstance
        odict['_ForwardRBFInstance'] = self._ForwardRBFInstance
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

        if self.NumControlPoints <= NumberOfControlPointsToTriggerMultiprocessing:
            Pool = nornir_pools.GetGlobalThreadPool()
        else:
            Pool = nornir_pools.GetGlobalMultithreadingPool()

        ForwardTask = Pool.add_task("Solve forward RBF transform", OneWayRBFWithLinearCorrection_GPUComponent, self.SourcePoints,
                                    self.TargetPoints)
        ReverseTask = Pool.add_task("Solve reverse RBF transform", OneWayRBFWithLinearCorrection_GPUComponent, self.TargetPoints,
                                    self.SourcePoints)

        super(MeshWithRBFFallback_GPUComponent, self).InitializeDataStructures()

        self._ForwardRBFInstance = ForwardTask.wait_return()
        self._ReverseRBFInstance = ReverseTask.wait_return()

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

    def Transform(self, points, return_cp: bool = False, **kwargs):
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
            # if return_cp:
            #     return TransformedPoints
            # else:
            #     return TransformedPoints.get()
            return TransformedPoints

        TransformedPoints = cp.asarray(TransformedPoints) if not isinstance(TransformedPoints,
                                                                            cp.ndarray) else TransformedPoints
        (GoodPoints, InvalidIndicies, ValidIndicies) = utils.InvalidIndicies_GPU(TransformedPoints)

        if (len(InvalidIndicies) == 0):
            if return_cp:
                return TransformedPoints
            else:
                return TransformedPoints.get()
        else:
            if len(points) > 1:
                # print InvalidIndicies;
                BadPoints = points[InvalidIndicies]
            else:
                BadPoints = points

        # BadPoints = cp.asarray(BadPoints, dtype=numpy.float32)
        if not (BadPoints.dtype == numpy.float32 or BadPoints.dtype == numpy.float64):
            BadPoints = cp.asarray(BadPoints, dtype=numpy.float32)

        FixedPoints = self.ForwardRBFInstance.Transform(BadPoints)
        FixedPoints = cp.asarray(FixedPoints) if not isinstance(FixedPoints,
                                                                cp.ndarray) else FixedPoints

        TransformedPoints[InvalidIndicies] = FixedPoints
        if return_cp:
            return TransformedPoints
        else:
            return TransformedPoints.get()

    def InverseTransform(self, points, return_cp: bool = False, **kwargs):
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
            # if return_cp:
            #     return TransformedPoints
            # else:
            #     return TransformedPoints.get()
            return TransformedPoints

        TransformedPoints = cp.asarray(TransformedPoints) if not isinstance(TransformedPoints,
                                                                            cp.ndarray) else TransformedPoints
        (GoodPoints, InvalidIndicies, ValidIndicies) = utils.InvalidIndicies_GPU(TransformedPoints)

        if (len(InvalidIndicies) == 0):
            if return_cp:
                return TransformedPoints
            else:
                return TransformedPoints.get()
        else:
            if points.ndim > 1:
                BadPoints = points[InvalidIndicies]
            else:
                BadPoints = points  # This is likely no longer needed since this function always returns a 2D array now

        if not (BadPoints.dtype == numpy.float32 or BadPoints.dtype == numpy.float64):
            BadPoints = cp.asarray(BadPoints, dtype=numpy.float32)

        FixedPoints = self.ReverseRBFInstance.Transform(BadPoints)
        FixedPoints = cp.asarray(FixedPoints) if not isinstance(FixedPoints,
                                                                            cp.ndarray) else FixedPoints

        TransformedPoints[InvalidIndicies] = FixedPoints
        if return_cp:
            return TransformedPoints
        else:
            return TransformedPoints.get()

    def __init__(self, pointpairs):
        """
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX]
        """
        super(MeshWithRBFFallback_GPUComponent, self).__init__(pointpairs)

        self._ReverseRBFInstance = None
        self._ForwardRBFInstance = None

    @staticmethod
    def Load(TransformString, pixelSpacing=None):
        return nornir_imageregistration.transforms.factory.ParseMeshTransform(TransformString, pixelSpacing,
                                                                              use_cp=True)


if __name__ == '__main__':
    print("Test OneWayRBFWithLinearCorrection")
    p = numpy.array([[0, 0, 0, 0],
                  [0, 10, 0, -10],
                  [10, 0, -10, 0],
                  [10, 10, -10, -10]])

    (Fixed, Moving) = numpy.hsplit(p, 2)
    T = OneWayRBFWithLinearCorrection(Fixed, Moving)

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