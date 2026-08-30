'''
Created on Oct 18, 2012

@author: Jamesan
'''

from typing import Callable

import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp

import scipy.interpolate
import scipy.linalg

import nornir_imageregistration
from nornir_imageregistration.spatial_distance import array_to_numpy_host, cdist as pairwise_cdist
from nornir_imageregistration.transforms.controlpointbase import ControlPointsHaveDuplicatePositions
from nornir_imageregistration.transforms.transform_type import TransformType
from .triangulation import Triangulation, Triangulation_GPUComponent


def _is_singular_matrix_error(err: BaseException) -> bool:
    """True when a LinAlgError reports a singular system.

    The wording is not stable across backends, so an equality test against any
    one of these silently disables the rigid-fallback recovery:

      SciPy >= 1.11  "A singular matrix detected: slice(s) [0] are singular."
      SciPy older    "Matrix is singular."
      NumPy          "Singular matrix"
      CuPy           varies with the cuSOLVER status it wraps

    Matching on the word itself keeps every backend on the recovery path while
    still re-raising the shape and dtype errors that indicate a caller bug.
    """
    return 'singular' in ' '.join(str(arg) for arg in err.args).lower()


# How far the rigid shortcut may move a control point away from its target before it
# stops being an acceptable stand-in for the full RBF evaluation.
#
# The shortcut used to be chosen by asking whether the deviation weights were close to
# zero, which is not a safe test: those weights multiply the basis function r^2*log r,
# which is about 1.15e11 at a 100k-pixel section extent, so a weight of 1e-9 -- "zero"
# to allclose's default atol of 1e-8 -- still displaces a point by roughly 115 px. That
# let genuinely non-rigid warps take the shortcut, and the larger the section the worse
# it got: at a 100k extent a warp with a 10 px non-rigid component was accepted.
#
# Measuring the error in pixels instead separates the two cases with room to spare. A
# genuinely rigid warp reproduces its own control points to under 0.008 px even at a
# 100k extent, while every warp the weight test wrongly accepted was off by at least
# 0.13 px. This bound sits in that gap, and errs toward doing the full RBF evaluation.
_MAX_RIGID_EQUIVALENT_ERROR_PIXELS = 0.05


def _rigid_transform_is_equivalent(candidate, source_points, target_points) -> bool:
    """Whether *candidate* reproduces the control points closely enough to stand in.

    Checked on the host: this runs once per weight solve over the control points only,
    and the result is a single bool.
    """
    source = nornir_imageregistration.EnsureNumpyArray(source_points)
    target = nornir_imageregistration.EnsureNumpyArray(target_points)
    reproduced = nornir_imageregistration.EnsureNumpyArray(candidate.Transform(source))
    max_error = float(np.max(np.abs(np.asarray(reproduced, dtype=np.float64)
                                    - np.asarray(target, dtype=np.float64))))
    return max_error <= _MAX_RIGID_EQUIVALENT_ERROR_PIXELS


def _tps_beta_matrix(
        points: NDArray,
        basis_function: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        xp,
) -> NDArray:
    """Build the thin-plate spline Beta matrix on *xp* (pairwise RBF plus linear terms)."""
    # Duplicate check is N control points (typically hundreds). CuPy unique on
    # that size is slower than a 2KB download plus NumPy unique.
    if ControlPointsHaveDuplicatePositions(array_to_numpy_host(points)):
        raise ValueError("Cannot have duplicate points in transform")

    points = xp.asarray(points)
    num_pts = int(points.shape[0])
    beta = xp.zeros((num_pts + 3, num_pts + 3), dtype=np.float32)
    if num_pts >= 2:
        distances = pairwise_cdist(points, points)
        if xp is cp and cp.get_array_module(distances) is np:
            distances = cp.asarray(distances)
        eye = xp.eye(num_pts, dtype=bool)
        # r^2 log(r) is undefined at r=0; the old loop skipped the diagonal.
        dist_safe = xp.where(eye, xp.asarray(1.0, dtype=distances.dtype), distances)
        values = basis_function(dist_safe)
        values = xp.where(eye, xp.asarray(0.0, dtype=np.float32), values)
        beta[3:, :num_pts] = values.astype(beta.dtype, copy=False)
    beta[3:, num_pts] = points[:, 1]
    beta[3:, num_pts + 1] = points[:, 0]
    beta[3:, num_pts + 2] = 1
    beta[0, :num_pts] = points[:, 0]
    beta[1, :num_pts] = points[:, 1]
    beta[2, :num_pts] = 1
    return beta


class OneWayRBFWithLinearCorrection(Triangulation):

    @property
    def type(self) -> TransformType:
        return TransformType.MESH

    def __setstate__(self, state):
        super(OneWayRBFWithLinearCorrection, self).__setstate__(state)
        self._weights = state['_weights']
        rigid_transform = state.get('_rigid_transform', None)
        self._rigid_transform = None if rigid_transform is None else nornir_imageregistration.transforms.LoadTransform(
            rigid_transform)

        # TODO: Can I serialize a python function pointer?
        self.BasisFunction = OneWayRBFWithLinearCorrection.DefaultBasisFunction
        return

    def __getstate__(self):
        odict = super(OneWayRBFWithLinearCorrection, self).__getstate__()
        odict['_weights'] = self._weights  # type: ignore[assignment]

        if '_rigid_transform' not in odict:
            odict['_rigid_transform'] = self._rigid_transform.ToITKString() if self.UseRigidTransform else None  # type: ignore[assignment, union-attr]

        return odict

    @property
    def Weights(self):
        if self._weights is None:
            assert self.BasisFunction is not None
            self._weights, use_rigid_transform = self.CalculateRBFWeights(self.SourcePoints, self.TargetPoints,
                                                                          self.BasisFunction)
            self._rigid_transform = None
            if use_rigid_transform:
                candidate = nornir_imageregistration.transforms.converters.ConvertTransformToRigidTransform(
                    self)
                # Confirm the shortcut before adopting it; see
                # _MAX_RIGID_EQUIVALENT_ERROR_PIXELS for why the weight test alone is not
                # enough to establish rigid equivalence.
                if _rigid_transform_is_equivalent(candidate, self.SourcePoints, self.TargetPoints):
                    self._rigid_transform = candidate

        return self._weights

    def PrecomputeWeights(self) -> None:
        """Force RBF weight solve so later Transform calls do not pay init cost."""
        _ = self.Weights

    @property
    def UseRigidTransform(self) -> bool:
        return self._rigid_transform is not None

    @staticmethod
    def DefaultBasisFunction(distance: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.multiply(np.power(distance, 2), np.log(distance))

    def __init__(self, WarpedPoints: NDArray[np.floating], FixedPoints: NDArray[np.floating],
                 BasisFunction: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None):

        points = np.hstack((
            nornir_imageregistration.EnsureNumpyArray(FixedPoints),
            nornir_imageregistration.EnsureNumpyArray(WarpedPoints),
        ))
        super(OneWayRBFWithLinearCorrection, self).__init__(points)

        self._rigid_transform = None
        self._weights = None

        # self.ControlPoints = TargetPoints
        # self.SourcePoints = SourcePoints
        self.BasisFunction = BasisFunction
        if self.BasisFunction is None:
            self.BasisFunction = OneWayRBFWithLinearCorrection.DefaultBasisFunction

    @staticmethod
    def Load(TransformString: str, pixelSpacing: float | None = None):
        return nornir_imageregistration.transforms.factory.ParseMeshTransform(TransformString, pixelSpacing)

    def _GetMatrixWeightSums(self, Points: NDArray[np.floating], WarpedPoints: NDArray[np.floating],
                             MaxChunkSize: int = 65536):
        NumCtrlPts = len(WarpedPoints)
        NumPts = Points.shape[0]

        # This calculation has an NumPoints X NumWarpedPoints memory footprint when there are a large number of points
        if NumPts <= MaxChunkSize:
            Distances = pairwise_cdist(Points, WarpedPoints)

            # VectorBasisFunc = np.vectorize( self.BasisFunction)
            # FuncValues = VectorBasisFunc(Distances)

            # We have to check for zeros so we don't crash if the transformed point exactly matches our control point
            nonzero = Distances != 0
            FuncValues = np.zeros(Distances.shape)

            if np.all(nonzero):
                FuncValues = np.multiply(np.power(Distances, 2.0), np.log(Distances))
            else:
                FuncValues[nonzero] = np.multiply(np.power(Distances[nonzero], 2.0), np.log(Distances[nonzero]))

            del Distances

            MatrixWeightSumX = np.sum(self.Weights[0:NumCtrlPts] * FuncValues, axis=1)
            MatrixWeightSumY = np.sum(self.Weights[3 + NumCtrlPts:(NumCtrlPts * 2) + 3] * FuncValues, axis=1)

            del FuncValues

            return MatrixWeightSumX, MatrixWeightSumY
        else:
            # Cut the array into chunks and build the weights over a loop
            iStart = 0
            MatrixWeightSumX = np.zeros(NumPts)
            MatrixWeightSumY = np.zeros(NumPts)
            while iStart < NumPts:
                iEnd = iStart + MaxChunkSize
                if iEnd > Points.shape[0]:
                    iEnd = Points.shape[0]

                (MatrixWeightSumXChunk, MatrixWeightSumYChunk) = self._GetMatrixWeightSums(Points[iStart:iEnd, :],
                                                                                           WarpedPoints)

                # Failing these asserts means we are stomping earlier results
                assert (MatrixWeightSumX[iStart] == 0)
                assert (MatrixWeightSumY[iStart] == 0)

                MatrixWeightSumX[iStart:iEnd] = MatrixWeightSumXChunk
                MatrixWeightSumY[iStart:iEnd] = MatrixWeightSumYChunk

                iStart = iEnd

            return MatrixWeightSumX, MatrixWeightSumY

    def Transform(self, Points: NDArray[np.floating], **kwargs):
        # if self._rigid_transform is not None:
        #   return self._rigid_transform.Transform(Points)

        Points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(Points)

        # The lazy weight solve is what decides whether the rigid shortcut applies, so it
        # has to run before UseRigidTransform is read. Otherwise the first call on a fresh
        # transform is served by the RBF branch and every later call by the rigid branch,
        # which made Transform return a different answer for the same input array.
        _ = self.Weights

        NumCtrlPts = len(self.TargetPoints)

        if self.UseRigidTransform:
            return self._rigid_transform.Transform(Points)  # type: ignore[union-attr]
            # NumPts = Points.shape[0]
            # MatrixWeightSumX = np.zeros((1, NumPts))
            # MatrixWeightSumY = np.zeros((1, NumPts))
        else:
            (MatrixWeightSumX, MatrixWeightSumY) = self._GetMatrixWeightSums(Points, self.SourcePoints)

        # (UnchunkedMatrixWeightSumX, MatrixWeightSumY) = self._GetMatrixWeightSums(points, self.TargetPoints, self.SourcePoints, MaxChunkSize=32768000)
        # assert(MatrixWeightSumX == UnchunkedMatrixWeightSumX)

        Xa = Points[:, 1] * self.Weights[NumCtrlPts]
        Xb = Points[:, 0] * self.Weights[NumCtrlPts + 1]
        Xc = self.Weights[NumCtrlPts + 2]
        XBase = np.vstack((MatrixWeightSumX, Xa, Xb))
        Xf = np.sum(XBase, axis=0) + Xc

        del Xa
        del Xb
        del Xc
        del XBase
        del MatrixWeightSumX

        Ya = Points[:, 1] * self.Weights[NumCtrlPts + 3 + NumCtrlPts]
        Yb = Points[:, 0] * self.Weights[NumCtrlPts + NumCtrlPts + 3 + 1]
        Yc = self.Weights[NumCtrlPts + NumCtrlPts + 3 + 2]
        YBase = np.vstack((MatrixWeightSumY, Ya, Yb))
        Yf = np.sum(YBase, axis=0) + Yc

        del Ya
        del Yb
        del Yc
        del YBase
        del MatrixWeightSumY

        MatrixOutpoints = np.vstack((Xf, Yf)).transpose()

        #        for iPoint in range(0, points.shape[0]):
        #            Point = points[iPoint]
        #            PFuncVals = FuncValues[iPoint]
        #
        #            # WeightSumX = np.sum(self.Weights[0:NumCtrlPts] * PFuncVals)
        #            # WeightSumY = np.sum(self.Weights[3 + NumCtrlPts:(NumCtrlPts * 2) + 3] * PFuncVals)
        #            WeightSumX = MatrixWeightSumX[iPoint]
        #            WeightSumY = MatrixWeightSumY[iPoint]
        #
        #            X = WeightSumX + (Point[1] * self.Weights[NumCtrlPts]) + (Point[0] * self.Weights[NumCtrlPts + 1]) + self.Weights[NumCtrlPts + 2]
        #            Y = WeightSumY + (Point[1] * self.Weights[NumCtrlPts + 3 + NumCtrlPts]) + (Point[0] * self.Weights[NumCtrlPts + NumCtrlPts + 3 + 1]) + self.Weights[NumCtrlPts + NumCtrlPts + 3 + 2]
        #
        #            assert(round(X, 1) == round(Xf[iPoint], 1))
        #            assert(round(Y, 1) == round(Yf[iPoint], 1))
        #
        #            assert(round(MatrixOutpoints[iPoint, 0], 1) == round(X, 1))
        #            assert(round(MatrixOutpoints[iPoint, 1], 1) == round(Y, 1))
        #
        #            OutPoints[iPoint, :] = [X, Y]

        return MatrixOutpoints

    def InverseTransform(self, Points: NDArray, **kwargs):
        raise NotImplementedError("RBF Transform does not support inverse transformations")

    @staticmethod
    def CreateSolutionMatricies(ControlPoints: NDArray):

        NumPts = len(ControlPoints)

        ResultMatrixX = np.zeros([NumPts + 3])
        ResultMatrixY = np.zeros([NumPts + 3])

        ResultMatrixX[3:] = ControlPoints[:, 0]  # .reshape(NumPts,1)
        ResultMatrixY[3:] = ControlPoints[:, 1]  # .reshape(NumPts,1)

        return ResultMatrixX, ResultMatrixY

    @staticmethod
    def CreateBetaMatrix(points: NDArray, BasisFunction: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None):
        if BasisFunction is None:
            BasisFunction = OneWayRBFWithLinearCorrection.DefaultBasisFunction
        return _tps_beta_matrix(points, BasisFunction, np)

    @staticmethod
    def CalculateRBFWeights(WarpedPoints: NDArray, ControlPoints: NDArray,
                            BasisFunction: Callable[[NDArray[np.floating]], NDArray[np.floating]]) -> tuple[
        NDArray, bool]:
        '''
        For each axis this function fits a rigid transformation (with rotation) to the points and then assigns weights to the remaining errors in the fit.
        
        The weights matrix is broken down as follows for N control points
        Weights[0:N] = Fit of point deviation from the rigid transformation
        Weights[N:N+1] = Rotation component of transformation
        Weights[N+3] = Translation component of transformation.

        If Weights[0:N] ~= 0, then we can use a much faster and simpler rigid transformation with rotation to translate the data
        If additionally Weights[N:N+1] ~= 0, then we can simply translate the points as needed    
        '''
        use_rigid_transform = False

        BetaMatrix = OneWayRBFWithLinearCorrection.CreateBetaMatrix(WarpedPoints, BasisFunction)
        (SolutionMatrix_X, SolutionMatrix_Y) = OneWayRBFWithLinearCorrection.CreateSolutionMatricies(ControlPoints)

        try:
            with nornir_imageregistration.IgnoreLinAlgWarning() as context:
                # Solve both axes on this thread. Parallelize Forward/Reverse RBF builds at
                # MeshWithRBFFallback.InitializeDataStructures instead of nesting pool waits.
                WeightsX = scipy.linalg.solve(BetaMatrix, SolutionMatrix_X, overwrite_b=True, check_finite=False)
                WeightsY = scipy.linalg.solve(BetaMatrix, SolutionMatrix_Y, overwrite_b=True, check_finite=False)

            if np.allclose(WeightsX[0:-3], 0) and np.allclose(WeightsY[0:-3], 0):
                # prettyoutput.Log("RBF transform is approximately Rigid")
                use_rigid_transform = True

                # source_rotation_center, rotation_matrix, scale, translation, reflected = nornir_imageregistration.transforms.converters._kabsch_umeyama(ControlPoints, WarpedPoints)

            return np.hstack([WeightsX, WeightsY]), use_rigid_transform
        except np.linalg.LinAlgError as e:
            if _is_singular_matrix_error(e):
                # This is a distraction for now, but I should be able to fill in these weights correctly
                # rigid_components = nornir_imageregistration.transforms.converters.EstimateRigidComponentsFromControlPoints(ControlPoints,WarpedPoints)
                # source_rotation_center, rotation_matrix, scale, translation, reflected = nornir_imageregistration.transforms.converters._kabsch_umeyama(
                #     ControlPoints, WarpedPoints)
                result = nornir_imageregistration.transforms.converters.EstimateRigidComponentsFromControlPoints(
                    ControlPoints, WarpedPoints)
                rotation_matrix = nornir_imageregistration.transforms.RotationMatrix(np.radians(result.angle))
                WeightsY = np.zeros(SolutionMatrix_Y.shape)
                WeightsY[-3] = rotation_matrix[0, 1]
                WeightsY[-2] = result.scale
                WeightsY[-1] = result.translation[0]

                WeightsX = np.zeros(SolutionMatrix_X.shape)
                WeightsX[-3] = rotation_matrix[0, 0]
                WeightsX[-2] = result.scale
                WeightsX[-1] = result.translation[1]

                # WeightsY = np.zeros(SolutionMatrix_Y.shape)
                # WeightsY[-3] = rotation_matrix[1, 0]
                # WeightsY[-2] = scale
                # WeightsY[-1] = translation[0]
                #
                # WeightsX = np.zeros(SolutionMatrix_X.shape)
                # WeightsX[-3] = rotation_matrix[0, 0]
                # WeightsX[-2] = scale
                # WeightsX[-1] = translation[1]

                return np.hstack([WeightsX, WeightsY]), True
            else:
                raise

    @property
    def LinearComponents(self):
        """The angle of rotation for the linear portion of the transform"""
        nPoints = self.points.shape[0]
        rotate_x_component = self.Weights[nPoints]
        scale_x_component = self.Weights[nPoints + 1]
        translate_x_component = self.Weights[nPoints + 2]

        axis_offset = nPoints + 3
        rotate_y_component = self.Weights[axis_offset + nPoints]
        scale_y_component = self.Weights[axis_offset + nPoints + 1]
        translate_y_component = self.Weights[axis_offset + nPoints + 2]

        angle = np.arctan2(rotate_x_component, rotate_y_component)
        scale = [scale_y_component, scale_x_component]
        rotate = [rotate_y_component, rotate_x_component]
        translate = [translate_y_component, translate_x_component]
        source_rotation_center = np.mean(self.points[:, 2:], 0)

        return source_rotation_center, angle, translate, scale

    # def AddPoint(self, pointpair: NDArray[np.floating]):
    #     self.points = np.vstack((self.points, pointpair))
    #     self.OnTransformChanged()
    #
    # def AddPoints(self, new_points: NDArray[np.floating]):
    #     self.points = np.vstack((self.points, new_points))
    #     self.OnTransformChanged()
    #
    # def UpdatePointPair(self, index: int, pointpair: NDArray[np.floating]):
    #     self.points[index, :] = pointpair
    #     self.OnTransformChanged()
    #
    # def UpdateTargetPoints(self, index: int | NDArray[np.integer], points: NDArray[np.floating]):
    #     self.points[index, 0:2] = points
    #     self.OnFixedPointChanged()
    #
    # def UpdateSourcePoints(self, index: int | NDArray[np.integer], points: NDArray[np.floating]):
    #     self.points[index, 2:4] = points
    #     self.OnWarpedPointChanged()
    #
    # def RemovePoint(self, index: int):
    #     del self.points[index, :]
    #     self.OnTransformChanged()

    def OnFixedPointChanged(self):
        super(OneWayRBFWithLinearCorrection, self).OnFixedPointChanged()
        self._weights = None
        self._rigid_transform = None
        super(OneWayRBFWithLinearCorrection, self).OnTransformChanged()

    def OnWarpedPointChanged(self):
        super(OneWayRBFWithLinearCorrection, self).OnWarpedPointChanged()
        self._weights = None
        self._rigid_transform = None
        super(OneWayRBFWithLinearCorrection, self).OnTransformChanged()

    def ClearDataStructures(self):
        super(OneWayRBFWithLinearCorrection, self).ClearDataStructures()
        self._weights = None
        self._rigid_transform = None


class OneWayRBFWithLinearCorrection_GPUComponent(Triangulation_GPUComponent):

    @property
    def type(self) -> TransformType:
        return TransformType.MESH

    def __setstate__(self, state):
        super(OneWayRBFWithLinearCorrection_GPUComponent, self).__setstate__(state)
        self._weights = state['_weights']
        rigid_transform = state.get('_rigid_transform', None)
        self._rigid_transform = None if rigid_transform is None else nornir_imageregistration.transforms.LoadTransform(
            rigid_transform)

        # TODO: Can I serialize a python function pointer?
        self.BasisFunction = OneWayRBFWithLinearCorrection_GPUComponent.DefaultBasisFunction
        return

    def __getstate__(self):
        odict = super(OneWayRBFWithLinearCorrection_GPUComponent, self).__getstate__()
        odict['_weights'] = self._weights  # type: ignore[assignment]

        if '_rigid_transform' not in odict:
            odict['_rigid_transform'] = self._rigid_transform.ToITKString() if self.UseRigidTransform else None  # type: ignore[assignment, union-attr]

        return odict

    @property
    def Weights(self):
        if self._weights is None:
            assert self.BasisFunction is not None
            self._weights, use_rigid_transform = self.CalculateRBFWeights(self.SourcePoints, self.TargetPoints,
                                                                          self.BasisFunction)
            self._rigid_transform = None
            if use_rigid_transform:
                candidate = nornir_imageregistration.transforms.converters.ConvertTransformToRigidTransform(
                    self)
                # Confirm the shortcut before adopting it; see
                # _MAX_RIGID_EQUIVALENT_ERROR_PIXELS for why the weight test alone is not
                # enough to establish rigid equivalence.
                if _rigid_transform_is_equivalent(candidate, self.SourcePoints, self.TargetPoints):
                    self._rigid_transform = candidate

        return self._weights

    def PrecomputeWeights(self) -> None:
        """Force RBF weight solve so later Transform calls do not pay init cost."""
        _ = self.Weights

    @property
    def UseRigidTransform(self) -> bool:
        return self._rigid_transform is not None

    @staticmethod
    def DefaultBasisFunction(distance: NDArray[np.floating]) -> NDArray[np.floating]:
        # cdist / host paths may pass NumPy; CuPy ufuncs reject ndarray without an explicit transfer.
        d = cp.asarray(distance)
        return cp.multiply(cp.power(d, 2), cp.log(d))

    def __init__(self, WarpedPoints: NDArray[np.floating], FixedPoints: NDArray[np.floating],
                 BasisFunction: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None):

        points = cp.hstack((cp.asarray(FixedPoints), cp.asarray(WarpedPoints)))
        super(OneWayRBFWithLinearCorrection_GPUComponent, self).__init__(points)

        self._rigid_transform = None
        self._weights = None

        # self.ControlPoints = TargetPoints
        # self.SourcePoints = SourcePoints
        self.BasisFunction = BasisFunction
        if self.BasisFunction is None:
            self.BasisFunction = OneWayRBFWithLinearCorrection_GPUComponent.DefaultBasisFunction

    @staticmethod
    def Load(TransformString: str, pixelSpacing: float | None = None):
        return nornir_imageregistration.transforms.factory.ParseMeshTransform(TransformString, pixelSpacing)

    def _GetMatrixWeightSums(self, Points: NDArray[np.floating], WarpedPoints: NDArray[np.floating],
                             MaxChunkSize: int = 65536):
        NumCtrlPts = len(WarpedPoints)
        NumPts = Points.shape[0]

        # This calculation has an NumPoints X NumWarpedPoints memory footprint when there are a large number of points
        if NumPts <= MaxChunkSize:
            Distances = pairwise_cdist(Points, WarpedPoints)
            # pairwise_cdist can return NumPy even when inputs are CuPy; keep masks/ops on-device.
            if cp.get_array_module(Points) is cp:
                Distances = cp.asarray(Distances)

            # VectorBasisFunc = np.vectorize( self.BasisFunction)
            # FuncValues = VectorBasisFunc(Distances)

            # We have to check for zeros so we don't crash if the transformed point exactly matches our control point
            nonzero = Distances != 0
            FuncValues = cp.zeros(Distances.shape, dtype=Distances.dtype)

            if cp.all(nonzero):
                FuncValues = cp.multiply(cp.power(Distances, 2.0), cp.log(Distances))
            else:
                FuncValues[nonzero] = cp.multiply(cp.power(Distances[nonzero], 2.0), cp.log(Distances[nonzero]))

            del Distances

            MatrixWeightSumX = cp.sum(self.Weights[0:NumCtrlPts] * FuncValues, axis=1)
            MatrixWeightSumY = cp.sum(self.Weights[3 + NumCtrlPts:(NumCtrlPts * 2) + 3] * FuncValues, axis=1)

            del FuncValues

            return MatrixWeightSumX, MatrixWeightSumY
        else:
            # Cut the array into chunks and build the weights over a loop
            iStart = 0
            MatrixWeightSumX = cp.zeros(NumPts)
            MatrixWeightSumY = cp.zeros(NumPts)
            while iStart < NumPts:
                iEnd = iStart + MaxChunkSize
                if iEnd > Points.shape[0]:
                    iEnd = Points.shape[0]

                (MatrixWeightSumXChunk, MatrixWeightSumYChunk) = self._GetMatrixWeightSums(Points[iStart:iEnd, :],
                                                                                           WarpedPoints)

                # Failing these asserts means we are stomping earlier results
                assert (MatrixWeightSumX[iStart] == 0)
                assert (MatrixWeightSumY[iStart] == 0)

                MatrixWeightSumX[iStart:iEnd] = MatrixWeightSumXChunk
                MatrixWeightSumY[iStart:iEnd] = MatrixWeightSumYChunk

                iStart = iEnd

            return MatrixWeightSumX, MatrixWeightSumY

    def Transform(self, Points: NDArray[np.floating], **kwargs):
        # if self._rigid_transform is not None:
        #   return self._rigid_transform.Transform(Points)

        Points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(Points)

        # See the host mirror: the lazy weight solve decides whether the rigid shortcut
        # applies, so it has to run before UseRigidTransform is read.
        _ = self.Weights

        NumCtrlPts = len(self.TargetPoints)

        if self.UseRigidTransform:
            NumPts = Points.shape[0]
            MatrixWeightSumX = cp.zeros((1, NumPts))
            MatrixWeightSumY = cp.zeros((1, NumPts))
        else:
            (MatrixWeightSumX, MatrixWeightSumY) = self._GetMatrixWeightSums(Points, self.SourcePoints)

        # (UnchunkedMatrixWeightSumX, MatrixWeightSumY) = self._GetMatrixWeightSums(points, self.TargetPoints, self.SourcePoints, MaxChunkSize=32768000)
        # assert(MatrixWeightSumX == UnchunkedMatrixWeightSumX)

        Xa = Points[:, 1] * self.Weights[NumCtrlPts]
        Xb = Points[:, 0] * self.Weights[NumCtrlPts + 1]
        Xc = self.Weights[NumCtrlPts + 2]
        XBase = cp.vstack((MatrixWeightSumX, Xa, Xb))
        Xf = cp.sum(XBase, axis=0) + Xc

        del Xa
        del Xb
        del Xc
        del XBase
        del MatrixWeightSumX

        Ya = Points[:, 1] * self.Weights[NumCtrlPts + 3 + NumCtrlPts]
        Yb = Points[:, 0] * self.Weights[NumCtrlPts + NumCtrlPts + 3 + 1]
        Yc = self.Weights[NumCtrlPts + NumCtrlPts + 3 + 2]
        YBase = cp.vstack((MatrixWeightSumY, Ya, Yb))
        Yf = cp.sum(YBase, axis=0) + Yc

        del Ya
        del Yb
        del Yc
        del YBase
        del MatrixWeightSumY

        MatrixOutpoints = cp.vstack((Xf, Yf)).transpose()

        #        for iPoint in range(0, points.shape[0]):
        #            Point = points[iPoint]
        #            PFuncVals = FuncValues[iPoint]
        #
        #            # WeightSumX = np.sum(self.Weights[0:NumCtrlPts] * PFuncVals)
        #            # WeightSumY = np.sum(self.Weights[3 + NumCtrlPts:(NumCtrlPts * 2) + 3] * PFuncVals)
        #            WeightSumX = MatrixWeightSumX[iPoint]
        #            WeightSumY = MatrixWeightSumY[iPoint]
        #
        #            X = WeightSumX + (Point[1] * self.Weights[NumCtrlPts]) + (Point[0] * self.Weights[NumCtrlPts + 1]) + self.Weights[NumCtrlPts + 2]
        #            Y = WeightSumY + (Point[1] * self.Weights[NumCtrlPts + 3 + NumCtrlPts]) + (Point[0] * self.Weights[NumCtrlPts + NumCtrlPts + 3 + 1]) + self.Weights[NumCtrlPts + NumCtrlPts + 3 + 2]
        #
        #            assert(round(X, 1) == round(Xf[iPoint], 1))
        #            assert(round(Y, 1) == round(Yf[iPoint], 1))
        #
        #            assert(round(MatrixOutpoints[iPoint, 0], 1) == round(X, 1))
        #            assert(round(MatrixOutpoints[iPoint, 1], 1) == round(Y, 1))
        #
        #            OutPoints[iPoint, :] = [X, Y]

        return MatrixOutpoints

    def InverseTransform(self, Points: NDArray, **kwargs):
        raise NotImplementedError("RBF Transform does not support inverse transformations")

    @staticmethod
    def CreateSolutionMatricies(ControlPoints: NDArray):

        ControlPoints = cp.array(ControlPoints)
        NumPts = len(ControlPoints)

        ResultMatrixX = cp.zeros([NumPts + 3])
        ResultMatrixY = cp.zeros([NumPts + 3])

        ResultMatrixX[3:] = ControlPoints[:, 0]  # .reshape(NumPts,1)
        ResultMatrixY[3:] = ControlPoints[:, 1]  # .reshape(NumPts,1)

        return ResultMatrixX, ResultMatrixY

    @staticmethod
    def CreateBetaMatrix(points: NDArray, BasisFunction: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None):
        if BasisFunction is None:
            BasisFunction = OneWayRBFWithLinearCorrection_GPUComponent.DefaultBasisFunction
        return _tps_beta_matrix(points, BasisFunction, cp)

    @staticmethod
    def CalculateRBFWeights(WarpedPoints: NDArray, ControlPoints: NDArray,
                            BasisFunction: Callable[[NDArray[np.floating]], NDArray[np.floating]]) -> tuple[
        NDArray, bool]:
        '''
        For each axis this function fits a rigid transformation (with rotation) to the points and then assigns weights to the remaining errors in the fit.

        The weights matrix is broken down as follows for N control points
        Weights[0:N] = Fit of point deviation from the rigid transformation
        Weights[N:N+1] = Rotation component of transformation
        Weights[N+3] = Translation component of transformation.

        If Weights[0:N] ~= 0, then we can use a much faster and simpler rigid transformation with rotation to translate the data
        If additionally Weights[N:N+1] ~= 0, then we can simply translate the points as needed
        '''
        use_rigid_transform = False

        BetaMatrix = OneWayRBFWithLinearCorrection_GPUComponent.CreateBetaMatrix(WarpedPoints, BasisFunction)
        (SolutionMatrix_X, SolutionMatrix_Y) = OneWayRBFWithLinearCorrection_GPUComponent.CreateSolutionMatricies(
            ControlPoints)

        # thread_pool = nornir_pools.GetGlobalThreadPool()

        try:
            # Y_Task = thread_pool.add_task("WeightsY", scipy.linalg.solve, BetaMatrix, SolutionMatrix_Y,
            #                               overwrite_b=True,
            #                               check_finite=False)
            # WeightsX = np.linalg.solve(BetaMatrix, SolutionMatrix_X, overwrite_b=True, check_finite=False)
            # WeightsY = np.linalg.solve(BetaMatrix, SolutionMatrix_Y, overwrite_b=True, check_finite=False)
            WeightsX = cp.linalg.solve(BetaMatrix, SolutionMatrix_X)
            WeightsY = cp.linalg.solve(BetaMatrix, SolutionMatrix_Y)

            # WeightsY = Y_Task.wait_return()

            if cp.allclose(WeightsX[0:-3], 0) and cp.allclose(WeightsY[0:-3], 0):
                # prettyoutput.Log("RBF transform is approximately Rigid")
                use_rigid_transform = True

            # source_rotation_center, rotation_matrix, scale, translation, reflected = nornir_imageregistration.transforms.converters._kabsch_umeyama(ControlPoints, WarpedPoints)

            return cp.hstack([WeightsX, WeightsY]), use_rigid_transform
        except cp.linalg.LinAlgError as e:
            if _is_singular_matrix_error(e):
                wp_np = array_to_numpy_host(WarpedPoints)
                cc_np = array_to_numpy_host(ControlPoints)
                result = nornir_imageregistration.transforms.converters.EstimateRigidComponentsFromControlPoints(
                    cc_np, wp_np)
                rotation_matrix = nornir_imageregistration.transforms.RotationMatrix(np.radians(result.angle))

                WeightsY = cp.zeros(SolutionMatrix_Y.shape)
                WeightsY[-3] = rotation_matrix[0, 1]
                WeightsY[-2] = result.scale
                WeightsY[-1] = result.translation[0]

                WeightsX = cp.zeros(SolutionMatrix_X.shape)
                WeightsX[-3] = rotation_matrix[0, 0]
                WeightsX[-2] = result.scale
                WeightsX[-1] = result.translation[1]

                return cp.hstack([WeightsX, WeightsY]), True
            else:
                raise

    @property
    def LinearComponents(self):
        """The angle of rotation for the linear portion of the transform"""
        xp = cp.get_array_module(self.Weights)
        nPoints = self.points.shape[0]
        rotate_x_component = self.Weights[nPoints]
        scale_x_component = self.Weights[nPoints + 1]
        translate_x_component = self.Weights[nPoints + 2]

        axis_offset = nPoints + 3
        rotate_y_component = self.Weights[axis_offset + nPoints]
        scale_y_component = self.Weights[axis_offset + nPoints + 1]
        translate_y_component = self.Weights[axis_offset + nPoints + 2]

        angle = xp.arctan2(rotate_x_component, rotate_y_component)
        scale = [scale_y_component, scale_x_component]
        rotate = [rotate_y_component, rotate_x_component]
        translate = [translate_y_component, translate_x_component]
        source_rotation_center = xp.mean(self.points[:, 2:], axis=0)

        return source_rotation_center, angle, translate, scale

    # def AddPoint(self, pointpair: NDArray[np.floating]):
    #     self.points = np.vstack((self.points, pointpair))
    #     self.OnTransformChanged()
    #
    # def AddPoints(self, new_points: NDArray[np.floating]):
    #     self.points = np.vstack((self.points, new_points))
    #     self.OnTransformChanged()
    #
    # def UpdatePointPair(self, index: int, pointpair: NDArray[np.floating]):
    #     self.points[index, :] = pointpair
    #     self.OnTransformChanged()
    #
    # def UpdateTargetPoints(self, index: int | NDArray[np.integer], points: NDArray[np.floating]):
    #     self.points[index, 0:2] = points
    #     self.OnFixedPointChanged()
    #
    # def UpdateSourcePoints(self, index: int | NDArray[np.integer], points: NDArray[np.floating]):
    #     self.points[index, 2:4] = points
    #     self.OnWarpedPointChanged()
    #
    # def RemovePoint(self, index: int):
    #     del self.points[index, :]
    #     self.OnTransformChanged()

    def OnFixedPointChanged(self):
        super(OneWayRBFWithLinearCorrection_GPUComponent, self).OnFixedPointChanged()
        self._weights = None
        self._rigid_transform = None
        super(OneWayRBFWithLinearCorrection_GPUComponent, self).OnTransformChanged()

    def OnWarpedPointChanged(self):
        super(OneWayRBFWithLinearCorrection_GPUComponent, self).OnWarpedPointChanged()
        self._weights = None
        self._rigid_transform = None
        super(OneWayRBFWithLinearCorrection_GPUComponent, self).OnTransformChanged()

    def ClearDataStructures(self):
        super(OneWayRBFWithLinearCorrection_GPUComponent, self).ClearDataStructures()
        self._weights = None
        self._rigid_transform = None

#
# class RBFTransform(triangulation.Triangulation):
#    '''
#    classdocs
#    '''
#
#    def OnTransformChanged(self):
#
#        ForwardTask = transformbase.TransformBase.ThreadPool.add_task("Solve forward RBF transform", OneWayRBFWithLinearCorrection, self.SourcePoints, self.TargetPoints)
#        ReverseTask = transformbase.TransformBase.ThreadPool.add_task("Solve reverse RBF transform", OneWayRBFWithLinearCorrection, self.TargetPoints, self.SourcePoints)
#
#        self.ForwardRBFInstance = ForwardTask.wait_return()
#        self.ReverseRBFInstance = ReverseTask.wait_return()
#
#        super(RBFTransform, self).OnTransformChanged()
#
#        # self.ForwardRBFInstance = OneWayRBFWithLinearCorrection(self.SourcePoints, self.TargetPoints)
#        # self.ReverseRBFInstance = OneWayRBFWithLinearCorrection(self.TargetPoints, self.SourcePoints)
#
#
#
# #        self.ForwardRBFInstanceX = scipy.interpolate.Rbf(self.SourcePoints[:, 0], self.SourcePoints[:, 1], self.TargetPoints[:, 0], function='gaussian')
# #        self.ForwardRBFInstanceY = scipy.interpolate.Rbf(self.SourcePoints[:, 0], self.SourcePoints[:, 1], self.TargetPoints[:, 1], function='gaussian')
# #        self.ReverseRBFInstanceX = scipy.interpolate.Rbf(self.TargetPoints[:, 0], self.TargetPoints[:, 1], self.SourcePoints[:, 0], function='gaussian')
# #        self.ReverseRBFInstanceY = scipy.interpolate.Rbf(self.TargetPoints[:, 0], self.TargetPoints[:, 1], self.SourcePoints[:, 1], function='gaussian')
#
#    @classmethod
#    def InvalidIndicies(self, points):
#        '''Removes rows with a NAN value and returns a list of indicies'''
#        invalidIndicies = []
#        for i in range(len(points) - 1, -1, -1):
#            Row = points[i]
#            for iCol in range(0, len(Row)):
#                if(math.isnan(Row[iCol])):
#                    invalidIndicies.append(i)
#                    points = np.delete(points, i, 0)
#                    break
#
#        invalidIndicies.reverse()
#        return (points, invalidIndicies)
#
#
#    def Transform(self, points):
#        if len(points) == 0:
#            return []
#
#        points = np.array(points)
#
#        TransformedPoints = super(RBFTransform, self).Transform(points)
#        (GoodPoints, InvalidIndicies) = RBFTransform.InvalidIndicies(TransformedPoints)
#
#        if(len(InvalidIndicies) == 0):
#            return TransformedPoints
#        else:
#            if len(points) > 1:
#                # print InvalidIndicies
#                BadPoints = points[InvalidIndicies]
#            else:
#                BadPoints = points
#
#        BadPoints = np.array(BadPoints)
#        TargetPoints = self.ForwardRBFInstance.Transform(BadPoints)
#
#        TransformedPoints[InvalidIndicies] = TargetPoints
#        return TransformedPoints
#
#    def InverseTransform(self, points):
#        if len(points) == 0:
#            return []
#
#        points = np.array(points)
#
#        TransformedPoints = super(RBFTransform, self).InverseTransform(points)
#        (GoodPoints, InvalidIndicies) = RBFTransform.InvalidIndicies(TransformedPoints)
#
#        if(len(InvalidIndicies) == 0):
#            return TransformedPoints
#        else:
#            if len(points) > 1:
#                BadPoints = points[InvalidIndicies]
#            else:
#                BadPoints = points
#
#        BadPoints = np.array(BadPoints)
#
#        TargetPoints = self.ReverseRBFInstance.Transform(BadPoints)
#
#        TransformedPoints[InvalidIndicies] = TargetPoints
#        return TransformedPoints
#
#    def __init__(self, pointpairs):
#        '''
#        Constructor
#        '''
#        super(RBFTransform, self).__init__(pointpairs)

#
# if __name__ == '__main__':
#    p = np.array([[0, 0, 0, 0],
#                  [0, 10, 0, -10],
#                  [10, 0, -10, 0],
#                  [10, 10, -10, -10]])
#
#    (Fixed, Moving) = np.hsplit(p, 2)
#    T = OneWayRBFWithLinearCorrection(Fixed, Moving)
#
#    warpedPoints = [[0, 0], [-5, -5]]
#    fp = T.ViewTransform(warpedPoints)
#    print("__Transform " + str(warpedPoints) + " to " + str(fp))
#    wp = T.InverseTransform(fp)
#
#
#    print "Fixed Verts"
#    print T.FixedTriangles
#    print "\nWarped Verts"
#    print T.WarpedTriangles
#
#    T.AddPoint([5, 5, -5, -5])
#    print "\nPoint added"
#    print "Fixed Verts"
#    print T.FixedTriangles
#    print "\nWarped Verts"
#    print T.WarpedTriangles
#
#    T.AddPoint([5, 5, 5, 5])
#    print "\nDuplicate Point added"
#    print "Fixed Verts"
#    print T.FixedTriangles
#    print "\nWarped Verts"
#    print T.WarpedTriangles
#
#    warpedPoint = [[-5, -5]]
#    fp = T.ViewTransform(warpedPoint)
#    print("__Transform " + str(warpedPoint) + " to " + str(fp))
#    wp = T.InverseTransform(fp)
#
#    T.UpdatePoint(3, [10, 15, -10, -15])
#    print "\nPoint updated"
#    print "Fixed Verts"
#    print T.FixedTriangles
#    print "\nWarped Verts"
#    print T.WarpedTriangles
#
#    warpedPoint = [[-9, -14]]
#    fp = T.ViewTransform(warpedPoint)
#    print("__Transform " + str(warpedPoint) + " to " + str(fp))
#    wp = T.InverseTransform(fp)
#
#    T.RemovePoint(1)
#    print "\nPoint removed"
#    print "Fixed Verts"
#    print T.FixedTriangles
#    print "\nWarped Verts"
#    print T.WarpedTriangles
#
#
#
#
#    print "\nFixedPointsInRect"
#    print T.GetFixedPointsRect([-1, -1, 14, 4])
