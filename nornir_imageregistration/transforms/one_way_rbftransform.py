'''
Created on Oct 18, 2012

@author: Jamesan
'''

from typing import Callable

import numpy as np
from numpy.typing import NDArray
import scipy.interpolate
import scipy.linalg
import scipy.spatial

import nornir_imageregistration
import nornir_pools
from nornir_imageregistration.transforms.transform_type import TransformType
from .triangulation import Triangulation


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
        odict['_weights'] = self._weights

        if '_rigid_transform' not in odict:
            odict['_rigid_transform'] = self._rigid_transform.ToITKString() if self.UseRigidTransform else None

        return odict

    @property
    def Weights(self):
        if self._weights is None:
            self._weights, use_rigid_transform = self.CalculateRBFWeights(self.SourcePoints, self.TargetPoints,
                                                                          self.BasisFunction)
            if use_rigid_transform:
                self._rigid_transform = nornir_imageregistration.transforms.converters.ConvertTransformToRigidTransform(
                    self)
            else:
                self._rigid_transform = None

        return self._weights

    @property
    def UseRigidTransform(self) -> bool:
        return self._rigid_transform is not None

    @staticmethod
    def DefaultBasisFunction(distance: NDArray[float]) -> NDArray[float]:
        return np.multiply(np.power(distance, 2), np.log(distance))

    def __init__(self, WarpedPoints: NDArray[float], FixedPoints: NDArray[float],
                 BasisFunction: Callable[[NDArray[float]], NDArray[float]] | None = None):

        points = np.hstack((FixedPoints, WarpedPoints))
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

    def _GetMatrixWeightSums(self, Points: NDArray[float], WarpedPoints: NDArray[float], MaxChunkSize: int = 65536):
        NumCtrlPts = len(WarpedPoints)
        NumPts = Points.shape[0]

        # This calculation has an NumPoints X NumWarpedPoints memory footprint when there are a large number of points
        if NumPts <= MaxChunkSize:
            Distances = scipy.spatial.distance.cdist(Points, WarpedPoints)

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

    def Transform(self, Points: NDArray[float], **kwargs):
        # if self._rigid_transform is not None:
        #   return self._rigid_transform.Transform(Points)

        Points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(Points)

        NumCtrlPts = len(self.TargetPoints)

        if self.UseRigidTransform:
            NumPts = Points.shape[0]
            MatrixWeightSumX = np.zeros((1, NumPts))
            MatrixWeightSumY = np.zeros((1, NumPts))
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
        raise NotImplemented("RBF Transform does not support inverse transformations")

    @staticmethod
    def CreateSolutionMatricies(ControlPoints: NDArray):

        NumPts = len(ControlPoints)

        ResultMatrixX = np.zeros([NumPts + 3])
        ResultMatrixY = np.zeros([NumPts + 3])

        ResultMatrixX[3:] = ControlPoints[:, 0]  # .reshape(NumPts,1)
        ResultMatrixY[3:] = ControlPoints[:, 1]  # .reshape(NumPts,1)

        return ResultMatrixX, ResultMatrixY

    @staticmethod
    def CreateBetaMatrix(points: NDArray, BasisFunction: Callable[[NDArray[float]], NDArray[float]] = None):
        # if BasisFunction is None:
        #    BasisFunction = OneWayRBFWithLinearCorrection.DefaultBasisFunction

        NumPts = len(points)
        BetaMatrix = np.zeros([NumPts + 3, NumPts + 3], dtype=np.float32)

        for iRow in range(3, NumPts + 3):
            iPointA = iRow - 3

            p = points[list(range((iPointA + 1), NumPts))]
            dList = scipy.spatial.distance.cdist([points[iPointA]], p)

            dList = dList.ravel()
            if dList.shape[0] >= 1:
                if np.min(dList) <= 0:
                    raise ValueError("Cannot have duplicate points in transform")

            valueList = BasisFunction(dList)
            # valueList = np.power(dList, 2)
            # valueList = np.multiply(valueList, np.log(dList))
            # valueList = valueList.ravel()

            BetaMatrix[iRow, list(range(iPointA + 1, NumPts))] = valueList
            BetaMatrix[list(range(iPointA + 1 + 3, NumPts + 3)), iRow - 3] = valueList

            #            for iCol in range(iPointA+1, NumPts):
            #                iPointB = iCol
            #                dist = scipy.spatial.distance.euclidean(points[iPointA], points[iPointB])
            #                value = BasisFunction(dist)
            #                BetaMatrix[iRow, iCol] = value
            #                BetaMatrix[iCol + 3, iRow - 3] = value

            BetaMatrix[iRow, NumPts] = points[iPointA][1]
            BetaMatrix[iRow, NumPts + 1] = points[iPointA][0]
            BetaMatrix[iRow, NumPts + 2] = 1

        for iCol in range(0, NumPts):
            BetaMatrix[0, iCol] = points[iCol][0]
            BetaMatrix[1, iCol] = points[iCol][1]
            BetaMatrix[2, iCol] = 1

        return BetaMatrix

    @staticmethod
    def CalculateRBFWeights(WarpedPoints: NDArray, ControlPoints: NDArray,
                            BasisFunction: Callable[[NDArray[float]], NDArray[float]]) -> tuple[NDArray, bool]:
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

        thread_pool = nornir_pools.GetGlobalThreadPool()

        try:
            Y_Task = thread_pool.add_task("WeightsY", scipy.linalg.solve, BetaMatrix, SolutionMatrix_Y,
                                          overwrite_b=True,
                                          check_finite=False)
            WeightsX = scipy.linalg.solve(BetaMatrix, SolutionMatrix_X, overwrite_b=True, check_finite=False)
            WeightsY = Y_Task.wait_return()

            if np.allclose(WeightsX[0:-3], 0) and np.allclose(WeightsY[0:-3], 0):
                # prettyoutput.Log("RBF transform is approximately Rigid")
                use_rigid_transform = True

            # source_rotation_center, rotation_matrix, scale, translation, reflected = nornir_imageregistration.transforms.converters._kabsch_umeyama(ControlPoints, WarpedPoints)

            return np.hstack([WeightsX, WeightsY]), use_rigid_transform
        except np.linalg.LinAlgError as e:
            if e.args[0] == 'Matrix is singular.':
                # This is a distraction for now, but I should be able to fill in these weights correctly
                source_rotation_center, rotation_matrix, scale, translation, reflected = nornir_imageregistration.transforms.converters._kabsch_umeyama(
                    ControlPoints, WarpedPoints)

                WeightsY = np.zeros(SolutionMatrix_Y.shape)
                WeightsY[-3] = rotation_matrix[1, 0]
                WeightsY[-2] = scale
                WeightsY[-1] = translation[0]

                WeightsX = np.zeros(SolutionMatrix_X.shape)
                WeightsX[-3] = rotation_matrix[0, 0]
                WeightsX[-2] = scale
                WeightsX[-1] = translation[1]

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

    # def AddPoint(self, pointpair: NDArray[float]):
    #     self.points = np.vstack((self.points, pointpair))
    #     self.OnTransformChanged()
    #
    # def AddPoints(self, new_points: NDArray[float]):
    #     self.points = np.vstack((self.points, new_points))
    #     self.OnTransformChanged()
    #
    # def UpdatePointPair(self, index: int, pointpair: NDArray[float]):
    #     self.points[index, :] = pointpair
    #     self.OnTransformChanged()
    #
    # def UpdateTargetPoints(self, index: int | NDArray[int], points: NDArray[float]):
    #     self.points[index, 0:2] = points
    #     self.OnFixedPointChanged()
    #
    # def UpdateSourcePoints(self, index: int | NDArray[int], points: NDArray[float]):
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
