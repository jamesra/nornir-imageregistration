'''
Created on Oct 18, 2012

@author: Jamesan
'''

import math

from nornir_imageregistration.transforms import triangulation
import numpy
import scipy.interpolate

import nornir_pools
#import nornir_shared
import scipy.linalg 
import scipy.spatial

from . import utils


class RBFWithLinearCorrection(triangulation.Triangulation):

    def __getstate__(self):
        odict = super(RBFWithLinearCorrection, self).__getstate__()

        odict['_Weights'] = self._Weights

        return odict

    @property
    def Weights(self):
        return self._Weights

    @classmethod
    def DefaultBasisFunction(cls, distance):
        if distance == 0:
            return 0

        return distance * distance * math.log(distance)

    def __init__(self, WarpedPoints, FixedPoints, BasisFunction=None):

        points = numpy.hstack((FixedPoints, WarpedPoints))
        super(RBFWithLinearCorrection, self).__init__(points)

        # self.ControlPoints = TargetPoints
        # self.SourcePoints = SourcePoints
        self.BasisFunction = BasisFunction
        if(self.BasisFunction is None):
            self.BasisFunction = RBFWithLinearCorrection.DefaultBasisFunction

        self._Weights = self.CalculateRBFWeights(WarpedPoints, FixedPoints, self.BasisFunction)

    def OnPointsAddedToTransform(self, new_points):
        '''Update our data structures to account for added control points'''
        super(RBFWithLinearCorrection, self).OnPointsAddedToTransform(new_points)
        self._Weights = self.CalculateRBFWeights(self.SourcePoints, self.TargetPoints, self.BasisFunction)

    def _GetMatrixWeightSums(self, Points, WarpedPoints, MaxChunkSize=65536):
        NumCtrlPts = len(WarpedPoints)
        NumPts = Points.shape[0]

        # This calculation has an NumPoints X NumWarpedPoints memory footprint when there are a large number of points
        if NumPts <= MaxChunkSize:
            Distances = scipy.spatial.distance.cdist(Points, WarpedPoints)

            # VectorBasisFunc = numpy.vectorize( self.BasisFunction)
            # FuncValues = VectorBasisFunc(Distances)

            #We have to check for zeros so we don't crash if the transformed point exactly matches our control point
            nonzero = Distances != 0
            FuncValues = numpy.zeros(Distances.shape)
            
            if numpy.all(nonzero):
                FuncValues = numpy.multiply(numpy.power(Distances, 2.0), numpy.log(Distances))
            else: 
                FuncValues[nonzero] = numpy.multiply(numpy.power(Distances[nonzero], 2.0), numpy.log(Distances[nonzero]))

            del Distances

            MatrixWeightSumX = numpy.sum(self.Weights[0:NumCtrlPts] * FuncValues, axis=1)
            MatrixWeightSumY = numpy.sum(self.Weights[3 + NumCtrlPts:(NumCtrlPts * 2) + 3] * FuncValues, axis=1)

            del FuncValues

            return (MatrixWeightSumX, MatrixWeightSumY)
        else:
            # Cut the array into chunks and build the weights over a loop
            iStart = 0
            MatrixWeightSumX = numpy.zeros((NumPts))
            MatrixWeightSumY = numpy.zeros((NumPts))
            while iStart < NumPts:
                iEnd = iStart + MaxChunkSize
                if iEnd > Points.shape[0]:
                    iEnd = Points.shape[0]

                (MatrixWeightSumXChunk, MatrixWeightSumYChunk) = self._GetMatrixWeightSums(Points[iStart:iEnd, :],
                                                                                      WarpedPoints)

                # Failing these asserts means we are stomping earlier results
                assert(MatrixWeightSumX[iStart] == 0)
                assert(MatrixWeightSumY[iStart] == 0)

                MatrixWeightSumX[iStart:iEnd] = MatrixWeightSumXChunk
                MatrixWeightSumY[iStart:iEnd] = MatrixWeightSumYChunk

                iStart = iEnd

            return (MatrixWeightSumX, MatrixWeightSumY)

    def Transform(self, Points, **kwargs):

        Points = utils.EnsurePointsAre2DNumpyArray(Points)

        NumCtrlPts = len(self.TargetPoints)

        (MatrixWeightSumX, MatrixWeightSumY) = self._GetMatrixWeightSums(Points, self.SourcePoints)
        # (UnchunkedMatrixWeightSumX, MatrixWeightSumY) = self._GetMatrixWeightSums(Points, self.TargetPoints, self.SourcePoints, MaxChunkSize=32768000)
        # assert(MatrixWeightSumX == UnchunkedMatrixWeightSumX)

        Xa = Points[:, 1] * self.Weights[NumCtrlPts]
        Xb = Points[:, 0] * self.Weights[NumCtrlPts + 1]
        Xc = self.Weights[NumCtrlPts + 2]
        XBase = numpy.vstack((MatrixWeightSumX, Xa, Xb))
        Xf = numpy.sum(XBase, axis=0) + Xc

        del Xa
        del Xb
        del Xc
        del XBase
        del MatrixWeightSumX

        Ya = Points[:, 1] * self.Weights[NumCtrlPts + 3 + NumCtrlPts]
        Yb = Points[:, 0] * self.Weights[NumCtrlPts + NumCtrlPts + 3 + 1]
        Yc = self.Weights[NumCtrlPts + NumCtrlPts + 3 + 2]
        YBase = numpy.vstack((MatrixWeightSumY, Ya, Yb))
        Yf = numpy.sum(YBase, axis=0) + Yc

        del Ya
        del Yb
        del Yc
        del YBase
        del MatrixWeightSumY

        MatrixOutpoints = numpy.vstack((Xf, Yf)).transpose()

#        for iPoint in range(0, Points.shape[0]):
#            Point = Points[iPoint]
#            PFuncVals = FuncValues[iPoint]
#
#            # WeightSumX = numpy.sum(self.Weights[0:NumCtrlPts] * PFuncVals)
#            # WeightSumY = numpy.sum(self.Weights[3 + NumCtrlPts:(NumCtrlPts * 2) + 3] * PFuncVals)
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
    
    def InverseTransform(self, Points, **kwargs):
        raise NotImplemented("RBF Transform does not support inverse transformations")

    @classmethod
    def CreateSolutionMatricies(cls, ControlPoints):

        NumPts = len(ControlPoints)

        ResultMatrixX = numpy.zeros([NumPts + 3])
        ResultMatrixY = numpy.zeros([NumPts + 3])

        ResultMatrixX[3:] = ControlPoints[:, 0]  # .reshape(NumPts,1)
        ResultMatrixY[3:] = ControlPoints[:, 1]  # .reshape(NumPts,1)

        return (ResultMatrixX, ResultMatrixY)

    @classmethod
    def CreateBetaMatrix(cls, Points, BasisFunction=None):
        if BasisFunction is None:
            BasisFunction = RBFWithLinearCorrection.DefaultBasisFunction

        NumPts = len(Points)
        BetaMatrix = numpy.zeros([NumPts + 3, NumPts + 3], dtype=numpy.float32)

        for iRow in range(3, NumPts + 3):
            iPointA = iRow - 3

            p = Points[list(range((iPointA + 1), NumPts))]
            dList = scipy.spatial.distance.cdist([Points[iPointA]], p)


            valueList = numpy.power(dList, 2)
            assert(numpy.all(dList > 0), "Cannot have duplicate points in transform")
            valueList = numpy.multiply(valueList, numpy.log(dList))
            valueList = valueList.ravel()

            BetaMatrix[iRow, list(range(iPointA + 1, NumPts))] = valueList
            BetaMatrix[list(range(iPointA + 1 + 3, NumPts + 3)), iRow - 3] = valueList

#            for iCol in range(iPointA+1, NumPts):
#                iPointB = iCol
#                dist = scipy.spatial.distance.euclidean(Points[iPointA], Points[iPointB])
#                value = BasisFunction(dist)
#                BetaMatrix[iRow, iCol] = value
#                BetaMatrix[iCol + 3, iRow - 3] = value

            BetaMatrix[iRow, NumPts] = Points[iPointA][1]
            BetaMatrix[iRow, NumPts + 1] = Points[iPointA][0]
            BetaMatrix[iRow, NumPts + 2] = 1

        for iCol in range(0, NumPts):
            BetaMatrix[0, iCol] = Points[iCol][0]
            BetaMatrix[1, iCol] = Points[iCol][1]
            BetaMatrix[2, iCol] = 1

        return BetaMatrix

    def CalculateRBFWeights(self, WarpedPoints, ControlPoints, BasisFunction=None):

        BetaMatrix = RBFWithLinearCorrection.CreateBetaMatrix(WarpedPoints, BasisFunction)
        (SolutionMatrix_X, SolutionMatrix_Y) = RBFWithLinearCorrection.CreateSolutionMatricies(ControlPoints)

        thread_pool = nornir_pools.GetGlobalThreadPool()

        Y_Task = thread_pool.add_task("WeightsY", scipy.linalg.solve, BetaMatrix, SolutionMatrix_Y)
        WeightsX = scipy.linalg .solve(BetaMatrix, SolutionMatrix_X)
        WeightsY = Y_Task.wait_return()

        return numpy.hstack([WeightsX, WeightsY])

#
# class RBFTransform(triangulation.Triangulation):
#    '''
#    classdocs
#    '''
#
#    def OnTransformChanged(self):
#
#        ForwardTask = transformbase.TransformBase.ThreadPool.add_task("Solve forward RBF transform", RBFWithLinearCorrection, self.SourcePoints, self.TargetPoints)
#        ReverseTask = transformbase.TransformBase.ThreadPool.add_task("Solve reverse RBF transform", RBFWithLinearCorrection, self.TargetPoints, self.SourcePoints)
#
#        self.ForwardRBFInstance = ForwardTask.wait_return()
#        self.ReverseRBFInstance = ReverseTask.wait_return()
#
#        super(RBFTransform, self).OnTransformChanged()
#
#        # self.ForwardRBFInstance = RBFWithLinearCorrection(self.SourcePoints, self.TargetPoints)
#        # self.ReverseRBFInstance = RBFWithLinearCorrection(self.TargetPoints, self.SourcePoints)
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
#                    points = numpy.delete(points, i, 0)
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
#        points = numpy.array(points)
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
#        BadPoints = numpy.array(BadPoints)
#        TargetPoints = self.ForwardRBFInstance.Transform(BadPoints)
#
#        TransformedPoints[InvalidIndicies] = TargetPoints
#        return TransformedPoints
#
#    def InverseTransform(self, points):
#        if len(points) == 0:
#            return []
#
#        points = numpy.array(points)
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
#        BadPoints = numpy.array(BadPoints)
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
#    p = numpy.array([[0, 0, 0, 0],
#                  [0, 10, 0, -10],
#                  [10, 0, -10, 0],
#                  [10, 10, -10, -10]])
#
#    (Fixed, Moving) = numpy.hsplit(p, 2)
#    T = RBFWithLinearCorrection(Fixed, Moving)
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


