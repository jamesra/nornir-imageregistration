'''
Created on Oct 18, 2012

@author: Jamesan
'''

import math

from nornir_imageregistration.spatial import iPoint
from nornir_imageregistration.transforms import triangulation
import numpy
import scipy.interpolate

import scipy.linalg as linalg
import scipy.spatial as spatial

from . import utils


class ScipyRbf(triangulation.Triangulation):

    def __getstate__(self):
        odict = super(ScipyRbf, self).__getstate__()

        odict['_ForwardRbfiX'] = self._ForwardRbfiX
        odict['_ForwardRbfiY'] = self._ForwardRbfiY
        odict['_ReverseRbfiX'] = self._ReverseRbfiX
        odict['_ReverseRbfiY'] = self._ReverseRbfiY

        return odict 
    
    @property
    def ForwardRbfiX(self):
        if self._ForwardRbfiX is None:
            self._ForwardRbfiX = scipy.interpolate.Rbf(self.TargetPoints[:, iPoint.Y], self.TargetPoints[:, iPoint.X], self.SourcePoints[:, iPoint.X])
             
        return self._ForwardRbfiX
    
    @property
    def ForwardRbfiY(self):
        if self._ForwardRbfiY is None:
            self._ForwardRbfiY = scipy.interpolate.Rbf(self.TargetPoints[:, iPoint.Y], self.TargetPoints[:, iPoint.X], self.SourcePoints[:, iPoint.Y])
             
        return self._ForwardRbfiY
    
    @property
    def ReverseRbfiX(self):
        if self._ReverseRbfiX is None:
            self._ReverseRbfiX = scipy.interpolate.Rbf(self.SourcePoints[:, iPoint.Y], self.SourcePoints[:, iPoint.X], self.TargetPoints[:, iPoint.X])

        return self._ReverseRbfiX 

    @property
    def ReverseRbfiY(self):
        if self._ReverseRbfiY is None:
            self._ReverseRbfiY = scipy.interpolate.Rbf(self.SourcePoints[:, iPoint.Y], self.SourcePoints[:, iPoint.X], self.TargetPoints[:, iPoint.Y])

        return self._ReverseRbfiY  

    def __init__(self, WarpedPoints, FixedPoints, BasisFunction=None):

        points = numpy.hstack((FixedPoints, WarpedPoints))
        super(ScipyRbf, self).__init__(points)

        self._ForwardRbfiX = None
        self._ForwardRbfiY = None
        self._ReverseRbfiX = None
        self._ReverseRbfiY = None


    def Transform(self, Points, MaxChunkSize=65536):

        Points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(Points)

        NumPts = Points.shape[0]

        # This calculation has an NumPoints X NumWarpedPoints memory footprint when there are a large number of points
        if NumPts <= MaxChunkSize:
            transformedX = self.ForwardRbfiX(Points[:, iPoint.Y], Points[:, iPoint.X])
            transformedY = self.ForwardRbfiY(Points[:, iPoint.Y], Points[:, iPoint.X]) 

            return numpy.vstack((transformedY, transformedX)).transpose()
        else:
            iStart = 0
            transformedData = numpy.zeros((NumPts, 2))

            while iStart < NumPts:
                iEnd = iStart + MaxChunkSize
                if iEnd > Points.shape[0]:
                    iEnd = Points.shape[0]

                transformedChunk = self.Transform(Points[iStart:iEnd, :])

                # Failing these asserts means we are stomping earlier results
                assert(transformedData[iStart, 0] == 0)
                assert(transformedData[iStart, 1] == 0)

                transformedData[iStart:iEnd, :] = transformedChunk

                del transformedChunk

                iStart = iEnd

            return transformedData

    def InverseTransform(self, Points):

        Points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(Points)

        transformedX = self.ReverseRbfiX(Points[:, iPoint.Y], Points[:, iPoint.X])
        transformedY = self.ReverseRbfiY(Points[:, iPoint.Y], Points[:, iPoint.X]) 

        return numpy.vstack((transformedY, transformedX)).transpose()


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


