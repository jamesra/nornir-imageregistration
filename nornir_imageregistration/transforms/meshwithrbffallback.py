'''
Created on Oct 18, 2012

@author: Jamesan
'''

import math

import numpy
import scipy.interpolate

from nornir_imageregistration.transforms import base, triangulation
from nornir_imageregistration.transforms.rbftransform import \
    RBFWithLinearCorrection
import nornir_pools as pools
import scipy.linalg as linalg
import scipy.spatial as spatial
import utils


class MeshWithRBFFallback(triangulation.Triangulation):
    '''
    classdocs
    '''

    @property
    def ReverseRBFInstance(self):
        if self._ReverseRBFInstance is None:
            self._ReverseRBFInstance = RBFWithLinearCorrection(self.FixedPoints, self.WarpedPoints)

        return self._ReverseRBFInstance

    @property
    def ForwardRBFInstance(self):
        if self._ForwardRBFInstance is None:
            self._ForwardRBFInstance = RBFWithLinearCorrection(self.WarpedPoints, self.FixedPoints)

        return self._ForwardRBFInstance

    def UpdateDataStructures(self):

        Pool = pools.GetGlobalThreadPool()
        ForwardTask = Pool.add_task("Solve forward RBF transform", RBFWithLinearCorrection, self.WarpedPoints, self.FixedPoints)
        ReverseTask = Pool.add_task("Solve reverse RBF transform", RBFWithLinearCorrection, self.FixedPoints, self.WarpedPoints)

        super(MeshWithRBFFallback, self).UpdateDataStructures()

        self._ForwardRBFInstance = ForwardTask.wait_return()
        self._ReverseRBFInstance = ReverseTask.wait_return()



    def ClearDataStructures(self):
        '''Something about the transform has changed, for example the points. 
           Clear out our data structures so we do not use bad data'''

        super(MeshWithRBFFallback, self).ClearDataStructures()

        self._ForwardRBFInstance = None
        self._ReverseRBFInstance = None


    def Transform(self, points, **kwargs):
        if len(points) == 0:
            return [];

        points = numpy.asarray(points, dtype=numpy.float32);

        TransformedPoints = super(MeshWithRBFFallback, self).Transform(points)
        (GoodPoints, InvalidIndicies) = utils.InvalidIndicies(TransformedPoints)

        if(len(InvalidIndicies) == 0):
            return TransformedPoints;
        else:
            if len(points) > 1:
                # print InvalidIndicies;
                BadPoints = points[InvalidIndicies];
            else:
                BadPoints = points;

        BadPoints = numpy.asarray(BadPoints, dtype=numpy.float32);
        FixedPoints = self.ForwardRBFInstance.Transform(BadPoints);

        TransformedPoints[InvalidIndicies] = FixedPoints;
        return TransformedPoints;

    def InverseTransform(self, points, **kwargs):
        if len(points) == 0:
            return [];

        points = numpy.asarray(points, dtype=numpy.float32);

        TransformedPoints = super(MeshWithRBFFallback, self).InverseTransform(points);
        (GoodPoints, InvalidIndicies) = utils.InvalidIndicies(TransformedPoints);

        if(len(InvalidIndicies) == 0):
            return TransformedPoints;
        else:
            if len(points) > 1:
                BadPoints = points[InvalidIndicies];
            else:
                BadPoints = points;

        BadPoints = numpy.asarray(BadPoints, dtype=numpy.float32);

        FixedPoints = self.ReverseRBFInstance.Transform(BadPoints);

        TransformedPoints[InvalidIndicies] = FixedPoints;
        return TransformedPoints;

    def __init__(self, pointpairs):
        '''
        Constructor
        '''
        super(MeshWithRBFFallback, self).__init__(pointpairs);

        self._ReverseRBFInstance = None
        self._ForwardRBFInstance = None


if __name__ == '__main__':
    p = numpy.array([[0, 0, 0, 0],
                  [0, 10, 0, -10],
                  [10, 0, -10, 0],
                  [10, 10, -10, -10]])

    (Fixed, Moving) = numpy.hsplit(p, 2);
    T = RBFWithLinearCorrection(Fixed, Moving);

    warpedPoints = [[0, 0], [-5, -5]];
    fp = T.ViewTransform(warpedPoints);
    print("__Transform " + str(warpedPoints) + " to " + str(fp))
    wp = T.InverseTransform(fp);


    print "Fixed Verts"
    print T.FixedTriangles;
    print "\nWarped Verts"
    print T.WarpedTriangles;

    T.AddPoint([5, 5, -5, -5]);
    print "\nPoint added"
    print "Fixed Verts"
    print T.FixedTriangles;
    print "\nWarped Verts"
    print T.WarpedTriangles;

    T.AddPoint([5, 5, 5, 5]);
    print "\nDuplicate Point added"
    print "Fixed Verts"
    print T.FixedTriangles;
    print "\nWarped Verts"
    print T.WarpedTriangles;

    warpedPoint = [[-5, -5]];
    fp = T.ViewTransform(warpedPoint);
    print("__Transform " + str(warpedPoint) + " to " + str(fp))
    wp = T.InverseTransform(fp);

    T.UpdatePoint(3, [10, 15, -10, -15]);
    print "\nPoint updated"
    print "Fixed Verts"
    print T.FixedTriangles;
    print "\nWarped Verts"
    print T.WarpedTriangles;

    warpedPoint = [[-9, -14]];
    fp = T.ViewTransform(warpedPoint);
    print("__Transform " + str(warpedPoint) + " to " + str(fp))
    wp = T.InverseTransform(fp);

    T.RemovePoint(1);
    print "\nPoint removed"
    print "Fixed Verts"
    print T.FixedTriangles;
    print "\nWarped Verts"
    print T.WarpedTriangles;




    print "\nFixedPointsInRect"
    print T.GetFixedPointsRect([-1, -1, 14, 4]);


