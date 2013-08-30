'''
Created on Oct 18, 2012

@author: Jamesan
'''

import numpy as np
from base import *
import math

from scipy.spatial import  *
from scipy.interpolate import griddata
import operator
import logging
import copy
from nornir_imageregistration.transforms.utils import InvalidIndicies
import utils

class Triangulation(Base):
    '''
    Triangulation transform has an nx4 array of points, with rows organized as
    [controlx controly warpedx warpedy]
    '''

    def __getstate__(self):
        odict = {};

        odict['_points'] = self._points;

        return odict;

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary);
        self.OnChangeEventListeners = [];
        self.UpdateDataStructures();

    @classmethod
    def RemoveDuplicates(cls, points):
        '''Returns tuple of the array sorted on fixed x,y without duplicates'''

        (points, InvalidIndicies) = utils.InvalidIndicies(points)

        DuplicateRemoved = False;
        points = np.around(points, 3);
        sortedpoints = sorted(points, key=operator.itemgetter(0, 1))
        for i in range(len(sortedpoints) - 1, 0, -1):
            lastP = sortedpoints[i - 1];
            testP = sortedpoints[i];

            if lastP[0] == testP[0]:
                if lastP[1] == testP[1]:
                    DuplicateRemoved = True;
                    sortedpoints = np.delete(sortedpoints, i, 0);
                    i = i + 1;

        return np.array(sortedpoints)

    @property
    def WarpedKDTree(self):
        if self._WarpedKDTree is None:
            self._WarpedKDTree = KDTree(self.WarpedPoints);

        return self._WarpedKDTree

    @property
    def FixedKDTree(self):
        if self._FixedKDTree is None:
            self._FixedKDTree = KDTree(self.FixedPoints);

        return self._FixedKDTree

    @property
    def fixedtri(self):
        if self._fixedtri is None:
            self._fixedtri = Delaunay(self.FixedPoints)

        return self._fixedtri

    @property
    def warpedtri(self):
        if self._warpedtri is None:
            self._warpedtri = Delaunay(self.WarpedPoints)

        return self._warpedtri

    def AddTransform(self, mappedTransform):
        '''Take the control points of the mapped transform and map them through our transform so the control points are in our controlpoint space'''
        mappedControlPoints = mappedTransform.FixedPoints;
        txMappedControlPoints = self.Transform(mappedControlPoints);

        pointPairs = np.hstack((txMappedControlPoints, mappedTransform.WarpedPoints));

        newTransform = copy.deepcopy(mappedTransform)
        newTransform.points = pointPairs;

        return newTransform;

    def Transform(self, points, **kwargs):
        transPoints = None;

        method = kwargs.get('method', 'linear')

        try:
            transPoints = griddata(self.WarpedPoints, self.FixedPoints, points, method=method);
        except:
            log = logging.getLogger(str(self.__class__));
            log.warning("Could not transform points: " + str(points));
            transPoints = None;

        return transPoints;

    def InverseTransform(self, points, **kwargs):
        transPoints = None;

        method = kwargs.get('method', 'linear')

        try:
            transPoints = griddata(self.FixedPoints, self.WarpedPoints, points, method=method);
        except:
            log = logging.getLogger(str(self.__class__));
            log.warning("Could not transform points: " + str(points));
            transPoints = None;

        return transPoints;

    def AddPoint(self, pointpair):
        '''Add the point and return the index'''
        self.points = np.append(self.points, [pointpair], 0);
        self.points = Triangulation.RemoveDuplicates(self.points);
        self.OnTransformChanged();

        Distance, index = self.NearestFixedPoint([pointpair[0], pointpair[1]]);
        return index

    def UpdatePointPair(self, index, pointpair):
        self.points[index, :] = pointpair;
        self.points = Triangulation.RemoveDuplicates(self.points);

        Distance, index = self.NearestFixedPoint([pointpair[0], pointpair[1]]);
        return index

        self.OnTransformChanged()

    def UpdateFixedPoint(self, index, point):
        self.points[index, 0:2] = point;
        self.points = Triangulation.RemoveDuplicates(self.points);
        self.OnTransformChanged()

        Distance, index = self.NearestFixedPoint([point[0], point[1]]);
        return index

    def UpdateWarpedPoint(self, index, point):
        self.points[index, 2:4] = point;
        self.points = Triangulation.RemoveDuplicates(self.points);
        self.OnTransformChanged()

        Distance, index = self.NearestWarpedPoint([point[0], point[1]]);
        return index

    def RemovePoint(self, index):
        if(self.points.shape[0] <= 3):
            return;  # Cannot have fewer than three points

        self.points = np.delete(self.points, index, 0);
        self.points = Triangulation.RemoveDuplicates(self.points);
        self.OnTransformChanged();

    def OnTransformChanged(self):
        self.ClearDataStructures()
        super(Triangulation, self).OnTransformChanged()

    def UpdateDataStructures(self):
        '''This optional method performs all computationally intense data structure creation
           If not run these data structures should be initialized in a lazy fashion by the class
           If it is known that the data structures will be needed this function can be faster
           since computations can be performed in parallel'''

        MPool = pools.GetMultithreadingPool("Transforms")
        TPool = pools.GetGlobalThreadPool()
        FixedTriTask = MPool.add_task("Fixed Triangle Delaunay", Delaunay, self.FixedPoints)
        WarpedTriTask = MPool.add_task("Warped Triangle Delaunay", Delaunay, self.WarpedPoints)

        # Cannot pickle KDTree, so use Python's thread pool

        FixedKDTask = TPool.add_task("Fixed KDTree", KDTree, self.FixedPoints)
        WarpedKDTask = TPool.add_task("Warped KDTree", KDTree, self.WarpedPoints)

        self._fixedtri = FixedTriTask.wait_return()
        self._warpedtri = WarpedTriTask.wait_return()
        self._WarpedKDTree = WarpedKDTask.wait_return()
        self._FixedKDTree = FixedKDTask.wait_return()

    def ClearDataStructures(self):
        '''Something about the transform has changed, for example the points. 
           Clear out our data structures so we do not use bad data'''

        self._fixedtri = None
        self._warpedtri = None
        self._WarpedKDTree = None
        self._FixedKDTree = None

    def NearestFixedPoint(self, points):
        '''Return the fixed points nearest to the query points'''
        return self.FixedKDTree.query(points);

    def NearestWarpedPoint(self, points):
        '''Return the warped points nearest to the query points'''
        return self.WarpedKDTree.query(points);

    def TranslateFixed(self, offset):
        '''Translate all fixed points by the specified amount'''

        self.points[:, 0:2] = self.points[:, 0:2] + offset
        self.OnTransformChanged();

    def TranslateWarped(self, offset):
        '''Translate all warped points by the specified amount'''
        self.points[:, 2:4] = self.points[:, 2:4] + offset
        self.OnTransformChanged();

    def RotateWarped(self, rangle, rotationCenter):
        '''Rotate all warped points about a center by a given angle'''
        temp = self.points[:, 2:4] - rotationCenter

        temp = np.hstack((temp, np.zeros((temp.shape[0], 1))))

        rmatrix = utils.RotationMatrix(rangle)

        rotatedtemp = temp * rmatrix
        rotatedtemp = rotatedtemp[:, 0:2] + rotationCenter
        self.points[:, 2:4] = rotatedtemp
        self.OnTransformChanged();

    def Scale(self, scalar):
        '''Scale both warped and control space by scalar'''
        self.points = self.points * scalar
        self.OnTransformChanged()

    @property
    def FixedPoints(self):
        return self.points[:, 0:2];

    @property
    def WarpedPoints(self):
        return self.points[:, 2:4];

    @property
    def ControlPointBoundingBox(self):
        cp = self.FixedPoints;

        minX = np.min(cp[:, 1]);
        maxX = np.max(cp[:, 1]);
        minY = np.min(cp[:, 0]);
        maxY = np.max(cp[:, 0]);

        return (minX, minY, maxX, maxY);

    @property
    def MappedPointBoundingBox(self):
        cp = self.WarpedPoints;

        minX = np.min(cp[:, 1]);
        maxX = np.max(cp[:, 1]);
        minY = np.min(cp[:, 0]);
        maxY = np.max(cp[:, 0]);

        return (minX, minY, maxX, maxY);

    @property
    def points(self):
        return self._points;

    @points.setter
    def points(self, val):
        self._points = np.array(val, dtype=np.float32)
        self.OnTransformChanged();

    def GetFixedPointsRect(self, bounds):
        '''bounds = [left bottom right top]'''
        return self.GetPointPairsInRect(self.FixedPoints, bounds);

    def GetWarpedPointsInRect(self, bounds):
        '''bounds = [left bottom right top]'''
        return self.GetPointPairsInRect(self.WarpedPoints, bounds);

    def GetPointPairsInRect(self, points, bounds):
        FixedPoints = [];
        WarpedPoints = [];

        # TODO: Matrix version
        # points[:, 0] >= bounds[0] and points[:]

        for iPoint in range(0, points.shape[0]):
            y, x = points[iPoint, :]
            if(x >= bounds[0] and x <= bounds[2] and y >= bounds[1] and y <= bounds[3]):
                FixedPoints.append([self.points[iPoint, 0:2]])
                WarpedPoints.append([self.points[iPoint, 2:4]])

        return (FixedPoints, WarpedPoints);

    @property
    def FixedTriangles(self):
        return self.fixedtri.vertices;

    @property
    def WarpedTriangles(self):
        return self.warpedtri.vertices;

    @property
    def MappedBounds(self):
        mapPoints = self.WarpedPoints;

        minX = np.min(mapPoints[:, 1]);
        maxX = np.max(mapPoints[:, 1]);
        minY = np.min(mapPoints[:, 0]);
        maxY = np.max(mapPoints[:, 0]);

        return (minX, minY, maxX, maxY);

    @property
    def ControlBounds(self):
        ctrlPoints = self.ControlBounds;

        minX = np.min(ctrlPoints[:, 1]);
        maxX = np.max(ctrlPoints[:, 1]);
        minY = np.min(ctrlPoints[:, 0]);
        maxY = np.max(ctrlPoints[:, 0]);

        return (minX, minY, maxX, maxY);


    def __init__(self, pointpairs):
        '''
        Constructor, expects at least three point pairs
        Point pair is (ControlX, ControlY, MappedX, MappedY)
        '''
        super(Triangulation, self).__init__();

        self._points = np.array(pointpairs, dtype=np.float32);
        self._fixedtri = None
        self._warpedtri = None
        self._WarpedKDTree = None
        self._FixedKDTree = None


    @classmethod
    def load(cls, variableParams, fixedParams):

        points = np.array.fromiter(variableParams);
        points.reshape(variableParams / 2, 2);

