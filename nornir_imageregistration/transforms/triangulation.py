'''
Created on Oct 18, 2012

@author: Jamesan
'''

import copy
import logging
import math
import operator

from scipy.interpolate import griddata
from scipy.spatial import *

from nornir_imageregistration.transforms.utils import InvalidIndicies
import nornir_imageregistration.spatial as spatial
import numpy as np

from . import utils
from .base import *


def AddTransforms(BToC_Unaltered_Transform, AToB_mapped_Transform, create_copy=True):
    '''Takes the control points of a mapping from A to B and returns control points mapping from A to C
    :param bool create_copy: True if a new transform should be returned.  If false replace the passed A to B transform points.  Default is True.  
    :return: ndarray of points that can be assigned as control points for a transform'''
    
    mappedControlPoints = AToB_mapped_Transform.FixedPoints
    txMappedControlPoints = BToC_Unaltered_Transform.Transform(mappedControlPoints)
    
    AToC_pointPairs = np.hstack( (txMappedControlPoints, AToB_mapped_Transform.WarpedPoints) )
    
    newTransform = None
    if create_copy:
        newTransform = copy.deepcopy(AToB_mapped_Transform)
        newTransform.points = AToC_pointPairs
        return newTransform
    else:
        AToB_mapped_Transform.points = AToC_pointPairs
        return AToB_mapped_Transform 


class Triangulation(Base):
    '''
    Triangulation transform has an nx4 array of points, with rows organized as
    [controlx controly warpedx warpedy]
    '''

    def __getstate__(self):
        odict = {}
        odict['_points'] = self._points

        return odict

    def __setstate__(self, dictionary):         
        self.__dict__.update(dictionary)
        self.OnChangeEventListeners = []
        self.OnTransformChanged()

    @classmethod
    def RemoveDuplicates(cls, points):
        '''Returns tuple of the array sorted on fixed x,y without duplicates'''

        (points, indicies) = utils.InvalidIndicies(points)

        DuplicateRemoved = False
        points = np.around(points, 3)
        sortedpoints = sorted(points, key=operator.itemgetter(0, 1))
        for i in range(len(sortedpoints) - 1, 0, -1):
            lastP = sortedpoints[i - 1]
            testP = sortedpoints[i]

            if lastP[0] == testP[0]:
                if lastP[1] == testP[1]:
                    DuplicateRemoved = True
                    sortedpoints = np.delete(sortedpoints, i, 0)
                    i = i + 1

        return np.asarray(sortedpoints)

    @property
    def WarpedKDTree(self):
        if self._WarpedKDTree is None:
            self._WarpedKDTree = KDTree(self.WarpedPoints)

        return self._WarpedKDTree

    @property
    def FixedKDTree(self):
        if self._FixedKDTree is None:
            self._FixedKDTree = KDTree(self.FixedPoints)

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
     
    def AddTransform(self, mappedTransform, create_copy=True):
        '''Take the control points of the mapped transform and map them through our transform so the control points are in our controlpoint space''' 
        return AddTransforms(self, mappedTransform, create_copy)


    def Transform(self, points, **kwargs):
        '''Map points from the fixed space to the warped space'''
        transPoints = None

        method = kwargs.get('method', 'linear')

        try:
            transPoints = griddata(self.WarpedPoints, self.FixedPoints, points, method=method)
        except:
            log = logging.getLogger(str(self.__class__))
            log.warning("Could not transform points: " + str(points))
            transPoints = None

        return transPoints

    def InverseTransform(self, points, **kwargs):
        '''Map points from the warped space to the fixed space'''
        transPoints = None

        method = kwargs.get('method', 'linear')

        try:
            transPoints = griddata(self.FixedPoints, self.WarpedPoints, points, method=method)
        except:
            log = logging.getLogger(str(self.__class__))
            log.warning("Could not transform points: " + str(points))
            transPoints = None

        return transPoints

    def AddPoint(self, pointpair):
        '''Add the point and return the index'''
        self.points = np.append(self.points, [pointpair], 0)
        self.points = Triangulation.RemoveDuplicates(self.points)
        self.OnTransformChanged()

        Distance, index = self.NearestFixedPoint([pointpair[0], pointpair[1]])
        return index

    def UpdatePointPair(self, index, pointpair):
        self.points[index, :] = pointpair
        self.points = Triangulation.RemoveDuplicates(self.points)

        Distance, index = self.NearestFixedPoint([pointpair[0], pointpair[1]])
        return index

        self.OnTransformChanged()

    def UpdateFixedPoint(self, index, point):
        self.points[index, 0:2] = point
        self.points = Triangulation.RemoveDuplicates(self.points)
        self.OnTransformChanged()

        Distance, index = self.NearestFixedPoint(point)
        return index

    def UpdateWarpedPoint(self, index, point):
        self.points[index, 2:4] = point
        self.points = Triangulation.RemoveDuplicates(self.points)
        self.OnTransformChanged()

        Distance, index = self.NearestWarpedPoint(point)
        return index

    def RemovePoint(self, index):
        if(self.points.shape[0] <= 3):
            return  # Cannot have fewer than three points

        self.points = np.delete(self.points, index, 0)
        self.points = Triangulation.RemoveDuplicates(self.points)
        self.OnTransformChanged()

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
        self._FixedBoundingBox = None
        self._MappedBoundingBox = None

    def NearestFixedPoint(self, points):
        '''Return the fixed points nearest to the query points'''
        return self.FixedKDTree.query(points)

    def NearestWarpedPoint(self, points):
        '''Return the warped points nearest to the query points'''
        return self.WarpedKDTree.query(points)

    def TranslateFixed(self, offset):
        '''Translate all fixed points by the specified amount'''

        self.points[:, 0:2] = self.points[:, 0:2] + offset
        self.OnTransformChanged()

    def TranslateWarped(self, offset):
        '''Translate all warped points by the specified amount'''
        self.points[:, 2:4] = self.points[:, 2:4] + offset
        self.OnTransformChanged()

    def RotateWarped(self, rangle, rotationCenter):
        '''Rotate all warped points about a center by a given angle'''
        temp = self.points[:, 2:4] - rotationCenter

        temp = np.hstack((temp, np.zeros((temp.shape[0], 1))))

        rmatrix = utils.RotationMatrix(rangle)

        rotatedtemp = temp * rmatrix
        rotatedtemp = rotatedtemp[:, 0:2] + rotationCenter
        self.points[:, 2:4] = rotatedtemp
        self.OnTransformChanged()

    def Scale(self, scalar):
        '''Scale both warped and control space by scalar'''
        self.points = self.points * scalar
        self.OnTransformChanged()

    @property
    def FixedPoints(self):
        ''' [[Y1, X1],
             [Y2, X2],
             [Yn, Xn]]'''
        return self.points[:, 0:2]

    @property
    def WarpedPoints(self):
        ''' [[Y1, X1],
             [Y2, X2],
             [Yn, Xn]]'''
        return self.points[:, 2:4]

    @property
    def FixedBoundingBox(self):
        '''
        :return: (minY, minX, maxY, maxX)
        '''
        if self._FixedBoundingBox is None:
            self._FixedBoundingBox = spatial.BoundingPrimitiveFromPoints(self.FixedPoints)

        return self._FixedBoundingBox

    @property
    def MappedBoundingBox(self):
        '''
        :return: (minY, minX, maxY, maxX)
        '''
        if self._MappedBoundingBox is None:
            self._MappedBoundingBox = spatial.BoundingPrimitiveFromPoints(self.WarpedPoints)

        return self._MappedBoundingBox

    @property
    def FixedBoundingBoxWidth(self):
        raise DeprecationWarning("FixedBoundingBoxWidth is deprecated.  Use FixedBoundingBox.Width instead")
        return self.FixedBoundingBox.Width

    @property
    def FixedBoundingBoxHeight(self):
        raise DeprecationWarning("FixedBoundingBoxHeight is deprecated.  Use FixedBoundingBox.Height instead")
        return self.FixedBoundingBox.Height

    @property
    def MappedBoundingBoxWidth(self):
        raise DeprecationWarning("MappedBoundingBoxWidth is deprecated.  Use MappedBoundingBox.Width instead")
        return self.MappedBoundingBox.Width

    @property
    def MappedBoundingBoxHeight(self):
        raise DeprecationWarning("MappedBoundingBoxHeight is deprecated.  Use MappedBoundingBox.Height instead")
        return self.MappedBoundingBox.Height

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, val):
        self._points = np.asarray(val, dtype=np.float32)
        self.OnTransformChanged()

    def GetFixedPointsRect(self, bounds):
        '''bounds = [left bottom right top]'''
        #return self.GetPointPairsInRect(self.FixedPoints, bounds)
        raise DeprecationWarning("This function was a typo, replace with GetFixedPointsInRect")
    
    def GetFixedPointsInRect(self, bounds):
        '''bounds = [left bottom right top]'''
        return self.GetPointPairsInRect(self.FixedPoints, bounds)

    def GetWarpedPointsInRect(self, bounds):
        '''bounds = [left bottom right top]'''
        return self.GetPointPairsInRect(self.WarpedPoints, bounds)
    
    def GetPointsInFixedRect(self, bounds):
        '''bounds = [left bottom right top]'''
        return self.GetPointPairsInRect(self.FixedPoints, bounds)

    def GetPointsInWarpedRect(self, bounds):
        '''bounds = [left bottom right top]'''
        return self.GetPointPairsInRect(self.WarpedPoints, bounds)

    def GetPointPairsInRect(self, points, bounds):
        OutputPoints = None
         
        for iPoint in range(0, points.shape[0]):
            y, x = points[iPoint, :]
            if(x >= bounds[spatial.iRect.MinX] and x <= bounds[spatial.iRect.MaxX] and y >= bounds[spatial.iRect.MinY] and y <= bounds[spatial.iRect.MaxY]):
                PointPair = self.points[iPoint, :] 
                if(OutputPoints is None):
                    OutputPoints = PointPair
                else:
                    OutputPoints = np.vstack((OutputPoints, PointPair))

        if not OutputPoints is None:
            if OutputPoints.ndim == 1:
                OutputPoints = np.reshape(OutputPoints,(1, OutputPoints.shape[0]))
                
        return OutputPoints

    @property
    def FixedTriangles(self):
        return self.fixedtri.vertices

    @property
    def WarpedTriangles(self):
        return self.warpedtri.vertices

    @property
    def MappedBounds(self):
        raise DeprecationWarning("MappedBounds is replaced by MappedBoundingBox")

    @property
    def ControlBounds(self):

        raise DeprecationWarning("ControlBounds is replaced by FixedBoundingBox")

    def __init__(self, pointpairs):
        '''
        Constructor, expects at least three point pairs
        Point pair is (ControlX, ControlY, MappedX, MappedY)
        '''
        super(Triangulation, self).__init__()

        self._points = np.asarray(pointpairs, dtype=np.float32)
        self._fixedtri = None
        self._warpedtri = None
        self._WarpedKDTree = None
        self._FixedKDTree = None
        self._FixedBoundingBox = None
        self._MappedBoundingBox = None 
        

    @classmethod
    def load(cls, variableParams, fixedParams):

        points = np.array.fromiter(variableParams)
        points.reshape(variableParams / 2, 2)

