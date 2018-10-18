'''
Created on Oct 18, 2012

@author: Jamesan
'''

import sys
import copy
import logging
import math
import operator
 
from . import utils
from .base import *

import nornir_imageregistration.transforms
from nornir_imageregistration.transforms.utils import InvalidIndicies
import scipy
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.spatial import *
import scipy.spatial

import nornir_imageregistration.spatial as spatial
import numpy as np
from nornir_imageregistration.transforms.utils import EnsurePointsAre4xN_NumpyArray
 

def distance(A, B):
    '''Distance between two arrays of points with equal numbers'''
    Delta = A - B
    Delta_Square = np.square(Delta)
    Delta_Sum = np.sum(Delta_Square, 1)
    Distances = np.sqrt(Delta_Sum)
    return Distances


def CentroidToVertexDistance(Centroids, TriangleVerts):
    '''
    :param ndarray Centroids: An Nx2 array of centroid points
    :param ndarray TriangleVerts: An Nx3x2 array of verticies of triangles  
    '''
    numCentroids = Centroids.shape[0]
    distance = np.zeros((numCentroids))
    for i in range(0, Centroids.shape[0]):
        distances = scipy.spatial.distance.cdist([Centroids[i]], TriangleVerts[i])
        distance[i] = np.min(distances)

    return distance


def AddTransforms(BToC_Unaltered_Transform, AToB_mapped_Transform, EnrichTolerance=None, create_copy=True):
    '''Takes the control points of a mapping from A to B and returns control points mapping from A to C
    :param bool create_copy: True if a new transform should be returned.  If false replace the passed A to B transform points.  Default is True.  
    :return: ndarray of points that can be assigned as control points for a transform'''

    if AToB_mapped_Transform.points.shape[0] < 250 and EnrichTolerance:
        return _AddAndEnrichTransforms(BToC_Unaltered_Transform, AToB_mapped_Transform, epsilon=EnrichTolerance, create_copy=create_copy) 
    else:
        return _AddMeshTransforms(BToC_Unaltered_Transform, AToB_mapped_Transform, create_copy)


def _AddMeshTransforms(BToC_Unaltered_Transform, AToB_mapped_Transform, create_copy=True):
    mappedControlPoints = AToB_mapped_Transform.FixedPoints
    txMappedControlPoints = BToC_Unaltered_Transform.Transform(mappedControlPoints)

    AToC_pointPairs = np.hstack((txMappedControlPoints, AToB_mapped_Transform.WarpedPoints))

    newTransform = None
    if create_copy:
        newTransform = copy.deepcopy(AToB_mapped_Transform)
        newTransform.points = AToC_pointPairs
        return newTransform
    else:
        AToB_mapped_Transform.points = AToC_pointPairs
        return AToB_mapped_Transform


def _AddAndEnrichTransforms(BToC_Unaltered_Transform, AToB_mapped_Transform, epsilon=None, create_copy=True):

    A_To_B_Transform = AToB_mapped_Transform
    B_To_C_Transform = BToC_Unaltered_Transform
    
    print("Begin enrichment with %d verticies" % np.shape(A_To_B_Transform.points)[0])

    PointsAdded = True
    while PointsAdded:

        A_To_C_Transform = _AddMeshTransforms(BToC_Unaltered_Transform, A_To_B_Transform, create_copy=True)

        A_Centroids = A_To_B_Transform.GetWarpedCentroids()

     #   B_Centroids = A_To_B_Transform.Transform(A_Centroids)
        # Get the centroids from B using A-B transform that correspond to A_Centroids
        B_Centroids = A_To_B_Transform.GetFixedCentroids(A_To_B_Transform.WarpedTriangles)

        # Warp the same centroids using both A->C and A->B transforms
        OC_Centroids = B_To_C_Transform.Transform(B_Centroids)
        AC_Centroids = A_To_C_Transform.Transform(A_Centroids)

        # Measure the discrepancy in the the results and create a bool array indicating which centroids failed
        Distances = distance(OC_Centroids, AC_Centroids)
        CentroidMisplaced = Distances > epsilon

        # In extreme distortion we don't want to add new control points forever or converge on existing control points. 
        # So ignore centroids falling too close to an existing vertex        
        A_CentroidTriangles = A_To_B_Transform.WarpedPoints[A_To_B_Transform.WarpedTriangles]
        CentroidVertexDistances = CentroidToVertexDistance(A_Centroids, A_CentroidTriangles)
        CentroidFarEnough = CentroidVertexDistances > epsilon

        # Add new verticies for the qualifying centroids
        AddCentroid = np.logical_and(CentroidMisplaced, CentroidFarEnough)
        PointsAdded = np.any(AddCentroid)

        if PointsAdded:
            New_ControlPoints = np.hstack((B_Centroids[AddCentroid], A_Centroids[AddCentroid]))
            starting_num_points = A_To_B_Transform.points.shape[0]
            A_To_B_Transform.AddPoints(New_ControlPoints)
            ending_num_points = A_To_B_Transform.points.shape[0]
            
            # If we have the same number of points after adding we must have had some duplicates in either fixed or warped space.  Continue onward
            if starting_num_points == ending_num_points:
                break 

            print("Mean Centroid Error: %g" % np.mean(Distances[AddCentroid]))
            print("Added %d centroids, %d centroids OK" % (np.sum(AddCentroid), np.shape(AddCentroid)[0] - np.sum(AddCentroid)))
            print("Total Verticies %d" % np.shape(A_To_B_Transform.points)[0])
            
            # TODO: Preserve the array indicating passing centroids to the next loop and do not repeat the test to save time.
            
    print("End enrichment") 

    if create_copy:
        output_transform = copy.deepcopy(AToB_mapped_Transform)
        output_transform.points = A_To_C_Transform.points
        return output_transform
    else:
        AToB_mapped_Transform.points = A_To_C_Transform.points
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
    def FindDuplicates(cls, points, new_points):
        '''Returns a bool array indicating which new_points already exist in points'''

        # (new_points, invalid_indicies) = utils.InvalidIndicies(new_points)

        round_points = np.around(points, 3)
        round_new_points = np.around(new_points, 3)

        sortedpoints = sorted(round_points, key=operator.itemgetter(0, 1))
        sorted_new_points = sorted(round_new_points, key=operator.itemgetter(0, 1)) 

        numPoints = sortedpoints.shape[0]
        numNew = new_points.shape[0]

        iPnt = 0
        iNew = 0

        invalid_indicies = np.zeros((1, numNew), dtype=bool)

        while iNew < numNew:
            testNew = sorted_new_points[iNew]

            while iPnt < numPoints:
                testPoint = sortedpoints[iPnt]

                if testPoint[0] == testNew[0]:
                    if testPoint[1] == testNew[1]:
                        invalid_indicies[iNew] = True
                        break
                    elif testPoint[1] > testNew[1]:
                        break

                if testPoint[0] > testNew[0]:
                    break

                iPnt = iPnt + 1

            iNew = iNew + 1

        return invalid_indicies

    @classmethod
    def RemoveDuplicates(cls, points):
        '''Returns tuple of the array sorted on fixed x,y without duplicates'''

        (points, indicies) = utils.InvalidIndicies(points)

        # The original implementation returned a sorted array.  I had to remove
        # that behavior because the change in index was breaking the existing 
        # triangulations the transform was caching.

        points = np.around(points, 3)
        indicies = sorted(range(len(points)), key=lambda k: points[k, 1])
        sortedpoints = sorted(enumerate(points), key=operator.itemgetter(0, 1))
        duplicate_indicies = []
        for i in range(len(sortedpoints) - 1, 0, -1):
            lastP = sortedpoints[i - 1]
            testP = sortedpoints[i]

            if lastP[0] == testP[0] and lastP[1] == testP[1]:
                sortedpoints = np.delete(sortedpoints, i, 0)
                duplicate_indicies.append(indicies[i])

        unduplicatedPoints = np.delete(points, duplicate_indicies, 0)
        return unduplicatedPoints

    @property
    def WarpedKDTree(self):
        if self._WarpedKDTree is None:
            self._WarpedKDTree = cKDTree(self.WarpedPoints)

        return self._WarpedKDTree

    @property
    def FixedKDTree(self):
        if self._FixedKDTree is None:
            self._FixedKDTree = cKDTree(self.FixedPoints)

        return self._FixedKDTree

    @property
    def fixedtri(self):
        if self._fixedtri is None:
            # try:
            # self._fixedtri = Delaunay(self.FixedPoints, incremental =True)
            # except:
            self._fixedtri = Delaunay(self.FixedPoints, incremental=False)

        return self._fixedtri

    @property
    def warpedtri(self):
        if self._warpedtri is None:
            # try:
            # self._warpedtri = Delaunay(self.WarpedPoints, incremental =True)
            # except:
            self._warpedtri = Delaunay(self.WarpedPoints, incremental=False)

        return self._warpedtri
    
    @property
    def NumControlPoints(self):
        if self._points is None:
            return 0

        return self._points.shape[0]

    @property
    def ForwardInterpolator(self):
        if self._ForwardInterpolator is None:
            self._ForwardInterpolator = LinearNDInterpolator(self.warpedtri, self.FixedPoints)

        return self._ForwardInterpolator

    @property
    def InverseInterpolator(self):
        if self._InverseInterpolator is None:
            self._InverseInterpolator = LinearNDInterpolator(self.fixedtri, self.WarpedPoints)

        return self._InverseInterpolator

    @classmethod
    def EnsurePointsAre2DNumpyArray(cls, points):
        raise DeprecationWarning('EnsurePointsAre2DNumpyArray should use utility method')
        return utils.EnsurePointsAre2DNumpyArray(points)

    @classmethod
    def EnsurePointsAre4xN_NumpyArray(cls, points):
        raise DeprecationWarning('EnsurePointsAre4xN_NumpyArray should use utility method')
        return EnsurePointsAre4xN_NumpyArray(points)

    def AddTransform(self, mappedTransform, EnrichTolerance=None, create_copy=True):
        '''Take the control points of the mapped transform and map them through our transform so the control points are in our controlpoint space''' 
        return AddTransforms(self, mappedTransform, EnrichTolerance=EnrichTolerance, create_copy=create_copy)

    def Transform(self, points, **kwargs):
        '''Map points from the warped space to fixed space'''
        transPoints = None

        method = kwargs.get('method', 'linear')

        points = utils.EnsurePointsAre2DNumpyArray(points)

        try: 
            transPoints = self.ForwardInterpolator(points)
        except:
            log = logging.getLogger(str(self.__class__))
            log.warning("Could not transform points: " + str(points))
            transPoints = None
            self._ForwardInterpolator = None

        return transPoints

    def InverseTransform(self, points, **kwargs):
        '''Map points from the fixed space to the warped space'''
        transPoints = None

        method = kwargs.get('method', 'linear')

        points = utils.EnsurePointsAre2DNumpyArray(points)

        try:
            transPoints = self.InverseInterpolator(points)
        except:
            log = logging.getLogger(str(self.__class__))
            log.warning("Could not transform points: " + str(points))
            transPoints = None
            self._InverseInterpolator = None

        return transPoints

    def FindDuplicateFixedPoints(self, new_points, epsilon=0):
        '''Using our control point KDTree, ensure the new points are not duplicates
        :return: An index array of duplicates
        '''
        distance, index = self.FixedKDTree.query(new_points)
        same = distance <= 0
        return same

    def AddPoints(self, new_points):
        '''Add the point and return the index'''
        numPts = self.NumControlPoints
        new_points = utils.EnsurePointsAre4xN_NumpyArray(new_points)

        duplicates = self.FindDuplicateFixedPoints(new_points[:, 0:2])
        new_points = new_points[~duplicates, :]

        if(new_points.shape[0] == 0):
            return

        self._points = np.append(self.points, new_points, 0)
        # self._points = Triangulation.RemoveDuplicates(self._points)

        # We won't see a change in the number of points if the new point was a duplicate
        if self.NumControlPoints != numPts:
            self.OnPointsAddedToTransform(new_points)

        return

    def AddPoint(self, pointpair):
        '''Add the point and return the index'''
        new_points = utils.EnsurePointsAre4xN_NumpyArray(pointpair)
        self.AddPoints(new_points)

        Distance, index = self.NearestFixedPoint([pointpair[0], pointpair[1]])
        return index

    def UpdatePointPair(self, index, pointpair):
        self._points[index, :] = pointpair
        self._points = Triangulation.RemoveDuplicates(self.points)
        self.OnTransformChanged()

        Distance, index = self.NearestFixedPoint([pointpair[0], pointpair[1]])
        return index

    def UpdateFixedPoint(self, index, point):
        self._points[index, 0:2] = point
        self._points = Triangulation.RemoveDuplicates(self._points)
        self.OnFixedPointChanged()

        Distance, index = self.NearestFixedPoint(point)
        return index

    def UpdateWarpedPoint(self, index, point):
        self._points[index, 2:4] = point
        self._points = Triangulation.RemoveDuplicates(self._points)
        self.OnWarpedPointChanged()

        Distance, index = self.NearestWarpedPoint(point)
        return index

    def RemovePoint(self, index):
        if(self._points.shape[0] <= 3):
            return  # Cannot have fewer than three points

        self._points = np.delete(self._points, index, 0)
        self._points = Triangulation.RemoveDuplicates(self._points)
        self.OnTransformChanged()

    def InitializeDataStructures(self):
        '''This optional method performs all computationally intense data structure creation
           If not run these data structures should be initialized in a lazy fashion by the class
           If it is known that the data structures will be needed this function can be faster
           since computations can be performed in parallel'''

        MPool = nornir_pools.GetGlobalMultithreadingPool()
        TPool = nornir_pools.GetGlobalThreadPool()
        FixedTriTask = MPool.add_task("Fixed Triangle Delaunay", Delaunay, self.FixedPoints)
        WarpedTriTask = MPool.add_task("Warped Triangle Delaunay", Delaunay, self.WarpedPoints)

        # Cannot pickle KDTree, so use Python's thread pool

        FixedKDTask = TPool.add_task("Fixed KDTree", cKDTree, self.FixedPoints)
        # WarpedKDTask = TPool.add_task("Warped KDTree", KDTree, self.WarpedPoints)

        self._WarpedKDTree = cKDTree(self.WarpedPoints)

        # MPool.wait_completion()

        self._FixedKDTree = FixedKDTask.wait_return()

        # self._FixedKDTree = cKDTree(self.FixedPoints)

        self._fixedtri = FixedTriTask.wait_return()
        self._warpedtri = WarpedTriTask.wait_return()

    def OnPointsAddedToTransform(self, new_points):
        '''Similiar to OnTransformChanged, but optimized to handle the case of points being added'''

        self.OnTransformChanged()
        return

# 
#         if(self._fixedtri is None or 
#            self._warpedtri is None):
#             self.OnTransformChanged()
#             return
# 
#         self._WarpedKDTree = None
#         self._FixedKDTree = None
#         self._FixedBoundingBox = None
#         self._MappedBoundingBox = None
#         self._ForwardInterpolator = None
#         self._InverseInterpolator = None
# 
#         self._fixedtri.add_points(new_points[:,0:2])
#         self._warpedtri.add_points(new_points[:,2:4])
#         super(Triangulation, self).OnTransformChanged()

    def OnFixedPointChanged(self):
        self._FixedKDTree = None
        self._fixedtri = None
        self._FixedBoundingBox = None
        self._ForwardInterpolator = None
        self._InverseInterpolator = None

        super(Triangulation, self).OnTransformChanged()

    def OnWarpedPointChanged(self):
        self._WarpedKDTree = None
        self._MappedBoundingBox = None
        self._warpedtri = None
        self._ForwardInterpolator = None
        self._InverseInterpolator = None

        super(Triangulation, self).OnTransformChanged()

    def OnTransformChanged(self):
        self.ClearDataStructures()
        super(Triangulation, self).OnTransformChanged()

    def ClearDataStructures(self):
        '''Something about the transform has changed, for example the points. 
           Clear out our data structures so we do not use bad data'''

        self._fixedtri = None
        self._warpedtri = None
        self._WarpedKDTree = None
        self._FixedKDTree = None
        self._FixedBoundingBox = None
        self._MappedBoundingBox = None
        self._ForwardInterpolator = None
        self._InverseInterpolator = None

    def NearestFixedPoint(self, points):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return self.FixedKDTree.query(points)

    def NearestWarpedPoint(self, points):
        '''Return the warped points nearest to the query points
        :return: Distance, Index'''
        return self.WarpedKDTree.query(points)

    def TranslateFixed(self, offset):
        '''Translate all fixed points by the specified amount'''

        self._points[:, 0:2] = self._points[:, 0:2] + offset
        self.OnFixedPointChanged()

    def TranslateWarped(self, offset):
        '''Translate all warped points by the specified amount'''
        self._points[:, 2:4] = self._points[:, 2:4] + offset
        self.OnWarpedPointChanged()

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
        self._points = self.points * scalar
        self.OnTransformChanged()
        
    def ScaleWarped(self, scalar):
        '''Scale both warped and control space by scalar'''
        self._points[:, 2:4] = self._points[:, 2:4] * scalar
        self.OnTransformChanged()
        
    def ScaleFixed(self, scalar):
        '''Scale both warped and control space by scalar'''
        self._points[:, 0:2] = self._points[:, 0:2] * scalar
        self.OnTransformChanged()

    @property
    def FixedPoints(self):
        ''' [[Y1, X1],
             [Y2, X2],
             [Yn, Xn]]'''
        return self._points[:, 0:2]

    @property
    def WarpedPoints(self):
        ''' [[Y1, X1],
             [Y2, X2],
             [Yn, Xn]]'''
        return self._points[:, 2:4]

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
        # return self.GetPointPairsInRect(self.FixedPoints, bounds)
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
                PointPair = self._points[iPoint, :] 
                if(OutputPoints is None):
                    OutputPoints = PointPair
                else:
                    OutputPoints = np.vstack((OutputPoints, PointPair))

        if not OutputPoints is None:
            if OutputPoints.ndim == 1:
                OutputPoints = np.reshape(OutputPoints, (1, OutputPoints.shape[0]))

        return OutputPoints

    @property
    def FixedTriangles(self):
        return self.fixedtri.vertices

    @property
    def WarpedTriangles(self):
        return self.warpedtri.vertices

    def GetFixedCentroids(self, triangles=None):
        '''Centroids of fixed triangles'''
        if triangles is None:
            triangles = self.FixedTriangles

        fixedTriangleVerticies = self.FixedPoints[triangles]
        swappedTriangleVerticies = np.swapaxes(fixedTriangleVerticies, 0, 2)
        Centroids = np.mean(swappedTriangleVerticies, 1)
        return np.swapaxes(Centroids, 0, 1)
    
    def GetWarpedCentroids(self, triangles=None):
        '''Centroids of warped triangles'''
        if triangles is None:
            triangles = self.WarpedTriangles

        warpedTriangleVerticies = self.WarpedPoints[triangles]
        swappedTriangleVerticies = np.swapaxes(warpedTriangleVerticies, 0, 2)
        Centroids = np.mean(swappedTriangleVerticies, 1)
        return np.swapaxes(Centroids, 0, 1)

    @property
    def MappedBounds(self):
        raise DeprecationWarning("MappedBounds is replaced by MappedBoundingBox")

    @property
    def ControlBounds(self): 
        raise DeprecationWarning("ControlBounds is replaced by FixedBoundingBox")

    def __init__(self, pointpairs):
        '''
        Constructor requires at least three point pairs
        :param ndarray pointpairs: [ControlY, ControlX, MappedY, MappedX] 
        '''
        super(Triangulation, self).__init__()

        self._points = np.asarray(pointpairs, dtype=np.float32)
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
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

