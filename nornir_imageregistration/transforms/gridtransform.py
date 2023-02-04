import copy
import logging
import operator

import numpy as np
from numpy.typing import NDArray
import scipy
import scipy.spatial
from scipy.interpolate import griddata, LinearNDInterpolator, CloughTocher2DInterpolator, RegularGridInterpolator

import nornir_pools
import nornir_imageregistration
from . import utils
from .base import IDiscreteTransform, ITransformChangeEvents, ITransform, ITransformScaling, ITransformTranslation, \
    IControlPoints, TransformType, ITransformTargetRotation, ITargetSpaceControlPointEdit
from nornir_imageregistration.transforms.controlpointbase import ControlPointBase
from nornir_imageregistration.grid_subdivision import ITKGridDivision

from nornir_imageregistration.transforms.utils import InvalidIndicies


class GridTransform(ITransformScaling, ITransformTranslation, ITransformTargetRotation, ITargetSpaceControlPointEdit, ControlPointBase):

    def type(self) -> TransformType:
        return TransformType.GRID

    def grid_dims(self) -> tuple[int, int]:
        return self._grid.grid_dims

    def Load(self, TransformString: str, pixelSpacing=None):
        """
        Creates an instance of the transform from the TransformString
        """
        return nornir_imageregistration.transforms.factory.LoadTransform(TransformString, pixelSpacing)

    def __getstate__(self):
        odict = {'_points': self._points}

        return odict

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary)
        self.OnChangeEventListeners = []
        self.OnTransformChanged()

    def __init__(self,
                 grid: ITKGridDivision):

        self._grid = grid
        control_points = np.hstack((grid.TargetPoints, grid.SourcePoints))

        super(GridTransform, self).__init__(control_points)

        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._FixedKDTree = None
        self._WarpedKDTree = None
        self._fixedtri = None

        pass

    @property
    def WarpedKDTree(self):
        if self._WarpedKDTree is None:
            self._WarpedKDTree = scipy.spatial.cKDTree(self.SourcePoints)

        return self._WarpedKDTree

    @property
    def FixedKDTree(self):
        if self._FixedKDTree is None:
            self._FixedKDTree = scipy.spatial.cKDTree(self.TargetPoints)

        return self._FixedKDTree

    @property
    def fixedtri(self):
        if self._fixedtri is None:
            # try:
            # self._fixedtri = Delaunay(self.TargetPoints, incremental =True)
            # except:
            self._fixedtri = scipy.spatial.Delaunay(self.TargetPoints, incremental=False)

        return self._fixedtri

    def NearestFixedPoint(self, points: NDArray[float]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return self.FixedKDTree.query(points)

    def NearestWarpedPoint(self, points: NDArray[float]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return self.WarpedKDTree.query(points)

    def Scale(self, scalar):
        '''Scale both warped and control space by scalar'''
        self._points = self._points * scalar
        self.OnTransformChanged()

    def ScaleWarped(self, scalar):
        '''Scale source space control points by scalar'''
        self._points[:, 2:4] = self._points[:, 2:4] * scalar
        self.OnTransformChanged()

    def ScaleFixed(self, scalar):
        '''Scale target space control points by scalar'''
        self._points[:, 0:2] = self._points[:, 0:2] * scalar
        self.OnTransformChanged()

    def TranslateFixed(self, offset: NDArray[float]):
        '''Translate all fixed points by the specified amount'''

        self._points[:, 0:2] = self._points[:, 0:2] + offset
        self.OnFixedPointChanged()

    def TranslateWarped(self, offset: NDArray[float]):
        '''Translate all warped points by the specified amount'''
        self._points[:, 2:4] = self._points[:, 2:4] + offset
        self.OnWarpedPointChanged()

    @property
    def ForwardInterpolator(self):
        if self._ForwardInterpolator is None:
            self._ForwardInterpolator = RegularGridInterpolator(self._grid.axis_points,
                                                                np.reshape(self._grid.TargetPoints, (self._grid.grid_dims[0], self._grid.grid_dims[1], 2)),
                                                                bounds_error=False)
            
        return self._ForwardInterpolator

    @property
    def InverseInterpolator(self):
        if self._InverseInterpolator is None:
            self._InverseInterpolator = LinearNDInterpolator(self.fixedtri, self.SourcePoints)

        return self._InverseInterpolator

    def Transform(self, points, **kwargs):
        '''Map points from the warped space to fixed space'''
        transPoints = None

        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        transPoints = self.ForwardInterpolator(points)
        return transPoints

    def InverseTransform(self, points, **kwargs):
        '''Map points from the fixed space to the warped space'''
        transPoints = None

        method = kwargs.get('method', 'linear')

        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        try:
            transPoints = self.InverseInterpolator(points)
        except Exception as e: # This is usually a scipy.spatial._qhull.QhullError:
            log = logging.getLogger(str(self.__class__))
            log.warning("Could not transform points: " + str(points))
            transPoints = None
            self._InverseInterpolator = None

            # This was added for the case where all points in the triangulation are colinear.
            transPoints = np.empty(points.shape)
            transPoints[:] = np.NaN

        return transPoints

    @property
    def FixedTriangles(self):
        return self.fixedtri.vertices

    def GetFixedCentroids(self, triangles=None):
        '''Centroids of fixed triangles'''
        if triangles is None:
            triangles = self.FixedTriangles

        fixedTriangleVerticies = self.TargetPoints[triangles]
        swappedTriangleVerticies = np.swapaxes(fixedTriangleVerticies, 0, 2)
        Centroids = np.mean(swappedTriangleVerticies, 1)
        return np.swapaxes(Centroids, 0, 1)

    def RotateTargetPoints(self, rangle: float, rotationCenter: NDArray[float]):
        '''Rotate all warped points about a center by a given angle'''
        self._points[:, 0:2] = ControlPointBase.RotatePoints(self.TargetPoints, rangle, rotationCenter)
        self.OnTransformChanged()

    def UpdateTargetPoints(self, index: int | NDArray[int], point: NDArray[float]):
        self._points[index, 0:2] = point
        self.OnFixedPointChanged()

    def OnFixedPointChanged(self):
        super(GridTransform, self).OnFixedPointChanged()
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._fixedtri = None
        self._FixedKDTree = None
        super(GridTransform, self).OnTransformChanged()

    def OnWarpedPointChanged(self):
        raise NotImplementedError("Grid transforms have a fixed grid of points, they should not change")

    def ClearDataStructures(self):
        '''Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data'''
        super(GridTransform, self).ClearDataStructures()
        self._fixedtri = None
        self._FixedKDTree = None
        self._ForwardInterpolator = None
        self._InverseInterpolator = None