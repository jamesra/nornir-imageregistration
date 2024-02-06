import logging

import numpy as np
try:
    import cupy as cp
    #import cupyx
    from cupyx.scipy.interpolate import RegularGridInterpolator as cuRegularGridInterpolator
    from cupyx.scipy.interpolate import RBFInterpolator as cuRBFInterpolator
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
    #import nornir_imageregistration.cupyx_thunk as cupyx
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
    #import nornir_imageregistration.cupyx_thunk as cupyx
from numpy.typing import NDArray
import scipy
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
import scipy.spatial

import nornir_imageregistration
from nornir_imageregistration.grid_subdivision import ITKGridDivision
from nornir_imageregistration.transforms import float_to_shortest_string
from nornir_imageregistration.transforms.controlpointbase import ControlPointBase, ControlPointBase_GPUComponent
from .base import ITransformScaling, ITransformRelativeScaling, \
    ITransformTranslation, \
    TransformType, ITransformTargetRotation, ITargetSpaceControlPointEdit, IGridTransform, \
    ITriangulatedTargetSpace


class GridTransform(ITransformScaling, ITransformRelativeScaling, ITransformTranslation,
                    ITransformTargetRotation, ITargetSpaceControlPointEdit,
                    IGridTransform, ITriangulatedTargetSpace, ControlPointBase):

    @property
    def type(self) -> TransformType:
        return TransformType.GRID

    @property
    def grid(self) -> ITKGridDivision:
        return self._grid

    @property
    def grid_dims(self) -> tuple[int, int]:
        return self._grid.grid_dims

    def Load(self, TransformString: str, pixelSpacing=None):
        """
        Creates an instance of the transform from the TransformString
        """
        return nornir_imageregistration.transforms.factory.LoadTransform(TransformString, pixelSpacing)

    def __getstate__(self):
        odict = {'_points': self._points,
                 '_grid': self._grid}

        return odict

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary)
        self.OnChangeEventListeners = []
        self.OnTransformChanged()

    def __init__(self,
                 grid: ITKGridDivision):

        self._grid = grid
        try:
            control_points = np.hstack((grid.TargetPoints, grid.SourcePoints))
        except:
            print(f'Invalid grid:\n{grid.TargetPoints}\n\n{grid.SourcePoints}')
            raise

        super(GridTransform, self).__init__(control_points)

        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._FixedKDTree = None
        self._WarpedKDTree = None
        self._fixedtri = None
        pass

    def ToITKString(self):
        numPoints = self.SourcePoints.shape[0]
        (bottom, left, top, right) = self.MappedBoundingBox.ToTuple()
        image_width = (
                right - left)  # We remove one because a 10x10 image is mappped from 0,0 to 10,10, which means the bounding box will be Left=0, Right=10, and width is 11 unless we correct for it.
        image_height = (top - bottom)

        YDim = int(self.grid.grid_dims[0]) - 1  # For whatever reason ITK subtracts one from the dimensions
        XDim = int(self.grid.grid_dims[1]) - 1  # For whatever reason ITK subtracts one from the dimensions
        
        if self.points.shape[0] != self.grid.grid_dims.prod():
            raise ValueError("Grid transform number of points does not match grid dimensions")

        output = ["GridTransform_double_2_2 vp " + str(numPoints * 2)]
        template = " %(cx)s %(cy)s"
        NumAdded = int(0)
        for CY, CX, MY, MX in self.points:
            pstr = template % {'cx': float_to_shortest_string(CX, 3), 'cy': float_to_shortest_string(CY, 3)}
            output.append(pstr)
            NumAdded += 1

        # ITK expects the image dimensions to be the actual dimensions of the image.  So if an image is 1024 pixels wide
        # then 1024 should be written to the file.
        output.append(f" fp 7 0 {YDim:d} {XDim:d} {left:g} {bottom:g} {image_width:g} {image_height:g}")
        transform_string = ''.join(output)

        return transform_string

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

    @property
    def target_space_trianglulation(self) -> scipy.spatial.Delaunay:
        return self.fixedtri

    def NearestFixedPoint(self, points: NDArray[np.floating]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return self.FixedKDTree.query(points)

    def NearestTargetPoint(self, points: NDArray[np.floating]):
        '''Return the target points nearest to the query points
        :return: Distance, Index
        '''
        return self.FixedKDTree.query(points)

    def NearestWarpedPoint(self, points: NDArray[np.floating]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return self.WarpedKDTree.query(points)

    def NearestSourcePoint(self, points: NDArray[np.floating]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return self.WarpedKDTree.query(points)

    def Scale(self, scalar):
        '''Scale both warped and control space by scalar'''
        self._points *= scalar
        self.OnTransformChanged()

    def ScaleWarped(self, scalar):
        '''Scale source space control points by scalar'''
        self._points[:, 2:4] = self._points[:, 2:4] * scalar
        self.OnTransformChanged()

    def ScaleFixed(self, scalar):
        '''Scale target space control points by scalar'''
        self._points[:, 0:2] = self._points[:, 0:2] * scalar
        self.OnTransformChanged()

    def TranslateFixed(self, offset: NDArray[np.floating]):
        '''Translate all fixed points by the specified amount'''

        self._points[:, 0:2] = self._points[:, 0:2] + offset
        self.OnFixedPointChanged()

    def TranslateWarped(self, offset: NDArray[np.floating]):
        '''Translate all warped points by the specified amount'''
        self._points[:, 2:4] = self._points[:, 2:4] + offset
        self.OnWarpedPointChanged()

    def GetPointPairsInRect(self, points: NDArray[np.floating], bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        OutputPoints = None

        bounds = nornir_imageregistration.Rectangle.PrimitiveToRectangle(bounds).ToArray()

        for iPoint in range(0, points.shape[0]):
            y, x = points[iPoint, :]
            if nornir_imageregistration.Rectangle.contains(bounds, (y, x)):
                PointPair = self._points[iPoint, :]
                if OutputPoints is None:
                    OutputPoints = PointPair
                else:
                    OutputPoints = np.vstack((OutputPoints, PointPair))

        if OutputPoints is not None:
            if OutputPoints.ndim == 1:
                OutputPoints = np.reshape(OutputPoints, (1, OutputPoints.shape[0]))

        return OutputPoints

    def GetFixedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetWarpedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def GetPointPairsInFixedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetPointPairsInWarpedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def PointPairsToWarpedPoints(self, points: NDArray[np.floating]):
        '''Return the warped points from a set of target-source point pairs'''
        return points[:, 2:4]

    def PointPairsToTargetPoints(self, points: NDArray[np.floating]):
        '''Return the target points from a set of target-source point pairs'''
        return points[:, 0:2]

    @property
    def ForwardInterpolator(self):
        if self._ForwardInterpolator is None:
            self._ForwardInterpolator = RegularGridInterpolator(self._grid.axis_points,
                                                                np.reshape(self.TargetPoints, (
                                                                self._grid.grid_dims[0], self._grid.grid_dims[1], 2)),
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
        except Exception as e:  # This is usually a scipy.spatial._qhull.QhullError:
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
        return self.fixedtri.simplices

    def GetFixedCentroids(self, triangles=None):
        '''Centroids of fixed triangles'''
        if triangles is None:
            triangles = self.FixedTriangles

        fixedTriangleVerticies = self.TargetPoints[triangles]
        swappedTriangleVerticies = np.swapaxes(fixedTriangleVerticies, 0, 2)
        Centroids = np.mean(swappedTriangleVerticies, 1)
        return np.swapaxes(Centroids, 0, 1)

    def RotateTargetPoints(self, rangle: float, rotationCenter: NDArray[np.floating] | None):
        '''Rotate all warped points about a center by a given angle'''
        self._points[:, 0:2] = ControlPointBase.RotatePoints(self.TargetPoints, rangle, rotationCenter)
        self.OnTransformChanged()

    def UpdateTargetPointsByIndex(self, index: int | NDArray[np.integer], point: NDArray[np.floating]) -> int | NDArray[np.integer]:
        self._points[index, 0:2] = point
        self.OnFixedPointChanged()
        return index

    def UpdateTargetPointsByPosition(self, old_points: NDArray[np.floating], points: NDArray[np.floating]) -> int | NDArray[np.integer]:
        old_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(old_points)
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
        distance, index = self.NearestFixedPoint(old_points)
        return self.UpdateTargetPointsByIndex(index, points)

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


class GridTransform_GPUComponent(ITransformScaling, ITransformRelativeScaling, ITransformTranslation,
                    ITransformTargetRotation, ITargetSpaceControlPointEdit,
                    IGridTransform, ITriangulatedTargetSpace, ControlPointBase):

    @property
    def type(self) -> TransformType:
        return TransformType.GRID

    @property
    def grid(self) -> ITKGridDivision:
        return self._grid

    @property
    def grid_dims(self) -> tuple[int, int]:
        return self._grid.grid_dims

    def Load(self, TransformString: str, pixelSpacing=None):
        """
        Creates an instance of the transform from the TransformString
        """
        return nornir_imageregistration.transforms.factory.LoadTransform(TransformString, pixelSpacing)

    def __getstate__(self):
        odict = {'_points': self._points,
                 '_grid': self._grid}

        return odict

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary)
        self.OnChangeEventListeners = []
        self.OnTransformChanged()

    def __init__(self,
                 grid: ITKGridDivision):

        self._grid = grid
        try:
            control_points = np.hstack((grid.TargetPoints, grid.SourcePoints))
        except:
            print(f'Invalid grid:\n{grid.TargetPoints}\n\n{grid.SourcePoints}')
            raise

        super(GridTransform_GPUComponent, self).__init__(control_points)

        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._FixedKDTree = None
        self._WarpedKDTree = None
        self._fixedtri = None
        pass

    def ToITKString(self):
        numPoints = self.SourcePoints.shape[0]
        (bottom, left, top, right) = self.MappedBoundingBox.ToTuple()
        image_width = (
                right - left)  # We remove one because a 10x10 image is mappped from 0,0 to 10,10, which means the bounding box will be Left=0, Right=10, and width is 11 unless we correct for it.
        image_height = (top - bottom)

        YDim = int(self.grid.grid_dims[0]) - 1  # For whatever reason ITK subtracts one from the dimensions
        XDim = int(self.grid.grid_dims[1]) - 1  # For whatever reason ITK subtracts one from the dimensions

        output = ["GridTransform_double_2_2 vp " + str(numPoints * 2)]
        template = " %(cx)s %(cy)s"
        NumAdded = int(0)
        for CY, CX, MY, MX in self.points:
            pstr = template % {'cx': float_to_shortest_string(CX, 3), 'cy': float_to_shortest_string(CY, 3)}
            output.append(pstr)
            NumAdded += 1

        # ITK expects the image dimensions to be the actual dimensions of the image.  So if an image is 1024 pixels wide
        # then 1024 should be written to the file.
        output.append(f" fp 7 0 {YDim:d} {XDim:d} {left:g} {bottom:g} {image_width:g} {image_height:g}")
        transform_string = ''.join(output)

        return transform_string

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

    @property
    def target_space_trianglulation(self) -> scipy.spatial.Delaunay:
        return self.fixedtri

    def NearestFixedPoint(self, points: NDArray[np.floating]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return self.FixedKDTree.query(points)

    def NearestTargetPoint(self, points: NDArray[np.floating]):
        '''Return the target points nearest to the query points
        :return: Distance, Index
        '''
        return self.FixedKDTree.query(points)

    def NearestWarpedPoint(self, points: NDArray[np.floating]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return self.WarpedKDTree.query(points)

    def NearestSourcePoint(self, points: NDArray[np.floating]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return self.WarpedKDTree.query(points)

    def Scale(self, scalar):
        '''Scale both warped and control space by scalar'''
        self._points *= scalar
        self.OnTransformChanged()

    def ScaleWarped(self, scalar):
        '''Scale source space control points by scalar'''
        self._points[:, 2:4] = self._points[:, 2:4] * scalar
        self.OnTransformChanged()

    def ScaleFixed(self, scalar):
        '''Scale target space control points by scalar'''
        self._points[:, 0:2] = self._points[:, 0:2] * scalar
        self.OnTransformChanged()

    def TranslateFixed(self, offset: NDArray[np.floating]):
        '''Translate all fixed points by the specified amount'''

        self._points[:, 0:2] = self._points[:, 0:2] + offset
        self.OnFixedPointChanged()

    def TranslateWarped(self, offset: NDArray[np.floating]):
        '''Translate all warped points by the specified amount'''
        self._points[:, 2:4] = self._points[:, 2:4] + offset
        self.OnWarpedPointChanged()

    def GetPointPairsInRect(self, points: NDArray[np.floating], bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        OutputPoints = None

        bounds = nornir_imageregistration.Rectangle.PrimitiveToRectangle(bounds).ToArray()

        for iPoint in range(0, points.shape[0]):
            y, x = points[iPoint, :]
            if nornir_imageregistration.Rectangle.contains(bounds, (y, x)):
                PointPair = self._points[iPoint, :]
                if OutputPoints is None:
                    OutputPoints = PointPair
                else:
                    OutputPoints = cp.vstack((OutputPoints, PointPair))

        if OutputPoints is not None:
            if OutputPoints.ndim == 1:
                OutputPoints = cp.reshape(OutputPoints, (1, OutputPoints.shape[0]))

        return OutputPoints

    def GetFixedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetWarpedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def GetPointPairsInFixedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetPointPairsInWarpedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def PointPairsToWarpedPoints(self, points: NDArray[np.floating]):
        '''Return the warped points from a set of target-source point pairs'''
        return points[:, 2:4]

    def PointPairsToTargetPoints(self, points: NDArray[np.floating]):
        '''Return the target points from a set of target-source point pairs'''
        return points[:, 0:2]

    @property
    def ForwardInterpolator(self):
        if self._ForwardInterpolator is None:
            self._ForwardInterpolator = cuRegularGridInterpolator(cp.array(self._grid.axis_points),
                                                                  cp.reshape(self.TargetPoints, (self._grid.grid_dims[0], self._grid.grid_dims[1], 2)),
                                                                  bounds_error=False)

        return self._ForwardInterpolator

    @property
    def InverseInterpolator(self):
        if self._InverseInterpolator is None:
            self._InverseInterpolator = LinearNDInterpolator(self.fixedtri, self.SourcePoints)

        return self._InverseInterpolator

    def Transform(self, points,  **kwargs):
        '''Map points from the warped space to fixed space'''
        transPoints = None

        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)
        transPoints = self.ForwardInterpolator(points)
        return transPoints

    def InverseTransform(self, points, **kwargs):
        '''Map points from the fixed space to the warped space'''
        transPoints = None

        method = kwargs.get('method', 'linear')

        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

        try:
            transPoints = self.InverseInterpolator(points)
        except Exception as e:  # This is usually a scipy.spatial._qhull.QhullError:
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
        return self.fixedtri.simplices

    def GetFixedCentroids(self, triangles=None):
        '''Centroids of fixed triangles'''
        if triangles is None:
            triangles = self.FixedTriangles

        fixedTriangleVerticies = self.TargetPoints[triangles]
        swappedTriangleVerticies = np.swapaxes(fixedTriangleVerticies, 0, 2)
        Centroids = np.mean(swappedTriangleVerticies, 1)
        return np.swapaxes(Centroids, 0, 1)

    def RotateTargetPoints(self, rangle: float, rotationCenter: NDArray[np.floating] | None):
        '''Rotate all warped points about a center by a given angle'''
        self._points[:, 0:2] = ControlPointBase.RotatePoints(self.TargetPoints, rangle, rotationCenter)
        self.OnTransformChanged()

    def UpdateTargetPointsByIndex(self, index: int | NDArray[np.integer], point: NDArray[np.floating]) -> int | NDArray[np.integer]:
        self._points[index, 0:2] = point
        self.OnFixedPointChanged()
        return index

    def UpdateTargetPointsByPosition(self, old_points: NDArray[np.floating], points: NDArray[np.floating]) -> int | NDArray[np.integer]:
        old_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(old_points)
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
        distance, index = self.NearestFixedPoint(old_points)
        return self.UpdateTargetPointsByIndex(index, points)

    def OnFixedPointChanged(self):
        super(GridTransform_GPUComponent, self).OnFixedPointChanged()
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        self._fixedtri = None
        self._FixedKDTree = None
        super(GridTransform_GPUComponent, self).OnTransformChanged()

    def OnWarpedPointChanged(self):
        raise NotImplementedError("Grid transforms have a fixed grid of points, they should not change")

    def ClearDataStructures(self):
        '''Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data'''
        super(GridTransform_GPUComponent, self).ClearDataStructures()
        self._fixedtri = None
        self._FixedKDTree = None
        self._ForwardInterpolator = None
        self._InverseInterpolator = None


class GridTransform_GPU(ITransformScaling, ITransformRelativeScaling, ITransformTranslation,
                    ITransformTargetRotation, ITargetSpaceControlPointEdit,
                    IGridTransform, ControlPointBase_GPUComponent):

    @property
    def type(self) -> TransformType:
        return TransformType.GRID

    @property
    def grid(self) -> ITKGridDivision:
        return self._grid

    @property
    def grid_dims(self) -> tuple[int, int]:
        return self._grid.grid_dims

    def Load(self, TransformString: str, pixelSpacing=None):
        """
        Creates an instance of the transform from the TransformString
        """
        return nornir_imageregistration.transforms.factory.LoadTransform(TransformString, pixelSpacing)

    def __getstate__(self):
        odict = {'_points': self._points,
                 '_grid': self._grid}

        return odict

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary)
        self.OnChangeEventListeners = []
        self.OnTransformChanged()

    def __init__(self,
                 grid: ITKGridDivision):

        self._grid = grid
        try:
            control_points = cp.hstack((grid.TargetPoints, grid.SourcePoints))
        except:
            print(f'Invalid grid:\n{grid.TargetPoints}\n\n{grid.SourcePoints}')
            raise

        super(GridTransform_GPU, self).__init__(control_points)

        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        pass

    def ToITKString(self):
        numPoints = self.SourcePoints.shape[0]
        (bottom, left, top, right) = self.MappedBoundingBox.ToTuple()
        image_width = (
                right - left)  # We remove one because a 10x10 image is mappped from 0,0 to 10,10, which means the bounding box will be Left=0, Right=10, and width is 11 unless we correct for it.
        image_height = (top - bottom)

        YDim = int(self.grid.grid_dims[0]) - 1  # For whatever reason ITK subtracts one from the dimensions
        XDim = int(self.grid.grid_dims[1]) - 1  # For whatever reason ITK subtracts one from the dimensions

        output = ["GridTransform_double_2_2 vp " + str(numPoints * 2)]
        template = " %(cx)s %(cy)s"
        NumAdded = int(0)
        for CY, CX, MY, MX in self.points:
            pstr = template % {'cx': float_to_shortest_string(CX, 3), 'cy': float_to_shortest_string(CY, 3)}
            output.append(pstr)
            NumAdded += 1

        # ITK expects the image dimensions to be the actual dimensions of the image.  So if an image is 1024 pixels wide
        # then 1024 should be written to the file.
        output.append(f" fp 7 0 {YDim:d} {XDim:d} {left:g} {bottom:g} {image_width:g} {image_height:g}")
        transform_string = ''.join(output)

        return transform_string

    def NearestFixedPoint(self, points: NDArray[np.floating]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return None

    def NearestWarpedPoint(self, points: NDArray[np.floating]):
        '''Return the fixed points nearest to the query points
        :return: Distance, Index
        '''
        return None

    def Scale(self, scalar):
        '''Scale both warped and control space by scalar'''
        self._points *= scalar
        self.OnTransformChanged()

    def ScaleWarped(self, scalar):
        '''Scale source space control points by scalar'''
        self._points[:, 2:4] = self._points[:, 2:4] * scalar
        self.OnTransformChanged()

    def ScaleFixed(self, scalar):
        '''Scale target space control points by scalar'''
        self._points[:, 0:2] = self._points[:, 0:2] * scalar
        self.OnTransformChanged()

    def TranslateFixed(self, offset: NDArray[np.floating]):
        '''Translate all fixed points by the specified amount'''

        self._points[:, 0:2] = self._points[:, 0:2] + cp.asarray(offset)
        self.OnFixedPointChanged()

    def TranslateWarped(self, offset: NDArray[np.floating]):
        '''Translate all warped points by the specified amount'''
        self._points[:, 2:4] = self._points[:, 2:4] + cp.asarray(offset)
        self.OnWarpedPointChanged()

    def GetPointPairsInRect(self, points: NDArray[np.floating], bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        OutputPoints = None

        bounds = nornir_imageregistration.Rectangle.PrimitiveToRectangle(bounds).ToArray()

        for iPoint in range(0, points.shape[0]):
            y, x = points[iPoint, :]
            if nornir_imageregistration.Rectangle.contains(bounds, (y, x)):
                PointPair = self._points[iPoint, :]
                if OutputPoints is None:
                    OutputPoints = PointPair
                else:
                    OutputPoints = cp.vstack((OutputPoints, PointPair))

        if OutputPoints is not None:
            if OutputPoints.ndim == 1:
                OutputPoints = cp.reshape(OutputPoints, (1, OutputPoints.shape[0]))

        return OutputPoints

    def GetFixedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetWarpedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def GetPointPairsInFixedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetPointPairsInWarpedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def PointPairsToWarpedPoints(self, points: NDArray[np.floating]):
        '''Return the warped points from a set of target-source point pairs'''
        return points[:, 2:4]

    def PointPairsToTargetPoints(self, points: NDArray[np.floating]):
        '''Return the target points from a set of target-source point pairs'''
        return points[:, 0:2]

    @property
    def ForwardInterpolator(self):
        if self._ForwardInterpolator is None:
            self._ForwardInterpolator = cuRegularGridInterpolator(self._grid.axis_points,
                                                                cp.reshape(self.TargetPoints, (
                                                                self._grid.grid_dims[0], self._grid.grid_dims[1], 2)),
                                                                bounds_error=False)

        return self._ForwardInterpolator

    @property
    def InverseInterpolator(self):
        if self._InverseInterpolator is None:
            self._InverseInterpolator = cuRBFInterpolator(self.TargetPoints, self.SourcePoints)

        return self._InverseInterpolator

    def Transform(self, points, **kwargs):
        '''Map points from the warped space to fixed space'''
        transPoints = None

        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)
        transPoints = self.ForwardInterpolator(points)
        return transPoints

    def InverseTransform(self, points, **kwargs):
        '''Map points from the fixed space to the warped space'''
        transPoints = None

        points = nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)
        transPoints = self.InverseInterpolator(points)
        return transPoints

    def RotateTargetPoints(self, rangle: float, rotationCenter: NDArray[np.floating] | None):
        '''Rotate all warped points about a center by a given angle'''
        self._points[:, 0:2] = ControlPointBase_GPUComponent.RotatePoints(self.TargetPoints, rangle, rotationCenter)
        self.OnTransformChanged()

    def UpdateTargetPointsByIndex(self, index: int | NDArray[np.integer], point: NDArray[np.floating]) -> int | NDArray[np.integer]:
        self._points[index, 0:2] = point
        self.OnFixedPointChanged()
        return index

    def UpdateTargetPointsByPosition(self, old_points: NDArray[np.floating], points: NDArray[np.floating]) -> int | NDArray[np.integer]:
        old_points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(old_points)
        points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
        distance, index = self.NearestFixedPoint(old_points)
        return self.UpdateTargetPointsByIndex(index, points)

    def OnFixedPointChanged(self):
        super(GridTransform_GPU, self).OnFixedPointChanged()
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
        super(GridTransform_GPU, self).OnTransformChanged()

    def OnWarpedPointChanged(self):
        raise NotImplementedError("Grid transforms have a fixed grid of points, they should not change")

    def ClearDataStructures(self):
        '''Something about the transform has changed, for example the points.
           Clear out our data structures so we do not use bad data'''
        super(GridTransform_GPU, self).ClearDataStructures()
        self._ForwardInterpolator = None
        self._InverseInterpolator = None
