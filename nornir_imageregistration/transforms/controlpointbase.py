from abc import ABCMeta, abstractmethod
import operator

import numpy as np
try:
    import cupy as cp
    #import cupyx
except ModuleNotFoundError:
    import cupy_thunk as cp
    #import cupyx_thunk as cupyx
except ImportError:
    import cupy_thunk as cp
    #import cupyx_thunk as cupyx
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.transforms import utils
from nornir_imageregistration.transforms.base import IControlPoints, IDiscreteTransform
from nornir_imageregistration.transforms.defaulttransformchangeevents import DefaultTransformChangeEvents


class ControlPointBase(IControlPoints, IDiscreteTransform, DefaultTransformChangeEvents, metaclass=ABCMeta):
    def __init__(self, pointpairs: NDArray[np.floating]):
        super(ControlPointBase, self).__init__()
        self._points = nornir_imageregistration.EnsurePointsAre4xN_NumpyArray(pointpairs, dtype=np.float32)
        self._MappedBoundingBox = None
        self._FixedBoundingBox = None

    def __getstate__(self):
        odict = {'_points': self._points}

        return odict

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary)
        self.OnChangeEventListeners = []
        self.OnTransformChanged()

    @staticmethod
    def FindDuplicates(points: NDArray[np.floating], new_points: NDArray[np.floating]) -> NDArray[np.bool_]:
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

                iPnt += 1

            iNew += 1

        return invalid_indicies

    @staticmethod
    def RemoveDuplicateControlPoints(points: NDArray[np.floating]) -> NDArray[np.floating]:
        '''Returns a copy of the array sorted in fixed space x,y without duplicates'''

        (points, invalid_indicies, valid_indicies) = utils.InvalidIndicies(points)

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

    @classmethod
    def EnsurePointsAre2DNumpyArray(cls, points):
        raise DeprecationWarning('EnsurePointsAre2DNumpyArray should use utility method')
        return nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

    @classmethod
    def EnsurePointsAre4xN_NumpyArray(cls, points):
        raise DeprecationWarning('EnsurePointsAre4xN_NumpyArray should use utility method')
        return nornir_imageregistration.EnsurePointsAre4xN_NumpyArray(points)

    def FindDuplicateFixedPoints(self, new_points, epsilon: float = 0):
        '''Using our control point KDTree, ensure the new points are not duplicates
        :return: An index array of duplicates
        '''
        distance, index = self.FixedKDTree.query(new_points)
        same = distance <= 0
        return same

    def OnTransformChanged(self):
        self.ClearDataStructures()
        super(ControlPointBase, self).OnTransformChanged()

    def GetFixedPointsRect(self, bounds):
        '''bounds = [left bottom right top]'''
        # return self.GetPointPairsInRect(self.TargetPoints, bounds)
        raise DeprecationWarning("This function was a typo, replace with GetFixedPointsInRect")

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

    def GetPointsInFixedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetPointsInWarpedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def GetPointPairsInTargetRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetPointPairsInSourceRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def PointPairsToWarpedPoints(self, points: NDArray[np.floating]):
        '''Return the warped points from a set of target-source point pairs'''
        return points[:, 2:4]

    def PointPairsToTargetPoints(self, points: NDArray[np.floating]):
        '''Return the target points from a set of target-source point pairs'''
        return points[:, 0:2]

    @property
    def MappedBounds(self):
        raise DeprecationWarning("MappedBounds is replaced by MappedBoundingBox")

    @property
    def NumControlPoints(self) -> int:
        if self._points is None:
            return 0

        return self._points.shape[0]

    @property
    def FixedBoundingBox(self):
        '''
        :return: (minY, minX, maxY, maxX)
        '''
        if self._FixedBoundingBox is None:
            self._FixedBoundingBox = nornir_imageregistration.BoundingPrimitiveFromPoints(self.TargetPoints)

        return self._FixedBoundingBox

    @property
    def points(self) -> NDArray[np.floating]:
        return self._points

    @points.setter
    def points(self, val):
        self._points = np.asarray(val, dtype=np.float32)
        self.OnTransformChanged()

    @property
    def FixedBoundingBoxHeight(self):
        raise DeprecationWarning("FixedBoundingBoxHeight is deprecated.  Use FixedBoundingBox.Height instead")
        return self.FixedBoundingBox.Height

    @property
    def MappedBoundingBoxWidth(self):
        raise DeprecationWarning("MappedBoundingBoxWidth is deprecated.  Use MappedBoundingBox.Width instead")
        return self.MappedBoundingBox.Width

    @property
    def SourcePoints(self) -> NDArray[np.floating]:
        ''' [[Y1, X1],
             [Y2, X2],
             [Yn, Xn]]'''
        return self._points[:, 2:4]

    @property
    def MappedBoundingBox(self):
        '''
        :return: (minY, minX, maxY, maxX)
        '''
        if self._MappedBoundingBox is None:
            self._MappedBoundingBox = nornir_imageregistration.BoundingRectangleFromPoints(self.SourcePoints)

        return self._MappedBoundingBox

    @property
    def FixedBoundingBoxWidth(self):
        raise DeprecationWarning("FixedBoundingBoxWidth is deprecated.  Use FixedBoundingBox.Width instead")
        return self.FixedBoundingBox.Width

    @property
    def TargetPoints(self) -> NDArray[np.floating]:
        ''' [[Y1, X1],
             [Y2, X2],
             [Yn, Xn]]'''
        return self._points[:, 0:2]

    @property
    def ControlBounds(self):
        raise DeprecationWarning("ControlBounds is replaced by FixedBoundingBox")

    @abstractmethod
    def OnFixedPointChanged(self):
        self._FixedBoundingBox = None

    @abstractmethod
    def OnWarpedPointChanged(self):
        self._MappedBoundingBox = None

    @abstractmethod
    def ClearDataStructures(self):
        """Something about the transform has changed, for example the points.
        Clear out our data structures so we do not use bad data"""
        self._FixedBoundingBox = None
        self._MappedBoundingBox = None

    @staticmethod
    def RotatePoints(points: NDArray[np.floating], rangle: float, rotationCenter: NDArray[np.floating] | None):
        '''Rotate all points about a center by a given angle'''

        rt = nornir_imageregistration.transforms.Rigid(target_offset=(0, 0),
                                                       source_rotation_center=rotationCenter,
                                                       angle=rangle)
        rotated = rt.Transform(points)
        return rotated
        # temp = points - rotationCenter
        #
        # temp = np.hstack((temp, np.zeros((temp.shape[0], 1))))
        #
        # rmatrix = utils.RotationMatrix(rangle)
        #
        # rotatedtemp = (self.forward_rotation_matrix @ centered_points.T).T
        # rotatedtemp = rotatedtemp[:, 0:2] + rotationCenter
        # return rotatedtemp



class ControlPointBase_GPUComponent(IControlPoints, IDiscreteTransform, DefaultTransformChangeEvents, metaclass=ABCMeta):
    def __init__(self, pointpairs: NDArray[np.floating]):
        super(ControlPointBase_GPUComponent, self).__init__()
        self._points = nornir_imageregistration.EnsurePointsAre4xN_CuPyArray(pointpairs, dtype=np.float32)
        self._MappedBoundingBox = None
        self._FixedBoundingBox = None

    def __getstate__(self):
        odict = {'_points': self._points}

        return odict

    def __setstate__(self, dictionary: dict):
        self.__dict__.update(dictionary)
        self.OnChangeEventListeners = []
        self.OnTransformChanged()
 

    @staticmethod
    def FindDuplicates(points: NDArray[np.floating], new_points: NDArray[np.floating]) -> NDArray[np.bool_]:
        '''Returns a bool array indicating which new_points already exist in points'''

        # (new_points, invalid_indicies, valid_indices) = utils.InvalidIndicies_GPU(new_points)

        round_points = cp.around(points, 3)
        round_new_points = cp.around(new_points, 3)

        sortedpoints = sorted(round_points, key=operator.itemgetter(0, 1))
        sorted_new_points = sorted(round_new_points, key=operator.itemgetter(0, 1))

        numPoints = sortedpoints.shape[0]
        numNew = new_points.shape[0]

        iPnt = 0
        iNew = 0

        invalid_indicies = cp.zeros((1, numNew), dtype=bool)

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

                iPnt += 1

            iNew += 1

        return invalid_indicies

    @staticmethod
    def RemoveDuplicateControlPoints(points: NDArray[np.floating]) -> NDArray[np.floating]:
        '''Returns a copy of the array sorted in fixed space x,y without duplicates'''

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

    @classmethod
    def EnsurePointsAre2DCuPyArray(cls, points):
        raise DeprecationWarning('EnsurePointsAre2DNumpyArray should use utility method')
        return nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)

    @classmethod
    def EnsurePointsAre4xN_CuPyArray(cls, points):
        raise DeprecationWarning('EnsurePointsAre4xN_CuPyArray should use utility method')
        return nornir_imageregistration.EnsurePointsAre4xN_CuPyArray(points)

    def FindDuplicateFixedPoints(self, new_points, epsilon: float = 0):
        '''Using our control point KDTree, ensure the new points are not duplicates
        :return: An index array of duplicates
        '''
        distance, index = self.FixedKDTree.query(new_points)
        same = distance <= 0
        return same

    def OnTransformChanged(self):
        self.ClearDataStructures()
        super(ControlPointBase_GPUComponent, self).OnTransformChanged()

    def GetFixedPointsRect(self, bounds):
        '''bounds = [left bottom right top]'''
        # return self.GetPointPairsInRect(self.TargetPoints, bounds)
        raise DeprecationWarning("This function was a typo, replace with GetFixedPointsInRect")

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

    def GetPointsInFixedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetPointsInWarpedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def GetPointPairsInTargetRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetPointPairsInSourceRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        '''bounds = [bottom left top right]'''
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def PointPairsToWarpedPoints(self, points: NDArray[np.floating]):
        '''Return the warped points from a set of target-source point pairs'''
        return points[:, 2:4]

    def PointPairsToTargetPoints(self, points: NDArray[np.floating]):
        '''Return the target points from a set of target-source point pairs'''
        return points[:, 0:2]

    @property
    def MappedBounds(self):
        raise DeprecationWarning("MappedBounds is replaced by MappedBoundingBox")

    @property
    def NumControlPoints(self) -> int:
        if self._points is None:
            return 0

        return self._points.shape[0]

    @property
    def FixedBoundingBox(self):
        '''
        :return: (minY, minX, maxY, maxX)
        '''
        if self._FixedBoundingBox is None:
            self._FixedBoundingBox = nornir_imageregistration.BoundingPrimitiveFromPoints(self.TargetPoints)

        return self._FixedBoundingBox

    @property
    def points(self) -> NDArray[np.floating]:
        return self._points

    @points.setter
    def points(self, val):
        self._points = cp.asarray(val, dtype=np.float32)
        self.OnTransformChanged()

    @property
    def FixedBoundingBoxHeight(self):
        raise DeprecationWarning("FixedBoundingBoxHeight is deprecated.  Use FixedBoundingBox.Height instead")
        return self.FixedBoundingBox.Height

    @property
    def MappedBoundingBoxWidth(self):
        raise DeprecationWarning("MappedBoundingBoxWidth is deprecated.  Use MappedBoundingBox.Width instead")
        return self.MappedBoundingBox.Width

    @property
    def SourcePoints(self) -> NDArray[np.floating]:
        ''' [[Y1, X1],
             [Y2, X2],
             [Yn, Xn]]'''
        return self._points[:, 2:4]

    @property
    def MappedBoundingBox(self):
        '''
        :return: (minY, minX, maxY, maxX)
        '''
        if self._MappedBoundingBox is None:
            self._MappedBoundingBox = nornir_imageregistration.BoundingPrimitiveFromPoints(self.SourcePoints)

        return self._MappedBoundingBox

    @property
    def FixedBoundingBoxWidth(self):
        raise DeprecationWarning("FixedBoundingBoxWidth is deprecated.  Use FixedBoundingBox.Width instead")
        return self.FixedBoundingBox.Width

    @property
    def TargetPoints(self) -> NDArray[np.floating]:
        ''' [[Y1, X1],
             [Y2, X2],
             [Yn, Xn]]'''
        return self._points[:, 0:2]

    @property
    def ControlBounds(self):
        raise DeprecationWarning("ControlBounds is replaced by FixedBoundingBox")

    @abstractmethod
    def OnFixedPointChanged(self):
        self._FixedBoundingBox = None

    @abstractmethod
    def OnWarpedPointChanged(self):
        self._MappedBoundingBox = None

    @abstractmethod
    def ClearDataStructures(self):
        """Something about the transform has changed, for example the points.
        Clear out our data structures so we do not use bad data"""
        self._FixedBoundingBox = None
        self._MappedBoundingBox = None

    @staticmethod
    def RotatePoints(points, rangle: float, rotationCenter: NDArray[np.floating]):
        '''Rotate all points about a center by a given angle'''

        rt = nornir_imageregistration.transforms.Rigid_GPU(target_offset=(0,0),
                                                       source_rotation_center=rotationCenter,
                                                       angle=rangle)
        rotated = rt.Transform(points)
        return rotated
        # temp = points - rotationCenter
        #
        # temp = np.hstack((temp, np.zeros((temp.shape[0], 1))))
        #
        # rmatrix = utils.RotationMatrix(rangle)
        #
        # rotatedtemp = (self.forward_rotation_matrix @ centered_points.T).T
        # rotatedtemp = rotatedtemp[:, 0:2] + rotationCenter
        # return rotatedtemp