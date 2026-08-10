from abc import ABCMeta, abstractmethod

import numpy as np
from typing import Any

try:
    import cupy as cp
    # import cupyx
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
    # import nornir_imageregistration.cupyx_thunk as cupyx
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
    # import nornir_imageregistration.cupyx_thunk as cupyx
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.spatial_distance import array_to_numpy_host
from nornir_imageregistration.transforms import utils
from nornir_imageregistration.transforms.base import IControlPoints, IDiscreteTransform, ITransformFlip
from nornir_imageregistration.transforms.defaulttransformchangeevents import DefaultTransformChangeEvents


def _as_bool_scalar(value: Any) -> bool:
    """Convert a 0-d NumPy/CuPy boolean result to a Python bool."""
    item = getattr(value, "item", None)
    if callable(item):
        return bool(item())
    return bool(value)


def GroupControlPointIndicesByPosition(
        points: NDArray[np.floating],
        *,
        decimals: int = 3,
) -> list[list[int]]:
    """Group row indices that share the same rounded fixed-space (y, x).

    Returns a list of lists. Each child list is the indices of points at one
    position (first-seen order of positions; indices within a group in ascending
    input order). Singleton positions are included as length-1 lists.

    Rounding and uniqueness run on the input array backend (``xp``). Only the
    compact inverse/index vectors are brought to the host to assemble Python lists.
    """
    xp = cp.get_array_module(points)
    pts = xp.asarray(points)
    if pts.ndim == 1:
        pts = xp.atleast_2d(pts)
    if pts.shape[0] == 0:
        return []

    rounded = xp.around(pts[:, 0:2], decimals)
    _unique, first_idx, inverse = xp.unique(
        rounded, axis=0, return_index=True, return_inverse=True
    )

    # Host assembly of list[list[int]]; transfer only int index vectors.
    first_idx_h = array_to_numpy_host(first_idx)
    inverse_h = array_to_numpy_host(inverse)
    appearance_order = np.argsort(first_idx_h)
    label_to_group = np.empty(appearance_order.shape[0], dtype=np.intp)
    label_to_group[appearance_order] = np.arange(appearance_order.shape[0], dtype=np.intp)

    groups: list[list[int]] = [[] for _ in range(appearance_order.shape[0])]
    for i, label in enumerate(inverse_h.tolist()):
        groups[int(label_to_group[int(label)])].append(i)
    return groups


def ControlPointsHaveDuplicatePositions(
        points: NDArray[np.floating],
        *,
        decimals: int = 3,
) -> bool:
    """Return True if any rounded fixed-space (y, x) position appears more than once.

    Stays on the input backend; faster than building index groups when only a
    yes/no answer is needed (e.g. RBF ``CreateBetaMatrix``).
    """
    xp = cp.get_array_module(points)
    pts = xp.asarray(points)
    if pts.ndim == 1:
        pts = xp.atleast_2d(pts)
    if pts.shape[0] <= 1:
        return False

    rounded = xp.around(pts[:, 0:2], decimals)
    _unique, counts = xp.unique(rounded, axis=0, return_counts=True)
    return _as_bool_scalar(xp.any(counts > 1))


class ControlPointBase(IControlPoints, IDiscreteTransform, ITransformFlip, DefaultTransformChangeEvents,
                       metaclass=ABCMeta):
    def __init__(self, pointpairs: NDArray[np.floating]):
        """
        :param pointpairs: [TargetY TargetX SourceY SourceX]
        """
        super(ControlPointBase, self).__init__()
        self._points = nornir_imageregistration.EnsurePointsAre4xN_NumpyArray(pointpairs, dtype=np.float32)
        self._SourceBoundingBox = None
        self._TargetBoundingBox = None

    def __getstate__(self):
        odict = {'_points': self._points}
        return odict

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary)  # type: ignore[attr-defined]
        self.OnChangeEventListeners = []
        self.OnTransformChanged()

    @staticmethod
    def FindDuplicates(points: NDArray[np.floating]) -> list[list[int]]:
        """Return index groups of duplicate fixed-space (y, x) positions.

        Each child list contains the indices of points that share one rounded
        position (3 decimals). Only groups with two or more indices are returned.
        """
        return [group for group in GroupControlPointIndicesByPosition(points) if len(group) > 1]

    @staticmethod
    def RemoveDuplicateControlPoints(points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return a copy of *points* without duplicate fixed-space (y, x) coordinates.

        First occurrence order is preserved. Coordinates are rounded to 3 decimals
        before comparison.
        """
        (points, _invalid_indices, _valid_indices) = utils.InvalidIndices(points)
        if points.shape[0] == 0:
            return points.copy()

        groups = GroupControlPointIndicesByPosition(points)
        keep_idx = [group[0] for group in groups]
        xp = cp.get_array_module(points)
        return xp.asarray(points)[xp.asarray(keep_idx, dtype=xp.intp)]

    @classmethod
    def EnsurePointsAre2DNumpyArray(cls, points):
        raise DeprecationWarning('EnsurePointsAre2DNumpyArray should use utility method')
        return nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)

    @classmethod
    def EnsurePointsAre4xN_NumpyArray(cls, points):
        raise DeprecationWarning('EnsurePointsAre4xN_NumpyArray should use utility method')
        return nornir_imageregistration.EnsurePointsAre4xN_NumpyArray(points)

    def FindDuplicateFixedPoints(self, new_points, epsilon: float = 0):
        """Return a boolean mask of *new_points* already present as fixed points.

        Points whose FixedKDTree distance is ``<= epsilon`` are duplicates.
        """
        distance, index = self.FixedKDTree.query(new_points)  # type: ignore[attr-defined]
        same = distance <= epsilon
        getter = getattr(same, "get", None)
        same_np = np.atleast_1d(np.asarray(getter()) if callable(getter) else np.asarray(same))
        return same_np.astype(bool, copy=False)

    def OnTransformChanged(self):
        try:
            from nornir_imageregistration import interactive_edit
            if interactive_edit.in_progress():
                super(ControlPointBase, self).OnTransformChanged()
                return
        except ImportError:
            pass
        self.ClearDataStructures()
        super(ControlPointBase, self).OnTransformChanged()

    def GetPointPairsInRect(self, points: NDArray[np.floating],
                            bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
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
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetWarpedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def GetPointsInFixedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetPointsInWarpedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def GetPointPairsInTargetRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetPointPairsInSourceRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def PointPairsToWarpedPoints(self, points: NDArray[np.floating]):
        """Return the warped points from a set of target-source point pairs"""
        return points[:, 2:4]

    def PointPairsToTargetPoints(self, points: NDArray[np.floating]):
        """Return the target points from a set of target-source point pairs"""
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
        """
        :return: (minY, minX, maxY, maxX)
        """
        return self.TargetBoundingBox

    @property
    def TargetBoundingBox(self):
        """
        :return: (minY, minX, maxY, maxX)
        """
        if self._TargetBoundingBox is None:
            self._TargetBoundingBox = nornir_imageregistration.BoundingPrimitiveFromPoints(self.TargetPoints)

        return self._TargetBoundingBox

    @property
    def points(self) -> NDArray[np.floating]:
        return self._points

    @points.setter
    def points(self, val):
        self._points = nornir_imageregistration.EnsurePointsAre4xN_NumpyArray(val, dtype=np.float32)
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
        """ [[Y1, X1],
             [Y2, X2],
             [Yn, Xn]]"""
        return self._points[:, 2:4]

    @property
    def MappedBoundingBox(self):
        """
        :return: (minY, minX, maxY, maxX)
        """
        return self.SourceBoundingBox

    @property
    def SourceBoundingBox(self):
        """
        :return: (minY, minX, maxY, maxX)
        """
        if self._SourceBoundingBox is None:
            self._SourceBoundingBox = nornir_imageregistration.spatial.BoundingRectangleFromPoints(self.SourcePoints)

        return self._SourceBoundingBox

    @property
    def FixedBoundingBoxWidth(self):
        raise DeprecationWarning("FixedBoundingBoxWidth is deprecated.  Use FixedBoundingBox.Width instead")
        return self.FixedBoundingBox.Width

    @property
    def TargetPoints(self) -> NDArray[np.floating]:
        """ [[Y1, X1],
             [Y2, X2],
             [Yn, Xn]]"""
        return self._points[:, 0:2]

    @property
    def ControlBounds(self):
        raise DeprecationWarning("ControlBounds is replaced by FixedBoundingBox")

    @abstractmethod
    def OnFixedPointChanged(self):
        self._TargetBoundingBox = None

    @abstractmethod
    def OnWarpedPointChanged(self):
        self._SourceBoundingBox = None

    @abstractmethod
    def ClearDataStructures(self):
        """Something about the transform has changed, for example the points.
        Clear out our data structures so we do not use bad data"""
        self._TargetBoundingBox = None
        self._SourceBoundingBox = None

    @staticmethod
    def RotatePoints(points: NDArray[np.floating], rangle: float, rotationCenter: NDArray[np.floating] | None):
        """Rotate all points about a center by a given angle"""

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

    def Flip(self):
        """Flip target and source X about each space's vertical midline."""
        target = np.asarray(self.TargetPoints, dtype=np.float32).copy()
        source = np.asarray(self.SourcePoints, dtype=np.float32).copy()

        target_center = (target.min(axis=0) + target.max(axis=0)) / 2.0
        source_center = (source.min(axis=0) + source.max(axis=0)) / 2.0

        target[:, 1] = -target[:, 1] + (2.0 * target_center[1])
        source[:, 1] = -source[:, 1] + (2.0 * source_center[1])

        self.points[:, 0:2] = target
        self.points[:, 2:4] = source
        self.OnTransformChanged()


class ControlPointBase_GPUComponent(IControlPoints, IDiscreteTransform, DefaultTransformChangeEvents,
                                    metaclass=ABCMeta):
    def __init__(self, pointpairs: NDArray[np.floating]):
        super(ControlPointBase_GPUComponent, self).__init__()
        self._points = nornir_imageregistration.EnsurePointsAre4xN_CuPyArray(pointpairs, dtype=np.float32)
        self._MappedBoundingBox = None
        self._FixedBoundingBox = None

    def __getstate__(self) -> dict[str, Any]:
        odict = {'_points': self._points} 
        return odict

    def __setstate__(self, dictionary: dict):
        self.__dict__.update(dictionary)  # type: ignore[attr-defined]
        self.OnChangeEventListeners = []
        self.OnTransformChanged()

    @staticmethod
    def FindDuplicates(points: NDArray[np.floating]) -> list[list[int]]:
        """Return index groups of duplicate fixed-space (y, x) positions.

        Each child list contains the indices of points that share one rounded
        position (3 decimals). Only groups with two or more indices are returned.
        """
        return [group for group in GroupControlPointIndicesByPosition(points) if len(group) > 1]

    @staticmethod
    def RemoveDuplicateControlPoints(points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return a copy of *points* without duplicate fixed-space (y, x) coordinates.

        First occurrence order is preserved. Coordinates are rounded to 3 decimals
        before comparison.
        """
        (points, _invalid_indices, _valid_indices) = utils.InvalidIndices(points)
        if points.shape[0] == 0:
            return points.copy()

        groups = GroupControlPointIndicesByPosition(points)
        keep_idx = [group[0] for group in groups]
        xp = cp.get_array_module(points)
        return xp.asarray(points)[xp.asarray(keep_idx, dtype=xp.intp)]

    @classmethod
    def EnsurePointsAre2DCuPyArray(cls, points):
        raise DeprecationWarning('EnsurePointsAre2DNumpyArray should use utility method')
        return nornir_imageregistration.EnsurePointsAre2DCuPyArray(points)

    @classmethod
    def EnsurePointsAre4xN_CuPyArray(cls, points):
        raise DeprecationWarning('EnsurePointsAre4xN_CuPyArray should use utility method')
        return nornir_imageregistration.EnsurePointsAre4xN_CuPyArray(points)

    def FindDuplicateFixedPoints(self, new_points, epsilon: float = 0):
        """Return a boolean mask of *new_points* already present as fixed points.

        Points whose FixedKDTree distance is ``<= epsilon`` are duplicates.
        """
        distance, index = self.FixedKDTree.query(new_points)  # type: ignore[attr-defined]
        same = distance <= epsilon
        getter = getattr(same, "get", None)
        same_np = np.atleast_1d(np.asarray(getter()) if callable(getter) else np.asarray(same))
        return same_np.astype(bool, copy=False)

    def OnTransformChanged(self):
        try:
            from nornir_imageregistration import interactive_edit
            if interactive_edit.in_progress():
                super(ControlPointBase_GPUComponent, self).OnTransformChanged()
                return
        except ImportError:
            pass
        self.ClearDataStructures()
        super(ControlPointBase_GPUComponent, self).OnTransformChanged()

    def GetFixedPointsRect(self, bounds):
        """bounds = [left bottom right top]"""
        # return self.GetPointPairsInRect(self.TargetPoints, bounds)
        raise DeprecationWarning("This function was a typo, replace with GetFixedPointsInRect")

    def GetPointPairsInRect(self, points: NDArray[np.floating],
                            bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
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
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetWarpedPointsInRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def GetPointsInFixedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetPointsInWarpedRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def GetPointPairsInTargetRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.TargetPoints, bounds)

    def GetPointPairsInSourceRect(self, bounds: nornir_imageregistration.Rectangle | NDArray[np.floating]):
        """bounds = [bottom left top right]"""
        return self.GetPointPairsInRect(self.SourcePoints, bounds)

    def PointPairsToWarpedPoints(self, points: NDArray[np.floating]):
        """Return the warped points from a set of target-source point pairs"""
        return points[:, 2:4]

    def PointPairsToTargetPoints(self, points: NDArray[np.floating]):
        """Return the target points from a set of target-source point pairs"""
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
        """
        :return: (minY, minX, maxY, maxX)
        """
        if self._FixedBoundingBox is None:
            self._FixedBoundingBox = nornir_imageregistration.BoundingPrimitiveFromPoints(self.TargetPoints)

        return self._FixedBoundingBox

    @property
    def TargetBoundingBox(self):
        """Bounding box of target-space control points (alias of FixedBoundingBox)."""
        return self.FixedBoundingBox

    @property
    def points(self) -> NDArray[np.floating]:
        return self._points

    @points.setter
    def points(self, val):
        self._points = nornir_imageregistration.EnsurePointsAre4xN_CuPyArray(val, dtype=np.float32)
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
        """ [[Y1, X1],
             [Y2, X2],
             [Yn, Xn]]"""
        return self._points[:, 2:4]

    @property
    def MappedBoundingBox(self):
        """
        :return: (minY, minX, maxY, maxX)
        """
        if self._MappedBoundingBox is None:
            self._MappedBoundingBox = nornir_imageregistration.BoundingPrimitiveFromPoints(self.SourcePoints)

        return self._MappedBoundingBox

    @property
    def SourceBoundingBox(self):
        """Bounding box of source-space control points (alias of MappedBoundingBox)."""
        return self.MappedBoundingBox

    @property
    def FixedBoundingBoxWidth(self):
        raise DeprecationWarning("FixedBoundingBoxWidth is deprecated.  Use FixedBoundingBox.Width instead")
        return self.FixedBoundingBox.Width

    @property
    def TargetPoints(self) -> NDArray[np.floating]:
        """ [[Y1, X1],
             [Y2, X2],
             [Yn, Xn]]"""
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
        """Rotate all points about a center by a given angle"""

        rt = nornir_imageregistration.transforms.Rigid_GPU(target_offset=(0, 0),  # type: ignore[attr-defined]
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

    def Flip(self):
        """Flip target and source X about each space's vertical midline."""
        xp = cp.get_array_module(self._points)
        target = xp.asarray(self.TargetPoints, dtype=xp.float32).copy()
        source = xp.asarray(self.SourcePoints, dtype=xp.float32).copy()

        target_center = (target.min(axis=0) + target.max(axis=0)) / 2.0
        source_center = (source.min(axis=0) + source.max(axis=0)) / 2.0

        target[:, 1] = -target[:, 1] + (2.0 * target_center[1])
        source[:, 1] = -source[:, 1] + (2.0 * source_center[1])

        self.points[:, 0:2] = target
        self.points[:, 2:4] = source
        self.OnTransformChanged()
