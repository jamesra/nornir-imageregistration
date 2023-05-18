"""
Created on Oct 22, 2018

@author: u0490822

Contains some helper classes for organizing a grid of points placed over an image.
Used with the slice-to-slice grid refinement code
"""

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration import IGrid
from nornir_imageregistration.transforms.base import ITransform


def build_coords_array(grid_dims: NDArray[int]) -> NDArray[int]:
    """
    Returns a Nx2 array containing each index into the grid of dimension grid_dims
    :param grid_dims:
    :return:
    """
    coords = np.zeros((np.prod(grid_dims), 2), dtype=int)
    col_array = np.array(range(0, grid_dims[1]), dtype=int)
    for iRow in range(grid_dims[0]):
        iRowStart = iRow * grid_dims[1]
        iRowEnd = (iRow + 1) * grid_dims[1]
        coords[iRowStart:iRowEnd, 0] = iRow
        coords[iRowStart:iRowEnd, 1] = col_array

    return coords


class GridDivisionBase(IGrid):
    """Abstract class for structures that divide images into grids of possibly overlapping cells"""

    def __init__(self):
        self._cell_size = None
        self._grid_dims = None
        self._grid_spacing = None
        self._coords = None
        self._TargetPoints = None
        self._SourcePoints = None
        self._source_shape = None
        self._axis_points = None

    @property
    def cell_size(self) -> NDArray[int]:
        return self._cell_size

    @property
    def grid_dims(self) -> NDArray[int]:
        return self._grid_dims

    @property
    def grid_spacing(self) -> NDArray[int]:
        return self._grid_spacing

    @property
    def coords(self) -> NDArray[int]:
        return self._coords

    @property
    def TargetPoints(self) -> NDArray[float]:
        return self._TargetPoints

    @TargetPoints.setter
    def TargetPoints(self, value: NDArray[float]):
        self._TargetPoints = value

    @property
    def SourcePoints(self) -> NDArray[float]:
        return self._SourcePoints

    @property
    def source_shape(self) -> NDArray[int]:
        return self._source_shape

    @property
    def axis_points(self) -> list[NDArray[float]]:
        """The points along the axis, in source space, where the grid lines intersect the axis"""
        return self._axis_points

    def __str__(self):
        return "grid_dims:{0},{1} grid_spacing:{2},{3} cell_size:{4},{5}".format(self._grid_dims[0],
                                                                                 self._grid_dims[1],
                                                                                 self._grid_spacing[0],
                                                                                 self._grid_spacing[1],
                                                                                 self._cell_size[0],
                                                                                 self._cell_size[1])

    @property
    def num_points(self) -> int:
        return self._coords.shape[0]

    @property
    def axis_points(self):
        """The points along the axis, in source space, where the grid lines intersect the axis"""
        return self._axis_points

    def PopulateTargetPoints(self, transform: ITransform):
        if transform is not None:
            self._TargetPoints = np.round(transform.Transform(self._SourcePoints), 3).astype(float, copy=False)
            return self._TargetPoints

    def RemoveMaskedPoints(self, mask: NDArray[bool]):
        """
        :param mask: a boolean mask that determines which points are kept.
        """

        self._coords = self._coords[mask, :]
        self._SourcePoints = self._SourcePoints[mask, :]

        if self._TargetPoints is not None:
            self._TargetPoints = self._TargetPoints[mask, :]

    def ApplyTargetImageMask(self, target_mask: NDArray[bool] | None):
        if target_mask is not None:
            self.FilterOutofBoundsTargetPoints(target_mask.shape)
            valid = nornir_imageregistration.index_with_array(target_mask, self._TargetPoints)

            self.RemoveMaskedPoints(valid)

    def __CalculateMaskedCells(self, mask: NDArray[bool], points: NDArray, min_unmasked_area: float = None):
        """
        :param ndarray mask: mask image used for calculation
        :param ndarray points: set of Nx2 coordinates for cell centers to test for masking
        :param float min_unmasked_area: Amount of cell area that must be valid according to mask.  If None, any cells with a single-unmasked pixel are valid
        """

        if points.shape[0] == 0:
            raise ValueError("points must have non-zero length")

        if min_unmasked_area is None:
            min_unmasked_area = 0

        cell_true_count = np.asarray([False] * points.shape[0], dtype=np.float64)
        half_cell = (self._cell_size / 2.0).astype(np.int32, copy=False)
        cell_area = np.prod(self._cell_size)

        origins = points - half_cell

        for iRow in range(0, points.shape[0]):
            o = origins[iRow, :]

            cell = nornir_imageregistration.CropImage(mask,
                                                      int(o[1]), int(o[0]),
                                                      int(self._cell_size[1]), int(self._cell_size[0]),
                                                      cval=False)
            cell_true_count[iRow] = np.count_nonzero(cell)

        overlaps = cell_true_count / float(cell_area)
        valid = overlaps > min_unmasked_area
        return valid

    def RemoveCellsUsingTargetImageMask(self, target_mask: NDArray[bool], min_unmasked_area: float):
        """
        :param ndarray target_mask: mask image used for calculation
        :param float min_unmasked_area: Amount of cell area that must be valid according to mask
        """
        if target_mask is not None:
            valid = self.__CalculateMaskedCells(mask=target_mask, points=self._TargetPoints,
                                                min_unmasked_area=min_unmasked_area)
            self.RemoveMaskedPoints(valid)

    def ApplySourceImageMask(self, source_mask: NDArray[bool] | None):
        if source_mask is not None:
            self.FilterOutofBoundsSourcePoints(source_mask.shape)
            valid = nornir_imageregistration.index_with_array(source_mask, self._SourcePoints)
            self.RemoveMaskedPoints(valid)

    def RemoveCellsUsingSourceImageMask(self, source_mask: NDArray[bool], min_unmasked_area: float):
        """
        :param ndarray source_mask: mask image used for calculation
        :param float min_unmasked_area: Amount of cell area that must be valid according to mask
        """
        if source_mask is not None:
            valid = self.__CalculateMaskedCells(mask=source_mask, points=self._SourcePoints,
                                                min_unmasked_area=min_unmasked_area)
            self.RemoveMaskedPoints(valid)

    def FilterOutofBoundsTargetPoints(self, target_shape: NDArray | None = None):
        valid_inbounds = np.logical_and(np.all(self._TargetPoints >= np.asarray((0, 0)), 1),
                                        np.all(self._TargetPoints < target_shape, 1))
        self.RemoveMaskedPoints(valid_inbounds)

    def FilterOutofBoundsSourcePoints(self, source_shape: NDArray):
        if source_shape is None:
            source_shape = self._source_shape

        valid_inbounds = np.logical_and(np.all(self._SourcePoints >= np.asarray((0, 0)), 1),
                                        np.all(self._SourcePoints < source_shape, 1))
        self.RemoveMaskedPoints(valid_inbounds)


class ITKGridDivision(GridDivisionBase):
    """
     Align the grid so the centers of the edge cells touch the edge of the image.  This grid should have a cell center
     at each corner of the image
    """

    def __init__(self,
                 source_shape: NDArray[int] | tuple[int, int],
                 cell_size: NDArray[int] | tuple[int, int] | None = None,
                 grid_dims: NDArray[int] | tuple[int, int] | None = None,
                 grid_spacing: NDArray[int] | tuple[int, int] | None = None,
                 transform=None):
        """
        Divides an image into a grid, of possibly overlapping cells.

        :param source_shape: (Rows, Columns) of image we are dividing
        :param cell_size: The dimensions of each grid cell
        :param grid_dims: The number of (Rows, Columns) in the grid
        :param grid_spacing: The distance between the centers of grid cells, possibly allowing overlapping cells

        """
        super(ITKGridDivision, self).__init__()

        source_shape = np.asarray(source_shape, np.int64)

        if cell_size is None:
            if grid_dims is None and grid_spacing is None:
                raise ValueError("cell_size must be specified if grid_dims and grid_spacing are not specified")
        else:
            self._cell_size = np.asarray(cell_size, np.int32)

        if grid_dims is not None and grid_spacing is not None:
            raise ValueError("Either grid_dims or grid_spacing must be specified but not both")

        # We want the coordinates of grid centers to go from edge to edge in the image because ITK expects this behavior
        # Due to this fact we do not guarantee the grid_spacing requested
        if grid_dims is None and grid_spacing is None:
            self._grid_dims = nornir_imageregistration.TileGridShape(source_shape,
                                                                     cell_size) + 1  # Add one because ITK Grid transform centers the boundary points on the edge and not the center
        elif grid_spacing is None:
            self._grid_dims = np.asarray(grid_dims, np.int32)
        elif grid_dims is None:
            self._grid_dims = nornir_imageregistration.TileGridShape(source_shape, grid_spacing) + 1

        if self._cell_size is None:  # Estimate a reasonable cell_size with overlap if it has not been determined, (passed grid dimensions only perhaps)
            self._cell_size = nornir_imageregistration.NearestPowerOfTwo(self._grid_dims)

        # Future Jamie, you spent a lot of time getting the grid spacing calculation correct for some reason.  It should have been obvious but don't mess with it again.
        self._grid_spacing = source_shape / (
                    self._grid_dims - 1)  # - 1 on grid_dims because we want the points at the edges of the image

        self._axis_points = [range(n) * self._grid_spacing[i] for i, n in enumerate(self._grid_dims)]

        self._coords = build_coords_array(self._grid_dims)

        self._SourcePoints = self._coords * self._grid_spacing
        # self.SourcePoints = np.floor(self.SourcePoints).astype(np.int64)

        if self._SourcePoints.shape[0] == 0:
            raise ValueError(
                "No source points generated.  Source Shape: {source_shape} Cell Size: {cell_size} Grid Dims: {grid_dims} Grid Spacing: {grid_spacing}")

        self._source_shape = source_shape

        self._TargetPoints = self.PopulateTargetPoints(transform) if transform is not None else None


class CenteredGridDivision(GridDivisionBase):
    """
    Align the grid so the edges of the edge cells touch the edge of the image
    """

    def __init__(self, source_shape: NDArray[int] | tuple[int, int],
                 cell_size: NDArray[int] | tuple[int, int],
                 grid_dims: NDArray[int] | tuple[int, int] | None = None,
                 grid_spacing: NDArray[int] | tuple[int, int] | None = None,
                 transform=None):
        """
        Divides an image into a grid, of possibly overlapping cells.

        :param source_shape: (Rows, Columns) of image we are dividing
        :param cell_size: The dimensions of each grid cell
        :param grid_dims: The number of (Rows, Columns) in the grid
        :param grid_spacing: The distance between the centers of grid cells, possibly allowing overlapping cells
        """
        super(CenteredGridDivision, self).__init__()

        self._cell_size = np.asarray(cell_size, np.int32)
        source_shape = np.asarray(source_shape, np.int64)

        if cell_size is None:
            if grid_dims is None and grid_spacing is None:
                raise ValueError("cell_size must be specified if grid_dims and grid_spacing are not specified")

        if grid_dims is not None and grid_spacing is not None:
            raise ValueError("Either grid_dims or grid_spacing must be specified but not both")

        if grid_dims is None and grid_spacing is None:
            self._grid_spacing = cell_size
            self._grid_dims = nornir_imageregistration.TileGridShape(source_shape, self._grid_spacing)
        elif grid_spacing is None:
            self._grid_dims = np.asarray(grid_dims, np.int32)
            self._grid_spacing = np.asarray((source_shape - 1) / self._grid_dims, np.int64)
        elif grid_dims is None:
            self._grid_spacing = np.asarray(grid_spacing, np.int64)
            self._grid_dims = nornir_imageregistration.TileGridShape(source_shape, self._grid_spacing)

        self._axis_points = [range(n) * self._grid_spacing[i] for i, n in enumerate(self._grid_dims)]

        self._coords = build_coords_array(self._grid_dims)

        self._SourcePoints = self._coords * self._grid_spacing
        self._SourcePoints = self._SourcePoints + (self._grid_spacing / 2.0)
        # Grid dimensions round up, so if we are larger than image find out by how much and adjust the points so they are centered on the image
        overage = ((self._grid_dims * self._grid_spacing) - source_shape) / 2.0
        # Scale the overage amount according to cell position on the grid so the cells remain on the grid but have uniform additional overlap
        overage_adjustment = overage * (self._coords / np.max(self._coords, 0))
        self._SourcePoints -= overage_adjustment
        # self.SourcePoints = np.round(self.SourcePoints - overage_adjustment).astype(np.int64)

        self._source_shape = source_shape

        if self._SourcePoints.shape[0] == 0:
            raise ValueError(
                "No source points generated.  Source Shape: {source_shape} Cell Size: {cell_size} Grid Dims: {grid_dims} Grid Spacing: {grid_spacing}")

        self._TargetPoints = self.PopulateTargetPoints(transform) if transform is not None else None
