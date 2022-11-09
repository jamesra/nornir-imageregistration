"""
Created on Oct 22, 2018

@author: u0490822

Contains some helper classes for organizing a grid of points placed over an image.
Used with the slice-to-slice grid refinement code
"""

import numpy as np
from numpy.typing import NDArray
import nornir_imageregistration


class GridDivisionBase(object):
    """Abstract class for structures that divide images into grids of possibly overlapping cells"""

    def __init__(self):
        self.cell_size = None
        self.grid_dims = None
        self.grid_spacing = None
        self.coords = None
        self.TargetPoints = None
        self.SourcePoints = None
        self.source_shape = None

    def __str__(self):
        return "grid_dims:{0},{1} grid_spacing:{2},{3} cell_size:{4},{5}".format(self.grid_dims[0],
                                                                                 self.grid_dims[1],
                                                                                 self.grid_spacing[0],
                                                                                 self.grid_spacing[1],
                                                                                 self.cell_size[0],
                                                                                 self.cell_size[1])

    @property
    def num_points(self) -> int:
        return self.coords.shape[0]

    def PopulateTargetPoints(self, transform: nornir_imageregistration.ITransform):
        if transform is not None:
            self.TargetPoints = np.round(transform.Transform(self.SourcePoints), 3).astype(np.float)
            return self.TargetPoints

    def RemoveMaskedPoints(self, mask: NDArray[bool]):
        """
        :param mask: a boolean mask that determines which points are kept.
        """

        self.coords = self.coords[mask, :]
        self.SourcePoints = self.SourcePoints[mask, :]

        if self.TargetPoints is not None:
            self.TargetPoints = self.TargetPoints[mask, :]

    def ApplyTargetImageMask(self, target_mask: NDArray[bool] | None):
        if target_mask is not None:
            self.FilterOutofBoundsTargetPoints(target_mask.shape)
            valid = nornir_imageregistration.index_with_array(target_mask, self.TargetPoints)

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
        half_cell = (self.cell_size / 2.0).astype(np.int32)
        cell_area = np.prod(self.cell_size)

        origins = points - half_cell

        for iRow in range(0, points.shape[0]):
            o = origins[iRow, :]

            cell = nornir_imageregistration.CropImage(mask,
                                                      int(o[1]), int(o[0]),
                                                      int(self.cell_size[1]), int(self.cell_size[0]),
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
            valid = self.__CalculateMaskedCells(mask=target_mask, points=self.TargetPoints,
                                                min_unmasked_area=min_unmasked_area)
            self.RemoveMaskedPoints(valid)

    def ApplySourceImageMask(self, source_mask: NDArray[bool] | None):
        if source_mask is not None:
            self.FilterOutofBoundsSourcePoints(source_mask.shape)
            valid = nornir_imageregistration.index_with_array(source_mask, self.SourcePoints)
            self.RemoveMaskedPoints(valid)

    def RemoveCellsUsingSourceImageMask(self, source_mask: NDArray[bool], min_unmasked_area: float):
        """
        :param ndarray source_mask: mask image used for calculation
        :param float min_unmasked_area: Amount of cell area that must be valid according to mask
        """
        if source_mask is not None:
            valid = self.__CalculateMaskedCells(mask=source_mask, points=self.SourcePoints,
                                                min_unmasked_area=min_unmasked_area)
            self.RemoveMaskedPoints(valid)

    def FilterOutofBoundsTargetPoints(self, target_shape: NDArray | None = None):
        valid_inbounds = np.logical_and(np.all(self.TargetPoints >= np.asarray((0, 0)), 1),
                                        np.all(self.TargetPoints < target_shape, 1))
        self.RemoveMaskedPoints(valid_inbounds)

    def FilterOutofBoundsSourcePoints(self, source_shape: NDArray):
        if source_shape is None:
            source_shape = self.source_shape

        valid_inbounds = np.logical_and(np.all(self.SourcePoints >= np.asarray((0, 0)), 1),
                                        np.all(self.SourcePoints < source_shape, 1))
        self.RemoveMaskedPoints(valid_inbounds)


class ITKGridDivision(GridDivisionBase):
    """
     Align the grid so the centers of the edge cells touch the edge of the image
    """

    def __init__(self,
                 source_shape: NDArray[int] | tuple[int, int],
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
        super(ITKGridDivision, self).__init__()

        self.cell_size = np.asarray(cell_size, np.int32)
        source_shape = np.asarray(source_shape, np.int64)

        if cell_size is None:
            raise ValueError("cell_size must be specified")

        if grid_dims is not None and grid_spacing is not None:
            raise ValueError("Either grid_dims or grid_spacing must be specified but not both")

        # We want the coordinates of grid centers to go from edge to edge in the image because ITK expects this behavior
        # Due to this fact we do not guarantee the grid_spacing requested
        if grid_dims is None and grid_spacing is None:
            self.grid_dims = nornir_imageregistration.TileGridShape(source_shape,
                                                                    cell_size) + 1  # Add one because we center the boundary points on the edge and not the center
        elif grid_spacing is None:
            self.grid_dims = np.asarray(grid_dims, np.int32)
        elif grid_dims is None:
            self.grid_dims = nornir_imageregistration.TileGridShape(source_shape, grid_spacing)

        self.grid_spacing = source_shape / (self.grid_dims - 1)

        self.coords = [np.asarray((iRow, iCol), dtype=np.int64) for iRow in range(self.grid_dims[0]) for iCol in
                       range(self.grid_dims[1])]
        self.coords = np.vstack(self.coords)

        self.SourcePoints = self.coords * self.grid_spacing
        self.SourcePoints = np.floor(self.SourcePoints).astype(np.int64)

        if self.SourcePoints.shape[0] == 0:
            raise ValueError(
                "No source points generated.  Source Shape: {source_shape} Cell Size: {cell_size} Grid Dims: {grid_dims} Grid Spacing: {grid_spacing}")

        self.source_shape = source_shape

        if transform is not None:
            self.TargetPoints = self.PopulateTargetPoints(transform)
        else:
            self.TargetPoints = None


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

        self.cell_size = np.asarray(cell_size, np.int32)
        source_shape = np.asarray(source_shape, np.int64)

        if cell_size is None:
            raise ValueError("cell_size must be specified")

        if grid_dims is not None and grid_spacing is not None:
            raise ValueError("Either grid_dims or grid_spacing must be specified but not both")

        if grid_dims is None and grid_spacing is None:
            self.grid_spacing = cell_size
            self.grid_dims = nornir_imageregistration.TileGridShape(source_shape, self.grid_spacing)
        elif grid_spacing is None:
            self.grid_dims = np.asarray(grid_dims, np.int32)
            self.grid_spacing = np.asarray((source_shape - 1) / self.grid_dims, np.int64)
        elif grid_dims is None:
            self.grid_spacing = np.asarray(grid_spacing, np.int64)
            self.grid_dims = nornir_imageregistration.TileGridShape(source_shape, self.grid_spacing)

        self.coords = [np.asarray((iRow, iCol), dtype=np.int64) for iRow in range(self.grid_dims[0]) for iCol in
                       range(self.grid_dims[1])]
        self.coords = np.vstack(self.coords)

        self.SourcePoints = self.coords * self.grid_spacing
        self.SourcePoints = self.SourcePoints + (self.grid_spacing / 2.0)
        # Grid dimensions round up, so if we are larger than image find out by how much and adjust the points so they are centered on the image
        overage = ((self.grid_dims * self.grid_spacing) - source_shape) / 2.0
        # Scale the overage amount according to cell position on the grid so the cells remain on the grid but have uniform additional overlap
        overage_adjustment = overage * (self.coords / np.max(self.coords, 0))
        self.SourcePoints = np.round(self.SourcePoints - overage_adjustment).astype(np.int64)
        # self.SourcePoints = np.round(self.SourcePoints - overage).astype(np.int64)

        self.source_shape = source_shape

        if self.SourcePoints.shape[0] == 0:
            raise ValueError(
                "No source points generated.  Source Shape: {source_shape} Cell Size: {cell_size} Grid Dims: {grid_dims} Grid Spacing: {grid_spacing}")

        if transform is not None:
            self.TargetPoints = self.PopulateTargetPoints(transform)
        else:
            self.TargetPoints = None
