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

from nornir_shared import prettyoutput
from nornir_shared.mathhelper import NearestPowerOfTwo

try:
    import cupy as cp
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp


def build_coords_array(grid_dims: NDArray[np.integer]) -> NDArray[np.integer]:
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

    _cell_size: NDArray[np.integer]
    _grid_dims: NDArray[np.integer]
    _grid_spacing: NDArray[np.integer]
    _coords: NDArray[np.floating]
    _TargetPoints: NDArray[np.floating]
    _SourcePoints: NDArray[np.floating]
    _source_shape: NDArray[np.integer]
    _axis_points: list[NDArray[np.floating]]

    @property
    def cell_size(self) -> NDArray[np.integer]:
        """Area we sample around each point in the grid.  3x3 samples 1.5 pixels in each direction from the center point"""
        return self._cell_size

    @property
    def grid_dims(self) -> NDArray[np.integer]:
        """Shape of the gride in rows, columns"""
        return self._grid_dims

    @property
    def grid_spacing(self) -> NDArray[np.integer]:
        """Spacing of the grid"""
        return self._grid_spacing

    @property
    def coords(self) -> NDArray[np.floating]:
        return self._coords

    @property
    def TargetPoints(self) -> NDArray[np.floating]:
        """Location of center of each cell in target space"""
        return self._TargetPoints

    @TargetPoints.setter
    def TargetPoints(self, value: NDArray[np.floating]):
        if value.shape[0] != self.grid_dims.prod():
            raise ValueError(
                f"Number of points must match grid dimensions, got {value.shape[0]} expected {self.grid_dims.prod()}")
        self._TargetPoints = value

    @property
    def SourcePoints(self) -> NDArray[np.floating]:
        """Location of center of each cell in source space"""
        return self._SourcePoints

    @property
    def source_shape(self) -> NDArray[np.integer]:
        """(Rows, Columns) of image we are dividing"""
        return self._source_shape

    @property
    def num_points(self) -> int:
        return self._coords.shape[0]

    @property
    def axis_points(self):
        """The points along the axis, in source space, where the grid lines intersect the axis"""
        return self._axis_points

    def PopulateTargetPoints(self, transform: ITransform) -> NDArray[np.floating] | None:
        if transform is not None:
            self._TargetPoints = np.round(transform.Transform(self._SourcePoints), 3).astype(np.float32, copy=False)
            if cp.get_array_module(self.TargetPoints) == cp:  # type: ignore[operator]
                self._TargetPoints = self._TargetPoints.get()  # type: ignore[attr-defined]
            return self._TargetPoints
        return None

    @staticmethod
    def _mask_summary(mask: NDArray) -> str:
        """Compact mask diagnostics for operator-facing error messages."""
        mask_host = np.asarray(nornir_imageregistration.EnsureNumpyArray(mask))
        if mask_host.size == 0:
            return "mask_shape=(); mask_true_fraction=n/a"
        true_count = int(np.count_nonzero(mask_host))
        return (
            f"mask_shape={tuple(int(s) for s in mask_host.shape)}; "
            f"mask_true_count={true_count}; "
            f"mask_true_fraction={true_count / float(mask_host.size):.4f}"
        )

    def RemoveMaskedPoints(self, mask: NDArray[np.bool_], *, context: str | None = None,
                           allow_empty: bool = False) -> int:
        """
        Keep only points where ``mask`` is True.

        :param mask: a boolean mask that determines which points are kept.
        :param context: optional operator-facing description of which filter emptied the grid.
        :param allow_empty: when True, emptying the grid clears points and returns 0 instead of raising.
        :return: number of points remaining after filtering.
        """
        keep = np.asarray(mask, dtype=bool).reshape(-1)
        if keep.size != self.num_points:
            raise ValueError(
                f"Point keep-mask length {keep.size} does not match grid point count {self.num_points}")

        if not np.any(keep):
            msg = (
                f"Masking operation removes all {self.num_points} points from grid refinement"
                + (f" ({context})" if context else "")
                + f". cell_size={np.asarray(self.cell_size).tolist()}; "
                f"grid_spacing={np.asarray(self.grid_spacing).tolist()}. "
                "Check that blended tissue masks still have unmasked area under the "
                "control-point cells (extrema masking can erase thin tissue), that the "
                "current transform places points on tissue, and that min_unmasked_area "
                "is not set too high."
            )
            if allow_empty:
                prettyoutput.Log(msg + " Continuing with an empty unfinalized set.")
                empty_ix = np.zeros(0, dtype=bool)
                self._coords = self._coords[empty_ix, :]
                self._SourcePoints = self._SourcePoints[empty_ix, :]
                if self._TargetPoints is not None:
                    self._TargetPoints = self._TargetPoints[empty_ix, :]
                return 0

            prettyoutput.LogErr(msg)
            raise ValueError(msg)

        self._coords = self._coords[keep, :]
        self._SourcePoints = self._SourcePoints[keep, :]

        if self._TargetPoints is not None:
            self._TargetPoints = self._TargetPoints[keep, :]
        return int(self.num_points)

    def ApplyTargetImageMask(self, target_mask: NDArray[np.bool_] | None):
        if target_mask is not None:
            self.FilterOutofBoundsTargetPoints(target_mask.shape)
            valid = nornir_imageregistration.index_with_array(target_mask, self._TargetPoints)

            self.RemoveMaskedPoints(
                valid,
                context=f"target image mask at point centers; {self._mask_summary(target_mask)}")

    def __CalculateMaskedCells(self, mask: NDArray[np.bool_], points: NDArray, min_unmasked_area: float | None = None):
        """
        :param ndarray mask: mask image used for calculation
        :param ndarray points: set of Nx2 coordinates for cell centers to test for masking
        :param float min_unmasked_area: Amount of cell area that must be valid according to mask.  If None, any cells with a single-unmasked pixel are valid
        """
        # Mask cell crops and overlap tests stay on the host: points are small and
        # CropImage / count_nonzero need a shared array module with the keep-mask.
        points_host = np.asarray(nornir_imageregistration.EnsureNumpyArray(points), dtype=np.float64)
        mask_host = np.asarray(nornir_imageregistration.EnsureNumpyArray(mask))

        if points_host.shape[0] == 0:
            raise ValueError("points must have non-zero length")

        if min_unmasked_area is None:
            min_unmasked_area = 0

        cell_true_count = np.zeros(points_host.shape[0], dtype=np.float64)
        half_cell = np.asarray(self._cell_size, dtype=np.float64) / 2.0
        cell_area = float(np.prod(self._cell_size))

        origins = points_host - half_cell
        # #region agent log
        _dbg_point_rows: list[dict] = []
        # #endregion

        for iRow in range(0, points_host.shape[0]):
            o = origins[iRow, :]

            cell = nornir_imageregistration.CropImage(mask_host,
                                                      int(o[1]), int(o[0]),
                                                      int(self._cell_size[1]), int(self._cell_size[0]),
                                                      cval=False)
            cell_true_count[iRow] = float(np.count_nonzero(cell))
            # #region agent log
            cy, cx = float(points_host[iRow, 0]), float(points_host[iRow, 1])
            in_bounds = (0 <= cy < mask_host.shape[0]) and (0 <= cx < mask_host.shape[1])
            center_val = None
            if in_bounds:
                center_val = bool(mask_host[int(cy), int(cx)])
            # swapped-axis probe: if coords were interpreted as (x,y) instead of (y,x)
            swapped_in = (0 <= cx < mask_host.shape[0]) and (0 <= cy < mask_host.shape[1])
            swapped_val = bool(mask_host[int(cx), int(cy)]) if swapped_in else None
            _dbg_point_rows.append({
                "i": iRow,
                "cy": cy, "cx": cx,
                "origin_y": float(o[0]), "origin_x": float(o[1]),
                "overlap": float(cell_true_count[iRow] / cell_area),
                "true_count": float(cell_true_count[iRow]),
                "in_bounds": in_bounds,
                "center_mask": center_val,
                "swapped_in_bounds": swapped_in,
                "swapped_center_mask": swapped_val,
            })
            # #endregion

        overlaps = cell_true_count / cell_area
        valid = overlaps > min_unmasked_area
        # #region agent log
        try:
            import json, time
            with open("/workspace/.cursor/debug-ec0d67.log", "a", encoding="utf-8") as _f:
                _f.write(json.dumps({
                    "sessionId": "ec0d67",
                    "hypothesisId": "A,B,E,F",
                    "location": "grid_subdivision.py:__CalculateMaskedCells",
                    "message": "per-point cell mask overlaps",
                    "data": {
                        "mask_shape": [int(s) for s in mask_host.shape],
                        "min_unmasked_area": float(min_unmasked_area),
                        "cell_size": [int(x) for x in np.asarray(self._cell_size).tolist()],
                        "n_points": int(points_host.shape[0]),
                        "n_valid": int(np.count_nonzero(valid)),
                        "overlap_min": float(np.min(overlaps)),
                        "overlap_max": float(np.max(overlaps)),
                        "overlap_mean": float(np.mean(overlaps)),
                        "points": _dbg_point_rows,
                    },
                    "timestamp": int(time.time() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion
        return valid

    def RemoveCellsUsingTargetImageMask(self, target_mask: NDArray[np.bool_], min_unmasked_area: float,
                                        *, allow_empty: bool = False) -> int:
        """
        :param ndarray target_mask: mask image used for calculation
        :param float min_unmasked_area: Amount of cell area that must be valid according to mask
        :param allow_empty: when True, emptying the grid is allowed (returns 0)
        :return: number of points remaining after filtering (unchanged if mask is None)
        """
        if target_mask is not None:
            points_before = self.num_points
            valid = self.__CalculateMaskedCells(mask=target_mask, points=self._TargetPoints,
                                                min_unmasked_area=min_unmasked_area)
            return self.RemoveMaskedPoints(
                valid,
                allow_empty=allow_empty,
                context=(
                    f"target cell tissue mask; points_before={points_before}; "
                    f"min_unmasked_area={float(min_unmasked_area):g}; "
                    f"{self._mask_summary(target_mask)}"
                ))
        return int(self.num_points)

    def ApplySourceImageMask(self, source_mask: NDArray[np.bool_] | None):
        if source_mask is not None:
            self.FilterOutofBoundsSourcePoints(source_mask.shape)
            valid = nornir_imageregistration.index_with_array(source_mask, self._SourcePoints)
            self.RemoveMaskedPoints(
                valid,
                context=f"source image mask at point centers; {self._mask_summary(source_mask)}")

    def RemoveCellsUsingSourceImageMask(self, source_mask: NDArray[np.bool_], min_unmasked_area: float,
                                        *, allow_empty: bool = False) -> int:
        """
        :param ndarray source_mask: mask image used for calculation
        :param float min_unmasked_area: Amount of cell area that must be valid according to mask
        :param allow_empty: when True, emptying the grid is allowed (returns 0)
        :return: number of points remaining after filtering (unchanged if mask is None)
        """
        if source_mask is not None:
            points_before = self.num_points
            valid = self.__CalculateMaskedCells(mask=source_mask, points=self._SourcePoints,
                                                min_unmasked_area=min_unmasked_area)
            return self.RemoveMaskedPoints(
                valid,
                allow_empty=allow_empty,
                context=(
                    f"source cell tissue mask; points_before={points_before}; "
                    f"min_unmasked_area={float(min_unmasked_area):g}; "
                    f"{self._mask_summary(source_mask)}"
                ))
        return int(self.num_points)

    def FilterOutofBoundsTargetPoints(self, target_shape: NDArray[np.integer] | tuple[int, int] | None = None,
                                      *, allow_empty: bool = False) -> int:

        xp = nornir_imageregistration.GetComputationModule() if target_shape is None else cp.get_array_module(
            target_shape)  # type: ignore[arg-type]

        if not isinstance(target_shape, np.ndarray):
            target_shape = xp.asarray(target_shape)

        points_before = self.num_points
        valid_inbounds = xp.logical_and(xp.all(self._TargetPoints >= xp.asarray((0, 0)), 1),
                                        xp.all(self._TargetPoints < target_shape, 1))  # type: ignore[operator]
        return self.RemoveMaskedPoints(
            np.asarray(nornir_imageregistration.EnsureNumpyArray(valid_inbounds), dtype=bool),
            allow_empty=allow_empty,
            context=(
                f"target out-of-bounds filter; points_before={points_before}; "
                f"target_shape={tuple(int(s) for s in np.asarray(nornir_imageregistration.EnsureNumpyArray(target_shape)).tolist())}"
            ))

    def FilterOutofBoundsSourcePoints(self, source_shape: NDArray | tuple[int, int] | None = None,
                                      *, allow_empty: bool = False) -> int:
        xp = nornir_imageregistration.GetComputationModule() if source_shape is None else cp.get_array_module(
            source_shape)  # type: ignore[arg-type]

        if source_shape is None:
            source_shape = xp.asarray(self._source_shape)
        elif not isinstance(source_shape, np.ndarray):
            source_shape = xp.asarray(source_shape)

        points_before = self.num_points
        valid_inbounds = xp.logical_and(xp.all(self._SourcePoints >= xp.asarray((0, 0)), 1),
                                        xp.all(self._SourcePoints < source_shape, 1))  # type: ignore[operator]
        return self.RemoveMaskedPoints(
            np.asarray(nornir_imageregistration.EnsureNumpyArray(valid_inbounds), dtype=bool),
            allow_empty=allow_empty,
            context=(
                f"source out-of-bounds filter; points_before={points_before}; "
                f"source_shape={tuple(int(s) for s in np.asarray(nornir_imageregistration.EnsureNumpyArray(source_shape)).tolist())}"
            ))

    def __str__(self):
        return f"grid_dims:{self._grid_dims[0]},{self._grid_dims[1]} grid_spacing:{self._grid_spacing[0]},{self._grid_spacing[1]} cell_size:{self._cell_size[0]},{self._cell_size[1]}"


class ITKGridDivision(GridDivisionBase):
    """
     Align the grid so the centers of the edge cells touch the edge of the image.  This grid should have a cell center
     at each corner of the image
    """

    def __init__(self,
                 source_shape: NDArray[np.integer] | tuple[int, int],
                 cell_size: NDArray[np.integer] | tuple[int, int] | None = None,
                 grid_dims: NDArray[np.integer] | tuple[int, int] | None = None,
                 grid_spacing: NDArray[np.integer] | tuple[int, int] | None = None,
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
                                                                     cell_size) + 1  # type: ignore[arg-type]  # Add one because ITK Grid transform centers the boundary points on the edge and not the center
        elif grid_spacing is None:
            self._grid_dims = np.asarray(grid_dims, np.int32)
        elif grid_dims is None:
            self._grid_dims = nornir_imageregistration.TileGridShape(source_shape, grid_spacing) + 1

        if cell_size is None:  # Estimate a reasonable cell_size with overlap if it has not been determined, (passed grid dimensions only perhaps)
            self._cell_size = NearestPowerOfTwo(self._grid_dims)

        # Future Jamie, you spent a lot of time getting the grid spacing calculation correct for some reason.  It should have been obvious but don't mess with it again.
        self._grid_spacing = source_shape / (  # type: ignore[assignment]
                self._grid_dims - 1)  # - 1 on grid_dims because we want the points at the edges of the image

        self._axis_points = [range(n) * self._grid_spacing[i] for i, n in enumerate(self._grid_dims)]

        self._coords = build_coords_array(self._grid_dims)  # type: ignore[assignment]

        self._SourcePoints = self._coords * self._grid_spacing  # type: ignore[assignment]
        # self.SourcePoints = np.floor(self.SourcePoints).astype(np.int64)

        if self._SourcePoints.shape[0] == 0:
            raise ValueError(
                "No source points generated.  Source Shape: {source_shape} Cell Size: {cell_size} Grid Dims: {grid_dims} Grid Spacing: {grid_spacing}")

        self._source_shape = source_shape

        self._TargetPoints = self.PopulateTargetPoints(transform) if transform is not None else None  # type: ignore[assignment]


class CenteredGridDivision(GridDivisionBase):
    """
    Align the grid so the edges of the edge cells touch the edge of the image
    """

    def __init__(self, source_shape: NDArray[np.integer] | tuple[int, int],
                 cell_size: NDArray[np.integer] | tuple[int, int],
                 grid_dims: NDArray[np.integer] | tuple[int, int] | None = None,
                 grid_spacing: NDArray[np.integer] | tuple[int, int] | None = None,
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
            self._grid_spacing = cell_size  # type: ignore[assignment]
            self._grid_dims = nornir_imageregistration.TileGridShape(source_shape, self._grid_spacing)
        elif grid_spacing is None:
            self._grid_dims = np.asarray(grid_dims, np.int32)
            self._grid_spacing = np.asarray((source_shape - 1) / self._grid_dims, np.int64)  # type: ignore[assignment]
        elif grid_dims is None:
            self._grid_spacing = np.asarray(grid_spacing, np.int64)
            self._grid_dims = nornir_imageregistration.TileGridShape(source_shape, self._grid_spacing)

        self._axis_points = [range(n) * self._grid_spacing[i] for i, n in enumerate(self._grid_dims)]

        self._coords = build_coords_array(self._grid_dims)  # type: ignore[assignment]

        self._SourcePoints = self._coords * self._grid_spacing  # type: ignore[assignment]
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

        # self._SourcePoints = cp.asarray(self._SourcePoints) if nornir_imageregistration.UsingCupy() else self._SourcePoints
        self._TargetPoints = self.PopulateTargetPoints(transform) if transform is not None else None  # type: ignore[assignment]
