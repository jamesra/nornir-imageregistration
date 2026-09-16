"""Spring-layout relaxation for masked grid control points.

Builds a 4-neighbor mosaic :class:`~nornir_imageregistration.layout.Layout` over
grid TargetPoints, pins well-masked nodes, and propagates position updates to
masked (free) neighbors with a visit-once BFS after a seed node moves.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.igrid import IGrid
from nornir_imageregistration.layout import Layout
from nornir_imageregistration.grid_subdivision import (
    classify_fixed_grid_points,
)


def _index_to_row_col(index: int, cols: int) -> tuple[int, int]:
    return int(index // cols), int(index % cols)


def _row_col_to_index(row: int, col: int, cols: int) -> int:
    return int(row * cols + col)


def four_adjacent_indices(index: int, rows: int, cols: int) -> list[int]:
    """Return 4-neighbor control-point indices for a flat row-major grid index."""
    row, col = _index_to_row_col(index, cols)
    neighbors: list[int] = []
    if col + 1 < cols:
        neighbors.append(_row_col_to_index(row, col + 1, cols))
    if col - 1 >= 0:
        neighbors.append(_row_col_to_index(row, col - 1, cols))
    if row + 1 < rows:
        neighbors.append(_row_col_to_index(row + 1, col, cols))
    if row - 1 >= 0:
        neighbors.append(_row_col_to_index(row - 1, col, cols))
    return neighbors


def create_grid_spacing_layout(
        target_points: NDArray[np.floating],
        grid_dims: NDArray[np.integer] | Sequence[int],
        grid_spacing: NDArray[np.floating] | Sequence[float],
) -> Layout:
    """Create a 4-neighbor Layout with axis-aligned rest offsets from *grid_spacing*.

    Rest offsets are replaced by :func:`set_local_similarity_rest_offsets` before
    a masked propagation wave.
    """
    points = np.asarray(target_points, dtype=np.float64)
    rows = int(grid_dims[0])
    cols = int(grid_dims[1])
    expected = rows * cols
    if points.shape[0] != expected:
        raise ValueError(
            f"target_points length {points.shape[0]} does not match grid_dims {rows}x{cols}")

    spacing = np.asarray(grid_spacing, dtype=np.float64).reshape(2)
    layout = Layout()
    for i in range(expected):
        layout.CreateNode(i, points[i].copy())

    for row in range(rows):
        for col in range(cols):
            i = _row_col_to_index(row, col, cols)
            if col + 1 < cols:
                j = _row_col_to_index(row, col + 1, cols)
                layout.SetOffset(i, j, offset=np.array([0.0, float(spacing[1])]), weight=1.0)
            if row + 1 < rows:
                j = _row_col_to_index(row + 1, col, cols)
                layout.SetOffset(i, j, offset=np.array([float(spacing[0]), 0.0]), weight=1.0)
    return layout


def sync_layout_positions(layout: Layout, target_points: NDArray[np.floating]) -> None:
    """Copy *target_points* into every Layout node Position."""
    points = np.asarray(target_points, dtype=np.float64)
    for node_id, node in layout.nodes.items():
        node.Position = points[int(node_id)].copy()


def set_local_similarity_rest_offsets(
        layout: Layout,
        target_points: NDArray[np.floating],
        fixed_mask: NDArray[np.bool_],
        grid_dims: NDArray[np.integer] | Sequence[int],
        grid_spacing: NDArray[np.floating] | Sequence[float],
) -> None:
    """Set spring rest offsets from nearby fixed–fixed pairs of the same lattice direction.

    For each right/down edge:
    - both endpoints fixed → rest = current Target delta (zero tension on anchors)
    - otherwise → nearest same-direction fixed–fixed vector (lattice distance from edge midpoint)
    - no fixed–fixed sample → axis-aligned *grid_spacing*
    """
    points = np.asarray(target_points, dtype=np.float64)
    fixed = np.asarray(fixed_mask, dtype=bool)
    rows = int(grid_dims[0])
    cols = int(grid_dims[1])
    spacing = np.asarray(grid_spacing, dtype=np.float64).reshape(2)

    right_samples: list[tuple[float, float, NDArray[np.floating]]] = []
    down_samples: list[tuple[float, float, NDArray[np.floating]]] = []

    for row in range(rows):
        for col in range(cols):
            i = _row_col_to_index(row, col, cols)
            if col + 1 < cols:
                j = _row_col_to_index(row, col + 1, cols)
                if fixed[i] and fixed[j]:
                    mid_r = row + 0.0
                    mid_c = col + 0.5
                    right_samples.append((mid_r, mid_c, points[j] - points[i]))
            if row + 1 < rows:
                j = _row_col_to_index(row + 1, col, cols)
                if fixed[i] and fixed[j]:
                    mid_r = row + 0.5
                    mid_c = col + 0.0
                    down_samples.append((mid_r, mid_c, points[j] - points[i]))

    def _nearest(
            samples: list[tuple[float, float, NDArray[np.floating]]],
            mid_r: float,
            mid_c: float,
            fallback: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        if not samples:
            return fallback
        best_vec = samples[0][2]
        best_d2 = float('inf')
        for sr, sc, vec in samples:
            d2 = (sr - mid_r) ** 2 + (sc - mid_c) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_vec = vec
        return np.asarray(best_vec, dtype=np.float64)

    right_fallback = np.array([0.0, float(spacing[1])], dtype=np.float64)
    down_fallback = np.array([float(spacing[0]), 0.0], dtype=np.float64)

    for row in range(rows):
        for col in range(cols):
            i = _row_col_to_index(row, col, cols)
            if col + 1 < cols:
                j = _row_col_to_index(row, col + 1, cols)
                if fixed[i] and fixed[j]:
                    rest = points[j] - points[i]
                else:
                    rest = _nearest(right_samples, row + 0.0, col + 0.5, right_fallback)
                layout.SetOffset(i, j, offset=np.asarray(rest, dtype=np.float64), weight=1.0)
            if row + 1 < rows:
                j = _row_col_to_index(row + 1, col, cols)
                if fixed[i] and fixed[j]:
                    rest = points[j] - points[i]
                else:
                    rest = _nearest(down_samples, row + 0.5, col + 0.0, down_fallback)
                layout.SetOffset(i, j, offset=np.asarray(rest, dtype=np.float64), weight=1.0)


def propagate_masked_grid_positions(
        target_points: NDArray[np.floating],
        fixed_mask: NDArray[np.bool_],
        seed_indices: Iterable[int],
        grid_dims: NDArray[np.integer] | Sequence[int],
        grid_spacing: NDArray[np.floating] | Sequence[float],
        layout: Layout | None = None,
) -> NDArray[np.floating]:
    """BFS-relax free neighbors of *seed_indices* once each; return new TargetPoints.

    Fixed nodes never move. Free nodes with no fixed neighbor in the connected
    component are left unchanged when they have no spring pull from a moved seed
    path (they may still move if an upstream free node moved). Nodes that are
    free but never reached from a seed are unchanged.
    """
    points = np.asarray(target_points, dtype=np.float64).copy()
    fixed = np.asarray(fixed_mask, dtype=bool)
    rows = int(grid_dims[0])
    cols = int(grid_dims[1])
    n = rows * cols
    if points.shape[0] != n or fixed.shape[0] != n:
        raise ValueError("target_points and fixed_mask must match grid_dims product")

    if not np.any(~fixed):
        return points

    seeds = [int(i) for i in seed_indices if 0 <= int(i) < n]
    if not seeds:
        return points

    if layout is None:
        layout = create_grid_spacing_layout(points, grid_dims, grid_spacing)
    else:
        sync_layout_positions(layout, points)

    set_local_similarity_rest_offsets(layout, points, fixed, grid_dims, grid_spacing)

    # Seeds already hold their updated positions in *points* / layout.
    # Mark them visited so a later free neighbor does not re-relax a seed.
    visited: set[int] = set(seeds)
    for seed in seeds:
        layout.nodes[seed].Position = points[seed].copy()

    queue: deque[int] = deque()

    for seed in seeds:
        for neighbor in four_adjacent_indices(seed, rows, cols):
            if not fixed[neighbor] and neighbor not in visited:
                queue.append(neighbor)

    while queue:
        node_id = queue.popleft()
        if node_id in visited or fixed[node_id]:
            continue
        visited.add(node_id)
        Layout.RelaxNode(layout, node_id, vector_scalar=1.0)
        points[node_id] = layout.nodes[node_id].Position.copy()
        for neighbor in four_adjacent_indices(node_id, rows, cols):
            if not fixed[neighbor] and neighbor not in visited:
                queue.append(neighbor)

    # Fixed rows stay identical to input (already copied).
    for i in range(n):
        if fixed[i]:
            points[i] = np.asarray(target_points[i], dtype=np.float64)
    return points


def build_fixed_mask_for_grid(
        grid: IGrid,
        min_unmasked: float,
        source_mask: NDArray[np.bool_] | None = None,
        target_mask: NDArray[np.bool_] | None = None,
        source_ok: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """Classify grid control points as fixed using cell unmasked fractions."""
    return classify_fixed_grid_points(
        source_points=np.asarray(grid.SourcePoints, dtype=np.float64),
        target_points=np.asarray(grid.TargetPoints, dtype=np.float64),
        cell_size=grid.cell_size,
        min_unmasked=min_unmasked,
        source_mask=source_mask,
        target_mask=target_mask,
        source_ok=source_ok,
    )
