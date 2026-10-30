"""
Batched grid-vertex measurement must not materialize one array per candidate cell.

The batched path used to accumulate a float cell and a bool mask per candidate
vertex and then stack them, so the transient high-water mark scaled with the
candidate count several times over. It now gates on preallocated mask stacks and
copies surviving cells straight into an exactly-sized float stack. These tests
pin that allocation contract and the slice geometry the rewrite factored out.
"""

from __future__ import annotations

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration import local_distortion_correction as ldc

CELL_SHAPE = np.asarray((32, 32), dtype=np.int64)
TILE_SIZE = 256


def _tile(seed: int, *, valid: np.ndarray | None = None) -> ldc._PrewarpedTile:
    """Build a textured host-side prewarped tile phase correlation will accept."""
    rng = np.random.default_rng(seed)
    image = rng.random((TILE_SIZE, TILE_SIZE)).astype(np.float64)
    if valid is None:
        valid = np.ones((TILE_SIZE, TILE_SIZE), dtype=bool)
    return ldc._PrewarpedTile(
        image=image,
        valid_mask=valid,
        origin=np.asarray((0, 0), dtype=np.int64))


def _grid(count_per_axis: int) -> np.ndarray:
    lin = np.linspace(40, TILE_SIZE - 40, count_per_axis)
    return np.asarray([(y, x) for y in lin for x in lin], dtype=np.float64)


class _ZerosSpy:
    """Record the shape of every ``numpy.zeros`` allocation."""

    def __init__(self):
        self.shapes: list[tuple[int, ...]] = []
        self._real = np.zeros

    def __call__(self, shape, *args, **kwargs):
        as_tuple = (shape,) if isinstance(shape, int) else tuple(
            int(v) for v in np.atleast_1d(np.asarray(shape)))
        self.shapes.append(as_tuple)
        return self._real(shape, *args, **kwargs)

    def cell_sized_count(self) -> int:
        cell = (int(CELL_SHAPE[0]), int(CELL_SHAPE[1]))
        return sum(1 for s in self.shapes if s == cell)

    def stack_count(self) -> int:
        cell = (int(CELL_SHAPE[0]), int(CELL_SHAPE[1]))
        return sum(1 for s in self.shapes if len(s) == 3 and s[1:] == cell)


def _run_with_spy(centers: np.ndarray) -> tuple[_ZerosSpy, tuple]:
    spy = _ZerosSpy()
    original = np.zeros
    np.zeros = spy  # type: ignore[assignment]
    try:
        result = ldc._measure_grid_vertex_displacements_batched(
            _tile(0), _tile(1), centers, CELL_SHAPE, 0.25)
    finally:
        np.zeros = original  # type: ignore[assignment]
    return spy, result


def test_no_per_cell_allocation():
    """No cell-shaped 2-D array is allocated: cells go straight into a stack."""
    spy, (_shifts, measured) = _run_with_spy(_grid(4))
    assert measured.sum() > 0, 'expected textured cells to measure'
    assert spy.cell_sized_count() == 0, (
        f'batched path allocated {spy.cell_sized_count()} per-cell arrays; '
        'cells must be written into the preallocated stack')


def test_cell_allocations_do_not_scale_with_vertex_count():
    """The number of cell-stack allocations is fixed, not one per candidate."""
    small_spy, (_s1, small_measured) = _run_with_spy(_grid(2))
    large_spy, (_s2, large_measured) = _run_with_spy(_grid(6))

    assert large_measured.size > small_measured.size
    assert large_measured.sum() > small_measured.sum()
    # 2 preallocated mask stacks + 2 exactly-sized float stacks.
    assert small_spy.stack_count() == large_spy.stack_count() == 4
    assert small_spy.cell_sized_count() == large_spy.cell_sized_count() == 0


def test_matches_serial_measurement():
    """Batched agrees with serial on which vertices measure and where they land.

    The batched peak finder interpolates to subpixel while the serial one returns
    the integer argmax, a pre-existing difference between the two paths, so the
    shifts are compared to within a pixel rather than exactly.
    """
    centers = _grid(4)
    moving, fixed = _tile(0), _tile(1)

    batched_shifts, batched_measured = ldc._measure_grid_vertex_displacements_batched(
        moving, fixed, centers, CELL_SHAPE, 0.25)
    serial_shifts, serial_measured = ldc._measure_grid_vertex_displacements(
        moving, fixed, centers, CELL_SHAPE, 0.25)

    np.testing.assert_array_equal(batched_measured, serial_measured)
    np.testing.assert_allclose(
        batched_shifts[batched_measured], serial_shifts[serial_measured], atol=1.0)


def test_partial_coverage_rejects_the_same_vertices():
    """Masked-out tiles gate identically before and after the mask preallocation."""
    valid = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    valid[:TILE_SIZE // 2, :] = True
    moving = _tile(0, valid=valid)
    fixed = _tile(1)
    centers = _grid(5)

    batched_shifts, batched_measured = ldc._measure_grid_vertex_displacements_batched(
        moving, fixed, centers, CELL_SHAPE, 0.25)
    _serial_shifts, serial_measured = ldc._measure_grid_vertex_displacements(
        moving, fixed, centers, CELL_SHAPE, 0.25)

    assert 0 < batched_measured.sum() < batched_measured.size, \
        'expected the half-masked tile to reject some but not all vertices'
    np.testing.assert_array_equal(batched_measured, serial_measured)
    # Rejected vertices keep the zero-shift default.
    np.testing.assert_array_equal(batched_shifts[~batched_measured], 0.0)


def test_no_candidates_returns_empty_measurement():
    """Centers entirely off the fixed tile short-circuit before any stack."""
    centers = np.asarray([(-50.0, -50.0), (TILE_SIZE + 10.0, 5.0)], dtype=np.float64)
    shifts, measured = ldc._measure_grid_vertex_displacements_batched(
        _tile(0), _tile(1), centers, CELL_SHAPE, 0.25)
    assert not measured.any()
    np.testing.assert_array_equal(shifts, 0.0)


@pytest.mark.parametrize('center', [
    (128.0, 128.0),        # fully interior
    (8.0, 128.0),          # clipped at the top
    (128.0, 8.0),          # clipped at the left
    (TILE_SIZE - 8.0, 128.0),  # clipped at the bottom
    (128.0, TILE_SIZE - 8.0),  # clipped at the right
    (8.0, 8.0),            # clipped in two axes at once
])
def test_window_geometry_matches_extraction(center):
    """``_refinement_cell_window`` reproduces the extractor's slice geometry."""
    tile = _tile(3)
    center_arr = np.asarray(center, dtype=np.float64)
    cell, valid = ldc._extract_refinement_cell_and_mask(tile, center_arr, CELL_SHAPE)
    window = ldc._refinement_cell_window(tile, center_arr, CELL_SHAPE)
    assert window is not None

    (ws0, we0, ws1, we1), (rs0, re0, rs1, re1) = window
    rebuilt = np.zeros(cell.shape, dtype=cell.dtype)
    rebuilt_valid = np.zeros(cell.shape, dtype=bool)
    rebuilt[ws0:we0, ws1:we1] = tile.image[rs0:re0, rs1:re1]
    rebuilt_valid[ws0:we0, ws1:we1] = tile.valid_mask[rs0:re0, rs1:re1]

    np.testing.assert_array_equal(rebuilt, cell)
    np.testing.assert_array_equal(rebuilt_valid, valid)


def test_window_is_none_off_tile():
    """A cell entirely off the tile yields no window and an all-zero extraction."""
    tile = _tile(3)
    center = np.asarray((-500.0, -500.0), dtype=np.float64)
    assert ldc._refinement_cell_window(tile, center, CELL_SHAPE) is None
    cell, valid = ldc._extract_refinement_cell_and_mask(tile, center, CELL_SHAPE)
    np.testing.assert_array_equal(cell, 0.0)
    assert not valid.any()
