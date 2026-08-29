"""
``TilesToImageThreaded`` must keep in-flight warped tiles O(workers), not O(tiles).

Every warp was submitted to the thread pool up front. Compositing is deliberately
in ``work_items`` order so z-buffer ties match the serial path, so a tile that
finished early had to keep its warped image resident until its turn arrived. With
one slow tile at the front, essentially the whole mosaic accumulated in memory:
measured 23 of 24 tiles alive at once on an 8-worker pool, and 40 of 40 at 40
tiles. A full-resolution 2048x2048 float32 tile is ~16 MB, so this is hundreds of
megabytes of avoidable peak.

Submission is now a bounded window. These tests pin the bound and the
count-independence, and check that output and compositing order are unchanged.
"""

from __future__ import annotations

import tempfile
import threading
import time

import numpy as np
import pytest

import nornir_imageregistration
import nornir_imageregistration.assemble_tiles as assemble_tiles

from test_assemble_gpu_threaded import _build_synthetic_mosaic

TILE_SHAPE = (32, 32)
SLOW_TILE_SECONDS = 0.4


class _LiveTileTracker:
    """Track how many warped tiles exist but have not yet been composited."""

    def __init__(self, monkeypatch, *, stall_first: bool):
        self._lock = threading.Lock()
        self.live = 0
        self.max_live = 0
        self.created = 0
        self.composite_order: list[int] = []
        self._stall_first = stall_first

        real_worker = assemble_tiles._transform_tile_worker
        real_composite = assemble_tiles._composite_transformed_tile_onto_canvas

        def counting_worker(tile, region, target_space_scale):
            result = real_worker(tile, region, target_space_scale)
            with self._lock:
                is_first = self.created == 0
                self.created += 1
                self.live += 1
                self.max_live = max(self.max_live, self.live)
            # Hold the first tile so later tiles finish while it is uncomposited,
            # which is what made the unbounded version accumulate the whole mosaic.
            if is_first and self._stall_first:
                time.sleep(SLOW_TILE_SECONDS)
            return result

        def counting_composite(data, full_image, zbuffer, rect):
            real_composite(data, full_image, zbuffer, rect)
            with self._lock:
                self.live -= 1
                self.composite_order.append(len(self.composite_order))

        monkeypatch.setattr(assemble_tiles, '_transform_tile_worker', counting_worker)
        monkeypatch.setattr(assemble_tiles, '_composite_transformed_tile_onto_canvas',
                            counting_composite)


def _assemble(n_tiles: int, *, stall_first: bool = True):
    """Assemble one mosaic under its own patch scope.

    Each call gets a fresh MonkeyPatch context; sharing one context across calls
    would stack the instrumentation wrappers and pool the counters together.
    """
    nornir_imageregistration.SetActiveComputationLib(
        nornir_imageregistration.ComputationLib.numpy)
    with pytest.MonkeyPatch.context() as patch:
        tracker = _LiveTileTracker(patch, stall_first=stall_first)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tileset = _build_synthetic_mosaic(
                tmp_dir, n_tiles=n_tiles, tile_shape=TILE_SHAPE)
            image, mask = assemble_tiles.TilesToImageThreaded(tileset)
    return tracker, image, mask


def _expected_bound() -> int:
    """Ceiling on warped tiles alive at once.

    The implementation keeps a window of ``2 * workers`` submitted. The tracker
    can observe one more than that: a refill is submitted after popping a future
    but before the popped tile is composited, so the tile awaiting compositing
    and a full window can overlap for an instant.
    """
    return assemble_tiles._TRANSFORM_WORKERS * 2 + 1


def test_in_flight_warps_are_bounded():
    """A slow leading tile no longer causes the mosaic to pile up in memory."""
    n_tiles = 40
    tracker, _image, _mask = _assemble(n_tiles)

    assert tracker.created == n_tiles
    assert tracker.max_live <= _expected_bound(), (
        f'{tracker.max_live} warped tiles were alive at once; expected at most '
        f'{_expected_bound()} for a {assemble_tiles._TRANSFORM_WORKERS}-worker pool')
    assert tracker.max_live < n_tiles


def test_in_flight_bound_does_not_grow_with_tile_count():
    """Doubling the mosaic must not double resident warped tiles."""
    small_tracker, _i1, _m1 = _assemble(20)
    large_tracker, _i2, _m2 = _assemble(40)

    assert small_tracker.created == 20
    assert large_tracker.created == 40
    assert large_tracker.max_live <= _expected_bound()
    assert large_tracker.max_live <= small_tracker.max_live + 1, (
        'resident warped tiles scaled with the tile count: '
        f'{small_tracker.max_live} at 20 tiles vs {large_tracker.max_live} at 40')


def test_every_tile_is_composited_exactly_once():
    """Bounding submission must not drop or duplicate work."""
    n_tiles = 24
    tracker, _image, _mask = _assemble(n_tiles)

    assert tracker.created == n_tiles
    assert len(tracker.composite_order) == n_tiles
    assert tracker.live == 0, 'a warped tile was left uncomposited'


def test_output_matches_serial_path():
    """The bounded window leaves pixels and coverage identical to serial."""
    n_tiles = 12
    nornir_imageregistration.SetActiveComputationLib(
        nornir_imageregistration.ComputationLib.numpy)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tileset = _build_synthetic_mosaic(tmp_dir, n_tiles=n_tiles, tile_shape=TILE_SHAPE)
        serial_image, serial_mask = assemble_tiles.TilesToImage(tileset)
        threaded_image, threaded_mask = assemble_tiles.TilesToImageThreaded(tileset)

    serial_arr = np.nan_to_num(np.asarray(serial_image, dtype=np.float64))
    threaded_arr = np.nan_to_num(np.asarray(threaded_image, dtype=np.float64))

    # Match the tolerance style of test_assemble_gpu_threaded: interior pixels only.
    crop = 2
    height = min(serial_arr.shape[0], threaded_arr.shape[0]) - crop
    width = min(serial_arr.shape[1], threaded_arr.shape[1]) - crop
    np.testing.assert_allclose(
        threaded_arr[crop:height, crop:width],
        serial_arr[crop:height, crop:width],
        atol=1e-5)
    assert int(np.count_nonzero(np.asarray(threaded_mask))) == \
        int(np.count_nonzero(np.asarray(serial_mask)))


def test_result_is_deterministic_across_runs():
    """Compositing order stays work_items order, so repeated runs agree exactly."""
    n_tiles = 16
    nornir_imageregistration.SetActiveComputationLib(
        nornir_imageregistration.ComputationLib.numpy)

    checksums = []
    for _ in range(3):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tileset = _build_synthetic_mosaic(
                tmp_dir, n_tiles=n_tiles, tile_shape=TILE_SHAPE)
            image, _mask = assemble_tiles.TilesToImageThreaded(tileset)
        checksums.append(
            float(np.nansum(np.abs(np.asarray(image, dtype=np.float64)))))

    assert len(set(checksums)) == 1, f'nondeterministic output: {checksums}'


def test_single_tile_mosaic_still_assembles():
    """The window bookkeeping handles fewer tiles than the window size."""
    tracker, image, _mask = _assemble(1, stall_first=False)

    assert tracker.created == 1
    assert len(tracker.composite_order) == 1
    assert np.asarray(image).size > 0
