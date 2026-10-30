"""Tiled TransformImage must not hold every warped tile resident at once.

The submission loop queued every tile, then ``mpool.wait_completion()`` blocked
until all of them had finished, and only then did the collection loop read the
first result. Each task returns its warped tile as a plain array -- shared memory
cannot travel worker->parent, see the note at the submission site -- so the parent
necessarily held the whole tile set simultaneously. Peak grew with the tile count
rather than the worker count:

    canvas 4096x4096   tiles  4   peak unconsumed tiles  4
    canvas 6144x6144   tiles  9   peak unconsumed tiles  9
    canvas 8192x8192   tiles 16   peak unconsumed tiles 16
    canvas 12288x12288 tiles 36   peak unconsumed tiles 36

At ~16 MiB per 2048x2048 float32 tile that is 576 MiB of tiles for the last row,
and a full section is far larger. With a bounded window (2 workers -> window 4)
the same canvases give 4, 5, 5, 5.

``task.wait_return()`` already blocks per task, so ``wait_completion()`` was never
needed for correctness -- it purely forced the all-resident peak. Same defect and
same fix as #37 for ``TilesToImageThreaded``.

Measured against a synchronous fake pool, which models the defect exactly: a task
produces its tile when submitted, so the count of completed-but-unread results is
the in-flight window. Real-pool equivalence was checked separately -- output
checksum 2367921.715735 on both trees, wall time unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

import nornir_imageregistration
import nornir_pools
from nornir_imageregistration import assemble

TILE = 2048


class _FakeTask:
    """Produces its tile on submission and reports when it is read."""

    def __init__(self, pool, name, kwargs):
        self.name = name
        area = kwargs['output_area']
        botleft = kwargs['output_botleft']
        self._pool = pool
        self._shape = (int(area[0]), int(area[1]))
        # A value unique to the tile's position, so placement can be checked.
        self._fill = float(int(botleft[0]) // TILE * 100 + int(botleft[1]) // TILE + 1)
        self._read = False
        pool.live += 1
        pool.peak_live = max(pool.peak_live, pool.live)
        pool.submitted += 1

    def wait_return(self):
        if not self._read:
            self._read = True
            self._pool.live -= 1
        if self._pool.fail_on_read:
            return None
        return np.full(self._shape, self._fill, dtype=np.float32)


class _FakePool:
    def __init__(self, max_workers: int, fail_on_read: bool = False,
                 explode_on_wait_completion: bool = False):
        self.max_workers = max_workers
        self.live = 0
        self.peak_live = 0
        self.submitted = 0
        self.fail_on_read = fail_on_read
        self._explode = explode_on_wait_completion

    def add_task(self, name, func, *args, **kwargs):
        return _FakeTask(self, name, kwargs)

    def wait_completion(self):
        if self._explode:
            raise AssertionError(
                'wait_completion() was called; that is what forced the all-resident peak')


def _run(canvas, pool):
    original = nornir_pools.GetGlobalLocalMachinePool
    nornir_pools.GetGlobalLocalMachinePool = lambda *a, **k: pool
    try:
        # The warped image has to exceed the tile size or TransformImage takes the
        # single-shot branch and never tiles at all.
        warped = np.zeros((TILE + 64, TILE + 64), dtype=np.float32)
        transform = nornir_imageregistration.transforms.Rigid(
            target_offset=(0, 0), source_rotation_center=(0, 0), angle=0)
        return assemble.TransformImage(
            transform=transform,
            fixedImageShape=np.asarray(canvas, dtype=np.int64),
            warpedImage=warped,
            CropUndefined=False)
    finally:
        nornir_pools.GetGlobalLocalMachinePool = original


def _tile_count(canvas):
    return -(-canvas[0] // TILE) * -(-canvas[1] // TILE)


# --- the bound -----------------------------------------------------------------

@pytest.mark.parametrize('canvas', [
    (TILE * 3, TILE * 3),
    (TILE * 4, TILE * 4),
    (TILE * 6, TILE * 6),
])
def test_peak_resident_tiles_does_not_grow_with_tile_count(canvas):
    pool = _FakePool(max_workers=2)
    _run(canvas, pool)

    # window is workers * 2, plus the one tile the parent is compositing.
    expected_bound = 2 * 2 + 1
    tiles = _tile_count(canvas)

    assert pool.submitted == tiles, 'every tile must still be submitted'
    assert pool.peak_live <= expected_bound, (
        f'{tiles} tiles left {pool.peak_live} resident, bound is {expected_bound}')


def test_the_bound_tracks_the_worker_count_not_the_canvas():
    canvas = (TILE * 6, TILE * 6)

    small = _FakePool(max_workers=1)
    _run(canvas, small)
    large = _FakePool(max_workers=8)
    _run(canvas, large)

    assert small.peak_live < large.peak_live, (
        'a wider pool should be allowed more tiles in flight')
    assert small.peak_live <= 1 * 2 + 1


def test_wait_completion_is_not_used():
    """It was redundant for correctness and is what forced the all-resident peak."""
    pool = _FakePool(max_workers=2, explode_on_wait_completion=True)
    _run((TILE * 3, TILE * 3), pool)


# --- output is unchanged -------------------------------------------------------

def test_every_tile_lands_in_its_own_region():
    canvas = (TILE * 3, TILE * 3)
    pool = _FakePool(max_workers=2)

    output = np.asarray(_run(canvas, pool))

    assert output.shape == canvas
    for row in range(3):
        for col in range(3):
            fill = float(row * 100 + col + 1)
            block = output[row * TILE:(row + 1) * TILE, col * TILE:(col + 1) * TILE]
            assert np.all(block == fill), f'tile ({row},{col}) misplaced'


def test_a_partial_edge_tile_is_still_covered():
    """The last row and column are short; they must not be dropped or overrun."""
    canvas = (TILE * 2 + 512, TILE * 2 + 300)
    pool = _FakePool(max_workers=2)

    output = np.asarray(_run(canvas, pool))

    assert output.shape == canvas
    assert pool.submitted == 9
    # Bottom-right corner belongs to the short tile, fill 2*100 + 2 + 1.
    assert output[-1, -1] == pytest.approx(203.0)


# --- failures still surface ----------------------------------------------------

def test_a_failed_tile_still_raises():
    pool = _FakePool(max_workers=2, fail_on_read=True)

    with pytest.raises(RuntimeError, match='Multiprocess tile assembly failed'):
        _run((TILE * 3, TILE * 3), pool)
