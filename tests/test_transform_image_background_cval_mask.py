"""
``TransformImage`` must not recompute the coverage mask the warp already gave it.

With ``enforce_background_cval`` set, the warp is asked for ``return_valid_mask``
and reports exactly which output pixels received a mapped sample. That mask was
discarded (``result = result[0]``) and then rebuilt by
``assembly_source_sample_mask``, which runs a second whole-canvas inverse
transform. Measured on a 1500x1500 canvas the rebuilt mask was pixel-identical to
the discarded one for identity, shifted and rotated transforms (0 of 2.25M pixels
differing) while costing 27-51% of the warp itself and 71-155 MiB of coordinate
arrays for a 2.1 MiB bool result.

These tests pin that the mask now comes from the warp, that output is unchanged,
and that the tiled branch accumulates per-tile coverage instead.
"""

from __future__ import annotations

import numpy as np
import pytest

import nornir_imageregistration
import nornir_imageregistration.assemble as assemble
from nornir_imageregistration.transforms.rigid import Rigid

CVAL = 0.0
SINGLE_TILE = 300      # <= 2048, so TransformImage takes the single-tile branch
TILED = 2200           # > 2048, so it takes the multi-tile branch


@pytest.fixture(autouse=True)
def _host_backend():
    nornir_imageregistration.SetActiveComputationLib(
        nornir_imageregistration.ComputationLib.numpy)


def _source(size: int) -> np.ndarray:
    rng = np.random.default_rng(11)
    # Offset away from cval so background is distinguishable from signal.
    return (rng.random((size, size), dtype=np.float32) + 0.5).astype(np.float32)


def _shape(size: int) -> np.ndarray:
    return np.asarray((size, size), dtype=np.int64)


class _InlineTask:
    """A pool task that has already run."""

    def __init__(self, name, func, args, kwargs):
        self.name = name
        self._result = func(*args, **kwargs)

    def wait_return(self):
        return self._result


class _InlinePool:
    """Synchronous stand-in for the local machine pool.

    The tiled branch is otherwise unreachable in tests: the real pool wedges in
    this environment, and process startup would dominate the run time anyway.
    """

    def __init__(self):
        self.tasks = []
        self.wait_completion_calls = 0

    def add_task(self, name, func, *args, **kwargs):
        task = _InlineTask(name, func, args, kwargs)
        self.tasks.append(task)
        return task

    def wait_completion(self):
        self.wait_completion_calls += 1


@pytest.fixture
def inline_pool(monkeypatch):
    pool = _InlinePool()
    monkeypatch.setattr(assemble.nornir_pools, 'GetGlobalLocalMachinePool',
                        lambda: pool)
    return pool


def _count_mask_recomputes(monkeypatch) -> list[int]:
    """Count calls to the whole-canvas mask recompute."""
    calls: list[int] = []
    real = assemble.assembly_source_sample_mask

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(assemble, 'assembly_source_sample_mask', counting)
    return calls


TRANSFORMS = {
    'identity': lambda size: Rigid(target_offset=(0.0, 0.0)),
    'shift': lambda size: Rigid(target_offset=(37.0, 53.0)),
    'negative_shift': lambda size: Rigid(target_offset=(-80.0, 20.0)),
    'rotated': lambda size: Rigid(target_offset=(0.0, 0.0), angle=0.20,
                                  source_rotation_center=(size / 2, size / 2)),
}


@pytest.mark.parametrize('name', sorted(TRANSFORMS))
def test_single_tile_output_matches_recomputed_mask(name, monkeypatch):
    """Using the warp's own mask leaves output identical to the recompute."""
    size = SINGLE_TILE
    transform = TRANSFORMS[name](size)
    src = _source(size)

    produced = assemble.TransformImage(
        transform, _shape(size), src, CropUndefined=False,
        enforce_background_cval=CVAL)

    # Reference: the old behaviour, mask rebuilt over the whole canvas.
    host = assemble.transform_for_host_assembly(transform)
    assert host is not None
    warped = assemble.SourceImageToTargetSpace(
        host, src, output_botleft=np.array([0, 0]), output_area=_shape(size),
        cval=CVAL, extrapolate=True, return_valid_mask=True)
    assert isinstance(warped, tuple), 'expected (image, valid_mask)'
    reference_image, _reference_mask = warped  # type: ignore[misc]
    reference = np.asarray(reference_image).copy()
    recomputed = assemble.assembly_source_sample_mask(
        host, _shape(size), src.shape[:2], extrapolate=True)
    reference[~np.asarray(recomputed)] = CVAL

    np.testing.assert_array_equal(np.asarray(produced), reference)


def test_single_tile_does_not_recompute_the_mask(monkeypatch):
    """The whole-canvas inverse transform is no longer run."""
    calls = _count_mask_recomputes(monkeypatch)
    size = SINGLE_TILE

    assemble.TransformImage(
        Rigid(target_offset=(37.0, 53.0)), _shape(size), _source(size),
        CropUndefined=False, enforce_background_cval=CVAL)

    assert calls == [], 'assembly_source_sample_mask was still called'


def test_background_is_actually_enforced(monkeypatch):
    """A shift leaves genuinely unmapped pixels, and they must hold cval."""
    size = SINGLE_TILE
    shift = 40.0
    produced = np.asarray(assemble.TransformImage(
        Rigid(target_offset=(shift, 0.0)), _shape(size), _source(size),
        CropUndefined=False, enforce_background_cval=CVAL))

    # Source values were offset to >= 0.5, so cval pixels are unambiguous.
    top_rows = produced[:int(shift) - 1, :]
    assert np.all(top_rows == CVAL), 'unmapped rows were not set to cval'
    assert np.any(produced != CVAL), 'everything was blanked'


def test_no_mask_work_when_cval_is_not_enforced(monkeypatch):
    """Without enforce_background_cval there is no mask to build at all."""
    calls = _count_mask_recomputes(monkeypatch)
    size = SINGLE_TILE

    assemble.TransformImage(
        Rigid(target_offset=(5.0, 5.0)), _shape(size), _source(size),
        CropUndefined=False)

    assert calls == []


def test_tiled_branch_accumulates_tile_coverage(inline_pool, monkeypatch):
    """The multi-tile path uses per-tile masks, not a canvas-wide recompute."""
    calls = _count_mask_recomputes(monkeypatch)
    size = TILED
    src = _source(size)
    transform = Rigid(target_offset=(37.0, 53.0))

    produced = np.asarray(assemble.TransformImage(
        transform, _shape(size), src, CropUndefined=False,
        enforce_background_cval=CVAL))

    assert len(inline_pool.tasks) > 1, 'expected the tiled branch'
    assert calls == [], 'tiled branch still recomputed the whole-canvas mask'
    assert produced.shape == (size, size)

    # Unmapped rows introduced by the shift must still be cval.
    assert np.all(produced[:36, :] == CVAL)
    assert np.any(produced != CVAL)


def test_tiled_branch_matches_recomputed_mask(inline_pool):
    """Accumulated per-tile coverage agrees with the analytic mask."""
    size = TILED
    src = _source(size)
    transform = Rigid(target_offset=(37.0, 53.0))

    produced = np.asarray(assemble.TransformImage(
        transform, _shape(size), src, CropUndefined=False,
        enforce_background_cval=CVAL))

    host = assemble.transform_for_host_assembly(transform)
    recomputed = np.asarray(assemble.assembly_source_sample_mask(
        host, _shape(size), src.shape[:2], extrapolate=True))

    # Every pixel the analytic mask calls unmapped must be cval in the output.
    assert np.all(produced[~recomputed] == CVAL)


def test_tiled_branch_falls_back_when_tiles_report_no_coverage(inline_pool, monkeypatch):
    """If tiles stop returning masks, the analytic recompute is used instead."""
    calls = _count_mask_recomputes(monkeypatch)
    size = TILED

    real_warp = assemble.SourceImageToTargetSpace

    def mask_stripping_warp(*args, **kwargs):
        kwargs['return_valid_mask'] = False
        result = real_warp(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    monkeypatch.setattr(assemble, 'SourceImageToTargetSpace', mask_stripping_warp)

    produced = np.asarray(assemble.TransformImage(
        Rigid(target_offset=(37.0, 53.0)), _shape(size), _source(size),
        CropUndefined=False, enforce_background_cval=CVAL))

    assert len(calls) == 1, 'expected one fallback recompute'
    assert np.all(produced[:36, :] == CVAL)
