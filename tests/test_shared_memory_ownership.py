"""
Shared memory can only be unlinked by the process that allocated it.

This pins the constraint behind ``return_shared_memory=False`` in tiled
``assemble.TransformImage``. ``unlink_shared_memory`` consults a module-level
registry populated by ``create_shared_memory_array``, so a name allocated in a
pool worker is unknown to the parent and cannot be released there. Measured
cross-process on Windows, the parent cannot even attach: the segment dies when
the worker task returns, and attach raises ``FileNotFoundError``.

Returning warped tiles as shared memory would therefore hand the parent metadata
for a dead segment. Pickling the tile back costs ~1% of the tile's own warp
(5.5 ms transfer vs 569 ms warp for 2048x2048 float32), so there is nothing to
win by trying.
"""

from __future__ import annotations

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration.core import _core


def _shape(rows: int, cols: int) -> np.ndarray:
    return np.asarray((rows, cols), dtype=np.int64)


def _owner_registry() -> dict:
    """The per-process record of segments this process allocated."""
    return getattr(_core, '__known_shared_memory_allocations')


def test_allocating_process_can_unlink_its_own_segment():
    """The owning process holds the registry entry and can release it."""
    meta, arr = nornir_imageregistration.create_shared_memory_array(
        _shape(16, 16), dtype=np.float32)
    arr.fill(1.25)

    registry = _owner_registry()
    assert meta.name in registry, 'allocation was not registered with the owner'

    del arr
    nornir_imageregistration.unlink_shared_memory(meta)
    assert meta.name not in registry, 'unlink left the registry entry behind'


def test_unlink_of_a_foreign_segment_does_not_release_it():
    """A name this process did not allocate cannot be unlinked here.

    This is the pool-worker case: the worker registers the segment, the parent
    receives only metadata, so the parent's unlink is a logged no-op.
    """
    meta, arr = nornir_imageregistration.create_shared_memory_array(
        _shape(16, 16), dtype=np.float32)
    arr.fill(2.5)

    registry = _owner_registry()
    # Simulate a segment owned by another process: metadata is visible, the
    # registry entry is not.
    owner_entry = registry.pop(meta.name)
    try:
        nornir_imageregistration.unlink_shared_memory(meta)
        assert meta.name not in registry
        # Still mapped, because nothing was actually released.
        still_readable = nornir_imageregistration.ImageParamToImageArray(meta)
        np.testing.assert_allclose(np.asarray(still_readable), 2.5)
    finally:
        registry[meta.name] = owner_entry
        del arr
        nornir_imageregistration.unlink_shared_memory(meta)


def test_unlink_is_a_silent_noop_on_a_plain_ndarray():
    """Why the call in the tile-collection loop was dead code."""
    tile = np.ones((8, 8), dtype=np.float32)
    nornir_imageregistration.unlink_shared_memory(tile)  # type: ignore[arg-type]
    np.testing.assert_allclose(tile, 1.0), 'array must be untouched'


def test_valid_mask_is_not_available_with_shared_memory():
    """The tiled path requests a valid mask, which shared memory cannot return.

    ``TransformImage`` passes ``return_valid_mask=True`` when
    ``enforce_background_cval`` is set, and that flag also applies cval to
    unmapped pixels, so it is load-bearing rather than merely informational.
    It is mutually exclusive with ``return_shared_memory``.
    """
    from nornir_imageregistration.assemble import _TransformImageUsingCoords

    with pytest.raises(ValueError, match='return_valid_mask'):
        _TransformImageUsingCoords(
            np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((4, 4), dtype=np.float32),
            output_origin=np.asarray((0, 0)), output_area=_shape(4, 4),
            return_shared_memory=True, return_valid_mask=True)
