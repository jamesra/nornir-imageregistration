"""The two assemble entry points disagreed about oversized output canvases.

``assemble_tiles.__CreateOutputBufferForArea`` refused an absurd canvas with a message
naming the byte count, the limit, the env var to change it, and the likely cause.
``assemble.TransformImage`` had no ceiling at all, so the same canvas reached
``np.zeros`` and surfaced as:

    MemoryError: Out of memory allocating 4,000,000,000,000 bytes
                 (allocated so far: 8,000,512 bytes)

which names a number with no indication that an exploded target-space transform is the
usual cause. On a machine with enough swap it thrashes for a long time first.

The guard now lives in ``assemble`` (the lower-level module; ``assemble_tiles`` imports
it, not the reverse) and both paths share one implementation, so the limit cannot drift
between them.

``TransformImage`` passes ``include_zbuffer=False`` because it allocates only an image,
whereas the tile-compositing path also allocates a companion float16 distance buffer.
"""
from __future__ import annotations

import os
from unittest import mock

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration import assemble, assemble_tiles

# Far larger than any real section: 1e12 pixels.
ABSURD_SIDE = 1_000_000


def _identity_transform():
    return nornir_imageregistration.transforms.RigidTranslation(target_offset=(0.0, 0.0))


def _small_source(side=64):
    return np.linspace(0.0, 1.0, side * side, dtype=np.float32).reshape(side, side)


# --- the guard is shared, not duplicated -------------------------------------

def test_both_modules_use_one_guard():
    """A copied limit is a limit that drifts."""
    assert assemble_tiles._raise_if_assemble_buffer_too_large is \
        assemble._raise_if_assemble_buffer_too_large
    assert assemble_tiles._max_assemble_buffer_bytes is \
        assemble._max_assemble_buffer_bytes
    assert assemble_tiles._DEFAULT_MAX_ASSEMBLE_BUFFER_BYTES == \
        assemble._DEFAULT_MAX_ASSEMBLE_BUFFER_BYTES


# --- TransformImage now refuses, rather than dying in the allocator ----------

def test_transform_image_refuses_an_absurd_canvas():
    with pytest.raises(ValueError) as excinfo:
        assemble.TransformImage(_identity_transform(),
                                np.array((ABSURD_SIDE, ABSURD_SIDE)),
                                _small_source(),
                                CropUndefined=False)

    message = str(excinfo.value)
    assert 'Refusing to allocate' in message
    assert 'NORNIR_MAX_ASSEMBLE_BUFFER_BYTES' in message
    # The diagnosis is the point of the message, not just the refusal.
    assert 'exploded target-space control points' in message


def test_transform_image_does_not_raise_memory_error():
    """MemoryError was the old behaviour and carries no diagnosis."""
    with pytest.raises(ValueError):
        assemble.TransformImage(_identity_transform(),
                                np.array((ABSURD_SIDE, ABSURD_SIDE)),
                                _small_source(),
                                CropUndefined=False)


def test_a_reasonable_canvas_still_assembles():
    source = _small_source(128)

    result = assemble.TransformImage(_identity_transform(),
                                     np.array((128, 128)),
                                     source,
                                     CropUndefined=False)

    assert nornir_imageregistration.EnsureNumpyArray(result).shape == (128, 128)


def test_the_limit_is_configurable_for_transform_image():
    """A low limit must be honoured, so the ceiling is operator-controllable."""
    source = _small_source(128)
    with mock.patch.dict(os.environ, {'NORNIR_MAX_ASSEMBLE_BUFFER_BYTES': '1024'}):
        with pytest.raises(ValueError, match='Refusing to allocate'):
            assemble.TransformImage(_identity_transform(),
                                     np.array((128, 128)),
                                     source,
                                     CropUndefined=False)


def test_raising_the_limit_permits_the_same_canvas():
    source = _small_source(128)
    with mock.patch.dict(os.environ,
                         {'NORNIR_MAX_ASSEMBLE_BUFFER_BYTES': str(64 * 1024 ** 2)}):
        result = assemble.TransformImage(_identity_transform(),
                                         np.array((128, 128)),
                                         source,
                                         CropUndefined=False)

    assert nornir_imageregistration.EnsureNumpyArray(result).shape == (128, 128)


# --- the guard itself --------------------------------------------------------

def test_the_zbuffer_is_only_counted_when_it_is_allocated():
    """TransformImage allocates no distance buffer, so it must not be charged for one."""
    side = 40_000  # 1.6e9 px: 6.4 GB as float32, plus 3.2 GB of float16 zbuffer.

    # Under the default 16 GiB limit the image alone fits, image+zbuffer also fits.
    assemble._raise_if_assemble_buffer_too_large(side, side, np.float32,
                                                 include_zbuffer=False)

    # Choose a limit that sits between the two totals to show they differ.
    image_only = side * side * 4
    with mock.patch.dict(os.environ,
                         {'NORNIR_MAX_ASSEMBLE_BUFFER_BYTES': str(image_only + 1)}):
        assemble._raise_if_assemble_buffer_too_large(side, side, np.float32,
                                                     include_zbuffer=False)
        with pytest.raises(ValueError):
            assemble._raise_if_assemble_buffer_too_large(side, side, np.float32,
                                                         include_zbuffer=True)


@pytest.mark.parametrize('height,width', [(0, 10), (10, 0), (-1, 10), (10, -1)])
def test_non_positive_dimensions_are_rejected(height, width):
    with pytest.raises(ValueError, match='must be positive'):
        assemble._raise_if_assemble_buffer_too_large(height, width, np.float32)


def test_a_normal_section_canvas_is_allowed():
    """The ceiling must not be so low that real sections trip it."""
    # 30k x 30k float32 is 3.6 GB of image plus 1.8 GB of zbuffer.
    assemble._raise_if_assemble_buffer_too_large(30_000, 30_000, np.float32)


def test_dtype_is_accounted_for():
    side = 50_000
    # uint8 fits under 16 GiB where float64 does not.
    assemble._raise_if_assemble_buffer_too_large(side, side, np.uint8,
                                                 include_zbuffer=False)
    with pytest.raises(ValueError):
        assemble._raise_if_assemble_buffer_too_large(side, side, np.float64,
                                                     include_zbuffer=False)
