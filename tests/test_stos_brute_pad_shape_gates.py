"""Regression: tuple shape gates match legacy np.array_equal pad-branch checks."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st


def _legacy_shape_gates(
    im_target: np.ndarray,
    rotated_source: np.ndarray,
    target_height: int,
    target_width: int,
) -> bool:
    if not np.array_equal(im_target.shape, np.array((target_height, target_width))):
        padded_target_shape = (target_height, target_width)
    else:
        padded_target_shape = im_target.shape
    if np.array_equal(rotated_source.shape, np.array((target_height, target_width))):
        rotated_padded_shape = rotated_source.shape
    else:
        rotated_padded_shape = (target_height, target_width)
    return bool(np.array_equal(padded_target_shape, rotated_padded_shape))


def _tuple_shape_gates(
    im_target: np.ndarray,
    rotated_source: np.ndarray,
    target_height: int,
    target_width: int,
) -> bool:
    desired = (target_height, target_width)
    if im_target.shape != desired:
        padded_target_shape = desired
    else:
        padded_target_shape = im_target.shape
    if rotated_source.shape == desired:
        rotated_padded_shape = rotated_source.shape
    else:
        rotated_padded_shape = desired
    return padded_target_shape == rotated_padded_shape


@given(
    im_h=st.integers(min_value=1, max_value=4096),
    im_w=st.integers(min_value=1, max_value=4096),
    rot_h=st.integers(min_value=1, max_value=4096),
    rot_w=st.integers(min_value=1, max_value=4096),
    th=st.integers(min_value=1, max_value=8192),
    tw=st.integers(min_value=1, max_value=8192),
)
def test_pad_shape_gates_tuple_matches_legacy(
    im_h: int, im_w: int, rot_h: int, rot_w: int, th: int, tw: int
) -> None:
    im = np.zeros((im_h, im_w), dtype=np.float32)
    rot = np.zeros((rot_h, rot_w), dtype=np.float32)
    assert _legacy_shape_gates(im, rot, th, tw) == _tuple_shape_gates(im, rot, th, tw)
