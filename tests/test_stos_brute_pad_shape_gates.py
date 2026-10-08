"""Regression: non-fixed ``_score_one_angle_core`` pad branch uses tuple shape gates."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import nornir_imageregistration
from hypothesis import given
from hypothesis import strategies as st
from nornir_imageregistration.stos_brute import (
    _non_fixed_correlation_frame_plan,
    _score_one_angle_core,
    pad_and_rotate_image,
)


def _legacy_frame_plan(
    im_target_shape: tuple[int, int],
    padded_target_shape: tuple[int, int],
    rotated_source_shape: tuple[int, int],
) -> tuple[tuple[int, int], bool, bool]:
    target_height = max(padded_target_shape[0], rotated_source_shape[0])
    target_width = max(padded_target_shape[1], rotated_source_shape[1])
    desired = (target_height, target_width)
    repad_target = not np.array_equal(
        im_target_shape, np.array((target_height, target_width))
    )
    reuse_rotated = np.array_equal(
        rotated_source_shape, np.array((target_height, target_width))
    )
    return desired, repad_target, reuse_rotated


@given(
    im_h=st.integers(min_value=1, max_value=4096),
    im_w=st.integers(min_value=1, max_value=4096),
    pad_h=st.integers(min_value=1, max_value=4096),
    pad_w=st.integers(min_value=1, max_value=4096),
    rot_h=st.integers(min_value=1, max_value=4096),
    rot_w=st.integers(min_value=1, max_value=4096),
)
def test_non_fixed_frame_plan_matches_legacy_gates(
    im_h: int, im_w: int, pad_h: int, pad_w: int, rot_h: int, rot_w: int
) -> None:
    im_shape = (im_h, im_w)
    padded_shape = (pad_h, pad_w)
    rotated_shape = (rot_h, rot_w)
    legacy = _legacy_frame_plan(im_shape, padded_shape, rotated_shape)
    plan = _non_fixed_correlation_frame_plan(im_shape, padded_shape, rotated_shape)
    assert plan.desired_shape == legacy[0]
    assert plan.repad_target == legacy[1]
    assert plan.reuse_rotated_without_pad == legacy[2]


def _letter_patch(shape: tuple[int, int], seed: int = 3) -> np.ndarray:
    img = np.full(shape, 0.25, dtype=np.float32)
    h, w = shape
    img[h // 4: 3 * h // 4, w // 3: w // 3 + max(1, h // 10)] = 0.95
    rng = np.random.default_rng(seed)
    img += 0.03 * rng.normal(size=shape).astype(np.float32)
    return img


def test_score_one_angle_core_non_fixed_avoids_legacy_array_equal_gates() -> None:
    """Reverting the pad branch to ``np.array_equal(..., np.array((h, w)))`` fails here."""

    target_shape = (64, 72)
    source_shape = (58, 66)
    target = _letter_patch(target_shape, seed=11)
    source = _letter_patch(source_shape, seed=17)
    target_stats = nornir_imageregistration.ImageStats.CalcStats(target)
    source_stats = nornir_imageregistration.ImageStats.CalcStats(source)
    real_array_equal = np.array_equal

    def guard_legacy_shape_gates(*args: object, **kwargs: object) -> bool:
        if len(args) >= 2 and isinstance(args[1], np.ndarray):
            second = args[1]
            if second.shape == (2,) and second.dtype.kind in "iu":
                raise AssertionError(
                    "non-fixed _score_one_angle_core must not use np.array_equal on shape tuples"
                )
        return bool(real_array_equal(*args, **kwargs))

    with patch("nornir_imageregistration.stos_brute.np.array_equal", side_effect=guard_legacy_shape_gates):
        record = _score_one_angle_core(
            target,
            source,
            target_shape,
            source_shape,
            angle=7.0,
            target_stats=target_stats,
            source_stats=source_stats,
            target_image_prepadded=False,
            min_overlap=0.5,
            fixed_shape=None,
        )
    assert record.weight > 0.0


def test_score_one_angle_core_non_fixed_frame_plan_drives_padding() -> None:
    target_shape = (48, 52)
    source_shape = (40, 44)
    target = _letter_patch(target_shape)
    source = _letter_patch(source_shape, seed=5)
    target_stats = nornir_imageregistration.ImageStats.CalcStats(target)
    source_stats = nornir_imageregistration.ImageStats.CalcStats(source)
    rotated = pad_and_rotate_image(
        image=source,
        angle=12.0,
        image_stats=source_stats,
        min_overlap=0.5,
    )
    padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
        target,
        image_median=target_stats.median,
        image_stddev=target_stats.std,
        min_overlap=0.5,
        original_shape=target_shape,
    )
    plan = _non_fixed_correlation_frame_plan(
        target_shape,
        tuple(int(s) for s in padded_target.shape),
        tuple(int(s) for s in rotated.shape),
    )
    pad_calls: list[tuple[str, tuple[int, int]]] = []
    real_pad = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation

    def recording_pad(image: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        label = "target" if image is target else "rotated"
        nh = kwargs.get("new_height")
        nw = kwargs.get("new_width")
        if nh is not None and nw is not None:
            pad_calls.append((label, (int(nh), int(nw))))
        return real_pad(image, *args, **kwargs)

    with patch.object(
        nornir_imageregistration.phasecorrelation,
        "pad_image_for_phase_correlation",
        side_effect=recording_pad,
    ):
        _score_one_angle_core(
            target,
            source,
            target_shape,
            source_shape,
            angle=12.0,
            target_stats=target_stats,
            source_stats=source_stats,
            target_image_prepadded=False,
            min_overlap=0.5,
            fixed_shape=None,
        )

    if plan.repad_target:
        assert ("target", plan.desired_shape) in pad_calls
    else:
        assert not any(label == "target" for label, _ in pad_calls)
    if plan.reuse_rotated_without_pad:
        assert not any(label == "rotated" for label, _ in pad_calls)
    else:
        assert ("rotated", plan.desired_shape) in pad_calls
