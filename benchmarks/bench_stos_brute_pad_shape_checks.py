"""Micro-benchmark: pad-branch shape gates in _score_one_angle_core (non-fixed_shape)."""

from __future__ import annotations

import os

import numpy as np
import pyperf

# Realistic section/tile dimensions after padding (powers of two near STOS ROI sizes).
_SHAPES: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = [
    ((512, 512), (600, 580), (640, 640)),
    ((1024, 1024), (1100, 1080), (1152, 1152)),
    ((2048, 2048), (2200, 2100), (2304, 2304)),
    ((4183, 4309), (4500, 4600), (8192, 8192)),
]


def _old_shape_gates(
    im_target: np.ndarray,
    rotated_source: np.ndarray,
    target_height: int,
    target_width: int,
) -> bool:
    needs_target_pad = not np.array_equal(im_target.shape, np.array((target_height, target_width)))
    rotated_ok = np.array_equal(rotated_source.shape, np.array((target_height, target_width)))
    if needs_target_pad:
        padded_target_shape = (target_height, target_width)
    else:
        padded_target_shape = im_target.shape
    if rotated_ok:
        rotated_padded_shape = rotated_source.shape
    else:
        rotated_padded_shape = (target_height, target_width)
    return np.array_equal(padded_target_shape, rotated_padded_shape)


def _new_shape_gates(
    im_target: np.ndarray,
    rotated_source: np.ndarray,
    target_height: int,
    target_width: int,
) -> bool:
    desired = (target_height, target_width)
    needs_target_pad = im_target.shape != desired
    rotated_ok = rotated_source.shape == desired
    if needs_target_pad:
        padded_target_shape = desired
    else:
        padded_target_shape = im_target.shape
    if rotated_ok:
        rotated_padded_shape = rotated_source.shape
    else:
        rotated_padded_shape = desired
    return padded_target_shape == rotated_padded_shape


def _make_arrays() -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    out: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    for im_shape, rot_shape, (th, tw) in _SHAPES:
        im = np.zeros(im_shape, dtype=np.float32)
        rot = np.zeros(rot_shape, dtype=np.float32)
        out.append((im, rot, th, tw))
    return out


_ANGLES_PER_SCORE: int = 180


def bench_old(cases: list[tuple[np.ndarray, np.ndarray, int, int]]) -> None:
    for _ in range(_ANGLES_PER_SCORE):
        for im, rot, th, tw in cases:
            _old_shape_gates(im, rot, th, tw)


def bench_new(cases: list[tuple[np.ndarray, np.ndarray, int, int]]) -> None:
    for _ in range(_ANGLES_PER_SCORE):
        for im, rot, th, tw in cases:
            _new_shape_gates(im, rot, th, tw)


def _variant_from_env() -> str:
    return os.environ.get("NORNIR_BENCH_SHAPE_GATES", "old")


def main() -> None:
    variant = _variant_from_env()
    runner = pyperf.Runner()
    cases = _make_arrays()

    for im, rot, th, tw in cases:
        assert _old_shape_gates(im, rot, th, tw) == _new_shape_gates(im, rot, th, tw)

    if variant == "old":
        runner.bench_func("stos_brute_pad_shape_gates_old", bench_old, cases)
    else:
        runner.bench_func("stos_brute_pad_shape_gates_new", bench_new, cases)


if __name__ == "__main__":
    main()
