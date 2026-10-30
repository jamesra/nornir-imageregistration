"""Clamp local ring Kabsch pose to the input STOS similarity.

The 9-point ring in ``ApproximateRigidTransformBySourcePoints`` linearizes the
current transform at a cell. Sparse or out-of-hull rings can invent a new
scale/angle/flip. Clip those components to the frozen input pose, then pin the
cell center so the prewarp window does not slide.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

RING_SCALE_FRACTION_MAX: float = 0.05
RING_ANGLE_MAX_DEGREES: float = 15.0
RING_ALLOW_FLIP_CHANGE: bool = False

_TWO_PI: float = 2.0 * math.pi
_MIN_REFERENCE_SCALE: float = 1e-15


@dataclass(frozen=True)
class RingReferencePose:
    """Global similarity the ring fit may not wander far from."""

    angle: float
    scale: float
    flip_ud: bool


@dataclass(frozen=True)
class RingPoseLimits:
    """Allowed deviation of a cell ring from ``RingReferencePose``."""

    scale_fraction_max: float = RING_SCALE_FRACTION_MAX
    angle_max_degrees: float = RING_ANGLE_MAX_DEGREES
    allow_flip_change: bool = RING_ALLOW_FLIP_CHANGE


def shortest_signed_angle_delta(angle: float, reference: float) -> float:
    """Return the shortest signed delta from ``reference`` to ``angle`` in ``(-pi, pi]``."""
    return float((angle - reference + math.pi) % _TWO_PI - math.pi)


def clamp_scale(scale: float, reference_scale: float, fraction_max: float) -> float:
    """Clip ``scale`` to ``reference_scale * [1 - f, 1 + f]``."""
    ref = max(abs(float(reference_scale)), _MIN_REFERENCE_SCALE)
    frac = max(0.0, float(fraction_max))
    lo = ref * (1.0 - frac)
    hi = ref * (1.0 + frac)
    return float(min(hi, max(lo, float(scale))))


def clamp_angle_radians(angle: float, reference_angle: float, max_delta_degrees: float) -> float:
    """Clip ``angle`` so its geodesic distance to ``reference_angle`` is at most ``max_delta_degrees``."""
    max_delta = math.radians(max(0.0, float(max_delta_degrees)))
    delta = shortest_signed_angle_delta(float(angle), float(reference_angle))
    if delta > max_delta:
        delta = max_delta
    elif delta < -max_delta:
        delta = -max_delta
    return float(reference_angle) + float(delta)


def clamp_flip(flip_ud: bool, reference_flip: bool, allow_flip_change: bool) -> bool:
    """Return ``flip_ud`` unless flip changes are frozen to the reference."""
    if allow_flip_change:
        return bool(flip_ud)
    return bool(reference_flip)


def clamp_similarity_to_reference(
        scale: float,
        angle: float,
        flip_ud: bool,
        reference: RingReferencePose,
        *,
        scale_fraction_max: float = RING_SCALE_FRACTION_MAX,
        angle_max_degrees: float = RING_ANGLE_MAX_DEGREES,
        allow_flip_change: bool = RING_ALLOW_FLIP_CHANGE) -> tuple[float, float, bool]:
    """Clip one similarity (scale, angle radians, flip) to ``reference``."""
    return (
        clamp_scale(scale, reference.scale, scale_fraction_max),
        clamp_angle_radians(angle, reference.angle, angle_max_degrees),
        clamp_flip(flip_ud, reference.flip_ud, allow_flip_change),
    )


def clamp_similarity_arrays(
        scales: NDArray[np.floating],
        angles: NDArray[np.floating],
        flips: NDArray[np.bool_],
        reference: RingReferencePose,
        *,
        scale_fraction_max: float = RING_SCALE_FRACTION_MAX,
        angle_max_degrees: float = RING_ANGLE_MAX_DEGREES,
        allow_flip_change: bool = RING_ALLOW_FLIP_CHANGE) -> tuple[
            NDArray[np.floating], NDArray[np.floating], NDArray[np.bool_]]:
    """Vectorized clip of ring Kabsch components to ``reference``."""
    scale_arr = np.asarray(scales, dtype=np.float64).reshape(-1)
    angle_arr = np.asarray(angles, dtype=np.float64).reshape(-1)
    flip_arr = np.asarray(flips, dtype=bool).reshape(-1)
    if not (scale_arr.shape == angle_arr.shape == flip_arr.shape):
        raise ValueError(
            "scales, angles, and flips must share shape; "
            f"got {scale_arr.shape}, {angle_arr.shape}, {flip_arr.shape}")

    ref_scale = max(abs(float(reference.scale)), _MIN_REFERENCE_SCALE)
    frac = max(0.0, float(scale_fraction_max))
    lo = ref_scale * (1.0 - frac)
    hi = ref_scale * (1.0 + frac)
    clamped_scales = np.clip(scale_arr, lo, hi)

    max_delta = math.radians(max(0.0, float(angle_max_degrees)))
    delta = (angle_arr - float(reference.angle) + math.pi) % _TWO_PI - math.pi
    delta = np.clip(delta, -max_delta, max_delta)
    clamped_angles = float(reference.angle) + delta

    if allow_flip_change:
        clamped_flips = flip_arr.copy()
    else:
        clamped_flips = np.full(flip_arr.shape, bool(reference.flip_ud), dtype=bool)

    return clamped_scales, clamped_angles, clamped_flips


def target_offset_pinning_source_to_target(
        source_yx: NDArray[np.floating],
        desired_target_yx: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return CS2D ``target_offset`` so a rotation center maps to ``desired_target_yx``.

    CenteredSimilarity2D maps ``c`` to ``c + t`` (scale/angle/flip of a zero
    delta are identity), so ``t = desired - c``. Callers that need float32
    rounding to match ``Transform`` should still apply a residual
    ``TranslateFixed`` after construction.
    """
    source = np.asarray(source_yx, dtype=np.float64).reshape(2)
    desired = np.asarray(desired_target_yx, dtype=np.float64).reshape(2)
    return desired - source


def reference_pose_from_transform(input_transform: Any) -> RingReferencePose:
    """Kabsch or copy the rigid pose encoded by ``input_transform``."""
    from nornir_imageregistration.transforms.base import IRigidTransform
    from nornir_imageregistration.transforms.converters import ConvertTransformToRigidTransform

    if isinstance(input_transform, IRigidTransform):
        return RingReferencePose(
            angle=float(input_transform.angle),
            scale=float(input_transform.scalar),
            flip_ud=bool(input_transform.flip_ud),
        )

    rigid = ConvertTransformToRigidTransform(input_transform)
    return RingReferencePose(
        angle=float(rigid.angle),
        scale=float(rigid.scalar),
        flip_ud=bool(rigid.flip_ud),
    )


def limits_from_settings(settings: Any | None) -> RingPoseLimits:
    """Read ring limits from a ``GridRefinement``-like object, else module defaults."""
    if settings is None:
        return RingPoseLimits()
    return RingPoseLimits(
        scale_fraction_max=float(getattr(
            settings, 'ring_scale_fraction_max', RING_SCALE_FRACTION_MAX)),
        angle_max_degrees=float(getattr(
            settings, 'ring_angle_max_degrees', RING_ANGLE_MAX_DEGREES)),
        allow_flip_change=bool(getattr(
            settings, 'ring_allow_flip_change', RING_ALLOW_FLIP_CHANGE)),
    )
