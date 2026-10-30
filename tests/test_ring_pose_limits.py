"""Clamp math for ring similarity vs a frozen input pose."""
from __future__ import annotations

import math
import unittest

from hypothesis import example, given, settings
from hypothesis import strategies as st

from nornir_imageregistration.refine_shared.ring_pose_limits import (
    RING_ALLOW_FLIP_CHANGE,
    RING_ANGLE_MAX_DEGREES,
    RING_SCALE_FRACTION_MAX,
    RingReferencePose,
    clamp_similarity_to_reference,
    shortest_signed_angle_delta,
)


class TestRingPoseClamp(unittest.TestCase):
    """Named defaults and geodesic clip of scale/angle/flip."""

    def test_scale_2_6_vs_ref_1_clips_to_1_05(self) -> None:
        ref = RingReferencePose(angle=math.pi, scale=1.0, flip_ud=False)
        scale, _angle, _flip = clamp_similarity_to_reference(
            2.6, math.pi, False, ref)
        self.assertAlmostEqual(scale, 1.0 * (1.0 + RING_SCALE_FRACTION_MAX))

    def test_angle_plus_90_vs_ref_180_clips_to_15_deg(self) -> None:
        ref = RingReferencePose(angle=math.pi, scale=1.0, flip_ud=False)
        _scale, angle, _flip = clamp_similarity_to_reference(
            1.0, math.pi + math.pi / 2.0, False, ref)
        delta_deg = math.degrees(abs(shortest_signed_angle_delta(angle, ref.angle)))
        self.assertAlmostEqual(delta_deg, RING_ANGLE_MAX_DEGREES, places=6)

    def test_flip_disagreement_keeps_reference_when_frozen(self) -> None:
        self.assertFalse(RING_ALLOW_FLIP_CHANGE)
        ref = RingReferencePose(angle=0.0, scale=1.0, flip_ud=False)
        _scale, _angle, flip = clamp_similarity_to_reference(
            1.0, 0.0, True, ref, allow_flip_change=False)
        self.assertFalse(flip)

    def test_flip_may_change_when_allowed(self) -> None:
        ref = RingReferencePose(angle=0.0, scale=1.0, flip_ud=False)
        _scale, _angle, flip = clamp_similarity_to_reference(
            1.0, 0.0, True, ref, allow_flip_change=True)
        self.assertTrue(flip)

    @given(
        ref_scale=st.floats(min_value=0.25, max_value=4.0, allow_nan=False, allow_infinity=False),
        fitted_scale=st.floats(min_value=0.05, max_value=8.0, allow_nan=False, allow_infinity=False),
        ref_angle=st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False),
        fitted_angle=st.floats(min_value=-2.0 * math.pi, max_value=2.0 * math.pi, allow_nan=False, allow_infinity=False),
        ref_flip=st.booleans(),
        fitted_flip=st.booleans(),
        fraction=st.floats(min_value=0.0, max_value=0.2, allow_nan=False, allow_infinity=False),
        max_deg=st.floats(min_value=0.0, max_value=45.0, allow_nan=False, allow_infinity=False),
    )
    @example(ref_scale=1.0, fitted_scale=2.6, ref_angle=math.pi, fitted_angle=math.pi / 2.0,
             ref_flip=False, fitted_flip=True, fraction=0.05, max_deg=15.0)
    @settings(max_examples=80, deadline=None)
    def test_clamp_respects_fraction_and_geodesic_cap(
            self,
            ref_scale: float,
            fitted_scale: float,
            ref_angle: float,
            fitted_angle: float,
            ref_flip: bool,
            fitted_flip: bool,
            fraction: float,
            max_deg: float) -> None:
        ref = RingReferencePose(angle=ref_angle, scale=ref_scale, flip_ud=ref_flip)
        scale, angle, flip = clamp_similarity_to_reference(
            fitted_scale, fitted_angle, fitted_flip, ref,
            scale_fraction_max=fraction,
            angle_max_degrees=max_deg,
            allow_flip_change=False)
        self.assertLessEqual(abs(scale / ref_scale - 1.0), fraction + 1e-9)
        self.assertLessEqual(
            abs(shortest_signed_angle_delta(angle, ref_angle)),
            math.radians(max_deg) + 1e-9)
        self.assertEqual(flip, ref_flip)


if __name__ == '__main__':
    unittest.main()
