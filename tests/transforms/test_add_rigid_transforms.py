"""Regression for #115 / C05-B009: rigid∘rigid must match sequential Transform."""
from __future__ import annotations

import unittest

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import nornir_imageregistration.transforms as T
from nornir_imageregistration.transforms import addition


_PROBE = np.array(
    [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [5.0, 5.0], [-3.0, 8.0], [12.0, -2.0]],
    dtype=np.float64,
)


def _max_compose_error(a_to_b, b_to_c, points=_PROBE) -> float:
    via = b_to_c.Transform(a_to_b.Transform(points))
    composed = addition._AddRigidTransforms(b_to_c, a_to_b)
    got = composed.Transform(points)
    return float(np.max(np.abs(np.asarray(via) - np.asarray(got))))


class TestAddRigidTransforms(unittest.TestCase):
    def test_pure_translation_still_adds_offsets(self):
        a_to_b = T.RigidTranslation(target_offset=(2.0, 3.0), source_rotation_center=(0.0, 0.0))
        b_to_c = T.RigidTranslation(target_offset=(1.0, -1.0), source_rotation_center=(4.0, 5.0))
        self.assertLess(_max_compose_error(a_to_b, b_to_c), 1e-5)

    def test_angles_same_center_match_sequential(self):
        a_to_b = T.Rigid(
            target_offset=(2.0, 3.0),
            source_rotation_center=(0.0, 0.0),
            angle=np.deg2rad(30),
        )
        b_to_c = T.Rigid(
            target_offset=(1.0, -1.0),
            source_rotation_center=(0.0, 0.0),
            angle=np.deg2rad(20),
        )
        err = _max_compose_error(a_to_b, b_to_c)
        self.assertLess(err, 1e-4, msg=f'compose error {err}')

    def test_angles_different_centers_match_sequential(self):
        a_to_b = T.Rigid(
            target_offset=(2.0, 3.0),
            source_rotation_center=(4.0, 5.0),
            angle=np.deg2rad(30),
        )
        b_to_c = T.Rigid(
            target_offset=(1.0, -1.0),
            source_rotation_center=(10.0, 2.0),
            angle=np.deg2rad(20),
        )
        err = _max_compose_error(a_to_b, b_to_c)
        self.assertLess(err, 1e-4, msg=f'compose error {err}')

    def test_old_offset_add_fails_for_rotated_pair(self):
        """Documents pre-fix behaviour: offset sum + angle sum is wrong."""
        a_to_b = T.Rigid(
            target_offset=(2.0, 3.0),
            source_rotation_center=(0.0, 0.0),
            angle=np.deg2rad(30),
        )
        b_to_c = T.Rigid(
            target_offset=(1.0, -1.0),
            source_rotation_center=(0.0, 0.0),
            angle=np.deg2rad(20),
        )
        via = b_to_c.Transform(a_to_b.Transform(_PROBE))
        legacy = T.Rigid(
            target_offset=a_to_b.target_offset + b_to_c.target_offset,
            source_rotation_center=a_to_b.source_space_center_of_rotation,
            angle=a_to_b.angle + b_to_c.angle,
        )
        legacy_err = float(np.max(np.abs(via - legacy.Transform(_PROBE))))
        self.assertGreater(legacy_err, 0.5)
        self.assertLess(_max_compose_error(a_to_b, b_to_c), 1e-4)

    @settings(max_examples=40, deadline=None)
    @given(
        angle_ab=st.floats(-np.pi / 2, np.pi / 2, allow_nan=False, allow_infinity=False),
        angle_bc=st.floats(-np.pi / 2, np.pi / 2, allow_nan=False, allow_infinity=False),
        ox_ab=st.floats(-20, 20, allow_nan=False, allow_infinity=False),
        oy_ab=st.floats(-20, 20, allow_nan=False, allow_infinity=False),
        ox_bc=st.floats(-20, 20, allow_nan=False, allow_infinity=False),
        oy_bc=st.floats(-20, 20, allow_nan=False, allow_infinity=False),
        cx_ab=st.floats(-30, 30, allow_nan=False, allow_infinity=False),
        cy_ab=st.floats(-30, 30, allow_nan=False, allow_infinity=False),
        cx_bc=st.floats(-30, 30, allow_nan=False, allow_infinity=False),
        cy_bc=st.floats(-30, 30, allow_nan=False, allow_infinity=False),
    )
    def test_compose_matches_sequential_hypothesis(
            self, angle_ab, angle_bc, ox_ab, oy_ab, ox_bc, oy_bc,
            cx_ab, cy_ab, cx_bc, cy_bc):
        a_to_b = T.Rigid(
            target_offset=(oy_ab, ox_ab),
            source_rotation_center=(cy_ab, cx_ab),
            angle=angle_ab,
        )
        b_to_c = T.Rigid(
            target_offset=(oy_bc, ox_bc),
            source_rotation_center=(cy_bc, cx_bc),
            angle=angle_bc,
        )
        self.assertLess(_max_compose_error(a_to_b, b_to_c), 1e-3)


if __name__ == '__main__':
    unittest.main()
