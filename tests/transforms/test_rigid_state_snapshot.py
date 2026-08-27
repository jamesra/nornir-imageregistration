"""Tests for GetRigidState / SetRigidState, the in-place gesture-undo contract.

Pyre shares one rigid model across all STOS windows and mutates it in place, so
cancelling a drag needs to restore parameters onto that same instance without
replacing it or disturbing its change subscribers.
"""

from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration import EnsureNumpyArray
from nornir_imageregistration.transforms.rigid import (
    CenteredSimilarity2DTransform,
    Rigid,
    RigidTranslation,
)


class TestRigidTranslationState(unittest.TestCase):
    """The translation-only model carries angle, offset and rotation center."""

    def test_round_trip_restores_offset(self):
        transform = RigidTranslation(target_offset=(10.0, 20.0))
        state = transform.GetRigidState()

        transform.TranslateWarped(np.array([5.0, -3.0]))
        self.assertFalse(np.allclose(transform.target_offset, (10.0, 20.0)))

        transform.SetRigidState(state)

        np.testing.assert_allclose(transform.target_offset, (10.0, 20.0))

    def test_snapshot_is_a_copy_not_a_view(self):
        """A snapshot aliasing live storage is the bug this API exists to avoid."""
        transform = RigidTranslation(target_offset=(10.0, 20.0))
        state = transform.GetRigidState()

        transform.TranslateWarped(np.array([100.0, 100.0]))

        np.testing.assert_allclose(state['target_offset'], (10.0, 20.0))

    def test_restore_preserves_object_identity(self):
        transform = RigidTranslation(target_offset=(1.0, 2.0))
        state = transform.GetRigidState()
        before = id(transform)

        transform.TranslateWarped(np.array([9.0, 9.0]))
        transform.SetRigidState(state)

        self.assertEqual(id(transform), before)


class TestRigidStateKeepsListeners(unittest.TestCase):
    """__setstate__ clears listeners; SetRigidState must not."""

    def setUp(self) -> None:
        self.transform = Rigid(target_offset=(4.0, 8.0),
                               source_rotation_center=(0.0, 0.0),
                               angle=0.25)
        self.calls: list[int] = []
        self.transform.AddOnChangeEventListener(lambda *a, **k: self.calls.append(1))

    def test_set_rigid_state_notifies_listeners(self):
        state = self.transform.GetRigidState()
        self.transform.TranslateWarped(np.array([3.0, 3.0]))
        self.calls.clear()

        self.transform.SetRigidState(state)

        self.assertTrue(self.calls, 'views must be told to repaint after an undo')

    def test_listeners_survive_the_restore(self):
        state = self.transform.GetRigidState()
        self.transform.SetRigidState(state)
        self.calls.clear()

        self.transform.TranslateWarped(np.array([1.0, 1.0]))

        self.assertTrue(self.calls,
                        'listeners were dropped, so later edits stop repainting')

    def test_setstate_by_contrast_drops_listeners(self):
        """Documents why the pickle hooks are unsuitable for a live model."""
        self.transform.__setstate__(self.transform.__getstate__())
        self.calls.clear()

        self.transform.TranslateWarped(np.array([1.0, 1.0]))

        self.assertFalse(self.calls)


class TestRigidStateFullParameters(unittest.TestCase):
    """Angle, flip and relative scale all have to come back."""

    def test_angle_is_restored(self):
        transform = Rigid(target_offset=(0.0, 0.0),
                          source_rotation_center=(5.0, 5.0),
                          angle=0.5)
        state = transform.GetRigidState()

        transform.RotateSourcePoints(0.75, None)
        self.assertNotAlmostEqual(transform.angle, 0.5)

        transform.SetRigidState(state)

        self.assertAlmostEqual(transform.angle, 0.5)

    def test_forward_matrix_matches_after_restore(self):
        transform = Rigid(target_offset=(2.0, 3.0),
                          source_rotation_center=(1.0, 1.0),
                          angle=0.2)
        # forward_matrix follows the active computation lib, so pull it to the
        # host before comparing.
        expected = np.array(EnsureNumpyArray(transform.forward_matrix), copy=True)
        state = transform.GetRigidState()

        transform.TranslateWarped(np.array([12.0, -7.0]))
        transform.RotateSourcePoints(0.4, None)
        transform.SetRigidState(state)

        np.testing.assert_allclose(EnsureNumpyArray(transform.forward_matrix),
                                   expected, atol=1e-6)

    def test_relative_scale_is_restored(self):
        transform = CenteredSimilarity2DTransform(target_offset=(0.0, 0.0),
                                                  source_rotation_center=(0.0, 0.0),
                                                  angle=0.0)
        state = transform.GetRigidState()
        original_scalar = transform.scalar

        transform.ScaleWarped(2.0)
        self.assertNotAlmostEqual(transform.scalar, original_scalar)

        transform.SetRigidState(state)

        self.assertAlmostEqual(transform.scalar, original_scalar)

    def test_transform_output_matches_after_restore(self):
        """The observable behavior, not just the stored fields."""
        transform = CenteredSimilarity2DTransform(target_offset=(3.0, 4.0),
                                                  source_rotation_center=(2.0, 2.0),
                                                  angle=0.1)
        probe = np.array([[10.0, 12.0], [0.0, 0.0], [-5.0, 7.0]])
        expected = np.array(EnsureNumpyArray(transform.Transform(probe)), copy=True)
        state = transform.GetRigidState()

        transform.TranslateWarped(np.array([25.0, 30.0]))
        transform.ScaleWarped(1.5)
        transform.SetRigidState(state)

        np.testing.assert_allclose(EnsureNumpyArray(transform.Transform(probe)),
                                   expected, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
