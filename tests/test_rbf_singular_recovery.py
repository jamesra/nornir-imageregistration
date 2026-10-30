"""Tests for the singular-matrix recovery in the RBF weight solve.

CalculateRBFWeights gates its rigid-transform fallback on the LinAlgError
message. It compared for equality against 'Matrix is singular.', which no
current backend emits, so the recovery below it was unreachable and a
degenerate control-point set escaped as LinAlgError. LinAlgError is not in the
(ValueError, FileNotFoundError, OSError) tuple that the SliceToVolume compose
loop catches, so it aborted the run instead of skipping the section.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import scipy.linalg

from nornir_imageregistration.transforms.one_way_rbftransform import (
    OneWayRBFWithLinearCorrection as RBF,
    _is_singular_matrix_error,
)

# The wording each backend actually produces.
SCIPY_MODERN = 'A singular matrix detected: slice(s) [0] are singular.'
SCIPY_LEGACY = 'Matrix is singular.'
NUMPY = 'Singular matrix'

SQUARE = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0]])
COLLINEAR = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])


class TestIsSingularMatrixError(unittest.TestCase):
    """The predicate has to span backends without swallowing caller bugs."""

    def test_recognizes_every_backend_wording(self):
        for message in (SCIPY_MODERN, SCIPY_LEGACY, NUMPY):
            with self.subTest(message=message):
                self.assertTrue(
                    _is_singular_matrix_error(np.linalg.LinAlgError(message)))

    def test_is_case_insensitive(self):
        self.assertTrue(
            _is_singular_matrix_error(np.linalg.LinAlgError('SINGULAR MATRIX')))

    def test_rejects_unrelated_linalg_errors(self):
        """A shape or dtype mistake is a caller bug and must keep propagating."""
        for message in ('Last 2 dimensions of the array must be square',
                        'Array must not contain infs or NaNs',
                        '0-dimensional array given'):
            with self.subTest(message=message):
                self.assertFalse(
                    _is_singular_matrix_error(np.linalg.LinAlgError(message)))

    def test_handles_error_with_no_args(self):
        self.assertFalse(_is_singular_matrix_error(np.linalg.LinAlgError()))


class TestSingularRecoveryIsReachable(unittest.TestCase):
    """A singular solve must reach the rigid fallback, not escape."""

    def _weights_with_solve_raising(self, message: str):
        """Force the solve to fail the way a backend reports singularity."""
        with mock.patch.object(scipy.linalg, 'solve',
                               side_effect=np.linalg.LinAlgError(message)):
            return RBF.CalculateRBFWeights(SQUARE, SQUARE + 5.0,
                                           RBF.DefaultBasisFunction)

    def test_modern_scipy_message_recovers(self):
        """This is the message that made the recovery dead code."""
        weights, use_rigid = self._weights_with_solve_raising(SCIPY_MODERN)

        self.assertTrue(use_rigid)
        self.assertTrue(np.all(np.isfinite(weights)))

    def test_legacy_and_numpy_messages_recover(self):
        for message in (SCIPY_LEGACY, NUMPY):
            with self.subTest(message=message):
                weights, use_rigid = self._weights_with_solve_raising(message)
                self.assertTrue(use_rigid)
                self.assertTrue(np.all(np.isfinite(weights)))

    def test_recovered_weights_describe_the_translation(self):
        """The fallback is a rigid fit, so the offset has to survive it."""
        offset = np.array([2.0, 7.0])
        with mock.patch.object(scipy.linalg, 'solve',
                               side_effect=np.linalg.LinAlgError(SCIPY_MODERN)):
            weights, use_rigid = RBF.CalculateRBFWeights(
                SQUARE, SQUARE + offset, RBF.DefaultBasisFunction)

        self.assertTrue(use_rigid)
        # Only the three linear terms per axis are populated by the fallback.
        half = len(weights) // 2
        self.assertTrue(np.allclose(weights[0:half - 3], 0.0))
        self.assertTrue(np.allclose(weights[half:-3], 0.0))

    def test_unrelated_linalg_error_still_propagates(self):
        with self.assertRaises(np.linalg.LinAlgError):
            self._weights_with_solve_raising(
                'Last 2 dimensions of the array must be square')


class TestDegenerateInputIsCatchable(unittest.TestCase):
    """The compose loop only catches ValueError, so degeneracy must land there."""

    def test_collinear_points_raise_value_error_not_linalg_error(self):
        with self.assertRaises(ValueError) as caught:
            RBF.CalculateRBFWeights(COLLINEAR, COLLINEAR * 1.1,
                                    RBF.DefaultBasisFunction)

        self.assertNotIsInstance(caught.exception, np.linalg.LinAlgError)

    def test_well_conditioned_input_is_untouched(self):
        """The common path must not change behavior."""
        weights, use_rigid = RBF.CalculateRBFWeights(SQUARE, SQUARE * 1.1,
                                                     RBF.DefaultBasisFunction)

        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertEqual(len(weights), 2 * (len(SQUARE) + 3))
        del use_rigid


if __name__ == '__main__':
    unittest.main()
