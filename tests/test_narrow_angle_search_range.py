"""``NarrowAngleSearchRangeWithResult`` survives float drift and narrow brackets.

Two defects, both reproduced before the fix:

1. The bracketing angle was located with ``sorted_angles.index(target_angle)``, an exact
   float equality test. The caller normally passes an angle that was selected from this
   same range, so it is usually an exact member -- but any arithmetic that rebuilds the
   range, or a narrowing through float32, leaves it a few ULP off and ``list.index``
   raised ``ValueError: x not in list``. Of a 19-angle 0.2-degree grid, 16 angles change
   value under a float32 round trip.

2. ``nSteps = int(refine_search_range / min_step_size)`` floors to 0 once the bracket is
   narrower than one step, and the next line divides by it. Nornir runs numpy with
   ``divide='raise'`` (``np.geterr()['divide'] == 'raise'``), so this was a hard
   ``FloatingPointError``, not an inf. With the caller's ``min_step_size=0.25`` it fires
   for any incoming range spaced finer than 0.125 degrees.

Two dead locals, ``iBelow`` and ``iAbove``, were assigned and never read; they are gone.

What is *not* a defect, and is now documented rather than "fixed": the returned range
collapses to just the target angle whenever the incoming angles are spaced at or below
twice ``min_step_size``. The range holds only points strictly between the two brackets,
and the brackets were scored on the previous pass, so once the bracket is two steps wide
there is genuinely no angle left that is both new and coarser than the floor. Measured
with the caller's ``min_step_size=0.25``:

===========  ========  ==================
spacing      nSteps    angles returned
===========  ========  ==================
0.125        1         1 (seed only)
0.200        1         1 (seed only)
0.250        2         1 (seed only)
0.300        2         1 (seed only)
0.500        4         3
1.000        8         7
===========  ========  ==================

So a refine pass over a range spaced 0.3 degrees or finer cannot improve on its seed.
That is worth knowing but is not this function's bug to fix.

See review issue #90.
"""

from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.stos_brute import NarrowAngleSearchRangeWithResult

# What the only production caller passes (stos_brute.py, _finalize_bruteforce_candidate).
_CALLER_MIN_STEP = 0.25


def _grid(spacing: float, count: int = 11) -> np.ndarray:
    """A symmetric angle grid centred on zero, as the settings supply."""
    half = count // 2
    return np.asarray([spacing * x for x in range(-half, half + 1)], dtype=float)


class TestFloatDriftNoLongerRaises(unittest.TestCase):
    """Defect 1: the lookup was an exact equality test."""

    def setUp(self):
        self.angles = _grid(0.5)
        self.target = float(self.angles[7])   # 1.0, an exact member

    def test_an_exact_member_still_works(self):
        result = NarrowAngleSearchRangeWithResult(
            self.angles, _CALLER_MIN_STEP, self.target)

        self.assertIn(self.target, result)

    def test_a_target_one_ulp_off_is_accepted(self):
        drifted = np.nextafter(self.target, 1e9)

        result = NarrowAngleSearchRangeWithResult(
            self.angles, _CALLER_MIN_STEP, drifted)

        self.assertIn(self.target, result)

    def test_a_target_off_by_a_picodegree_is_accepted(self):
        result = NarrowAngleSearchRangeWithResult(
            self.angles, _CALLER_MIN_STEP, self.target + 1e-12)

        self.assertIn(self.target, result)

    def test_a_target_that_went_through_float32_is_accepted(self):
        """16 of 19 angles on a 0.2-degree grid change under this round trip."""
        angles = _grid(0.2, 19)
        for index in (3, 7, 12, 16):
            exact = float(angles[index])
            through_f32 = float(np.float32(exact))
            with self.subTest(index=index, exact=exact):
                result = NarrowAngleSearchRangeWithResult(
                    angles, _CALLER_MIN_STEP, through_f32)
                self.assertIn(exact, result)

    def test_drift_does_not_leave_a_near_duplicate_in_the_result(self):
        """Snapping to the matched member avoids scoring one angle twice."""
        drifted = np.nextafter(self.target, 1e9)

        result = sorted(NarrowAngleSearchRangeWithResult(
            self.angles, _CALLER_MIN_STEP, drifted))

        gaps = np.diff(result)
        if gaps.size:
            self.assertGreater(float(gaps.min()), 1e-9,
                               f'two effectively identical angles present: {result}')
        self.assertNotIn(float(drifted), result)

    def test_an_angle_from_a_different_range_is_still_rejected(self):
        """Drift tolerance must not become "snap anything to an endpoint"."""
        with self.assertRaises(ValueError) as caught:
            NarrowAngleSearchRangeWithResult(self.angles, _CALLER_MIN_STEP, 99.0)

        self.assertIn('99.0', str(caught.exception))

    def test_the_rejection_message_names_the_range(self):
        with self.assertRaises(ValueError) as caught:
            NarrowAngleSearchRangeWithResult(self.angles, _CALLER_MIN_STEP, 99.0)

        message = str(caught.exception)
        self.assertIn('search range', message)
        self.assertNotIn('not in list', message,
                         'the opaque list.index message should be gone')


class TestNarrowBracketsNoLongerDivideByZero(unittest.TestCase):
    """Defect 2: int() floored nSteps to zero, then the next line divided by it."""

    def test_numpy_would_raise_on_that_divide(self):
        """Pins the premise: an inf would have been far less visible."""
        self.assertEqual(np.geterr()['divide'], 'raise')

    def test_spacings_finer_than_half_a_step_are_handled(self):
        for spacing in (0.001, 0.01, 0.05, 0.1, 0.124):
            with self.subTest(spacing=spacing):
                angles = _grid(spacing)
                result = NarrowAngleSearchRangeWithResult(
                    angles, _CALLER_MIN_STEP, float(angles[5]))

                self.assertGreaterEqual(len(result), 1)
                for angle in result:
                    self.assertTrue(np.isfinite(angle))

    def test_a_degenerate_bracket_returns_the_seed(self):
        angles = _grid(0.01)
        target = float(angles[5])

        result = NarrowAngleSearchRangeWithResult(angles, _CALLER_MIN_STEP, target)

        self.assertEqual(result, {target})

    def test_a_non_positive_min_step_is_rejected_rather_than_dividing(self):
        angles = _grid(0.5)
        for bad in (0.0, -0.25):
            with self.subTest(min_step=bad):
                with self.assertRaises(ValueError):
                    NarrowAngleSearchRangeWithResult(angles, bad, float(angles[5]))


class TestTheRefinedRangeIsStillCorrect(unittest.TestCase):
    """Behaviour that must survive the rewrite."""

    def test_the_seed_is_always_included(self):
        for spacing in (0.1, 0.25, 0.5, 1.0, 5.0):
            with self.subTest(spacing=spacing):
                angles = _grid(spacing)
                target = float(angles[5])

                result = NarrowAngleSearchRangeWithResult(
                    angles, _CALLER_MIN_STEP, target)

                self.assertIn(target, result)

    def test_the_range_stays_inside_the_bracketing_angles(self):
        angles = _grid(1.0)
        target = float(angles[5])

        result = NarrowAngleSearchRangeWithResult(angles, _CALLER_MIN_STEP, target)

        below, above = float(angles[4]), float(angles[6])
        for angle in result:
            self.assertGreaterEqual(angle, below)
            self.assertLessEqual(angle, above)

    def test_the_step_floor_is_respected(self):
        angles = _grid(5.0)

        result = sorted(NarrowAngleSearchRangeWithResult(
            angles, _CALLER_MIN_STEP, float(angles[5])))

        gaps = np.diff(result)
        self.assertGreaterEqual(float(gaps.min()), _CALLER_MIN_STEP - 1e-9)

    def test_a_seed_at_either_extreme_extrapolates_rather_than_failing(self):
        angles = _grid(1.0)
        for index in (0, len(angles) - 1):
            with self.subTest(index=index):
                target = float(angles[index])

                result = NarrowAngleSearchRangeWithResult(
                    angles, _CALLER_MIN_STEP, target)

                self.assertIn(target, result)
                self.assertTrue(all(np.isfinite(a) for a in result))

    def test_an_unsorted_input_gives_the_same_answer(self):
        angles = _grid(1.0)
        target = float(angles[5])
        shuffled = angles.copy()
        np.random.default_rng(11).shuffle(shuffled)

        self.assertEqual(
            NarrowAngleSearchRangeWithResult(angles, _CALLER_MIN_STEP, target),
            NarrowAngleSearchRangeWithResult(shuffled, _CALLER_MIN_STEP, target))

    def test_too_few_angles_is_still_rejected(self):
        with self.assertRaises(ValueError):
            NarrowAngleSearchRangeWithResult(np.asarray([1.0]), _CALLER_MIN_STEP, 1.0)


class TestTheCollapseIsDocumentedNotAccidental(unittest.TestCase):
    """The refine legitimately has nothing to offer on an already-fine grid."""

    def test_fine_grids_return_only_the_seed(self):
        for spacing in (0.125, 0.2, 0.25, 0.3):
            with self.subTest(spacing=spacing):
                angles = _grid(spacing)
                target = float(angles[5])

                result = NarrowAngleSearchRangeWithResult(
                    angles, _CALLER_MIN_STEP, target)

                self.assertEqual(result, {target},
                                 'a grid this fine has no coarser angle left to try')

    def test_coarse_grids_do_offer_new_angles(self):
        for spacing, expected in ((0.5, 3), (1.0, 7), (2.0, 15)):
            with self.subTest(spacing=spacing):
                angles = _grid(spacing)

                result = NarrowAngleSearchRangeWithResult(
                    angles, _CALLER_MIN_STEP, float(angles[5]))

                self.assertEqual(len(result), expected)

    def test_the_docstring_records_the_collapse(self):
        doc = NarrowAngleSearchRangeWithResult.__doc__ or ''

        self.assertIn('collapses', doc.lower())


class TestDeadLocalsAreGone(unittest.TestCase):
    def test_the_unused_index_locals_were_removed(self):
        import inspect

        from nornir_imageregistration import stos_brute

        source = inspect.getsource(stos_brute.NarrowAngleSearchRangeWithResult)

        for name in ('iBelow', 'iAbove'):
            with self.subTest(name=name):
                self.assertNotIn(name, source)

    def test_the_lookup_is_no_longer_list_index(self):
        import inspect

        from nornir_imageregistration import stos_brute

        source = inspect.getsource(stos_brute.NarrowAngleSearchRangeWithResult)
        code = '\n'.join(line for line in source.splitlines()
                         if not line.strip().startswith('#'))

        self.assertNotIn('.index(target_angle)', code)


if __name__ == '__main__':
    unittest.main()
