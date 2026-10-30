"""``find_peak``'s three no-peak exits are distinguishable in the log.

``find_peak`` returns a byte-identical ``FindPeakResult((0, 0), 0, 0.0, 0.0, 0.0)`` from
three unrelated conditions:

1. the overlap mask admits no pixels at all,
2. nothing in the correlation surface survives the cutoff,
3. the labelled components all sum to zero.

Before this change ``phasecorrelation.py`` had no ``import logging`` at all, so a caller
saw ``weight == 0`` and could not tell a misconfigured overlap window from a blank tile
from a flat correlation surface. All three now log a distinct reason at debug level.

The *result* is deliberately unchanged -- callers gate on the weight, and altering the
returned record would be a behaviour change rather than an observability one. Debug is
the right level because a blank or featureless tile is ordinary in a large mosaic and a
warning would fire per tile.

The same commit converts six ``print`` calls in ``stos_brute.py`` to the logger. Four sit
under ``if in_debug_mode():``, which is exactly the form already used twice in that same
file, so this makes the module internally consistent rather than introducing a
convention. The other two report a ``FloatingPointError`` and then return the same
zero-offset record, which on a worker process was printed to a stdout nobody was reading.

See review issue #91.
"""

from __future__ import annotations

import logging
import unittest

import numpy as np

from nornir_imageregistration.phasecorrelation import find_peak

_LOGGER_NAME = 'nornir_imageregistration.phasecorrelation'
_SIZE = 32

# The all-zero record every no-peak path returns.
_NO_PEAK = ((0.0, 0.0), 0, 0.0, 0.0, 0.0)


def _as_tuple(result) -> tuple:
    return (tuple(float(v) for v in result[0]), result[1],
            float(result[2]), float(result[3]), float(result[4]))


class _CaptureFixture(unittest.TestCase):
    """Collects records from the phasecorrelation logger."""

    def capture(self, image, **kwargs) -> tuple[tuple, list[str]]:
        with self.assertLogs(_LOGGER_NAME, level=logging.DEBUG) as caught:
            result = find_peak(image.copy(), **kwargs)
        return _as_tuple(result), [record.getMessage() for record in caught.records]

    @staticmethod
    def empty_mask_case():
        rng = np.random.default_rng(3)
        return rng.random((_SIZE, _SIZE)), np.zeros((_SIZE, _SIZE), dtype=bool)

    @staticmethod
    def flat_surface_case():
        return np.zeros((_SIZE, _SIZE), dtype=np.float64)

    @staticmethod
    def cancelling_components_case():
        """A +1/-1 pair with the cutoff forced to zero.

        Needs signed data: after ``correlation_image -= min()`` the production surface is
        non-negative, so this exit is reachable only through the public entry point.
        """
        image = np.zeros((_SIZE, _SIZE), dtype=np.float64)
        image[10, 10] = 1.0
        image[10, 11] = -1.0
        return image


class TestEachExitSaysWhichOneItIs(_CaptureFixture):

    def test_an_empty_overlap_mask_is_reported(self):
        image, mask = self.empty_mask_case()

        result, messages = self.capture(image, overlap_mask=mask)

        self.assertEqual(result, _NO_PEAK)
        self.assertTrue(any('overlap mask admits no pixels' in m for m in messages),
                        messages)

    def test_an_empty_cutoff_is_reported(self):
        result, messages = self.capture(self.flat_surface_case())

        self.assertEqual(result, _NO_PEAK)
        self.assertTrue(any('nothing survived the cutoff' in m for m in messages),
                        messages)

    def test_cancelling_components_are_reported(self):
        result, messages = self.capture(self.cancelling_components_case(), cutoff=0.0)

        self.assertEqual(result, _NO_PEAK)
        self.assertTrue(
            any('does not sum above zero' in m for m in messages),
            messages)

    def test_the_three_reasons_are_all_different(self):
        image, mask = self.empty_mask_case()
        _r1, first = self.capture(image, overlap_mask=mask)
        _r2, second = self.capture(self.flat_surface_case())
        _r3, third = self.capture(self.cancelling_components_case(), cutoff=0.0)

        reasons = {first[0], second[0], third[0]}

        self.assertEqual(len(reasons), 3,
                         f'the three exits should be distinguishable, got {reasons}')

    def test_the_messages_carry_useful_context(self):
        _result, messages = self.capture(self.flat_surface_case())

        joined = ' '.join(messages)
        self.assertIn('cutoff', joined)
        self.assertIn('image_shape', joined)


class TestTheReturnedRecordIsUnchanged(_CaptureFixture):
    """Observability only: nothing about the result may move."""

    def test_all_three_still_return_the_same_all_zero_record(self):
        image, mask = self.empty_mask_case()
        results = [
            self.capture(image, overlap_mask=mask)[0],
            self.capture(self.flat_surface_case())[0],
            self.capture(self.cancelling_components_case(), cutoff=0.0)[0],
        ]

        for index, result in enumerate(results):
            with self.subTest(exit=index):
                self.assertEqual(result, _NO_PEAK)

    def test_a_real_peak_is_untouched_and_logs_nothing(self):
        rows = np.arange(_SIZE)
        image = np.exp(-((rows[:, None] - 12.0) ** 2
                         + (rows[None, :] - 20.0) ** 2) / 8.0)

        with self.assertNoLogs(_LOGGER_NAME, level=logging.DEBUG):
            result = find_peak(image.copy())

        self.assertGreater(result[1], 0, 'a clear peak should carry weight')


class TestTheStrongestComponentIsTheOneTested(unittest.TestCase):
    """``center_of_mass`` divides by the selected label's sum, so that is what to guard.

    The old guard tested whether *every* label summed to zero. On a non-negative surface
    those are the same test, which is why the registration path never saw it: both
    ``_peak_from_correlation_image`` and ``_score_one_angle_core`` do
    ``correlation_image -= correlation_image.min()`` first.

    On a signed surface they diverge. ``argmax`` can select a component summing to zero
    while other components are negative, so the total is non-zero, the guard passes, and
    ``center_of_mass`` normalises by zero. Because nornir runs numpy with
    ``divide='raise'``, that surfaced as an unhandled ``FloatingPointError`` rather than
    the degenerate record every other unusable surface returns.

    Reaching this needs a negative cutoff value. ``cutoff`` is a fraction mapped to a
    percentile, so ``cutoff=0.0`` puts the threshold at the surface minimum; on signed
    input that is negative, and negative pixels are then labelled. See review issue #226.
    """

    @staticmethod
    def _surface_whose_strongest_component_sums_to_zero():
        """Components summing to 0 and -5 and -2. Zeros separate them: the labelling
        excludes exact zeros when the cutoff is <= 0."""
        image = np.zeros((16, 16), dtype=np.float64)
        image[2, 2], image[2, 3], image[3, 2] = 2.0, -1.0, -1.0   # sums to 0
        image[10, 10], image[10, 11] = -4.0, -1.0                 # sums to -5
        image[12, 2] = -2.0                                       # sums to -2
        return image

    def test_the_premise_holds_for_this_surface(self):
        """Guard the fixture itself: total non-zero, strongest component exactly zero."""
        import scipy as sp

        image = self._surface_whose_strongest_component_sums_to_zero()
        cutoff_value = float(np.percentile(image.ravel(), 0.0))
        self.assertLess(cutoff_value, 0.0, 'need a negative cutoff to label negatives')

        above = (image >= cutoff_value) & (image != 0)
        labels, count = sp.ndimage.label(above)
        sums = sp.ndimage.sum_labels(image, labels, np.arange(1, count + 1))

        self.assertNotEqual(float(sums.sum()), 0.0, 'the old guard must let this pass')
        self.assertEqual(float(sums[sums.argmax()]), 0.0, 'the divisor must be zero')

    def test_a_zero_sum_strongest_component_does_not_raise(self):
        result = find_peak(self._surface_whose_strongest_component_sums_to_zero(),
                           cutoff=0.0)

        self.assertEqual(_as_tuple(result), _NO_PEAK)

    def test_it_reports_why_rather_than_failing_silently(self):
        with self.assertLogs(_LOGGER_NAME, level=logging.DEBUG) as caught:
            find_peak(self._surface_whose_strongest_component_sums_to_zero(), cutoff=0.0)

        messages = [record.getMessage() for record in caught.records]
        self.assertTrue(any('does not sum above zero' in m for m in messages), messages)

    def test_smaller_signed_surfaces_are_handled_too(self):
        cases = {
            '1x5, components sum [0, -2]': np.array(
                [[1.0, -1.0, 0.0, -1.0, -1.0]], dtype=np.float64),
        }
        eight = np.zeros((8, 8), dtype=np.float64)
        eight[1, 1], eight[1, 2] = 3.0, -3.0
        eight[5, 5], eight[5, 6] = -1.0, -2.0
        cases['8x8, components sum [0, -3]'] = eight

        for name, image in cases.items():
            with self.subTest(case=name):
                self.assertEqual(_as_tuple(find_peak(image, cutoff=0.0)), _NO_PEAK)

    def test_an_all_negative_surface_reports_no_peak(self):
        """No component can sum above zero, so there is nothing to centre on."""
        image = np.zeros((16, 16), dtype=np.float64)
        image[4, 4], image[4, 5] = -1.0, -2.0
        image[9, 9] = -3.0

        self.assertEqual(_as_tuple(find_peak(image, cutoff=0.0)), _NO_PEAK)

    def test_a_genuine_peak_on_a_signed_surface_is_still_found(self):
        """The guard must reject only unusable surfaces, not every signed one."""
        image = np.zeros((16, 16), dtype=np.float64)
        image[6, 7], image[6, 8] = 5.0, 4.0    # strongest, sums to +9
        image[12, 2] = -3.0

        result = find_peak(image, cutoff=0.0)

        self.assertGreater(result[1], 0, 'a positive component should still win')

    def test_the_non_negative_path_is_bit_for_bit_unchanged(self):
        """While the surface is non-negative the new test equals the old one."""
        rows = np.arange(_SIZE)
        image = np.exp(-((rows[:, None] - 12.0) ** 2
                         + (rows[None, :] - 20.0) ** 2) / 8.0)

        result = _as_tuple(find_peak(image.copy()))

        self.assertEqual(result[0], (4.0, -4.0))
        self.assertGreater(result[1], 0)


class TestNothingIsWrittenToStdout(_CaptureFixture):
    """The module used to have no logger; make sure nothing reverts to print."""

    def test_the_no_peak_paths_print_nothing(self):
        import contextlib
        import io

        image, mask = self.empty_mask_case()
        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):
            find_peak(image.copy(), overlap_mask=mask)
            find_peak(self.flat_surface_case())

        self.assertEqual(buffer.getvalue(), '')

    def test_the_module_has_a_logger(self):
        import inspect

        from nornir_imageregistration import phasecorrelation

        source = inspect.getsource(phasecorrelation)

        self.assertIn('import logging', source)
        self.assertIn('getLogger', source)

    def test_no_print_calls_survive_outside_the_demo_block(self):
        """The prints at the foot of the file are a __main__ profiling demo."""
        import inspect

        from nornir_imageregistration import phasecorrelation

        source = inspect.getsource(phasecorrelation)
        library_part = source.split("if __name__ == '__main__':")[0]

        offenders = [line.strip() for line in library_part.splitlines()
                     if 'print(' in line and not line.strip().startswith('#')]

        self.assertEqual(offenders, [])


class TestStosBruteUsesTheLoggerToo(unittest.TestCase):
    """The second half of the finding: stos_brute printed instead of logging."""

    def test_no_print_calls_survive_outside_the_demo_block(self):
        import inspect

        from nornir_imageregistration import stos_brute

        source = inspect.getsource(stos_brute)
        library_part = source.split("if __name__ == '__main__':")[0]

        offenders = [line.strip() for line in library_part.splitlines()
                     if 'print(' in line and not line.strip().startswith('#')]

        self.assertEqual(offenders, [], f'still printing: {offenders}')

    def test_the_debug_prints_became_debug_logs(self):
        """Four sites under in_debug_mode() now match the two that already logged."""
        import inspect

        from nornir_imageregistration import stos_brute

        lines = inspect.getsource(stos_brute).splitlines()
        guards = [i for i, line in enumerate(lines)
                  if 'in_debug_mode()' in line and line.strip().startswith(('if', 'elif'))]

        self.assertGreaterEqual(len(guards), 6)
        for index in guards:
            follower = lines[index + 1].strip()
            with self.subTest(line=index + 1):
                self.assertFalse(follower.startswith('print('),
                                 f'line {index + 2} still prints: {follower}')

    def test_the_floating_point_handlers_warn(self):
        import inspect

        from nornir_imageregistration import stos_brute

        source = inspect.getsource(stos_brute)

        self.assertIn('Floating point error normalizing', source)
        occurrences = source.count('Floating point error normalizing')
        self.assertEqual(occurrences, 2, 'both handlers should warn')


if __name__ == '__main__':
    unittest.main()
