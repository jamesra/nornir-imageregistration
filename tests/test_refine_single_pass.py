"""A single-pass refine must return a transform, not raise for the sake of a log line.

``RefineTransform(..., num_iterations=1)`` always raised::

    File "local_distortion_correction.py", in RefineTransform
        f'Exponential Moving Average diagnostic inflection: {cutoff_ema.ema_value}\n')
    File "mathfuncs/ema.py", in ema_value
        raise ValueError("Cannot calculate EMA until at least one sample is collected")

The cause is not that the read beats the first sample -- ``cutoff_ema.add`` runs a
dozen lines *above* the log statement. It is that the window is
``settings.num_iterations // 2``, which is ``0`` for a single pass, and ``add``
saturates the sample count at the window size::

    self._num_samples_collected = (
        self._num_samples_collected + 1
        if self._num_samples_collected < self._num_samples   # 0 < 0 is False
        else self._num_samples)                             # so it stays 0

With a window of ``0`` the count never leaves ``0``, so ``ema_value`` raises no
matter how many samples arrive -- five ``add`` calls still left it at ``0``. The
value is diagnostic only: it is logged and never gates a decision, since
``transform_cutoff_value`` is hard-wired to ``-inf``. So a log line aborted the
refine it was describing.

Fixed in three layers: the window is floored at 1, ``EMA`` refuses a window below 1
at construction rather than failing at a distant read, and the diagnostic reports
"n/a" when there is no average yet so it can never abort a refine again.
"""
from __future__ import annotations

import os
import tempfile
import unittest

import pytest

from nornir_imageregistration import local_distortion_correction as ldc
from nornir_imageregistration.mathfuncs import EMA

_FIXTURE = os.path.join(
    os.path.dirname(__file__), 'fixtures', 'idoc_690_691', 'StosBrute16',
    '690-691_ctrl-TEM_Leveled_map-TEM_Leveled.stos')

_CELL = 256


class TestEmaWindow(unittest.TestCase):
    """The window arithmetic, without paying for a refine."""

    def test_a_zero_window_is_refused(self):
        with self.assertRaises(ValueError):
            EMA(0, 2)

    def test_a_negative_window_is_refused(self):
        with self.assertRaises(ValueError):
            EMA(-1, 2)

    def test_a_single_sample_window_averages(self):
        ema = EMA(1, 2)
        ema.add(5.0)
        self.assertEqual(ema.ema_value, 5.0)

    def test_the_window_a_single_pass_refine_asks_for_is_usable(self):
        """num_iterations // 2 is 0 for one pass, so the caller floors it at 1."""
        for num_iterations in (1, 2, 3, 4, 6):
            with self.subTest(num_iterations=num_iterations):
                ema = EMA(max(1, num_iterations // 2), 2)
                ema.add(1.0)
                self.assertTrue(ema.has_samples)
                self.assertEqual(ema.ema_value, 1.0)

    def test_has_samples_reports_before_any_sample(self):
        ema = EMA(2, 2)
        self.assertFalse(ema.has_samples)
        with self.assertRaises(ValueError):
            _ = ema.ema_value

        ema.add(3.0)
        self.assertTrue(ema.has_samples)
        self.assertEqual(ema.ema_value, 3.0)

    def test_samples_accumulate_up_to_the_window_and_then_saturate(self):
        ema = EMA(2, 2)
        for value in (1.0, 2.0, 3.0, 4.0):
            ema.add(value)
            self.assertTrue(ema.has_samples)


@pytest.mark.slow
class TestSinglePassRefine(unittest.TestCase):
    """The reported symptom, end to end."""

    def _refine(self, num_iterations: int) -> int:
        """Run refine and return the size of the stos file it wrote.

        ``RefineStosFile`` returns None by contract and reports through the file it
        writes, so the output is what says the refine finished.
        """
        with tempfile.TemporaryDirectory() as output_dir:
            output = os.path.join(output_dir, 'Final.stos')
            ldc.RefineStosFile(
                InputStos=_FIXTURE,
                OutputStosPath=output,
                num_iterations=num_iterations,
                cell_size=(_CELL, _CELL),
                grid_spacing=(_CELL // 2, _CELL // 2),
                angles_to_search=[0],
                max_travel_for_finalization=None,
                max_travel_for_finalization_improvement=None,
                min_alignment_overlap=0.5,
                min_unmasked_area=0.49)
            self.assertTrue(os.path.isfile(output), 'refine wrote no output')
            return os.path.getsize(output)

    def test_a_single_pass_refine_completes(self):
        self.assertGreater(self._refine(1), 0)

    def test_two_passes_still_complete(self):
        """The floor must not disturb the windows that already worked."""
        self.assertGreater(self._refine(2), 0)


if __name__ == '__main__':
    unittest.main()
