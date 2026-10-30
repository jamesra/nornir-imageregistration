"""Refine must run every pass the caller asked for.

``RefineTransform`` marks the final pass in two places:

* at the top of the loop, ``if i == settings.num_iterations`` -- correct, and
  reached on every path;
* near the bottom, after ``i += 1``, a second check commented "Check if the next
  pass is the last pass" but written ``if i == settings.num_iterations - 1``.

Because ``i`` has already been incremented there it holds the *next* pass number
(the cell-size calls just above pass ``pass_index=i - 1`` for that reason), so the
second check fired one pass early and the loop broke a pass short.

It hid because the block is skipped whenever a pass grows the cell size after
finding nothing usable, which is every pass at small cell sizes on this fixture.
Line tracing showed the offending line executing 0 times at cell 64, once
without firing at cell 128, and firing at cell 256 -- where refine ran 3 of 4
requested passes.

Asserted on the highest pass number reached rather than the number of progress
reports, because a separate defect can re-report a pass (see the duplicate
``pass 1`` in the review's ``C03-B008``), nondeterministically.
"""
from __future__ import annotations

import os
import re
import tempfile
import unittest

from nornir_imageregistration import local_distortion_correction as ldc

_FIXTURE = os.path.join(
    os.path.dirname(__file__), 'fixtures', 'idoc_690_691', 'StosBrute16',
    '690-691_ctrl-TEM_Leveled_map-TEM_Leveled.stos')

# Cell size matters: the off-by-one block is unreachable at 64 on this fixture
# because every pass takes the grow-cell-size shortcut instead.
_CELL = 256


def _passes_reached(num_iterations: int, cell: int = _CELL) -> list[int]:
    """Run refine and return the pass numbers it announced."""
    seen: list[int] = []
    original = ldc.report_progress

    def spy(callback, current, total, label, *args, **kwargs):
        match = re.match(r'Refine pass (\d+)/(\d+)', str(label))
        if match:
            seen.append(int(match.group(1)))
        return original(callback, current, total, label, *args, **kwargs)

    ldc.report_progress = spy
    try:
        with tempfile.TemporaryDirectory() as output_dir:
            ldc.RefineStosFile(
                InputStos=_FIXTURE,
                OutputStosPath=os.path.join(output_dir, 'Final.stos'),
                num_iterations=num_iterations,
                cell_size=(cell, cell),
                grid_spacing=(cell // 2, cell // 2),
                angles_to_search=[0],
                max_travel_for_finalization=None,
                max_travel_for_finalization_improvement=None,
                min_alignment_overlap=0.5,
                min_unmasked_area=0.49)
    finally:
        ldc.report_progress = original
    return seen


@unittest.skipUnless(os.path.isfile(_FIXTURE), f'Bundled STOS fixture missing: {_FIXTURE}')
class TestRefineRunsEveryRequestedPass(unittest.TestCase):

    def test_four_iterations_run_four_passes(self):
        """The reproducing case: this reached only pass 3 before the fix."""
        passes = _passes_reached(4)

        self.assertTrue(passes, 'no refine passes were announced')
        self.assertEqual(max(passes), 4, f'refine stopped at pass {max(passes)} of 4: {passes}')

    def test_five_iterations_run_five_passes(self):
        passes = _passes_reached(5)

        self.assertEqual(max(passes), 5, f'refine stopped at pass {max(passes)} of 5: {passes}')

    def test_passes_are_announced_in_order(self):
        passes = _passes_reached(4)

        self.assertEqual(passes, sorted(passes), f'pass numbers went backwards: {passes}')
        self.assertEqual(passes[0], 1, f'first announced pass was {passes[0]}')

    def test_two_iterations_run_two_passes(self):
        """Guard the boundary the old arithmetic happened to get right.

        ``num_iterations=1`` is not covered: it raises from the final-pass
        diagnostics before returning, on both sides of this fix, because the EMA
        has no samples yet on a single pass. Tracked separately.
        """
        passes = _passes_reached(2)

        self.assertEqual(max(passes), 2)


@unittest.skipUnless(os.path.isfile(_FIXTURE), f'Bundled STOS fixture missing: {_FIXTURE}')
class TestFinalPassIsMarkedByTheLoopHeader(unittest.TestCase):
    """The remaining mechanism must be the top-of-loop check, on every path."""

    def test_no_off_by_one_comparison_governs_final_pass(self):
        import inspect

        source = inspect.getsource(ldc.RefineTransform)

        # assertIn/assertNotIn would dump the whole function into the failure message.
        self.assertTrue('if i == settings.num_iterations:' in source,
                        'the top-of-loop final-pass check is gone')
        self.assertFalse('if i == settings.num_iterations - 1:' in source,
                         'the off-by-one final-pass check is back; it declares the '
                         'final pass early because i is already incremented')


if __name__ == '__main__':
    unittest.main()
