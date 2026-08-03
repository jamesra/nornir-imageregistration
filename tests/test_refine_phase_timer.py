"""Unit tests for refine phase-timer buckets and failure-mode stats."""

from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.refine_shared.failure_mode_stats import (
    identity_lock_mask,
    peak_direction_coherence,
    spatial_half_stats,
)
from nornir_imageregistration.refine_shared.phase_timer import RefinePhaseTimer


class TestRefinePhaseTimerBuckets(unittest.TestCase):
    """New measure-gap buckets are registered and record wall time."""

    def test_measure_gap_phases_listed(self) -> None:
        for name in ('grid_build', 'approx_rigid', 'record_assemble'):
            self.assertIn(name, RefinePhaseTimer.PHASES)

    def test_section_records_new_buckets(self) -> None:
        timer = RefinePhaseTimer(enabled=True)
        with timer.section('grid_build'):
            _ = sum(range(1000))
        with timer.section('approx_rigid'):
            _ = sum(range(1000))
        with timer.section('record_assemble'):
            _ = sum(range(1000))
        self.assertGreater(timer.totals['grid_build'], 0.0)
        self.assertGreater(timer.totals['approx_rigid'], 0.0)
        self.assertGreater(timer.totals['record_assemble'], 0.0)
        self.assertEqual(timer.counts['approx_rigid'], 1)


class TestFailureModeStats(unittest.TestCase):
    """Stats helpers matching 240-241 / 241-242 diagnosis."""

    def test_identity_lock_mask(self) -> None:
        locked = np.asarray([True, True, False, True])
        travel = np.asarray([0.0, 0.4, 0.0, 2.0])
        mask = identity_lock_mask(locked, travel, travel_eps=0.5)
        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_peak_direction_coherence_aligned(self) -> None:
        py = np.asarray([10.0, 12.0, 11.0, 9.0])
        px = np.asarray([40.0, 42.0, 41.0, 39.0])
        coh = peak_direction_coherence(py, px)
        self.assertGreater(coh, 0.95)

    def test_peak_direction_coherence_opposed(self) -> None:
        py = np.asarray([10.0, -10.0])
        px = np.asarray([0.0, 0.0])
        coh = peak_direction_coherence(py, px)
        self.assertLess(coh, 0.2)

    def test_spatial_half_stats_asymmetric(self) -> None:
        # Low-x: free high travel; high-x: identity locks — 241-242 shape.
        source_x = np.asarray([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
        locked = np.asarray([False, False, False, False, True, True, True, True])
        travel = np.asarray([16.0, 15.0, 17.0, 14.0, 0.0, 0.1, 0.0, 0.2])
        stats = spatial_half_stats(source_x, locked, travel)
        self.assertLess(stats['low_x']['lock_frac'], 0.1)
        self.assertGreater(stats['low_x']['travel_med'], 10.0)
        self.assertGreater(stats['high_x']['lock_frac'], 0.9)
        self.assertGreater(stats['high_x']['identity_lock_frac'], 0.9)
        self.assertLess(stats['high_x']['travel_med'], 0.5)


if __name__ == '__main__':
    unittest.main()
