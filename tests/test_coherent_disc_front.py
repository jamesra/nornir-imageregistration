"""Unit tests for coherent disc-front mesh raw-preserve."""

from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.alignment_record import EnhancedAlignmentRecord
from nornir_imageregistration.refine_shared.anchor_smooth import (
    AnchorSmoothSettings,
    smooth_peaks_from_locked_anchors,
)
from nornir_imageregistration.refine_shared.cell_roles import (
    DISC_FRONT_MIN_CELLS,
    coherent_discontinuity_raw_preserve_ids,
)
from nornir_imageregistration.refine_shared.peak_ratio_gates import soft_discontinuity_ids
from nornir_imageregistration.transforms.rigid import RigidTranslation


def _rec(
        key: tuple[int, int],
        *,
        peak: tuple[float, float],
        weight: float = 12.0,
        peak_ratio: float | None = 1.03,
) -> EnhancedAlignmentRecord:
    """Build a minimal alignment record for disc-front tests."""
    return EnhancedAlignmentRecord(
        ID=key,
        TargetPoint=np.asarray((float(key[0]) * 10.0, float(key[1]) * 10.0), dtype=np.float64),
        SourcePoint=np.asarray((float(key[0]) * 10.0, float(key[1]) * 10.0), dtype=np.float64),
        peak=np.asarray(peak, dtype=np.float64),
        weight=weight,
        angle=0.0,
        flipped_ud=False,
        peak_ratio=peak_ratio,
    )


class TestCoherentDiscontinuityRawPreserve(unittest.TestCase):
    """Spatial coherence gate for ambiguous tear-front mesh preserve."""

    def test_coherent_front_preserves_ambiguous_ratio(self) -> None:
        """A strip of active disc cells with shared direction enters the set."""
        max_travel = 5.0
        peak = (0.0, 12.0)
        records = [
            _rec((0, col), peak=peak, peak_ratio=1.03)
            for col in range(DISC_FRONT_MIN_CELLS)
        ]
        disc_ids = {rec.ID for rec in records}
        preserved = coherent_discontinuity_raw_preserve_ids(
            records, disc_ids, max_travel=max_travel)
        self.assertEqual(preserved, disc_ids)
        soft = soft_discontinuity_ids(records, disc_ids)
        self.assertEqual(soft, set())

    def test_isolated_ambiguous_disc_denied(self) -> None:
        """Small / isolated ambiguous disc dirt stays out of mesh preserve."""
        max_travel = 5.0
        records = [
            _rec((0, 0), peak=(0.0, 12.0), peak_ratio=1.03),
            _rec((0, 1), peak=(0.0, 12.0), peak_ratio=1.03),
            _rec((5, 5), peak=(0.0, 0.1), peak_ratio=1.5),
        ]
        disc_ids = {(0, 0), (0, 1)}
        preserved = coherent_discontinuity_raw_preserve_ids(
            records, disc_ids, max_travel=max_travel)
        self.assertEqual(preserved, set())

    def test_bimodal_connected_component_denied(self) -> None:
        """One connected component with opposing ± peaks fails coherence."""
        max_travel = 5.0
        records: list[EnhancedAlignmentRecord] = []
        # Six cells in a row: first three +x, last three -x → one component.
        for col in range(3):
            records.append(_rec((0, col), peak=(0.0, 12.0), peak_ratio=1.03))
        for col in range(3, 6):
            records.append(_rec((0, col), peak=(0.0, -12.0), peak_ratio=1.03))
        disc_ids = {rec.ID for rec in records}
        preserved = coherent_discontinuity_raw_preserve_ids(
            records, disc_ids, max_travel=max_travel)
        self.assertEqual(preserved, set())

    def test_soft_discontinuity_unchanged_for_coherent_ambiguous(self) -> None:
        """Lock soft floors still require pr >= PEAK_RATIO_MIN."""
        max_travel = 5.0
        records = [
            _rec((0, col), peak=(0.0, 12.0), peak_ratio=1.03)
            for col in range(DISC_FRONT_MIN_CELLS)
        ]
        disc_ids = {rec.ID for rec in records}
        coherent = coherent_discontinuity_raw_preserve_ids(
            records, disc_ids, max_travel=max_travel)
        soft = soft_discontinuity_ids(records, disc_ids)
        self.assertEqual(len(coherent), DISC_FRONT_MIN_CELLS)
        self.assertEqual(soft, set())

    def test_production_set_not_all_disc_on_mixed_grid(self) -> None:
        """Coherent front ∪ soft must not equal full discontinuity tags with dirt."""
        max_travel = 5.0
        front = [
            _rec((0, col), peak=(0.0, 12.0), peak_ratio=1.03)
            for col in range(DISC_FRONT_MIN_CELLS)
        ]
        dirt = [
            _rec((3, 0), peak=(0.0, 12.0), peak_ratio=1.03),
            _rec((3, 2), peak=(12.0, 0.0), peak_ratio=1.03),
            _rec((4, 1), peak=(0.0, -12.0), peak_ratio=1.03),
        ]
        records = front + dirt
        disc_ids = {rec.ID for rec in records}
        coherent = coherent_discontinuity_raw_preserve_ids(
            records, disc_ids, max_travel=max_travel)
        soft = soft_discontinuity_ids(records, disc_ids)
        production = coherent | soft
        self.assertNotEqual(production, disc_ids)
        self.assertTrue({rec.ID for rec in front}.issubset(production))
        self.assertFalse(any(rec.ID in production for rec in dirt))

    def test_anchor_smooth_keeps_coherent_front_raw_peaks(self) -> None:
        """Coherent-front IDs passed to anchor-smooth keep large raw peaks."""
        max_travel = 5.0
        raw_peak = (0.0, 40.0)
        front = [
            _rec((0, col), peak=raw_peak, peak_ratio=1.03)
            for col in range(DISC_FRONT_MIN_CELLS)
        ]
        disc_ids = {rec.ID for rec in front}
        coherent = coherent_discontinuity_raw_preserve_ids(
            front, disc_ids, max_travel=max_travel)
        self.assertEqual(coherent, disc_ids)

        transform = RigidTranslation((0.0, 0.0))
        finalized = {
            (1, 0): _rec((1, 0), peak=(0.0, 0.0), peak_ratio=1.5),
            (1, 2): _rec((1, 2), peak=(0.0, 0.0), peak_ratio=1.5),
            (1, 5): _rec((1, 5), peak=(0.0, 0.0), peak_ratio=1.5),
        }
        # Identity-anchored finalized cells (peak baked to zero).
        for key, rec in list(finalized.items()):
            finalized[key] = EnhancedAlignmentRecord(
                ID=rec.ID,
                TargetPoint=rec.SourcePoint.copy(),
                SourcePoint=rec.SourcePoint.copy(),
                peak=np.zeros(2, dtype=np.float64),
                weight=rec.weight,
                angle=0.0,
                flipped_ud=False,
                peak_ratio=1.5,
            )
        settings = AnchorSmoothSettings(min_anchor_count=3, median_radius=1)
        smoothed = smooth_peaks_from_locked_anchors(
            finalized, front, transform, settings, discontinuity_ids=coherent)
        by_id = {rec.ID: rec for rec in smoothed}
        for key in disc_ids:
            np.testing.assert_allclose(by_id[key].peak, raw_peak, rtol=0, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
