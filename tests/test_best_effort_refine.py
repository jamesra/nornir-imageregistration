"""Tests for pair-adaptive best-effort refine helpers and Role integration."""

from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.alignment_record import EnhancedAlignmentRecord
from nornir_imageregistration.refine_shared.best_effort import (
    apply_best_effort_ambiguous_mesh_promotion,
    assess_best_effort_mode,
    ranked_ambiguous_mesh_ids,
    zncc_decoy_radius_px,
)
from nornir_imageregistration.refine_shared.cell_roles import (
    RejectReason,
    Role,
    ZnccScore,
    classify_roles,
)
from nornir_imageregistration.refine_shared.cell_validity import is_alignable_cell


def _rec(
        key: tuple[int, int],
        *,
        peak: tuple[float, float],
        weight: float = 12.0,
        peak_ratio: float | None = 1.5,
) -> EnhancedAlignmentRecord:
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


class TestZnccDecoyRadius(unittest.TestCase):
    """Size-normalized decoy ring."""

    def test_matches_base_at_reference_cell(self) -> None:
        self.assertAlmostEqual(
            zncc_decoy_radius_px(128, 128, base_radius=10.0), 10.0)

    def test_scales_with_cell_size(self) -> None:
        r256 = zncc_decoy_radius_px(256, 256, base_radius=10.0)
        r512 = zncc_decoy_radius_px(512, 512, base_radius=10.0)
        self.assertAlmostEqual(r256, 20.0)
        self.assertAlmostEqual(r512, 40.0)


class TestAssessBestEffort(unittest.TestCase):
    """Relative tension trigger."""

    def test_inactive_when_settled_identity_only(self) -> None:
        records = [_rec((i, 0), peak=(0.0, 0.0), peak_ratio=2.0) for i in range(20)]
        lc = np.ones(20, dtype=bool)
        result = assess_best_effort_mode(
            records, lock_candidate=lc, max_travel=10.0, travel_eps=0.5)
        self.assertFalse(result.active)

    def test_active_when_identity_locks_and_high_travel_tension(self) -> None:
        records = (
            [_rec((i, 0), peak=(0.0, 0.0), peak_ratio=2.5) for i in range(16)]
            + [_rec((i, 1), peak=(8.0, 0.0), peak_ratio=1.05) for i in range(4)]
        )
        lc = np.asarray([True] * 16 + [False] * 4, dtype=bool)
        result = assess_best_effort_mode(
            records, lock_candidate=lc, max_travel=10.0, travel_eps=0.5)
        self.assertTrue(result.active)
        self.assertGreaterEqual(result.identity_lock_cand_frac, 0.5)
        self.assertGreaterEqual(result.high_travel_frac, 0.05)

    def test_inactive_when_movers_are_unique_not_ambiguous(self) -> None:
        """Healthy unique high-travel FREE cells must not trigger best-effort."""
        records = (
            [_rec((i, 0), peak=(0.0, 0.0), peak_ratio=2.5) for i in range(16)]
            + [_rec((i, 1), peak=(8.0, 0.0), peak_ratio=2.0) for i in range(4)]
        )
        lc = np.asarray([True] * 16 + [False] * 4, dtype=bool)
        result = assess_best_effort_mode(
            records, lock_candidate=lc, max_travel=10.0, travel_eps=0.5)
        self.assertFalse(result.active)


class TestBestEffortClassify(unittest.TestCase):
    """Identity-freeze quantile demotion."""

    def test_healthy_path_unchanged_without_best_effort(self) -> None:
        records = [_rec((0, 0), peak=(0.1, 0.0), peak_ratio=1.5)]
        result = classify_roles(
            records,
            transform_cutoff=float('-inf'),
            max_travel=2.0,
            zncc_by_id={(0, 0): ZnccScore(
                peak=0.5, decoy_med=0.0, decoy_max=0.1, prominence=10.0)},
            best_effort_active=False,
        )
        self.assertEqual(result.roles[0], Role.LOCKABLE)

    def test_best_effort_demotes_low_prominence_identity(self) -> None:
        # Four lock-cands at travel≈0 with prominences 5,6,7,20 — quantile 0.75 ≈ 7+
        records = [
            _rec((0, 0), peak=(0.0, 0.0), peak_ratio=2.0),
            _rec((0, 1), peak=(0.0, 0.0), peak_ratio=2.0),
            _rec((1, 0), peak=(0.0, 0.0), peak_ratio=2.0),
            _rec((1, 1), peak=(0.0, 0.0), peak_ratio=2.0),
        ]
        zncc = {
            (0, 0): ZnccScore(peak=0.4, decoy_med=0.0, decoy_max=0.05, prominence=5.0),
            (0, 1): ZnccScore(peak=0.4, decoy_med=0.0, decoy_max=0.05, prominence=6.0),
            (1, 0): ZnccScore(peak=0.4, decoy_med=0.0, decoy_max=0.05, prominence=7.0),
            (1, 1): ZnccScore(peak=0.5, decoy_med=0.0, decoy_max=0.05, prominence=20.0),
        }
        result = classify_roles(
            records,
            transform_cutoff=float('-inf'),
            max_travel=2.0,
            zncc_by_id=zncc,
            best_effort_active=True,
            identity_prominence_quantile=0.75,
        )
        self.assertEqual(result.role_by_id[(1, 1)], Role.LOCKABLE)
        self.assertEqual(result.role_by_id[(0, 0)], Role.IDENTITY_SUSPECT)
        self.assertEqual(result.role_by_id[(0, 1)], Role.IDENTITY_SUSPECT)

    def test_low_content_still_reject_with_best_effort(self) -> None:
        records = [_rec((0, 0), peak=(0.0, 0.0), peak_ratio=2.0)]
        result = classify_roles(
            records,
            transform_cutoff=float('-inf'),
            max_travel=2.0,
            zncc_by_id={(0, 0): ZnccScore(
                peak=0.5, decoy_med=0.0, decoy_max=0.1, prominence=20.0)},
            low_content_ids={(0, 0)},
            best_effort_active=True,
        )
        self.assertEqual(result.roles[0], Role.REJECT)
        self.assertEqual(result.reject_reasons[0], RejectReason.LOW_CONTENT)
        self.assertFalse(is_alignable_cell(np.zeros((32, 32), dtype=np.float64)))


class TestAmbiguousMeshPromotion(unittest.TestCase):
    """Ranked PEAK_AMBIGUOUS → FREE for mesh."""

    def test_promotes_upper_quantile_only(self) -> None:
        records = [
            _rec((0, 0), peak=(6.0, 0.0), peak_ratio=1.05),
            _rec((0, 1), peak=(6.0, 0.0), peak_ratio=1.10),
            _rec((1, 0), peak=(6.0, 0.0), peak_ratio=1.15),
            _rec((1, 1), peak=(6.0, 0.0), peak_ratio=1.18),
        ]
        reasons = [RejectReason.PEAK_AMBIGUOUS] * 4
        roles = [Role.REJECT] * 4
        promote = ranked_ambiguous_mesh_ids(
            records, reasons, max_travel=10.0, travel_eps=0.5, promote_quantile=0.5)
        self.assertIn((1, 1), promote)
        self.assertIn((1, 0), promote)
        self.assertNotIn((0, 0), promote)

        new_roles, new_reasons, by_id = apply_best_effort_ambiguous_mesh_promotion(
            roles, reasons, records, promote)
        self.assertEqual(by_id[(1, 1)], Role.FREE)
        self.assertEqual(new_reasons[0], RejectReason.PEAK_AMBIGUOUS)
        self.assertEqual(new_roles[0], Role.REJECT)

    def test_never_promotes_low_content(self) -> None:
        records = [_rec((0, 0), peak=(6.0, 0.0), peak_ratio=1.15)]
        reasons = [RejectReason.LOW_CONTENT]
        promote = ranked_ambiguous_mesh_ids(
            records, reasons, max_travel=10.0, travel_eps=0.5)
        self.assertEqual(promote, set())


if __name__ == '__main__':
    unittest.main()
