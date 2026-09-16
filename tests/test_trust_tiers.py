"""Record-snapshot unit tests for trusted-mesh trust tiers."""

from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.alignment_record import EnhancedAlignmentRecord
from nornir_imageregistration.refine_shared.peak_ratio_gates import PEAK_RATIO_MIN
from nornir_imageregistration.refine_shared.trust_tiers import (
    LOCK_FRAC_TRIGGER,
    TrustTier,
    assign_trust_tiers,
    demote_disagreeing,
    find_unique_clusters,
    low_lock_quality_flag,
    mesh_records_from_tiers,
    trusted_set_snapshot,
)


def _rec(
        key: tuple[int, int],
        *,
        peak: tuple[float, float] = (2.0, 0.0),
        weight: float = 10.0,
        peak_ratio: float | None = 2.0,
) -> EnhancedAlignmentRecord:
    return EnhancedAlignmentRecord(
        ID=key,
        TargetPoint=np.array((float(key[0]) * 10.0, float(key[1]) * 10.0), dtype=np.float64),
        SourcePoint=np.array((float(key[0]) * 10.0, float(key[1]) * 10.0), dtype=np.float64),
        peak=np.array(peak, dtype=np.float64),
        weight=weight,
        angle=0.0,
        flipped_ud=False,
        peak_ratio=peak_ratio,
    )


class TestTrustTiers(unittest.TestCase):
    def test_quality_flag_uses_final_lock_fraction(self) -> None:
        self.assertTrue(low_lock_quality_flag(4, 100))
        self.assertFalse(low_lock_quality_flag(5, 100))
        self.assertFalse(low_lock_quality_flag(
            1, 1, trigger=LOCK_FRAC_TRIGGER))

    def test_cluster_seeds_provisional_when_no_locks(self) -> None:
        records = [
            _rec((0, 0), peak=(10.0, 0.0)),
            _rec((0, 1), peak=(11.0, 0.5)),
            _rec((0, 2), peak=(9.5, -0.2)),
            _rec((5, 5), peak=(1.0, 20.0), peak_ratio=1.01),  # ambiguous / opposing
        ]
        clusters = find_unique_clusters(records)
        self.assertTrue(any(len(c) >= 3 for c in clusters))
        tiers = assign_trust_tiers(records, max_travel=12.0, cell_half_size=64.0)
        self.assertEqual(tiers[(0, 0)], TrustTier.PROVISIONAL)
        self.assertEqual(tiers[(0, 1)], TrustTier.PROVISIONAL)
        self.assertEqual(tiers[(0, 2)], TrustTier.PROVISIONAL)
        self.assertEqual(tiers[(5, 5)], TrustTier.UNTRUSTED)

    def test_locked_requires_unique_and_zncc(self) -> None:
        records = [
            _rec((0, 0), peak=(0.2, 0.0), peak_ratio=2.0),
            _rec((0, 1), peak=(0.3, 0.0), peak_ratio=None),
        ]
        tiers = assign_trust_tiers(
            records,
            locked_ids={(0, 0), (0, 1)},
            zncc_pass_ids={(0, 0)},
            max_travel=2.0,
            cell_half_size=64.0,
        )
        self.assertEqual(tiers[(0, 0)], TrustTier.LOCKED)
        self.assertEqual(tiers[(0, 1)], TrustTier.UNTRUSTED)

    def test_converged_candidate_requires_zncc_to_lock(self) -> None:
        records = [
            _rec((0, 0), peak=(0.2, 0.0)),
            _rec((0, 1), peak=(0.3, 0.0)),
        ]
        tiers = assign_trust_tiers(
            records,
            zncc_pass_ids={(0, 0)},
            converged_ids={(0, 0), (0, 1)},
            max_travel=2.0,
            cell_half_size=64.0,
        )
        self.assertEqual(tiers[(0, 0)], TrustTier.LOCKED)
        self.assertEqual(tiers[(0, 1)], TrustTier.UNTRUSTED)

    def test_demote_disagreeing_provisional(self) -> None:
        records = [
            _rec((0, 0), peak=(0.1, 0.0)),
            _rec((0, 1), peak=(40.0, 0.0)),
        ]
        tiers = {
            (0, 0): TrustTier.LOCKED,
            (0, 1): TrustTier.PROVISIONAL,
        }
        out = demote_disagreeing(
            tiers, records, max_travel=2.0, cell_half_size=8.0)
        self.assertEqual(out[(0, 1)], TrustTier.UNTRUSTED)

    def test_mesh_split_excludes_untrusted(self) -> None:
        records = [
            _rec((0, 0)),
            _rec((0, 1)),
            _rec((1, 0), peak_ratio=1.05),
        ]
        tiers = {
            (0, 0): TrustTier.LOCKED,
            (0, 1): TrustTier.PROVISIONAL,
            (1, 0): TrustTier.UNTRUSTED,
        }
        locked, provisional = mesh_records_from_tiers(records, tiers)
        self.assertEqual({r.ID for r in locked}, {(0, 0)})
        self.assertEqual({r.ID for r in provisional}, {(0, 1)})
        snap = trusted_set_snapshot(tiers)
        self.assertIn(((0, 0), int(TrustTier.LOCKED)), snap)
        self.assertNotIn(((1, 0), int(TrustTier.UNTRUSTED)), snap)

    def test_ambiguous_below_min_never_provisional(self) -> None:
        records = [
            _rec((0, 0), peak=(10.0, 0.0), peak_ratio=PEAK_RATIO_MIN - 0.2),
            _rec((0, 1), peak=(11.0, 0.0), peak_ratio=PEAK_RATIO_MIN - 0.2),
            _rec((0, 2), peak=(9.0, 0.0), peak_ratio=PEAK_RATIO_MIN - 0.2),
        ]
        tiers = assign_trust_tiers(records, max_travel=12.0, cell_half_size=64.0)
        self.assertTrue(all(t == TrustTier.UNTRUSTED for t in tiers.values()))

    def test_missing_ratio_never_shapes_mesh(self) -> None:
        records = [
            _rec((0, 0), peak_ratio=None),
            _rec((0, 1), peak_ratio=None),
            _rec((0, 2), peak_ratio=None),
        ]
        tiers = assign_trust_tiers(
            records,
            zncc_pass_ids={(0, 0), (0, 1), (0, 2)},
            max_travel=12.0,
            cell_half_size=64.0,
        )
        self.assertTrue(all(t == TrustTier.UNTRUSTED for t in tiers.values()))

    def test_cluster_seed_requires_zncc_when_available(self) -> None:
        records = [
            _rec((0, 0)),
            _rec((0, 1)),
            _rec((0, 2)),
        ]
        tiers = assign_trust_tiers(
            records,
            zncc_pass_ids={(0, 0), (0, 1)},
            max_travel=12.0,
            cell_half_size=64.0,
        )
        self.assertTrue(all(t == TrustTier.UNTRUSTED for t in tiers.values()))

    def test_provisional_can_agree_with_lock_two_hops_away(self) -> None:
        records = [
            _rec((0, 0), peak=(2.0, 0.0)),
            _rec((0, 2), peak=(2.5, 0.0)),
        ]
        tiers = assign_trust_tiers(
            records,
            locked_ids={(0, 0)},
            zncc_pass_ids={(0, 2)},
            max_travel=2.0,
            cell_half_size=8.0,
        )
        self.assertEqual(tiers[(0, 0)], TrustTier.LOCKED)
        self.assertEqual(tiers[(0, 2)], TrustTier.PROVISIONAL)


if __name__ == '__main__':
    unittest.main()
