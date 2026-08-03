"""Unit tests for STOS cell Role / FieldMode theory."""

from __future__ import annotations

import os
import unittest

import numpy as np

from nornir_imageregistration.alignment_record import EnhancedAlignmentRecord
from nornir_imageregistration.refine_shared.cell_roles import (
    DEFAULT_IDENTITY_ZNCC_MIN,
    FieldMode,
    RejectReason,
    Role,
    classify_field,
    classify_roles,
    exclude_reject_mesh_records,
    field_brand_identity_suspect_ids,
    unique_large_travel_raw_preserve_ids,
    active_unique_field_is_hot,
    masked_zncc,
)
from nornir_imageregistration.refine_shared.cell_validity import (
    DEFAULT_LOW_CONTENT_STD_MIN,
    is_alignable_cell,
)
from nornir_imageregistration.refine_shared.coherent_residual import (
    COHERENCE_MIN,
    LOCK_FRAC_TRIGGER,
    estimate_coherent_residual_translation,
)
from nornir_imageregistration.refine_shared.finalize import (
    FinalizeCandidateState,
    FinalizeSettings,
    evaluate_finalize_candidates,
)
from nornir_imageregistration.refine_shared.runtime_config import get_runtime_config
from nornir_imageregistration.refine_shared.source_content_cache import SourceContentCache


def _rec(
        key: tuple[int, int],
        *,
        peak: tuple[float, float],
        weight: float = 12.0,
        peak_ratio: float | None = 1.5,
        source_x: float | None = None,
) -> EnhancedAlignmentRecord:
    sx = float(key[1]) * 10.0 if source_x is None else float(source_x)
    return EnhancedAlignmentRecord(
        ID=key,
        TargetPoint=np.asarray((float(key[0]) * 10.0, sx), dtype=np.float64),
        SourcePoint=np.asarray((float(key[0]) * 10.0, sx), dtype=np.float64),
        peak=np.asarray(peak, dtype=np.float64),
        weight=weight,
        angle=0.0,
        flipped_ud=False,
        peak_ratio=peak_ratio,
    )


class TestMaskedZncc(unittest.TestCase):
    """Secondary ZNCC helper."""

    def test_identical_patches_near_one(self) -> None:
        rng = np.random.default_rng(0)
        patch = rng.normal(size=(32, 32))
        score = masked_zncc(patch, patch)
        self.assertGreater(score, 0.99)

    def test_uncorrelated_near_zero(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.normal(size=(32, 32))
        b = rng.normal(size=(32, 32))
        score = masked_zncc(a, b)
        self.assertLess(abs(score), 0.35)


class TestClassifyRoles(unittest.TestCase):
    """PC-pass / ZNCC Role branding."""

    def test_weight_bar_disabled_low_weight_zncc_pass_is_lockable(self) -> None:
        """With transform_cutoff=-inf, low PC weight still reaches ZNCC → LOCKABLE."""
        records = [_rec((0, 0), peak=(0.1, 0.0), weight=0.5, peak_ratio=1.5)]
        result = classify_roles(
            records,
            transform_cutoff=float('-inf'),
            max_travel=2.0,
            zncc_by_id={(0, 0): 0.9},
            identity_zncc_min=0.25,
        )
        self.assertEqual(result.roles[0], Role.LOCKABLE)
        self.assertTrue(bool(result.lock_candidate[0]))

    def test_weight_bar_disabled_ambiguous_still_reject(self) -> None:
        records = [_rec((0, 0), peak=(0.1, 0.0), weight=0.5, peak_ratio=1.0)]
        result = classify_roles(
            records,
            transform_cutoff=float('-inf'),
            max_travel=2.0,
            zncc_by_id={(0, 0): 0.9},
            identity_zncc_min=0.25,
        )
        self.assertEqual(result.roles[0], Role.REJECT)
        self.assertEqual(result.reject_reasons[0], RejectReason.PEAK_AMBIGUOUS)

    def test_ambiguous_is_reject(self) -> None:
        records = [_rec((0, 0), peak=(1.0, 0.0), peak_ratio=1.0)]
        result = classify_roles(
            records, transform_cutoff=10.0, max_travel=2.0, zncc_by_id={})
        self.assertEqual(result.roles[0], Role.REJECT)
        self.assertEqual(result.reject_reasons[0], RejectReason.PEAK_AMBIGUOUS)

    def test_free_when_not_lock_candidate(self) -> None:
        records = [_rec((0, 0), peak=(5.0, 0.0), weight=12.0, peak_ratio=1.5)]
        result = classify_roles(
            records, transform_cutoff=10.0, max_travel=2.0, zncc_by_id={})
        self.assertEqual(result.roles[0], Role.FREE)

    def test_pc_pass_zncc_fail_is_identity_suspect(self) -> None:
        records = [_rec((0, 0), peak=(0.1, 0.0), weight=12.0, peak_ratio=1.5)]
        result = classify_roles(
            records,
            transform_cutoff=10.0,
            max_travel=2.0,
            zncc_by_id={(0, 0): 0.05},
            identity_zncc_min=0.25,
        )
        self.assertEqual(result.roles[0], Role.IDENTITY_SUSPECT)
        self.assertEqual(result.n_zncc_fail, 1)

    def test_pc_pass_zncc_pass_is_lockable(self) -> None:
        records = [_rec((0, 0), peak=(0.1, 0.0), weight=12.0, peak_ratio=1.5)]
        result = classify_roles(
            records,
            transform_cutoff=10.0,
            max_travel=2.0,
            zncc_by_id={(0, 0): 0.9},
            identity_zncc_min=0.25,
        )
        self.assertEqual(result.roles[0], Role.LOCKABLE)

    def test_missing_zncc_on_lock_candidate_fails_closed(self) -> None:
        records = [_rec((0, 0), peak=(0.0, 0.0), weight=12.0, peak_ratio=1.5)]
        result = classify_roles(
            records, transform_cutoff=10.0, max_travel=2.0, zncc_by_id={})
        self.assertEqual(result.roles[0], Role.IDENTITY_SUSPECT)

    def test_low_content_reject(self) -> None:
        records = [_rec((0, 0), peak=(0.0, 0.0), peak_ratio=1.5)]
        result = classify_roles(
            records,
            transform_cutoff=10.0,
            max_travel=2.0,
            low_content_ids={(0, 0)},
            zncc_by_id={(0, 0): 0.9},
        )
        self.assertEqual(result.roles[0], Role.REJECT)
        self.assertEqual(result.reject_reasons[0], RejectReason.LOW_CONTENT)

    def test_role_not_sticky_across_remeasure(self) -> None:
        """Same ID can be REJECT then FREE when peak_ratio recovers."""
        bad = [_rec((1, 1), peak=(2.0, 0.0), peak_ratio=1.0)]
        good = [_rec((1, 1), peak=(3.0, 0.0), peak_ratio=1.4)]
        r1 = classify_roles(bad, transform_cutoff=10.0, max_travel=2.0)
        r2 = classify_roles(good, transform_cutoff=10.0, max_travel=2.0)
        self.assertEqual(r1.roles[0], Role.REJECT)
        self.assertEqual(r2.roles[0], Role.FREE)

    def test_env_identity_zncc_min_override(self) -> None:
        old = os.environ.get('NORNIR_REFINE_IDENTITY_ZNCC_MIN')
        try:
            os.environ['NORNIR_REFINE_IDENTITY_ZNCC_MIN'] = '0.8'
            get_runtime_config(refresh=True)
            records = [_rec((0, 0), peak=(0.0, 0.0), weight=12.0, peak_ratio=1.5)]
            result = classify_roles(
                records,
                transform_cutoff=10.0,
                max_travel=2.0,
                zncc_by_id={(0, 0): 0.5},
            )
            self.assertEqual(result.roles[0], Role.IDENTITY_SUSPECT)
            self.assertAlmostEqual(result.identity_zncc_min, 0.8)
        finally:
            if old is None:
                os.environ.pop('NORNIR_REFINE_IDENTITY_ZNCC_MIN', None)
            else:
                os.environ['NORNIR_REFINE_IDENTITY_ZNCC_MIN'] = old
            get_runtime_config(refresh=True)


class TestClassifyField(unittest.TestCase):
    """FieldMode detection."""

    def test_rigid_residual_on_coherent_low_locks(self) -> None:
        records = [
            _rec((0, i), peak=(-3.0, 40.0 + 0.1 * i), peak_ratio=1.4)
            for i in range(60)
        ]
        mode = classify_field(records, lock_fraction=0.01, max_travel=2.0)
        self.assertEqual(mode, FieldMode.RIGID_RESIDUAL)
        residual = estimate_coherent_residual_translation(records, lock_fraction=0.01)
        self.assertIsNotNone(residual)
        assert residual is not None
        self.assertGreaterEqual(residual.coherence, COHERENCE_MIN)

    def test_healthy_is_local(self) -> None:
        records = [
            _rec((0, i), peak=(0.2, 0.1), peak_ratio=1.5)
            for i in range(20)
        ]
        mode = classify_field(records, lock_fraction=0.35, max_travel=2.0)
        self.assertEqual(mode, FieldMode.LOCAL)

    def test_asymmetric_cold_half_brands_identity_despite_high_zncc(self) -> None:
        # Low-x traveling free; high-x near-identity lock-cands — ASYMMETRIC.
        free = [
            _rec((0, i), peak=(0.0, 16.0), peak_ratio=1.4, source_x=float(i))
            for i in range(10)
        ]
        cold = [
            _rec((1, i), peak=(0.0, 0.1), weight=12.0, peak_ratio=1.5, source_x=float(100 + i))
            for i in range(10)
        ]
        mode = classify_field(free + cold, lock_fraction=0.2, max_travel=2.0)
        self.assertEqual(mode, FieldMode.ASYMMETRIC)
        zncc = {(1, i): 0.75 for i in range(10)}
        zncc.update({(0, i): 0.9 for i in range(10)})
        result = classify_roles(
            free + cold,
            transform_cutoff=10.0,
            max_travel=20.0,
            zncc_by_id=zncc,
            field_mode=mode,
            travel_eps=0.5,
        )
        # Traveling free half stays FREE / non-suspect.
        self.assertTrue(all(r != Role.IDENTITY_SUSPECT for r in result.roles[:10]))
        # Cold identity lock-cands are suspects even with high ZNCC.
        self.assertTrue(all(r == Role.IDENTITY_SUSPECT for r in result.roles[10:]))

    def test_neighbor_active_unique_brands_identity_despite_high_zncc(self) -> None:
        # (0,1) identity lock-cand next to active unique (0,0).
        records = [
            _rec((0, 0), peak=(0.0, 12.0), peak_ratio=1.4, source_x=0.0),
            _rec((0, 1), peak=(0.0, 0.1), weight=12.0, peak_ratio=1.5, source_x=10.0),
        ]
        branded = field_brand_identity_suspect_ids(
            records, field_mode=FieldMode.LOCAL, max_travel=2.0, travel_eps=0.5)
        self.assertIn((0, 1), branded)
        result = classify_roles(
            records,
            transform_cutoff=10.0,
            max_travel=20.0,
            zncc_by_id={(0, 1): 0.9, (0, 0): 0.9},
            field_mode=FieldMode.LOCAL,
            field_suspect_ids=branded,
        )
        self.assertEqual(result.roles[1], Role.IDENTITY_SUSPECT)

    def test_hot_unique_field_brands_all_identity(self) -> None:
        """FOV-wide: many unique high-travel peaks → refuse travel≈0 locks."""
        travelers = [
            _rec((0, i), peak=(0.0, 15.0), peak_ratio=1.4, source_x=float(i))
            for i in range(25)
        ]
        identity = [
            _rec((1, i), peak=(0.0, 0.1), weight=12.0, peak_ratio=1.5, source_x=float(200 + i))
            for i in range(10)
        ]
        records = travelers + identity
        self.assertTrue(
            active_unique_field_is_hot(records, max_travel=2.0))
        branded = field_brand_identity_suspect_ids(
            records, field_mode=FieldMode.LOCAL, max_travel=2.0, travel_eps=0.5)
        for i in range(10):
            self.assertIn((1, i), branded)
        result = classify_roles(
            records,
            transform_cutoff=10.0,
            max_travel=20.0,
            zncc_by_id={(1, i): 0.9 for i in range(10)},
            field_mode=FieldMode.LOCAL,
            field_suspect_ids=branded,
            travel_eps=0.5,
        )
        self.assertTrue(all(r == Role.IDENTITY_SUSPECT for r in result.roles[25:]))

    def test_healthy_unique_field_not_hot(self) -> None:
        """Healthy free-unique travel ~2 px stays cold → no FOV-wide identity brand."""
        records = [
            _rec((0, i), peak=(0.2, 1.5), peak_ratio=1.4, source_x=float(i))
            for i in range(40)
        ] + [
            _rec((1, 0), peak=(0.0, 0.1), weight=12.0, peak_ratio=1.5, source_x=100.0)
        ]
        self.assertFalse(
            active_unique_field_is_hot(records, max_travel=11.0))

    def test_unique_large_travel_raw_preserve_ids(self) -> None:
        records = [
            _rec((0, 0), peak=(0.0, 15.0), peak_ratio=1.4),
            _rec((0, 1), peak=(0.0, 1.0), peak_ratio=1.4),
            _rec((0, 2), peak=(0.0, 15.0), peak_ratio=1.0),  # ambiguous
        ]
        ids = unique_large_travel_raw_preserve_ids(records, max_travel=2.0)
        self.assertEqual(ids, {(0, 0)})

    def test_disc_neighbor_brands_identity_suspect(self) -> None:
        """Identity cell next to an active disc neighbor is branded suspect."""
        # Disc at (0,1) with large travel; identity lock-cand at (0,0).
        records = [
            _rec((0, 0), peak=(0.0, 0.1), weight=12.0, peak_ratio=1.5),
            _rec((0, 1), peak=(0.0, 40.0), weight=5.0, peak_ratio=1.05),
            _rec((0, 2), peak=(0.0, 0.2), weight=12.0, peak_ratio=1.5),
        ]
        branded = field_brand_identity_suspect_ids(
            records,
            field_mode=FieldMode.LOCAL,
            max_travel=2.0,
            travel_eps=0.5,
            discontinuity_ids={(0, 1)},
        )
        self.assertIn((0, 0), branded)
        self.assertIn((0, 2), branded)

    def test_disc_neighbor_ignores_settled_disc(self) -> None:
        """Disc with travel ≤ max_travel does not brand neighbors."""
        records = [
            _rec((0, 0), peak=(0.0, 0.1), weight=12.0, peak_ratio=1.5),
            _rec((0, 1), peak=(0.0, 1.0), weight=5.0, peak_ratio=1.05),
        ]
        branded = field_brand_identity_suspect_ids(
            records,
            field_mode=FieldMode.LOCAL,
            max_travel=2.0,
            travel_eps=0.5,
            discontinuity_ids={(0, 1)},
        )
        self.assertNotIn((0, 0), branded)

    def test_asymmetric_is_diagnostic_only(self) -> None:
        # Free peaks disagree across halves → ASYMMETRIC; traveling half not suspect.
        free = [
            _rec((0, i), peak=(0.0, 16.0), peak_ratio=1.4, source_x=float(i))
            for i in range(10)
        ] + [
            _rec((1, i), peak=(0.0, 0.2), weight=1.0, peak_ratio=1.4, source_x=float(100 + i))
            for i in range(10)
        ]
        mode = classify_field(free, lock_fraction=0.2, max_travel=2.0)
        self.assertEqual(mode, FieldMode.ASYMMETRIC)
        # Low-x travelers are not lock-candidates (travel > max) → FREE, not suspects.
        result = classify_roles(
            free[:10],
            transform_cutoff=10.0,
            max_travel=2.0,
            zncc_by_id={(0, i): 0.9 for i in range(10)},
            field_mode=mode,
        )
        self.assertTrue(all(r == Role.FREE for r in result.roles))


class TestFinalizeLockableOnly(unittest.TestCase):
    """Finalize consumes Role.LOCKABLE set."""

    def setUp(self) -> None:
        self.settings = FinalizeSettings(
            max_travel_for_finalization=2.0,
            min_finalize_pass=2,
            finalize_stability_passes=1,
            finalize_stability_epsilon_px=0.5,
        )

    def test_identity_suspect_never_locks(self) -> None:
        records = [
            _rec((0, 0), peak=(0.0, 0.0), peak_ratio=1.5),
            _rec((0, 1), peak=(0.0, 0.1), peak_ratio=1.5),
        ]
        prior = {
            rec.ID: FinalizeCandidateState(
                peak=np.asarray(rec.peak, dtype=np.float64),
                weight=12.0,
                pass_index=2,
                consecutive_stable=1,
            )
            for rec in records
        }
        # Only (0,1) is LOCKABLE; (0,0) is IDENTITY_SUSPECT.
        result = evaluate_finalize_candidates(
            records,
            transform_cutoff=10.0,
            settings=self.settings,
            pass_index=3,
            prior_candidates=prior,
            lockable_ids={(0, 1)},
        )
        self.assertFalse(bool(result.lock_mask[0]))
        self.assertTrue(bool(result.lock_mask[1]))
        self.assertGreaterEqual(result.rejected_identity_suspect_count, 1)


class TestExcludeRejectMesh(unittest.TestCase):
    def test_drops_reject(self) -> None:
        records = [
            _rec((0, 0), peak=(1.0, 0.0), peak_ratio=1.0),
            _rec((0, 1), peak=(2.0, 0.0), peak_ratio=1.5),
            _rec((0, 2), peak=(3.0, 0.0), peak_ratio=1.5),
            _rec((0, 3), peak=(4.0, 0.0), peak_ratio=1.5),
        ]
        roles = [Role.REJECT, Role.FREE, Role.FREE, Role.FREE]
        kept, dropped = exclude_reject_mesh_records(records, roles, min_keep=3)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(kept), 3)


class TestLowContentGate(unittest.TestCase):
    def test_constant_not_alignable(self) -> None:
        cell = np.ones((16, 16), dtype=np.float64)
        self.assertFalse(is_alignable_cell(cell, min_std=DEFAULT_LOW_CONTENT_STD_MIN))

    def test_structured_alignable(self) -> None:
        rng = np.random.default_rng(2)
        cell = rng.normal(size=(16, 16))
        self.assertTrue(is_alignable_cell(cell, min_std=DEFAULT_LOW_CONTENT_STD_MIN))

    def test_source_cache_sticky(self) -> None:
        cache = SourceContentCache(min_std=1.0)
        self.assertFalse(cache.remember((0, 0), 0.1))
        self.assertTrue(cache.is_low_content((0, 0)))
        # Second remember does not revive.
        self.assertTrue(cache.is_low_content((0, 0)))

    def test_env_low_content_std_override(self) -> None:
        old = os.environ.get('NORNIR_REFINE_LOW_CONTENT_STD_MIN')
        try:
            os.environ['NORNIR_REFINE_LOW_CONTENT_STD_MIN'] = '5.0'
            get_runtime_config(refresh=True)
            cache = SourceContentCache()
            self.assertAlmostEqual(cache.min_std, 5.0)
        finally:
            if old is None:
                os.environ.pop('NORNIR_REFINE_LOW_CONTENT_STD_MIN', None)
            else:
                os.environ['NORNIR_REFINE_LOW_CONTENT_STD_MIN'] = old
            get_runtime_config(refresh=True)


class TestCoherentResidualStillWorks(unittest.TestCase):
    def test_skips_when_lock_fraction_healthy(self) -> None:
        records = [
            _rec((0, i), peak=(-3.0, 40.0), peak_ratio=1.4)
            for i in range(60)
        ]
        result = estimate_coherent_residual_translation(
            records, lock_fraction=LOCK_FRAC_TRIGGER)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
