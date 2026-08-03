"""Unit tests for STOS multi-criteria finalize / unlock gating."""

from __future__ import annotations

import os
import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.alignment_record import EnhancedAlignmentRecord
from nornir_imageregistration.refine_shared.finalize import (
    FinalizeCandidateState,
    FinalizeSettings,
    evaluate_finalize_candidates,
    filter_records_for_mesh_inclusion,
    legacy_finalize_mask,
    unlock_stale_finalized,
    use_legacy_finalize_gate,
)
from nornir_imageregistration.refine_shared.runtime_config import get_runtime_config
from nornir_imageregistration.transforms.rigid import RigidTranslation


def _rec(
        key: tuple[int, int],
        *,
        peak: tuple[float, float] = (0.1, 0.1),
        weight: float = 10.0,
        source: tuple[float, float] = (10.0, 10.0),
        target: tuple[float, float] | None = None,
        peak_ratio: float | None = None) -> EnhancedAlignmentRecord:
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target if target is not None else source, dtype=np.float64)
    return EnhancedAlignmentRecord(
        ID=key,
        TargetPoint=tgt,
        SourcePoint=src,
        peak=np.asarray(peak, dtype=np.float64),
        weight=weight,
        angle=0.0,
        flipped_ud=False,
        peak_ratio=peak_ratio,
    )


class TestEvaluateFinalizeCandidates(unittest.TestCase):
    """Multi-criteria lock gates."""

    def setUp(self) -> None:
        self.settings = FinalizeSettings(
            max_travel_for_finalization=2.0,
            min_finalize_pass=2,
            finalize_stability_passes=2,
            finalize_stability_epsilon_px=0.5,
            finalize_unlock_travel_multiplier=1.5,
        )

    def test_low_weight_small_travel_does_not_lock(self) -> None:
        """Seam-class failure: low weight + small travel must not lock."""
        records = [
            _rec((0, 0), peak=(0.2, 0.1), weight=1.0),
            _rec((0, 1), peak=(0.1, 0.2), weight=1.1),
            _rec((1, 0), peak=(0.0, 0.1), weight=0.9),
            _rec((1, 1), peak=(0.1, 0.0), weight=12.0),
            _rec((2, 0), peak=(0.2, 0.0), weight=11.5),
            _rec((2, 1), peak=(0.0, 0.2), weight=13.0),
        ]
        # High transform cutoff: only strong weights may lock.
        result = evaluate_finalize_candidates(
            records, transform_cutoff=10.0, settings=self.settings, pass_index=3,
            prior_candidates={
                (1, 1): FinalizeCandidateState(
                    peak=np.array((0.1, 0.0)), weight=12.0, pass_index=2, consecutive_stable=1),
                (2, 0): FinalizeCandidateState(
                    peak=np.array((0.2, 0.0)), weight=11.5, pass_index=2, consecutive_stable=1),
                (2, 1): FinalizeCandidateState(
                    peak=np.array((0.0, 0.2)), weight=13.0, pass_index=2, consecutive_stable=1),
            })
        # Low-weight cells must never lock.
        for i, rec in enumerate(records):
            if rec.weight < 10.0:
                self.assertFalse(bool(result.lock_mask[i]), msg=f'{rec.ID} weight={rec.weight}')

    def test_high_weight_stable_locks_on_second_pass(self) -> None:
        records = [
            _rec((0, 0), peak=(0.1, 0.1), weight=12.0),
            _rec((0, 1), peak=(0.2, 0.0), weight=11.0),
            _rec((1, 0), peak=(0.0, 0.2), weight=13.0),
        ]
        first = evaluate_finalize_candidates(
            records, transform_cutoff=10.0, settings=self.settings, pass_index=2)
        self.assertFalse(bool(np.any(first.lock_mask)))
        self.assertGreater(first.deferred_stability_count, 0)

        second = evaluate_finalize_candidates(
            records, transform_cutoff=10.0, settings=self.settings, pass_index=3,
            prior_candidates=first.candidates)
        self.assertTrue(bool(np.all(second.lock_mask)))

    def test_unstable_peak_deferred(self) -> None:
        records = [_rec((0, 0), peak=(0.1, 0.1), weight=12.0),
                   _rec((0, 1), peak=(0.1, 0.1), weight=12.0),
                   _rec((1, 0), peak=(0.1, 0.1), weight=12.0)]
        prior = {
            (0, 0): FinalizeCandidateState(
                peak=np.array((2.0, 2.0)), weight=12.0, pass_index=2, consecutive_stable=1),
            (0, 1): FinalizeCandidateState(
                peak=np.array((0.1, 0.1)), weight=12.0, pass_index=2, consecutive_stable=1),
            (1, 0): FinalizeCandidateState(
                peak=np.array((0.1, 0.1)), weight=12.0, pass_index=2, consecutive_stable=1),
        }
        result = evaluate_finalize_candidates(
            records, transform_cutoff=10.0, settings=self.settings, pass_index=3,
            prior_candidates=prior)
        self.assertFalse(bool(result.lock_mask[0]))
        self.assertEqual(result.candidates[(0, 0)].consecutive_stable, 1)
        self.assertTrue(bool(result.lock_mask[1]))
        self.assertTrue(bool(result.lock_mask[2]))

    def test_pass_delay_blocks_early_lock(self) -> None:
        records = [
            _rec((0, 0), peak=(0.1, 0.1), weight=12.0),
            _rec((0, 1), peak=(0.1, 0.1), weight=12.0),
            _rec((1, 0), peak=(0.1, 0.1), weight=12.0),
        ]
        result = evaluate_finalize_candidates(
            records, transform_cutoff=10.0, settings=self.settings, pass_index=1)
        self.assertFalse(bool(np.any(result.lock_mask)))
        self.assertGreater(result.rejected_pass_count, 0)

    def test_locked_set_subset_of_transform_cutoff(self) -> None:
        records = [
            _rec((0, 0), peak=(0.1, 0.1), weight=5.0),
            _rec((0, 1), peak=(0.1, 0.1), weight=12.0),
            _rec((1, 0), peak=(0.1, 0.1), weight=11.0),
            _rec((1, 1), peak=(0.1, 0.1), weight=3.0),
        ]
        prior = {
            key: FinalizeCandidateState(
                peak=np.array((0.1, 0.1)), weight=12.0, pass_index=2, consecutive_stable=1)
            for key in ((0, 1), (1, 0))
        }
        cutoff = 10.0
        result = evaluate_finalize_candidates(
            records, transform_cutoff=cutoff, settings=self.settings, pass_index=3,
            prior_candidates=prior)
        for i, rec in enumerate(records):
            if result.lock_mask[i]:
                self.assertGreaterEqual(rec.weight, cutoff)

    def test_weight_bar_disabled_allows_low_weight_lock(self) -> None:
        """With transform_cutoff=-inf, low weight is not rejected by the weight gate."""
        records = [
            _rec((0, 0), peak=(0.1, 0.1), weight=0.5),
            _rec((0, 1), peak=(0.1, 0.1), weight=0.4),
            _rec((1, 0), peak=(0.1, 0.1), weight=0.3),
        ]
        prior = {
            key: FinalizeCandidateState(
                peak=np.array((0.1, 0.1)), weight=0.5, pass_index=2, consecutive_stable=1)
            for key in ((0, 0), (0, 1), (1, 0))
        }
        result = evaluate_finalize_candidates(
            records,
            transform_cutoff=float('-inf'),
            settings=self.settings,
            pass_index=3,
            prior_candidates=prior,
            lockable_ids={(0, 0), (0, 1), (1, 0)},
        )
        self.assertEqual(result.rejected_weight_count, 0)
        self.assertTrue(bool(np.any(result.lock_mask)))


class TestPeakRatioFinalizeGates(unittest.TestCase):
    """Always-on peak_ratio hard-reject and early-lock."""

    def setUp(self) -> None:
        self.settings = FinalizeSettings(
            max_travel_for_finalization=2.0,
            min_finalize_pass=2,
            finalize_stability_passes=2,
            finalize_stability_epsilon_px=0.5,
        )

    def test_low_ratio_never_locks_and_counts_ambiguous(self) -> None:
        records = [
            _rec((0, 0), peak=(0.1, 0.1), weight=12.0, peak_ratio=1.05),
            _rec((0, 1), peak=(0.1, 0.1), weight=12.0, peak_ratio=1.05),
            _rec((1, 0), peak=(0.1, 0.1), weight=12.0, peak_ratio=1.05),
        ]
        prior = {
            key: FinalizeCandidateState(
                peak=np.array((0.1, 0.1)), weight=12.0, pass_index=2, consecutive_stable=1)
            for key in ((0, 0), (0, 1), (1, 0))
        }
        result = evaluate_finalize_candidates(
            records, transform_cutoff=10.0, settings=self.settings, pass_index=3,
            prior_candidates=prior)
        self.assertFalse(bool(np.any(result.lock_mask)))
        self.assertEqual(result.rejected_ambiguous_count, 3)

    def test_missing_ratio_does_not_hard_reject(self) -> None:
        records = [
            _rec((0, 0), peak=(0.1, 0.1), weight=12.0, peak_ratio=None),
            _rec((0, 1), peak=(0.1, 0.1), weight=12.0, peak_ratio=None),
            _rec((1, 0), peak=(0.1, 0.1), weight=12.0, peak_ratio=None),
        ]
        prior = {
            key: FinalizeCandidateState(
                peak=np.array((0.1, 0.1)), weight=12.0, pass_index=2, consecutive_stable=1)
            for key in ((0, 0), (0, 1), (1, 0))
        }
        result = evaluate_finalize_candidates(
            records, transform_cutoff=10.0, settings=self.settings, pass_index=3,
            prior_candidates=prior)
        self.assertTrue(bool(np.all(result.lock_mask)))
        self.assertEqual(result.rejected_ambiguous_count, 0)

    def test_high_ratio_early_locks_with_stability_one(self) -> None:
        records = [
            _rec((0, 0), peak=(0.1, 0.1), weight=12.0, peak_ratio=1.6),
            _rec((0, 1), peak=(0.1, 0.1), weight=12.0, peak_ratio=1.6),
            _rec((1, 0), peak=(0.1, 0.1), weight=12.0, peak_ratio=1.6),
        ]
        # Pass index 1 is normally blocked (min_finalize_pass=2); early ratio allows it.
        result = evaluate_finalize_candidates(
            records, transform_cutoff=10.0, settings=self.settings, pass_index=1)
        self.assertTrue(bool(np.all(result.lock_mask)))
        self.assertEqual(result.deferred_stability_count, 0)

    def test_low_ratio_disc_does_not_get_soft_weight(self) -> None:
        from nornir_imageregistration.refine_shared.discontinuity import per_record_max_travel
        from nornir_imageregistration.refine_shared.peak_ratio_gates import soft_discontinuity_ids

        records = [
            _rec((0, 0), peak=(1.0, 0.0), weight=20.0, peak_ratio=2.0),
            _rec((0, 1), peak=(5.0, 0.0), weight=3.0, peak_ratio=1.05),
        ]
        disc = {(0, 1)}
        soft_ids = soft_discontinuity_ids(records, disc)
        self.assertEqual(soft_ids, set())
        settings = FinalizeSettings(
            max_travel_for_finalization=10.0,
            min_finalize_pass=1,
            finalize_stability_passes=1,
        )
        limits = per_record_max_travel(
            records, base_max_travel=10.0, discontinuity_ids=soft_ids, relax_mult=2.5)
        result = evaluate_finalize_candidates(
            records,
            transform_cutoff=10.0,
            settings=settings,
            pass_index=2,
            per_record_max_travel=limits,
            soft_weight_cutoff=2.0,
            discontinuity_ids=soft_ids,
        )
        self.assertFalse(bool(result.lock_mask[1]))
        self.assertEqual(result.rejected_ambiguous_count, 1)


class TestExcludeAmbiguousMeshRecords(unittest.TestCase):
    """Phase 3: ambiguous free peaks must not reshape the mesh."""

    def test_excludes_ambiguous_keeps_clear(self) -> None:
        from nornir_imageregistration.refine_shared.peak_ratio_gates import (
            exclude_ambiguous_mesh_records,
        )
        records = [
            _rec((0, 0), peak=(0.1, 0.0), weight=12.0, peak_ratio=1.6),
            _rec((0, 1), peak=(40.0, 0.0), weight=12.0, peak_ratio=1.05),
            _rec((1, 0), peak=(0.2, 0.0), weight=12.0, peak_ratio=1.4),
            _rec((1, 1), peak=(50.0, 0.0), weight=12.0, peak_ratio=1.01),
        ]
        kept, dropped = exclude_ambiguous_mesh_records(records, min_keep=2)
        self.assertEqual(dropped, 2)
        self.assertEqual({r.ID for r in kept}, {(0, 0), (1, 0)})

    def test_emergency_fill_when_too_few_clear(self) -> None:
        from nornir_imageregistration.refine_shared.peak_ratio_gates import (
            exclude_ambiguous_mesh_records,
        )
        records = [
            _rec((0, 0), peak=(1.0, 0.0), weight=12.0, peak_ratio=1.05),
            _rec((0, 1), peak=(2.0, 0.0), weight=12.0, peak_ratio=1.05),
            _rec((1, 0), peak=(30.0, 0.0), weight=12.0, peak_ratio=1.05),
        ]
        kept, dropped = exclude_ambiguous_mesh_records(records, min_keep=2)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 1)
        # Lowest-travel ambiguous cells fill the min_keep budget.
        self.assertEqual({r.ID for r in kept}, {(0, 0), (0, 1)})

    def test_missing_ratio_not_excluded(self) -> None:
        from nornir_imageregistration.refine_shared.peak_ratio_gates import (
            exclude_ambiguous_mesh_records,
        )
        records = [
            _rec((0, 0), peak=(0.1, 0.0), weight=12.0, peak_ratio=None),
            _rec((0, 1), peak=(40.0, 0.0), weight=12.0, peak_ratio=1.05),
            _rec((1, 0), peak=(0.2, 0.0), weight=12.0, peak_ratio=None),
        ]
        kept, dropped = exclude_ambiguous_mesh_records(records, min_keep=2)
        self.assertEqual(dropped, 1)
        self.assertEqual({r.ID for r in kept}, {(0, 0), (1, 0)})


class TestUnlockStaleFinalized(unittest.TestCase):
    """Unlock when baked targets disagree with the current transform."""

    def test_unlock_when_transform_moves_away(self) -> None:
        settings = FinalizeSettings(
            max_travel_for_finalization=1.0,
            finalize_unlock_travel_multiplier=1.5,
        )
        # Identity transform predicts target ~= source.
        transform = RigidTranslation(target_offset=np.array((0.0, 0.0)))
        # Locked with a baked target far from prediction.
        locked = {
            (0, 0): _rec((0, 0), source=(10.0, 10.0), target=(20.0, 20.0),
                         peak=(0.0, 0.0), weight=12.0),
            (0, 1): _rec((0, 1), source=(30.0, 30.0), target=(30.2, 30.1),
                         peak=(0.0, 0.0), weight=12.0),
        }
        kept, unlocked = unlock_stale_finalized(locked, transform, settings)
        self.assertIn((0, 0), unlocked)
        self.assertIn((0, 1), kept)
        self.assertNotIn((0, 0), kept)

    def test_unlock_disabled_when_multiplier_zero(self) -> None:
        settings = FinalizeSettings(
            max_travel_for_finalization=1.0,
            finalize_unlock_travel_multiplier=0.0,
        )
        transform = RigidTranslation(target_offset=np.array((0.0, 0.0)))
        locked = {
            (0, 0): _rec((0, 0), source=(10.0, 10.0), target=(50.0, 50.0),
                         peak=(0.0, 0.0), weight=12.0),
        }
        kept, unlocked = unlock_stale_finalized(locked, transform, settings)
        self.assertEqual(unlocked, [])
        self.assertIn((0, 0), kept)


class TestFilterRecordsForMeshInclusion(unittest.TestCase):
    def test_drops_large_peak_outliers(self) -> None:
        records = [
            _rec((0, 0), peak=(0.5, 0.0), weight=12.0),
            _rec((0, 1), peak=(50.0, 0.0), weight=12.0),
            _rec((1, 0), peak=(0.0, 1.0), weight=12.0),
            _rec((1, 1), peak=(40.0, 40.0), weight=12.0),
        ]
        kept, dropped = filter_records_for_mesh_inclusion(records, max_travel=2.0, min_keep=2)
        self.assertEqual(dropped, 2)
        self.assertEqual({r.ID for r in kept}, {(0, 0), (1, 0)})


class TestLegacyFinalizeMask(unittest.TestCase):
    def test_legacy_allows_low_weight_small_travel(self) -> None:
        # Distance-primary: with an explicit low floor, weak + small travel locks.
        records = [
            _rec((0, 0), peak=(0.1, 0.1), weight=1.0),
            _rec((0, 1), peak=(0.1, 0.1), weight=10.0),
            _rec((1, 0), peak=(5.0, 5.0), weight=10.0),
        ]
        polyfit = np.linspace(0.0, 20.0, 101)
        mask = legacy_finalize_mask(
            records, max_travel_distance=2.0, polyfit_weights=polyfit, floor_percentile=2.0)
        self.assertTrue(bool(mask[0]))
        self.assertTrue(bool(mask[1]))
        self.assertFalse(bool(mask[2]))


class TestLegacyEnvFlag(unittest.TestCase):
    def test_use_legacy_finalize_gate_reads_env(self) -> None:
        old = os.environ.get('NORNIR_REFINE_FINALIZE_LEGACY')
        try:
            os.environ['NORNIR_REFINE_FINALIZE_LEGACY'] = '1'
            get_runtime_config(refresh=True)
            self.assertTrue(use_legacy_finalize_gate())
            os.environ['NORNIR_REFINE_FINALIZE_LEGACY'] = '0'
            get_runtime_config(refresh=True)
            self.assertFalse(use_legacy_finalize_gate())
        finally:
            if old is None:
                os.environ.pop('NORNIR_REFINE_FINALIZE_LEGACY', None)
            else:
                os.environ['NORNIR_REFINE_FINALIZE_LEGACY'] = old
            get_runtime_config(refresh=True)


class TestSyntheticRefineNoStep(unittest.TestCase):
    """Synthetic refine: injected weak-region measurements must not create a hard step."""

    def test_evaluate_rejects_weak_half_plane(self) -> None:
        # Left half: strong, right half: weak but small travel (legacy would lock).
        records = []
        for r in range(4):
            for c in range(6):
                weight = 12.0 if c < 3 else 1.0
                records.append(_rec((r, c), peak=(0.1, 0.1), weight=weight))
        settings = FinalizeSettings(
            max_travel_for_finalization=2.0,
            min_finalize_pass=2,
            finalize_stability_passes=1,
            finalize_stability_epsilon_px=0.5,
        )
        result = evaluate_finalize_candidates(
            records, transform_cutoff=8.0, settings=settings, pass_index=3)
        for i, rec in enumerate(records):
            col = rec.ID[1]
            if col >= 3:
                self.assertFalse(bool(result.lock_mask[i]))
            else:
                self.assertTrue(bool(result.lock_mask[i]))


@unittest.skipUnless(
    bool(os.environ.get('INPUT_NORNIR_DATA', '').strip()),
    'INPUT_NORNIR_DATA not set — optional RC2 / composite seam refine check',
)
class TestOptionalRc2FinalizeRegression(unittest.TestCase):
    """Optional smoke: refine settings construct and finalize helpers stay importable.

    Manual checklist (composite seam + one RC2 pair):
    1. Re-run refine-grid with SavePlots=True on the composite-seam STOS pair.
    2. Confirm pass logs show deferred_stability / unlocked counts; no early lock
       band along the vertical seam until weights are strong.
    3. Composite view: no left/right grid color discontinuity.
    4. Re-run one RC2 Brute64→Grid pair (e.g. 1453-1452 or section 1024); compare
       overlay / displacement RMS along the former seam.
    Launch tip: leave NORNIR_REFINE_FINALIZE_LEGACY unset (default multi-criteria).
    """

    def test_finalize_settings_defaults_from_grid_refinement(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        shape = (64, 64)
        image = np.full(shape, 0.5, dtype=nornir_imageregistration.default_image_dtype())
        stats = nornir_imageregistration.ImageStats.Create(image)
        settings = nornir_imageregistration.settings.GridRefinement(
            target_image=image,
            source_image=image.copy(),
            target_image_stats=stats,
            source_image_stats=stats,
            single_thread_processing=True,
        )
        fs = FinalizeSettings.from_grid_refinement(settings)
        self.assertEqual(fs.min_finalize_pass, 2)
        self.assertEqual(fs.finalize_stability_passes, 2)
        self.assertAlmostEqual(fs.finalize_unlock_travel_multiplier, 1.5)


if __name__ == '__main__':
    unittest.main()
