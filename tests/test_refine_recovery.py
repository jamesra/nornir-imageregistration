"""Unit tests for coherent residual + global FOV recovery."""

from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.alignment_record import EnhancedAlignmentRecord
from nornir_imageregistration.refine_shared.coherent_residual import (
    COHERENCE_MIN,
    LOCK_FRAC_TRIGGER,
    MIN_UNIQUE_PEAKS,
    diagnose_coherent_residual_translation,
    estimate_coherent_residual_translation,
    estimate_global_fov_residual_translation,
    should_attempt_global_fov_recovery,
    should_preserve_post_residual_transform,
    should_keep_prior_sparse_mesh,
)
from nornir_imageregistration import local_distortion_correction as ldc
from nornir_imageregistration.local_distortion_correction import (
    should_finish_on_empty_alignment_pass,
)
from nornir_imageregistration.transforms.rigid import RigidTranslation


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


class TestCoherentResidualTranslation(unittest.TestCase):
    """Track A / RIGID_RESIDUAL trigger conditions."""

    def test_triggers_on_coherent_unique_peaks_with_low_locks(self) -> None:
        records = [
            _rec((0, i), peak=(-3.0, 40.0 + 0.1 * i), peak_ratio=1.4)
            for i in range(60)
        ]
        result = estimate_coherent_residual_translation(records, lock_fraction=0.01)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreaterEqual(result.coherence, COHERENCE_MIN)
        self.assertGreaterEqual(result.n_unique, 50)
        self.assertGreaterEqual(result.n_inliers, MIN_UNIQUE_PEAKS)
        np.testing.assert_allclose(result.translation[0], -3.0, atol=0.5)
        self.assertGreater(float(result.translation[1]), 35.0)

    def test_skips_when_lock_fraction_healthy(self) -> None:
        records = [
            _rec((0, i), peak=(-3.0, 40.0), peak_ratio=1.4)
            for i in range(60)
        ]
        result = estimate_coherent_residual_translation(
            records, lock_fraction=LOCK_FRAC_TRIGGER)
        self.assertIsNone(result)

    def test_skips_when_peaks_oppose(self) -> None:
        records = [
            _rec((0, i), peak=(20.0 if i % 2 == 0 else -20.0, 0.0), peak_ratio=1.4)
            for i in range(60)
        ]
        result = estimate_coherent_residual_translation(records, lock_fraction=0.01)
        self.assertIsNone(result)

    def test_inliers_recover_240_241_like_outlier_veto(self) -> None:
        """Dominant +x cluster survives minority opposing outliers (pre-inlier coh ~0.66)."""
        dominant = [
            _rec((0, i), peak=(-0.1, 37.0 + 0.05 * i), peak_ratio=1.4)
            for i in range(128)
        ]
        outliers = [
            _rec((1, i), peak=(25.0, -30.0), peak_ratio=1.4)
            for i in range(36)
        ]
        records = dominant + outliers
        result = estimate_coherent_residual_translation(records, lock_fraction=0.01)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.n_unique, 164)
        self.assertGreaterEqual(result.n_inliers, MIN_UNIQUE_PEAKS)
        self.assertGreaterEqual(result.coherence, COHERENCE_MIN)
        self.assertGreater(float(result.translation[1]), 30.0)
        self.assertLess(abs(float(result.translation[0])), 5.0)

    def test_healthy_lock_fraction_never_fires_even_with_coherent_peaks(self) -> None:
        records = [
            _rec((0, i), peak=(-0.1, 37.0), peak_ratio=1.4)
            for i in range(100)
        ]
        result = estimate_coherent_residual_translation(records, lock_fraction=0.33)
        self.assertIsNone(result)

    def test_diagnose_skip_reason_on_wrap_peak_soup(self) -> None:
        """Opposing ±60-like unique peaks: Track A skips; global recovery should attempt."""
        records = []
        for i in range(40):
            if i % 4 == 0:
                peak = (0.0, 60.0)
            elif i % 4 == 1:
                peak = (0.0, -60.0)
            elif i % 4 == 2:
                peak = (60.0, 0.0)
            else:
                peak = (-60.0, 0.0)
            records.append(_rec((0, i), peak=peak, peak_ratio=1.4))
        diagnosis = diagnose_coherent_residual_translation(records, lock_fraction=0.01)
        self.assertIsNone(diagnosis.result)
        self.assertIsNotNone(diagnosis.skip_reason)
        self.assertTrue(should_attempt_global_fov_recovery(diagnosis, 0.01))
        self.assertFalse(should_attempt_global_fov_recovery(diagnosis, 0.33))

    def test_diagnose_skip_low_unique_count(self) -> None:
        records = [
            _rec((0, i), peak=(-0.1, 37.0), peak_ratio=1.4)
            for i in range(30)
        ]
        diagnosis = diagnose_coherent_residual_translation(records, lock_fraction=0.01)
        self.assertIsNone(diagnosis.result)
        self.assertIn('n_unique', str(diagnosis.skip_reason))
        self.assertEqual(diagnosis.n_unique, 30)
        self.assertTrue(should_attempt_global_fov_recovery(diagnosis, 0.01))

    def test_no_global_fov_when_all_cell_peaks_rejected(self) -> None:
        """n_unique==0: do not invent a TranslateFixed via whole-FOV PC."""
        # Ambiguous peaks (low peak_ratio) ⇒ diagnose reports n_unique=0.
        records = [
            _rec((0, i), peak=(0.0, 40.0), peak_ratio=1.05)
            for i in range(60)
        ]
        diagnosis = diagnose_coherent_residual_translation(records, lock_fraction=0.01)
        self.assertIsNone(diagnosis.result)
        self.assertEqual(diagnosis.n_unique, 0)
        self.assertFalse(should_attempt_global_fov_recovery(diagnosis, 0.01))


class TestGlobalFovResidual(unittest.TestCase):
    """Downsampled whole-FOV residual recovery."""

    def test_recovers_known_translation_on_shifted_noise(self) -> None:
        rng = np.random.default_rng(0)
        target = rng.normal(size=(256, 256))
        # Source is target shifted by (+8, -12) in (y, x); identity transform → residual.
        dy, dx = 8, -12
        source = np.roll(np.roll(target, dy, axis=0), dx, axis=1)
        transform = RigidTranslation(target_offset=np.asarray((0.0, 0.0), dtype=np.float64))
        peak = estimate_global_fov_residual_translation(
            transform, target, source, max_dim=128)
        self.assertIsNotNone(peak)
        assert peak is not None
        # Phase correlation peak sign follows find_offset convention; magnitude should match.
        self.assertLess(abs(abs(float(peak[0])) - abs(dy)), 2.0)
        self.assertLess(abs(abs(float(peak[1])) - abs(dx)), 2.0)


class TestPreservePostResidualTransform(unittest.TestCase):
    """Sparse mesh must not discard TranslateFixed after residual recovery."""

    def test_triggers_when_residual_applied_and_mesh_sparse(self) -> None:
        self.assertTrue(should_preserve_post_residual_transform(
            residual_applied=True,
            n_mesh=14,
            n_grid=6000,
            n_locks=0,
        ))

    def test_skips_when_residual_not_applied(self) -> None:
        self.assertFalse(should_preserve_post_residual_transform(
            residual_applied=False,
            n_mesh=14,
            n_grid=6000,
            n_locks=0,
        ))

    def test_skips_when_locks_healthy(self) -> None:
        self.assertFalse(should_preserve_post_residual_transform(
            residual_applied=True,
            n_mesh=14,
            n_grid=6000,
            n_locks=int(0.1 * 6000),
        ))

    def test_skips_when_mesh_dense(self) -> None:
        self.assertFalse(should_preserve_post_residual_transform(
            residual_applied=True,
            n_mesh=500,
            n_grid=6000,
            n_locks=0,
        ))

    def test_final_gate_must_use_fov_n_grid_not_sparse_control_set(self) -> None:
        """Regression: final path used len(control_records) as n_grid (~14).

        With 2 locks / 14 points, lock_frac looks healthy and preserve skips,
        then a 12-point mesh undoes TranslateFixed. FOV-sized n_grid keeps it.
        """
        # Buggy final-style denominator (sparse control set + locks).
        self.assertFalse(should_preserve_post_residual_transform(
            residual_applied=True,
            n_mesh=12,
            n_grid=14,
            n_locks=2,
        ))
        # Correct last-pass FOV denominator (Grid16 240-241 scale).
        self.assertTrue(should_preserve_post_residual_transform(
            residual_applied=True,
            n_mesh=12,
            n_grid=6262,
            n_locks=2,
        ))


class TestKeepPriorSparseMesh(unittest.TestCase):
    """Non-residual reject-soup must not replace a usable prior pose."""

    def test_keeps_rigid_prior_when_mesh_is_triangulation_floor(self) -> None:
        """784-782 style: 0 locks, ~3 emergency-filled rejects, rigid input."""
        self.assertTrue(should_keep_prior_sparse_mesh(
            residual_applied=False,
            n_mesh=3,
            n_grid=274,
            n_locks=0,
            n_prior_points=0,
        ))

    def test_keeps_dense_prior_when_mesh_collapses(self) -> None:
        self.assertTrue(should_keep_prior_sparse_mesh(
            residual_applied=False,
            n_mesh=3,
            n_grid=274,
            n_locks=0,
            n_prior_points=200,
        ))

    def test_allows_dense_rebuild(self) -> None:
        self.assertFalse(should_keep_prior_sparse_mesh(
            residual_applied=False,
            n_mesh=150,
            n_grid=274,
            n_locks=0,
            n_prior_points=0,
        ))

    def test_still_preserves_after_residual(self) -> None:
        self.assertTrue(should_keep_prior_sparse_mesh(
            residual_applied=True,
            n_mesh=14,
            n_grid=6000,
            n_locks=0,
            n_prior_points=6000,
        ))


class TestFinishOnEmptyAlignmentPass(unittest.TestCase):
    """Empty remasure after a prior pass must not abort the refine pipeline."""

    def test_pass_one_with_no_locks_is_hard_failure(self) -> None:
        self.assertFalse(should_finish_on_empty_alignment_pass(1, 0))

    def test_later_pass_keeps_prior_transform(self) -> None:
        self.assertTrue(should_finish_on_empty_alignment_pass(3, 0))

    def test_any_pass_with_locks_finishes(self) -> None:
        self.assertTrue(should_finish_on_empty_alignment_pass(1, 4))


class TestBuildMeshTransformOrKeep(unittest.TestCase):
    def test_keeps_prior_when_fewer_than_three_points(self) -> None:
        prior = RigidTranslation(np.asarray((1.0, 2.0), dtype=np.float64))
        records = [_rec((0, 0), peak=(3.0, 4.0))]
        transform, used, scores = ldc._build_mesh_transform_or_keep(
            records, prior_transform=prior, fixed_points=None)
        self.assertIs(transform, prior)
        self.assertEqual(len(used), 1)
        self.assertEqual(scores.shape[0], 1)

    def test_empty_fixed_points_are_zero_budget(self) -> None:
        empty = ldc.AlignRecordsToControlPoints([])
        self.assertEqual(empty.shape, (0, 4))
        self.assertEqual(ldc._fixed_point_count(empty), 0)
        self.assertEqual(ldc._fixed_point_count(None), 0)

    def test_control_point_count_rigid_is_zero(self) -> None:
        prior = RigidTranslation(np.asarray((1.0, 2.0), dtype=np.float64))
        self.assertEqual(ldc._control_point_count(prior), 0)

    def test_keeps_prior_when_zero_records(self) -> None:
        prior = RigidTranslation(np.asarray((1.0, 2.0), dtype=np.float64))
        transform, used, scores = ldc._build_mesh_transform_or_keep(
            [], prior_transform=prior, fixed_points=None)
        self.assertIs(transform, prior)
        self.assertEqual(len(used), 0)
        self.assertEqual(scores.shape[0], 0)


if __name__ == '__main__':
    unittest.main()
