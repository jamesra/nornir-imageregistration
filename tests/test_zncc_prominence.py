"""Tests for scale-free ZNCC prominence scoring used by the STOS lock gate."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from nornir_imageregistration import local_distortion_correction as ldc
from nornir_imageregistration.refine_shared.cell_roles import ZnccScore


class TestZnccProminenceStack(unittest.TestCase):
    """Batched prominence: true shift vs uncorrelated / masked."""

    def test_shifted_copy_high_prominence_any_cell_size(self) -> None:
        rng = np.random.default_rng(0)
        for cell in (32, 64, 128, 256):
            fixed = rng.normal(size=(cell, cell)).astype(np.float32)
            peak = np.array([3.0, -2.0], dtype=np.float64)
            # Moving is offset by +peak relative to fixed; scoring shifts by -peak.
            moving = np.roll(np.roll(fixed, -int(peak[0]), axis=0), -int(peak[1]), axis=1)
            from nornir_imageregistration.refine_shared.best_effort import zncc_decoy_radius_px
            radius = zncc_decoy_radius_px(cell, cell, base_radius=10.0)
            z_peak, _z_med, z_max, prom = ldc._zncc_prominence_stack(
                fixed[None], moving[None], peak.reshape(1, 2),
                decoy_radius=radius, travel_eps=0.5)
            self.assertGreater(float(z_peak[0]), float(z_max[0]), msg=f'cell={cell}')
            self.assertGreater(float(prom[0]), 4.0, msg=f'cell={cell} prom={prom[0]}')
            self.assertGreater(float(z_peak[0]), 0.9, msg=f'cell={cell}')

    def test_uncorrelated_prominence_near_zero(self) -> None:
        rng = np.random.default_rng(1)
        fixed = rng.normal(size=(64, 64)).astype(np.float32)
        moving = rng.normal(size=(64, 64)).astype(np.float32)
        peak = np.zeros((1, 2), dtype=np.float64)
        _z_peak, _z_med, _z_max, prom = ldc._zncc_prominence_stack(
            fixed[None], moving[None], peak, decoy_radius=5.0)
        self.assertLess(abs(float(prom[0])), 3.0)

    def test_invariant_under_gain_offset(self) -> None:
        rng = np.random.default_rng(2)
        fixed = rng.normal(size=(48, 48)).astype(np.float32)
        peak = np.array([2.0, 1.0], dtype=np.float64)
        moving = np.roll(np.roll(fixed, -int(peak[0]), axis=0), -int(peak[1]), axis=1)
        base = ldc._zncc_prominence_stack(
            fixed[None], moving[None], peak.reshape(1, 2), decoy_radius=5.0)
        remapped_fixed = (fixed * 3.5 + 12.0).astype(np.float32)
        remapped_moving = (moving * 3.5 + 12.0).astype(np.float32)
        remapped = ldc._zncc_prominence_stack(
            remapped_fixed[None], remapped_moving[None], peak.reshape(1, 2),
            decoy_radius=5.0)
        self.assertAlmostEqual(float(base[0][0]), float(remapped[0][0]), places=5)
        self.assertAlmostEqual(float(base[3][0]), float(remapped[3][0]), places=2)

    def test_invariant_under_partial_noise_mask(self) -> None:
        rng = np.random.default_rng(3)
        cell = 64
        fixed = rng.normal(size=(cell, cell)).astype(np.float32)
        peak = np.array([2.0, -1.0], dtype=np.float64)
        moving = np.roll(np.roll(fixed, -int(peak[0]), axis=0), -int(peak[1]), axis=1)
        valid = np.ones((cell, cell), dtype=bool)
        valid[: int(0.4 * cell), :] = False
        z_peak, _z_med, z_max, prom = ldc._zncc_prominence_stack(
            fixed[None], moving[None], peak.reshape(1, 2),
            valid_mask=valid[None], decoy_radius=5.0)
        self.assertGreater(float(z_peak[0]), float(z_max[0]))
        self.assertGreater(float(prom[0]), 4.0)
        self.assertGreater(float(z_peak[0]), 0.9)

    def test_round_shift_not_floor(self) -> None:
        """Fractional peaks must round; floor can over-shift by nearly 1 px."""
        rng = np.random.default_rng(4)
        fixed = rng.normal(size=(40, 40)).astype(np.float32)
        # True integer offset is 1 px. Claimed peak 1.4 → round(-1.4)=-1 aligns;
        # floor(-1.4)=-2 over-shifts.
        peak = np.array([1.4, 0.0], dtype=np.float64)
        moving = np.roll(fixed, -1, axis=0)
        shifted, _mask = ldc._shift_moving_stack_by_peaks(
            moving[None], peak.reshape(1, 2), travel_eps=0.01)
        zncc_round = float(ldc._masked_zncc_stack(fixed[None], shifted)[0])
        self.assertGreater(zncc_round, 0.95)
        self.assertEqual(int(np.round(-peak[0])), -1)
        self.assertEqual(int(np.floor(-peak[0])), -2)
        from nornir_imageregistration.refine_shared.cell_roles import masked_zncc
        import nornir_imageregistration as nir
        over = nir.CropImage(moving, 0, -2, 40, 40, cval=float(np.median(moving)))
        self.assertLess(masked_zncc(fixed, over), 0.2)


class TestComputeZnccUsesRoiCandidate(unittest.TestCase):
    """``_compute_zncc_for_candidates`` must score on the winning ROI kind."""

    def test_exact_roi_candidate_uses_exact_path(self) -> None:
        rng = np.random.default_rng(5)
        cell = 32
        fixed = rng.normal(size=(cell, cell)).astype(np.float32)
        moving_exact = np.roll(fixed, -2, axis=0)
        moving_rigid = rng.normal(size=(cell, cell)).astype(np.float32)
        rec = SimpleNamespace(
            ID=(1, 2),
            SourcePoint=np.array([10.0, 10.0]),
            TargetPoint=np.array([10.0, 10.0]),
            peak=np.array([2.0, 0.0], dtype=np.float64),
            roi_candidate='exact',
        )
        settings = SimpleNamespace(
            cell_size=(cell, cell),
            peak_ratio_exclusion_radius=5.0,
            ring_scale_fraction_max=0.05,
            ring_angle_max_degrees=5.0,
            ring_allow_flip_change=False,
            target_image=fixed,
            source_image=moving_exact,
            target_image_stats=None,
            source_image_stats=None,
        )

        def fake_batched(**_kwargs):
            nan = np.zeros((1, cell, cell), dtype=bool)
            return fixed[None], moving_rigid[None], nan

        def fake_exact(**_kwargs):
            nan = np.zeros((1, cell, cell), dtype=bool)
            return moving_exact[None], nan

        with patch.object(ldc, 'ApproximateRigidTransformBySourcePoints', return_value=[object()]), \
             patch.object(ldc, '_stos_settings_images', return_value=(fixed, moving_exact)), \
             patch.object(ldc, 'BuildAlignmentROIsBatched', side_effect=fake_batched), \
             patch.object(ldc, 'BuildExactMovingROIsBatched', side_effect=fake_exact):
            transform = object()
            scores = ldc._compute_zncc_for_candidates(
                [rec], {(1, 2)}, transform, settings, travel_eps=0.5)
        self.assertIn((1, 2), scores)
        score = scores[(1, 2)]
        self.assertIsInstance(score, ZnccScore)
        self.assertGreater(score.peak, 0.9)
        self.assertGreater(score.prominence, 4.0)


if __name__ == '__main__':
    unittest.main()
