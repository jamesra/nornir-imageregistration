"""Unit tests for log-polar Phase A helpers in stos_brute."""

from __future__ import annotations

import unittest
import unittest.mock

import numpy as np
import scipy.ndimage

import nornir_imageregistration
import nornir_imageregistration.stos_brute as stos_brute


class TestRpc3ScaleSearchBounds(unittest.TestCase):
    """Blind scale grid spans RPC3 manual corpus (module docstring in stos_brute)."""

    def test_tissue_grid_covers_manual_min_max(self) -> None:
        grid = stos_brute._SCALE_REFINE_TISSUE_GRID
        self.assertLessEqual(min(grid), stos_brute._RPC3_MANUAL_SCALE_MIN)
        self.assertGreaterEqual(max(grid), stos_brute._RPC3_MANUAL_SCALE_MAX)

    def test_blind_candidates_use_tissue_grid(self) -> None:
        candidates = stos_brute._scale_search_candidates(1.0, None, force_search=False)
        self.assertEqual(candidates, list(stos_brute._SCALE_REFINE_TISSUE_GRID))


class TestLogpolarWarpRadius(unittest.TestCase):
    def test_angle_radius_divisor_four(self) -> None:
        self.assertEqual(stos_brute._logpolar_warp_radius(1024, for_scale=False), 256)

    def test_scale_radius_divisor_two(self) -> None:
        self.assertEqual(stos_brute._logpolar_warp_radius(1024, for_scale=True), 512)

    def test_radius_minimum(self) -> None:
        self.assertEqual(stos_brute._logpolar_warp_radius(8, for_scale=False), 8)


class TestParabolicPeakIndex(unittest.TestCase):
    def test_refines_symmetric_peak(self) -> None:
        values = np.array([0.1, 0.9, 1.0, 0.9, 0.1], dtype=np.float32)
        refined = stos_brute._parabolic_peak_index(values, 2)
        self.assertAlmostEqual(refined, 2.0, delta=0.05)

    def test_refines_offset_peak(self) -> None:
        values = np.array([0.1, 0.5, 0.9, 1.0, 0.8, 0.2], dtype=np.float32)
        refined = stos_brute._parabolic_peak_index(values, 3)
        self.assertGreater(refined, 2.5)
        self.assertLess(refined, 3.5)


class TestRefineLogpolarPeakOffsets(unittest.TestCase):
    def test_refines_2d_gaussian_peak(self) -> None:
        size = 33
        yy, xx = np.mgrid[0:size, 0:size]
        center = (16.3, 17.7)
        corr = np.exp(-((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / 8.0).astype(np.float32)
        row_off = size * 0.5 - center[0]
        col_off = size * 0.5 - center[1]
        refined_row, refined_col = stos_brute._refine_logpolar_peak_offsets(corr, row_off, col_off)
        self.assertAlmostEqual(refined_row, row_off, delta=0.15)
        self.assertAlmostEqual(refined_col, col_off, delta=0.15)


class TestScaleAtFinalAngle(unittest.TestCase):
    def test_delegates_to_refine_scale_local(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        rng = np.random.default_rng(0)
        base = rng.standard_normal((64, 64)).astype(np.float32)
        source_h = nornir_imageregistration.ImagePermutationHelper(base)
        target_h = nornir_imageregistration.ImagePermutationHelper(base.copy())
        with unittest.mock.patch.object(stos_brute, '_refine_scale_local', return_value=0.99) as mock_refine:
            result = stos_brute._scale_at_final_angle(
                source_h.ImageWithMaskAsNoise,
                target_h.ImageWithMaskAsNoise,
                source_h.Stats,
                target_h.Stats,
                final_angle=5.0,
                scale_seed=1.0,
                min_overlap=0.5,
            )
        self.assertAlmostEqual(result, 0.99)
        mock_refine.assert_called_once()


class TestRefineScaleLocal(unittest.TestCase):
    def test_recovers_shrink_near_seed(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        rng = np.random.default_rng(3)
        base = rng.standard_normal((128, 128)).astype(np.float32)
        base -= base.min()
        base /= base.max() + 1e-6
        shrink = 0.96
        warped = scipy.ndimage.zoom(base, shrink, order=1)
        pad_y = (base.shape[0] - warped.shape[0]) // 2
        pad_x = (base.shape[1] - warped.shape[1]) // 2
        source = np.zeros_like(base)
        source[pad_y:pad_y + warped.shape[0], pad_x:pad_x + warped.shape[1]] = warped
        source_h = nornir_imageregistration.ImagePermutationHelper(source)
        target_h = nornir_imageregistration.ImagePermutationHelper(base)
        refined = stos_brute._refine_scale_local(
            source_h.ImageWithMaskAsNoise,
            target_h.ImageWithMaskAsNoise,
            source_h.Stats,
            target_h.Stats,
            angle=0.0,
            initial_scale=shrink,
            min_overlap=0.5,
            wide_search=False,
        )
        self.assertAlmostEqual(refined, shrink, delta=0.05)


class TestLogpolarScaleFromFullPlane(unittest.TestCase):
    def test_identity_rows_give_unit_scale(self) -> None:
        row = np.linspace(1.0, 2.0, 64, dtype=np.float32)
        lp = np.tile(row, (32, 1))
        scale, _ = stos_brute._logpolar_scale_from_full_plane(lp, lp, 0.0, radius=32, log_polar_width=64)
        self.assertAlmostEqual(scale, 1.0, delta=0.05)


class TestChooseLogpolarScale(unittest.TestCase):
    def test_prefers_half_when_full_disagrees(self) -> None:
        self.assertAlmostEqual(
            stos_brute._choose_logpolar_scale(1.0, 1.12, full_peak_ratio=2.0), 1.0)

    def test_uses_full_when_consistent_and_strong_peak(self) -> None:
        self.assertAlmostEqual(
            stos_brute._choose_logpolar_scale(0.98, 0.99, full_peak_ratio=1.5), 0.99)


class TestSyntheticShrinkGrid(unittest.TestCase):
    """Phase A9 synthetic grid: isotropic shrink without metadata."""

    def _shrink_pair(self, shrink: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        base = rng.standard_normal((256, 256)).astype(np.float32)
        base -= base.min()
        base /= base.max() + 1e-6
        warped = scipy.ndimage.zoom(base, shrink, order=1)
        if shrink <= 1.0:
            pad_y = (base.shape[0] - warped.shape[0]) // 2
            pad_x = (base.shape[1] - warped.shape[1]) // 2
            source = np.zeros_like(base)
            source[pad_y:pad_y + warped.shape[0], pad_x:pad_x + warped.shape[1]] = warped
        else:
            start_y = (warped.shape[0] - base.shape[0]) // 2
            start_x = (warped.shape[1] - base.shape[1]) // 2
            source = warped[
                start_y:start_y + base.shape[0],
                start_x:start_x + base.shape[1],
            ]
        return source, base

    def test_logpolar_shrink_grid(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        for shrink in (0.95, 0.97, 1.0, 1.03, 1.05):
            with self.subTest(shrink=shrink):
                source, base = self._shrink_pair(shrink, seed=int(shrink * 100))
                source_h = nornir_imageregistration.ImagePermutationHelper(source)
                target_h = nornir_imageregistration.ImagePermutationHelper(base)
                result = stos_brute._find_angle_and_scale_with_logpolar(
                    source_image=source_h.ImageWithMaskAsNoise,
                    target_image=target_h.ImageWithMaskAsNoise,
                    source_stats=source_h.Stats,
                    target_stats=target_h.Stats,
                )
                self.assertAlmostEqual(result.scale, shrink, delta=0.06)
                self.assertAlmostEqual(result.angle, 0.0, delta=4.0)


if __name__ == '__main__':
    unittest.main()
