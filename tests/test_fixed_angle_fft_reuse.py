"""Compatibility tests for fixed-size angle sweeps and reused target FFTs."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage

import nornir_imageregistration
from nornir_imageregistration import stos_brute
from nornir_imageregistration.phasecorrelation import image_phase_correlation
from nornir_imageregistration.stos_brute import (
    _find_best_angle,
    _fixed_correlation_shape,
    _rotated_aabb_shape,
    _score_one_angle_core,
    pad_and_rotate_image,
    rotate_image,
)

try:
    import cupy as cp

    _HAS_CUPY = True
except Exception:  # pragma: no cover - optional GPU dependency
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False


def _letter_f(shape: tuple[int, int], seed: int = 1) -> np.ndarray:
    """High-contrast anisotropic pattern so angle scoring has a clear winner."""
    img = np.full(shape, 0.2, dtype=np.float32)
    h, w = shape
    img[h // 5: 4 * h // 5, w // 4: w // 4 + h // 12] = 1.0
    img[h // 5: h // 5 + h // 12, w // 4: 3 * w // 5] = 1.0
    img[h // 2: h // 2 + h // 14, w // 4: w // 2] = 1.0
    rng = np.random.default_rng(seed)
    img += 0.05 * rng.normal(size=shape).astype(np.float32)
    return ndimage.gaussian_filter(img, 0.8).astype(np.float32)


def _center_to_shape(image: np.ndarray, shape: tuple[int, int], fill: float) -> np.ndarray:
    """Crop or pad *image* so its center content fills *shape*."""
    out = np.full(shape, fill, dtype=np.float32)
    sy = (image.shape[0] - shape[0]) // 2
    sx = (image.shape[1] - shape[1]) // 2
    if sy >= 0 and sx >= 0:
        out[:] = image[sy: sy + shape[0], sx: sx + shape[1]]
        return out
    dy = max(0, (shape[0] - image.shape[0]) // 2)
    dx = max(0, (shape[1] - image.shape[1]) // 2)
    y1 = dy + min(image.shape[0], shape[0] - dy)
    x1 = dx + min(image.shape[1], shape[1] - dx)
    out[dy:y1, dx:x1] = image[: y1 - dy, : x1 - dx]
    if image.shape[0] > shape[0] or image.shape[1] > shape[1]:
        sy = max(0, (image.shape[0] - shape[0]) // 2)
        sx = max(0, (image.shape[1] - shape[1]) // 2)
        return image[sy: sy + shape[0], sx: sx + shape[1]].astype(np.float32)
    return out


def _pair_for_angle(
        shape: tuple[int, int],
        offset_yx: tuple[float, float],
        angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Target is a translated letter; source is rotated by *-angle_deg* then cropped."""
    base = _letter_f(shape)
    stats = nornir_imageregistration.ImageStats.CalcStats(base)
    target = ndimage.shift(
        base, offset_yx, order=1, mode='constant', cval=float(stats.median)).astype(np.float32)
    if abs(angle_deg) < 1e-9:
        source = base.copy()
    else:
        rotated = rotate_image(base, -angle_deg, stats)
        source = _center_to_shape(rotated, shape, float(stats.median))
    return target, source.astype(np.float32)


def _pad_target_to_fixed(
        target: np.ndarray,
        target_stats: nornir_imageregistration.ImageStats,
        fixed_shape: tuple[int, int]) -> np.ndarray:
    return nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
        target,
        min_overlap=1.0,
        image_median=target_stats.median,
        image_stddev=target_stats.std,
        new_height=fixed_shape[0],
        new_width=fixed_shape[1],
        original_shape=target.shape)


class TestRotatedAabbHelper(unittest.TestCase):
    """AABB helper must cover SciPy/CuPyX reshape=True rotate output."""

    def test_aabb_covers_ndimage_rotate(self) -> None:
        angles = (0, 15, 30, 45, 90, 135)
        shapes = ((64, 64), (48, 80), (100, 60))
        for height, width in shapes:
            img = np.ones((height, width), dtype=np.float32)
            for angle in angles:
                rotated = ndimage.rotate(img, angle, reshape=True, order=1)
                aabb = _rotated_aabb_shape(height, width, float(angle))
                self.assertLessEqual(
                    rotated.shape[0], aabb[0],
                    msg=f'h={height} w={width} angle={angle}: rot={rotated.shape} aabb={aabb}')
                self.assertLessEqual(
                    rotated.shape[1], aabb[1],
                    msg=f'h={height} w={width} angle={angle}: rot={rotated.shape} aabb={aabb}')
                # Helper should not be wildly larger than SciPy (slack from ceil + 1px margin).
                self.assertLessEqual(aabb[0] - rotated.shape[0], 4)
                self.assertLessEqual(aabb[1] - rotated.shape[1], 4)


class TestFixedCorrelationShape(unittest.TestCase):
    """Fixed sweep frame covers every angle and stays tight for narrow refine ranges."""

    def test_fixed_shape_covers_sweep_and_narrow_is_smaller(self) -> None:
        source_shape = (80, 120)
        target_shape = (80, 120)
        min_overlap = 0.5
        full_angles = list(range(0, 360, 2))
        narrow_angles = list(range(-2, 3))

        fixed_full = _fixed_correlation_shape(target_shape, source_shape, full_angles, min_overlap)
        fixed_narrow = _fixed_correlation_shape(target_shape, source_shape, narrow_angles, min_overlap)

        img = np.ones(source_shape, dtype=np.float32)
        for angle in full_angles[::15]:
            rotated = ndimage.rotate(img, angle, reshape=True, order=1)
            self.assertLessEqual(rotated.shape[0], fixed_full[0])
            self.assertLessEqual(rotated.shape[1], fixed_full[1])

        for angle in narrow_angles:
            rotated = ndimage.rotate(img, angle, reshape=True, order=1)
            self.assertLessEqual(rotated.shape[0], fixed_narrow[0])
            self.assertLessEqual(rotated.shape[1], fixed_narrow[1])

        self.assertTrue(
            fixed_narrow[0] < fixed_full[0] or fixed_narrow[1] < fixed_full[1],
            msg=f'narrow {fixed_narrow} should be strictly smaller than full {fixed_full}')


class TestPeakOffsetAndRankingParity(unittest.TestCase):
    """Old per-angle sizing vs fixed-size + reused FFT: same geometry, not bit-identical weights."""

    def setUp(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

    def test_per_angle_peak_offset_parity(self) -> None:
        shape = (160, 160)
        offset = (6.0, -5.0)
        min_overlap = 0.5
        angles = [0.0, 15.0, 30.0, 45.0]
        fixed = _fixed_correlation_shape(shape, shape, angles, min_overlap)

        for angle in angles:
            target, source = _pair_for_angle(shape, offset, angle)
            target_stats = nornir_imageregistration.ImageStats.CalcStats(target)
            source_stats = nornir_imageregistration.ImageStats.CalcStats(source)
            padded_target = _pad_target_to_fixed(target, target_stats, fixed)
            fft_target = np.fft.fft2(padded_target - target_stats.mean)

            np.random.seed(1000 + int(angle))
            old = _score_one_angle_core(
                target, source, shape, shape, angle, target_stats, source_stats,
                target_image_prepadded=False, min_overlap=min_overlap, fixed_shape=None)
            np.random.seed(2000 + int(angle))
            new = _score_one_angle_core(
                padded_target, source, shape, shape, angle, target_stats, source_stats,
                target_image_prepadded=True, min_overlap=min_overlap,
                fixed_shape=fixed, fft_target=fft_target)

            delta = np.abs(np.asarray(old.peak, dtype=float) - np.asarray(new.peak, dtype=float))
            self.assertLessEqual(
                float(delta[0]), 1.0,
                msg=f'angle={angle}: old={old.peak} new={new.peak}')
            self.assertLessEqual(
                float(delta[1]), 1.0,
                msg=f'angle={angle}: old={old.peak} new={new.peak}')

    def test_best_angle_ranking_parity(self) -> None:
        shape = (128, 128)
        offset = (6.0, -5.0)
        true_angle = 20.0
        min_overlap = 0.5
        angle_range = list(range(0, 40, 2))

        target, source = _pair_for_angle(shape, offset, true_angle)
        target_stats = nornir_imageregistration.ImageStats.CalcStats(target)
        source_stats = nornir_imageregistration.ImageStats.CalcStats(source)

        old_records = [
            _score_one_angle_core(
                target, source, shape, shape, float(angle), target_stats, source_stats,
                target_image_prepadded=False, min_overlap=min_overlap, fixed_shape=None)
            for angle in angle_range
        ]
        old_best = max(old_records, key=lambda r: r.weight)
        new_best = _find_best_angle(
            source_image=source,
            target_image=target,
            source_stats=source_stats,
            target_stats=target_stats,
            angle_range=angle_range,
            min_overlap=min_overlap,
            SingleThread=True)

        self.assertLessEqual(abs(old_best.angle - true_angle), 2.0)
        self.assertLessEqual(abs(new_best.angle - true_angle), 2.0)


class TestReusedFftMatchesFresh(unittest.TestCase):
    """Cached target FFT must match a fresh FFT of the same padded target."""

    def setUp(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

    def test_reused_fft_matches_fresh_at_fixed_size(self) -> None:
        shape = (96, 96)
        min_overlap = 0.5
        angle = 15.0
        fixed = _fixed_correlation_shape(shape, shape, [0.0, angle, 45.0], min_overlap)
        target, source = _pair_for_angle(shape, (4.0, -3.0), angle)
        target_stats = nornir_imageregistration.ImageStats.CalcStats(target)
        source_stats = nornir_imageregistration.ImageStats.CalcStats(source)
        padded_target = _pad_target_to_fixed(target, target_stats, fixed)

        np.random.seed(123)
        rotated = pad_and_rotate_image(
            image=source,
            angle=angle,
            image_stats=source_stats,
            desired_shape=fixed,
            min_overlap=min_overlap)

        fft_target = np.fft.fft2(padded_target - target_stats.mean)
        corr_reused = image_phase_correlation(
            padded_target, rotated, target_stats.mean, source_stats.mean, 0.66,
            fft_target=fft_target)
        corr_fresh = image_phase_correlation(
            padded_target, rotated, target_stats.mean, source_stats.mean, 0.66,
            fft_target=None)
        np.testing.assert_allclose(corr_reused, corr_fresh, rtol=0, atol=0)

        # Same fixed-size scoring path with identical pad noise: cached vs fresh FFT.
        np.random.seed(99)
        with_cache = _score_one_angle_core(
            padded_target, source, shape, shape, angle, target_stats, source_stats,
            target_image_prepadded=True, min_overlap=min_overlap,
            fixed_shape=fixed, fft_target=fft_target)
        np.random.seed(99)
        without_cache = _score_one_angle_core(
            padded_target, source, shape, shape, angle, target_stats, source_stats,
            target_image_prepadded=True, min_overlap=min_overlap,
            fixed_shape=fixed, fft_target=None)

        np.testing.assert_allclose(
            np.asarray(with_cache.peak, dtype=float),
            np.asarray(without_cache.peak, dtype=float),
            atol=1e-5)
        self.assertAlmostEqual(with_cache.weight, without_cache.weight, places=5)


@unittest.skipUnless(_HAS_CUPY, 'CuPy not available')
class TestFixedSizeCupyParity(unittest.TestCase):
    """Fixed-size path on CuPy matches NumPy on peak offset."""

    def test_cupy_fixed_size_peak_matches_numpy(self) -> None:
        shape = (128, 128)
        offset = (6.0, -5.0)
        angle = 15.0
        min_overlap = 0.5
        angles = [0.0, angle, 45.0]
        fixed = _fixed_correlation_shape(shape, shape, angles, min_overlap)
        target, source = _pair_for_angle(shape, offset, angle)
        target_stats = nornir_imageregistration.ImageStats.CalcStats(target)
        source_stats = nornir_imageregistration.ImageStats.CalcStats(source)

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        padded_np = _pad_target_to_fixed(target, target_stats, fixed)
        fft_np = np.fft.fft2(padded_np - target_stats.mean)
        np.random.seed(7)
        rec_np = _score_one_angle_core(
            padded_np, source, shape, shape, angle, target_stats, source_stats,
            target_image_prepadded=True, min_overlap=min_overlap,
            fixed_shape=fixed, fft_target=fft_np)

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        try:
            target_cp = cp.asarray(target)
            source_cp = cp.asarray(source)
            padded_cp = cp.asarray(padded_np)
            fft_cp = cp.fft.fft2(padded_cp - target_stats.mean)
            np.random.seed(7)
            rec_cp = _score_one_angle_core(
                padded_cp, source_cp, shape, shape, angle, target_stats, source_stats,
                target_image_prepadded=True, min_overlap=min_overlap,
                fixed_shape=fixed, fft_target=fft_cp)
        finally:
            nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

        peak_np = np.asarray(rec_np.peak, dtype=float)
        peak_cp = np.asarray(rec_cp.peak, dtype=float)
        self.assertLessEqual(abs(peak_np[0] - peak_cp[0]), 1.0)
        self.assertLessEqual(abs(peak_np[1] - peak_cp[1]), 1.0)


if __name__ == '__main__':
    unittest.main()
