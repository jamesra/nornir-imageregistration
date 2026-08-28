"""The radial scale seed requires the DoG magnitude spectrum, not a raw one.

``_logpolar_fft_magnitude`` used to claim "A8: DoG for angle, raw for scale" while
all four call sites passed ``use_dog=True``.  Reading the code, that looks like the
scale path was never wired to the raw spectrum it was supposed to use.

Measurement says the call sites are right and the docstring was wrong.  A raw
magnitude spectrum is dominated by the DC and near-DC terms, which land at radius
0 in every log-polar warp no matter the scale, so the phase-correlation peak is
pinned at zero column shift and the estimate collapses to exactly 1.0 for every
input.  These tests lock that in, so the contradiction is not "fixed" later by
flipping the flag and silently disabling the scale seed.

Everything here avoids ``pad_image_for_phase_correlation``, which fills with
unseeded random noise and makes the full log-polar entry point nondeterministic
run to run.
"""
from __future__ import annotations

import unittest

import numpy as np
import skimage.filters
import skimage.transform

from nornir_imageregistration import stos_brute as sb

_SIZE = 512


def _texture(size: int = _SIZE, seed: int = 3) -> np.ndarray:
    """Deterministic multi-octave texture, so the radial profile carries information."""
    rng = np.random.default_rng(seed)
    img = rng.random((size, size))
    for sigma in (2, 6, 16):
        img = img + 0.7 * skimage.filters.gaussian(img, sigma=sigma)
    img -= img.min()
    img /= img.max()
    return (img * 255.0).astype(np.float32)


def _rescale_to_same_shape(img: np.ndarray, scale: float) -> np.ndarray:
    """Rescale then centre-crop/zero-pad back to ``img.shape``, deterministically."""
    scaled = skimage.transform.rescale(
        img, scale, order=3, preserve_range=True, anti_aliasing=True).astype(np.float32)
    out = np.zeros_like(img)
    h, w = img.shape
    sh, sw = scaled.shape
    if sh >= h:
        top = (sh - h) // 2
        left = (sw - w) // 2
        out[:] = scaled[top:top + h, left:left + w]
    else:
        top = (h - sh) // 2
        left = (w - sw) // 2
        out[top:top + sh, left:left + sw] = scaled
    return out


def _seed_for(target: np.ndarray, source: np.ndarray, use_dog: bool) -> float:
    """Radial scale seed, skipping the noise-padding stage entirely."""
    window = sb.HannWindowCache.GetOrCreate(target.shape)
    target_magnitude = sb._logpolar_fft_magnitude(target, window, use_dog=use_dog)
    source_magnitude = sb._logpolar_fft_magnitude(source, window, use_dog=use_dog)
    radius = sb._radial_fft_max_radius(max(target.shape))
    scale, _ratio = sb._estimate_scale_radial_fft(
        target_magnitude, source_magnitude, radius, target.shape)
    return scale


class TestRawSpectrumCarriesNoScaleInformation(unittest.TestCase):
    """The "documented" raw path is degenerate; this is why it must not be used."""

    def test_raw_returns_unity_for_every_scale(self):
        target = _texture()
        for scale in (0.92, 0.96, 1.0, 1.04, 1.08):
            with self.subTest(scale=scale):
                source = _rescale_to_same_shape(target, scale)

                self.assertAlmostEqual(_seed_for(target, source, use_dog=False), 1.0, places=3,
                                       msg='raw spectrum should collapse to unity')

    def test_raw_is_no_better_than_guessing_one(self):
        """Aggregate the same point as an error comparison, which is what matters."""
        target = _texture()
        scales = (0.92, 0.96, 1.04, 1.08)
        raw_error = 0.0
        dog_error = 0.0
        for scale in scales:
            source = _rescale_to_same_shape(target, scale)
            raw_error += abs(_seed_for(target, source, use_dog=False) - scale)
            dog_error += abs(_seed_for(target, source, use_dog=True) - scale)

        self.assertLess(dog_error, raw_error,
                        f'DoG total error {dog_error:.4f} should beat raw {raw_error:.4f}')


class TestDogSpectrumTracksScale(unittest.TestCase):
    """Positive control: with the DoG the seed is actually useful."""

    def test_seed_tracks_true_scale_inside_the_supported_band(self):
        target = _texture()
        for scale in (0.94, 0.97, 1.0, 1.03, 1.06):
            with self.subTest(scale=scale):
                source = _rescale_to_same_shape(target, scale)

                seed = _seed_for(target, source, use_dog=True)

                self.assertAlmostEqual(seed, scale, delta=0.02,
                                       msg=f'DoG seed {seed} should track {scale}')

    def test_seed_is_monotone_in_scale(self):
        target = _texture()
        scales = (0.94, 0.98, 1.02, 1.06)
        seeds = [_seed_for(target, _rescale_to_same_shape(target, s), use_dog=True) for s in scales]

        for earlier, later in zip(seeds, seeds[1:]):
            self.assertLessEqual(earlier, later + 1e-9, f'seeds not monotone: {seeds}')

    def test_seed_respects_the_clamp(self):
        """Out-of-band scales clamp rather than returning nonsense."""
        target = _texture()
        for scale in (0.6, 1.6):
            with self.subTest(scale=scale):
                seed = _seed_for(target, _rescale_to_same_shape(target, scale), use_dog=True)

                self.assertGreaterEqual(seed, sb._SCALE_REFINE_MIN)
                self.assertLessEqual(seed, sb._SCALE_REFINE_MAX)


class TestMagnitudeHelperIsPure(unittest.TestCase):
    """Angle and scale now share one spectrum; that is only safe if nothing mutates it."""

    def test_repeated_calls_are_bit_identical(self):
        image = _texture()
        window = sb.HannWindowCache.GetOrCreate(image.shape)

        first = sb._logpolar_fft_magnitude(image, window, use_dog=True)
        second = sb._logpolar_fft_magnitude(image, window, use_dog=True)

        np.testing.assert_array_equal(first, second)

    def test_call_does_not_mutate_its_inputs(self):
        image = _texture()
        window = sb.HannWindowCache.GetOrCreate(image.shape)
        image_before = image.copy()
        window_before = window.copy()

        sb._logpolar_fft_magnitude(image, window, use_dog=True)

        np.testing.assert_array_equal(image, image_before)
        np.testing.assert_array_equal(window, window_before)

    def test_scale_estimator_does_not_mutate_the_shared_spectrum(self):
        target = _texture()
        source = _rescale_to_same_shape(target, 1.05)
        window = sb.HannWindowCache.GetOrCreate(target.shape)
        target_magnitude = sb._logpolar_fft_magnitude(target, window, use_dog=True)
        source_magnitude = sb._logpolar_fft_magnitude(source, window, use_dog=True)
        target_before = target_magnitude.copy()
        source_before = source_magnitude.copy()

        sb._estimate_scale_radial_fft(
            target_magnitude, source_magnitude,
            sb._radial_fft_max_radius(max(target.shape)), target.shape)

        np.testing.assert_array_equal(target_magnitude, target_before)
        np.testing.assert_array_equal(source_magnitude, source_before)

    def test_warp_polar_does_not_mutate_the_shared_spectrum(self):
        """The angle path warps the same array the scale path later consumes."""
        image = _texture()
        window = sb.HannWindowCache.GetOrCreate(image.shape)
        magnitude = sb._logpolar_fft_magnitude(image, window, use_dog=True)
        before = magnitude.copy()

        skimage.transform.warp_polar(
            magnitude,
            radius=sb._logpolar_warp_radius(max(image.shape)),
            output_shape=image.shape,
            scaling='log',
            order=sb._LOGPOLAR_WARP_ORDER)

        np.testing.assert_array_equal(magnitude, before)


class TestCallSiteUsesOneDogSpectrum(unittest.TestCase):
    """Guard the call site, which is what a future "fix" would change.

    The tests above only prove raw is unusable; they would still pass if someone
    flipped the flag in ``_find_angle_and_scale_with_logpolar`` and silently
    disabled the scale seed.
    """

    def setUp(self):
        self.calls: list[bool] = []
        self.arrays: list[int] = []
        self.real = sb._logpolar_fft_magnitude

        def spy(padded_image, window, *, use_dog):
            self.calls.append(use_dog)
            self.arrays.append(id(padded_image))
            return self.real(padded_image, window, use_dog=use_dog)

        sb._logpolar_fft_magnitude = spy
        self.addCleanup(setattr, sb, '_logpolar_fft_magnitude', self.real)

    def _run_logpolar(self):
        import nornir_imageregistration

        target = _texture()
        source = _rescale_to_same_shape(target, 1.04)
        sb._find_angle_and_scale_with_logpolar(
            source, target,
            nornir_imageregistration.ImageStats.CalcStats(source),
            nornir_imageregistration.ImageStats.CalcStats(target),
            min_overlap=0.5)

    def test_every_spectrum_is_dog_filtered(self):
        self._run_logpolar()

        self.assertTrue(self.calls, 'spy never fired; the helper was not called')
        self.assertTrue(all(self.calls),
                        'a raw spectrum reached the log-polar path, which zeroes the '
                        'scale seed; see this module docstring')

    def test_spectrum_is_computed_once_per_image(self):
        """Angle and scale share one spectrum, so two images means two calls."""
        self._run_logpolar()

        self.assertEqual(len(self.calls), 2,
                         f'expected one DoG+FFT per image, got {len(self.calls)} calls')
        self.assertEqual(len(set(self.arrays)), 2, 'the two calls should be different images')


if __name__ == '__main__':
    unittest.main()
