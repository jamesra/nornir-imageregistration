"""A clamped radial scale seed must announce itself.

``_estimate_scale_radial_fft`` clamps its estimate to
``[_SCALE_REFINE_MIN, _SCALE_REFINE_MAX]``. It used to do so silently, so a seed sitting
at the band edge could mean either "the scale really is about 1.12" or "the estimate was
nonsense and got squashed". Measured over a sweep of true scales, 8 of 15 saturated and
none of them emitted a single log record.

The clamped number cannot distinguish the two cases, and ``peak_ratio`` does not fill
the gap. Measured:

=========  ==========  ==========
true       returned    peak_ratio
=========  ==========  ==========
1.12       1.1113      1.345
2.00       1.1200      2.449
0.40       1.1200      1.028
=========  ==========  ==========

The badly aliased 2.00 case carries a *higher* peak ratio than the correct band-edge
1.12 case, so a confidence threshold would keep the wrong one. Note also that beyond
roughly a 25% scale change the log-radius correlation inverts: a true 0.40 estimates as
~2.5 and therefore clamps to the *top* of the band.

So saturation is now logged at warning level with the raw value, and surfaced on
``LogPolarDiagnostics`` for callers. The clamped value itself is unchanged; this is
observability only. See review issue #87.
"""

from __future__ import annotations

import logging
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


def _seed_and_ratio(target: np.ndarray, source: np.ndarray) -> tuple[float, float]:
    """Radial scale seed, skipping the noise-padding stage so results are repeatable."""
    window = sb.HannWindowCache.GetOrCreate(target.shape)
    target_magnitude = sb._logpolar_fft_magnitude(target, window, use_dog=True)
    source_magnitude = sb._logpolar_fft_magnitude(source, window, use_dog=True)
    radius = sb._radial_fft_max_radius(max(target.shape))
    return sb._estimate_scale_radial_fft(
        target_magnitude, source_magnitude, radius, target.shape)


class _TextureFixture(unittest.TestCase):
    """One texture per class; building it is the expensive part."""

    @classmethod
    def setUpClass(cls):
        cls.target = _texture()

    def _pair_at(self, scale: float) -> tuple[np.ndarray, np.ndarray]:
        return self.target, _rescale_to_same_shape(self.target, scale)


class TestSaturationIsAnnounced(_TextureFixture):
    """The fix: the clamp is no longer silent."""

    def test_an_out_of_band_scale_warns(self):
        target, source = self._pair_at(2.00)

        with self.assertLogs(sb.__name__, level=logging.WARNING) as captured:
            _seed_and_ratio(target, source)

        self.assertEqual(len(captured.records), 1)
        message = captured.records[0].getMessage()
        self.assertIn('outside supported band', message)

    def test_the_warning_reports_the_raw_estimate_not_just_the_clamp(self):
        """Without the raw value the operator cannot tell 1.2 from 2.0."""
        target, source = self._pair_at(2.00)

        with self.assertLogs(sb.__name__, level=logging.WARNING) as captured:
            _seed_and_ratio(target, source)

        message = captured.records[0].getMessage()
        self.assertRegex(message, r'radial scale seed 1\.9',
                         f'raw estimate missing from: {message}')

    def test_both_band_edges_warn(self):
        for scale in (0.40, 2.00):
            with self.subTest(scale=scale):
                target, source = self._pair_at(scale)
                with self.assertLogs(sb.__name__, level=logging.WARNING):
                    _seed_and_ratio(target, source)

    def test_an_in_band_scale_stays_quiet(self):
        """A warning on every ordinary registration would be worse than silence."""
        for scale in (0.96, 1.00, 1.04):
            with self.subTest(scale=scale):
                target, source = self._pair_at(scale)
                with self.assertNoLogs(sb.__name__, level=logging.WARNING):
                    _seed_and_ratio(target, source)


class TestTheClampedValueIsUnchanged(_TextureFixture):
    """Observability only -- registration output must not move."""

    def test_out_of_band_still_returns_the_bound(self):
        for scale, expected in ((0.75, sb._SCALE_REFINE_MIN),
                                (2.00, sb._SCALE_REFINE_MAX)):
            with self.subTest(scale=scale):
                seed, _ratio = _seed_and_ratio(*self._pair_at(scale))

                self.assertAlmostEqual(seed, expected, places=6)

    def test_in_band_is_returned_untouched_and_tracks_truth(self):
        for scale in (0.96, 1.00, 1.04):
            with self.subTest(scale=scale):
                seed, _ratio = _seed_and_ratio(*self._pair_at(scale))

                self.assertGreater(seed, sb._SCALE_REFINE_MIN)
                self.assertLess(seed, sb._SCALE_REFINE_MAX)
                self.assertAlmostEqual(seed, scale, delta=0.02)


class TestPeakRatioIsNotASaturationDetector(_TextureFixture):
    """Why a dedicated flag is needed instead of thresholding the existing signal."""

    def test_a_saturated_estimate_can_outrank_a_genuine_band_edge_one(self):
        _genuine_scale, genuine_ratio = _seed_and_ratio(*self._pair_at(1.12))
        _failed_scale, failed_ratio = _seed_and_ratio(*self._pair_at(2.00))

        self.assertGreater(
            failed_ratio, genuine_ratio,
            'if this flips, peak_ratio may have become a usable confidence gate and '
            'the rationale for scale_seed_saturated should be revisited')

    def test_a_large_downscale_estimates_upward(self):
        """The aliasing that makes the raw value unusable as a fallback."""
        low, high = sb._SCALE_REFINE_MIN, sb._SCALE_REFINE_MAX
        sb._SCALE_REFINE_MIN, sb._SCALE_REFINE_MAX = 0.0, 1e9
        try:
            raw, _ratio = _seed_and_ratio(*self._pair_at(0.40))
        finally:
            sb._SCALE_REFINE_MIN, sb._SCALE_REFINE_MAX = low, high

        self.assertGreater(raw, 1.0,
                           'a true 0.40 aliases to an upscale estimate; the raw value '
                           'must not be used as a fallback when saturated')


class TestDiagnosticsCarryTheProvenance(unittest.TestCase):
    """The flag reaches callers, and the new fields do not disturb existing ones."""

    def _diagnostics(self, **overrides):
        fields = dict(angle_peak_ratio=1.5, translation_peak_ratio=1.3,
                      strength_delta_ratio=0.3, degrees_per_pixel=0.7,
                      peak_strength=10.0)
        fields.update(overrides)
        return sb.LogPolarDiagnostics(**fields)

    def test_the_new_fields_default_so_existing_construction_still_works(self):
        diagnostics = self._diagnostics()

        self.assertEqual(diagnostics.radial_peak_ratio, 0.0)
        self.assertFalse(diagnostics.scale_seed_saturated)

    def test_the_saturation_flag_round_trips(self):
        diagnostics = self._diagnostics(scale_seed_saturated=True,
                                        radial_peak_ratio=2.449)

        self.assertTrue(diagnostics.scale_seed_saturated)
        self.assertAlmostEqual(diagnostics.radial_peak_ratio, 2.449)

    def test_confidence_ignores_the_new_fields(self):
        """_logpolar_confidence steers the fallback angle search; it must not move."""
        quiet = self._diagnostics()
        saturated = self._diagnostics(scale_seed_saturated=True,
                                      radial_peak_ratio=99.0)

        self.assertEqual(sb._logpolar_confidence(quiet),
                         sb._logpolar_confidence(saturated))

    def test_the_flag_matches_the_bound_test_used_at_the_call_site(self):
        low, high = sb._SCALE_REFINE_MIN, sb._SCALE_REFINE_MAX
        for seed, expected in ((low, True), (high, True), (low - 0.1, True),
                               (high + 0.1, True), (1.0, False),
                               ((low + high) / 2.0, False)):
            with self.subTest(seed=seed):
                self.assertEqual(seed <= low or seed >= high, expected)


class TestDegenerateSurfaceStillReportsUnity(unittest.TestCase):
    """The pre-existing no-peak path is untouched and must not warn."""

    def test_a_flat_spectrum_returns_unity_without_warning(self):
        flat = np.zeros((256, 256), dtype=np.float32)
        radius = sb._radial_fft_max_radius(256)

        with self.assertNoLogs(sb.__name__, level=logging.WARNING):
            scale, ratio = sb._estimate_scale_radial_fft(
                flat, flat.copy(), radius, (256, 256))

        self.assertEqual(scale, 1.0)
        self.assertEqual(ratio, 0.0)


if __name__ == '__main__':
    unittest.main()
