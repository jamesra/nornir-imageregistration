"""Peak-search normalization must guard the range explicitly, not via seterr.

Both log-polar peak searches used to normalize with ``peak_search /=
peak_search.max()`` and catch ``FloatingPointError``.  That guard only works
because ``nornir_imageregistration`` sets ``np.seterr(invalid='raise',
divide='raise')`` at import, and only on the host:

* CuPy ignores ``seterr`` entirely, so a GPU surface yields NaN silently.
* Any caller running inside ``np.errstate(invalid='ignore')`` disarms it. The
  codebase does exactly that in ``core/_core.py`` and ``transforms/``.

``phasecorrelation.find_offset`` already checks the range explicitly; these tests
pin that behaviour for the log-polar pair.  The degenerate branch is forced by
substituting the correlation stage, because a flat-but-nonzero correlation
surface is not reachable through the public entry point with ordinary imagery.
"""
from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import stos_brute as sb

_N = 256


def _radius() -> int:
    return sb._radial_fft_max_radius(_N)


class _ConstantCorrelation:
    """Substitute for ``image_phase_correlation`` returning a flat nonzero surface.

    ``surface - surface.min()`` is then all zeros, so the normalization divides by
    zero. This is the exact input the guard exists for.
    """

    def __init__(self, value: float = 7.0):
        self.value = value
        self.calls = 0

    def __call__(self, target, source, *args, **kwargs):
        self.calls += 1
        return np.full(np.asarray(target).shape, self.value, dtype=np.float64)


class _NonFiniteCorrelation:
    def __init__(self, value: float = np.nan):
        self.value = value

    def __call__(self, target, source, *args, **kwargs):
        return np.full(np.asarray(target).shape, self.value, dtype=np.float64)


class _PatchCorrelation:
    """Context manager swapping the module-level correlation function."""

    def __init__(self, replacement):
        self.replacement = replacement
        self.module = nornir_imageregistration.phasecorrelation
        self.original = self.module.image_phase_correlation

    def __enter__(self):
        self.module.image_phase_correlation = self.replacement
        return self.replacement

    def __exit__(self, *exc):
        self.module.image_phase_correlation = self.original
        return False


class TestNormalizedPeakSearchSurface(unittest.TestCase):
    """Unit-level contract for the shared helper."""

    def test_healthy_surface_is_rescaled_to_unit_range(self):
        rng = np.random.default_rng(1)
        surface = rng.random((32, 32)) * 5.0 + 2.0

        result = sb._normalized_peak_search_surface(surface)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(float(result.min()), 0.0, places=6)
        self.assertAlmostEqual(float(result.max()), 1.0, places=6)

    def test_argmax_is_preserved(self):
        surface = np.zeros((16, 16))
        surface[3, 11] = 4.0

        result = sb._normalized_peak_search_surface(surface)

        assert result is not None
        self.assertEqual(np.unravel_index(int(np.argmax(result)), result.shape), (3, 11))

    def test_flat_surface_returns_none(self):
        for value in (0.0, 7.0, -3.5):
            with self.subTest(value=value):
                self.assertIsNone(
                    sb._normalized_peak_search_surface(np.full((16, 16), value)))

    def test_non_finite_surface_returns_none(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.assertIsNone(
                    sb._normalized_peak_search_surface(np.full((16, 16), value)))

    def test_does_not_mutate_its_input(self):
        rng = np.random.default_rng(2)
        surface = rng.random((16, 16))
        before = surface.copy()

        sb._normalized_peak_search_surface(surface)

        np.testing.assert_array_equal(surface, before)

    def test_guard_holds_with_error_state_disarmed(self):
        """The whole point: no reliance on np.seterr."""
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            self.assertIsNone(sb._normalized_peak_search_surface(np.full((16, 16), 7.0)))
            self.assertIsNone(sb._normalized_peak_search_surface(np.full((16, 16), np.nan)))


class TestScaleEstimatorHandlesDegenerateSurface(unittest.TestCase):

    def _estimate(self):
        flat = np.full((_N, _N), 5.0, dtype=np.float32)
        return sb._estimate_scale_radial_fft(flat, flat.copy(), _radius(), (_N, _N))

    def test_flat_correlation_returns_unity_and_zero_ratio(self):
        with _PatchCorrelation(_ConstantCorrelation()) as spy:
            scale, ratio = self._estimate()

        self.assertGreater(spy.calls, 0, 'premise: the substitute was not used')
        self.assertEqual(scale, 1.0)
        self.assertEqual(ratio, 0.0)

    def test_flat_correlation_is_finite_with_error_state_disarmed(self):
        with _PatchCorrelation(_ConstantCorrelation()):
            with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                scale, ratio = self._estimate()

        self.assertTrue(np.isfinite(scale), f'scale {scale} leaked non-finite')
        self.assertTrue(np.isfinite(ratio), f'ratio {ratio} leaked non-finite')
        self.assertEqual(scale, 1.0)

    def test_non_finite_correlation_is_finite_with_error_state_disarmed(self):
        with _PatchCorrelation(_NonFiniteCorrelation()):
            with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                scale, ratio = self._estimate()

        self.assertTrue(np.isfinite(scale), f'scale {scale} leaked non-finite')
        self.assertTrue(np.isfinite(ratio), f'ratio {ratio} leaked non-finite')

    def test_real_spectra_still_estimate_a_scale(self):
        """Positive control: the guard must not swallow working input."""
        rng = np.random.default_rng(4)
        target = rng.random((_N, _N)).astype(np.float32)
        window = sb.HannWindowCache.GetOrCreate(target.shape)
        magnitude = sb._logpolar_fft_magnitude(target, window, use_dog=True)

        scale, _ratio = sb._estimate_scale_radial_fft(
            magnitude, magnitude.copy(), _radius(), (_N, _N))

        self.assertTrue(np.isfinite(scale))
        self.assertGreaterEqual(scale, sb._SCALE_REFINE_MIN)
        self.assertLessEqual(scale, sb._SCALE_REFINE_MAX)


class TestAnglePathHandlesDegenerateSurface(unittest.TestCase):

    def _run(self):
        rng = np.random.default_rng(6)
        target = (rng.random((_N, _N)) * 255).astype(np.float32)
        source = np.roll(target, 5, axis=0)
        return sb._find_angle_and_scale_with_logpolar(
            source, target,
            nornir_imageregistration.ImageStats.CalcStats(source),
            nornir_imageregistration.ImageStats.CalcStats(target),
            min_overlap=0.5)

    def test_flat_correlation_reports_no_rotation(self):
        with _PatchCorrelation(_ConstantCorrelation()):
            with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                result = self._run()

        self.assertEqual(result.angle, 0)
        self.assertEqual(result.weight, 0)
        self.assertEqual(result.scale, 1.0)
        self.assertTrue(np.all(np.isfinite(np.asarray(result.translation, dtype=np.float64))))

    def test_non_finite_correlation_reports_no_rotation(self):
        with _PatchCorrelation(_NonFiniteCorrelation()):
            with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                result = self._run()

        self.assertEqual(result.weight, 0)
        self.assertTrue(np.isfinite(float(result.angle)))
        self.assertTrue(np.isfinite(float(result.scale)))


if __name__ == '__main__':
    unittest.main()
