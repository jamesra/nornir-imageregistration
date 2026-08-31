"""One log-polar registration must compute the DoG magnitude spectrum twice, not four times.

Review #93 found ``_find_angle_and_scale_with_logpolar`` calling
``_logpolar_fft_magnitude`` four times -- once per image for the angle estimate and again,
with byte-identical arguments, for the radial scale seed. Commit 1d577b7 consolidated it:
the angle warp and ``_estimate_scale_radial_fft`` now share ``target_freq_shift`` and
``source_freq_shift``.

Two is the floor -- one spectrum per image -- so this is not a "fewer is better" budget
but an exact contract, and it is easy to undo. The helper is pure, so a future change that
recomputes it for the radial seed produces bit-identical output and no test would notice;
only the call count reveals it. That is why this asserts the count rather than the result.

The cost is not hypothetical. The spectrum is a Difference-of-Gaussians (low_sigma=4,
high_sigma=20, so a 161-tap separable kernel) plus a full FFT over the padded frame. While
profiling #229, ``correlate1d`` under this helper was the single largest non-idle cost in
the whole brute registration.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import stos_brute

_TARGET = 'nornir_imageregistration.stos_brute._logpolar_fft_magnitude'


def _textured_pair(size: int = 64, rotation_columns: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A small textured image and a shifted copy, big enough to survive the warps."""
    rng = np.random.default_rng(20260830)
    target = rng.random((size, size), dtype=np.float32)
    # A roll gives a real correlation peak without needing an interpolated rotation.
    source = np.roll(target, rotation_columns, axis=1)
    return target, source


class _SpectrumCounter:
    """Counts calls and records the arguments, delegating to the real helper."""

    def __init__(self):
        self.real = stos_brute._logpolar_fft_magnitude
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, padded_image, window, **kwargs):
        self.calls.append(tuple(padded_image.shape))
        return self.real(padded_image, window, **kwargs)

    @property
    def count(self) -> int:
        return len(self.calls)


def _run_logpolar(target: np.ndarray, source: np.ndarray):
    return stos_brute._find_angle_and_scale_with_logpolar(
        source_image=source,
        target_image=target,
        source_stats=nornir_imageregistration.ImageStats.CalcStats(source),
        target_stats=nornir_imageregistration.ImageStats.CalcStats(target),
        min_overlap=0.5)


class TestTheSpectrumIsComputedOncePerImage(unittest.TestCase):

    def setUp(self):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        self.target, self.source = _textured_pair()

    def _count(self) -> _SpectrumCounter:
        counter = _SpectrumCounter()
        with mock.patch(_TARGET, counter):
            _run_logpolar(self.target, self.source)
        return counter

    def test_exactly_two_calls(self):
        counter = self._count()
        self.assertEqual(2, counter.count,
                         'one spectrum per image is the floor; 4 means the radial seed '
                         'recomputed what the angle estimate already had (review #93)')

    def test_one_call_per_distinct_image(self):
        # Both images pad to the same frame, so shapes alone cannot distinguish them --
        # but two calls at that shape is exactly right, and four is exactly the bug.
        counter = self._count()
        self.assertEqual(2, len(counter.calls))
        self.assertEqual(counter.calls[0], counter.calls[1],
                         'target and source pad to a common frame, so the two calls '
                         'should share a shape')

    def test_the_count_does_not_grow_with_repeated_runs(self):
        first = self._count().count
        second = self._count().count
        self.assertEqual(first, second, 'call count must not depend on cache warmth')

    def test_the_registration_still_produces_a_result(self):
        # Guarding a call count is worthless if the call itself stopped working.
        result = _run_logpolar(self.target, self.source)
        self.assertIsNotNone(result)
        self.assertTrue(np.isfinite(result.angle))
        self.assertTrue(np.isfinite(result.scale))
        self.assertGreater(result.scale, 0.0)


class TestTheCounterWouldCatchTheRegression(unittest.TestCase):
    """A count assertion is only worth having if it fails when the bug returns.

    Rather than trust that, this reintroduces #93's shape -- a second pair of calls with
    identical arguments -- and confirms the counter sees four.
    """

    def setUp(self):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        self.target, self.source = _textured_pair()

    def test_a_duplicated_spectrum_is_detected(self):
        counter = _SpectrumCounter()

        original = stos_brute._find_angle_and_scale_with_logpolar

        def duplicating_magnitude(padded_image, window, **kwargs):
            # Stand in for the pre-1d577b7 code: compute it, then compute it again with
            # the same arguments, exactly as the _angle / _radial pairs did.
            counter(padded_image, window, **kwargs)
            return counter(padded_image, window, **kwargs)

        with mock.patch(_TARGET, duplicating_magnitude):
            original(source_image=self.source, target_image=self.target,
                     source_stats=nornir_imageregistration.ImageStats.CalcStats(self.source),
                     target_stats=nornir_imageregistration.ImageStats.CalcStats(self.target),
                     min_overlap=0.5)

        self.assertEqual(4, counter.count,
                         'the duplicating stand-in should produce the 4 calls #93 '
                         'reported, proving the count assertion has teeth')


class TestTheSharedSpectrumReachesBothConsumers(unittest.TestCase):
    """The count could also be held at 2 by dropping the radial seed entirely.

    So assert the scale estimator is still called, and called with the same arrays the
    angle path used -- that is what "share one spectrum" means.
    """

    def setUp(self):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        self.target, self.source = _textured_pair()

    def test_the_radial_estimator_still_runs(self):
        real = stos_brute._estimate_scale_radial_fft
        seen: list[tuple] = []

        def recording(target_magnitude, source_magnitude, *args, **kwargs):
            seen.append((id(target_magnitude), id(source_magnitude)))
            return real(target_magnitude, source_magnitude, *args, **kwargs)

        with mock.patch('nornir_imageregistration.stos_brute._estimate_scale_radial_fft',
                        recording):
            _run_logpolar(self.target, self.source)

        self.assertEqual(1, len(seen),
                         'the radial scale seed must still run; holding the spectrum '
                         'count at 2 by removing it is not the fix #93 asked for')

    def test_the_radial_estimator_receives_the_angle_spectra(self):
        spectra: list[int] = []
        real_magnitude = stos_brute._logpolar_fft_magnitude

        def recording_magnitude(padded_image, window, **kwargs):
            out = real_magnitude(padded_image, window, **kwargs)
            spectra.append(id(out))
            return out

        real_radial = stos_brute._estimate_scale_radial_fft
        received: list[int] = []

        def recording_radial(target_magnitude, source_magnitude, *args, **kwargs):
            received.extend((id(target_magnitude), id(source_magnitude)))
            return real_radial(target_magnitude, source_magnitude, *args, **kwargs)

        with mock.patch(_TARGET, recording_magnitude), \
             mock.patch('nornir_imageregistration.stos_brute._estimate_scale_radial_fft',
                        recording_radial):
            _run_logpolar(self.target, self.source)

        self.assertEqual(2, len(spectra))
        self.assertEqual(spectra, received,
                         'the radial seed must be handed the same two spectrum objects '
                         'the angle estimate warped, not fresh ones')


if __name__ == '__main__':
    unittest.main()
