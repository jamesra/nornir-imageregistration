"""The Hann window cache is host-only because the log-polar path it feeds is host-only.

Review #96 read ``HannWindowCache`` storing numpy arrays and
``_find_angle_and_scale_with_logpolar`` passing them through
``_coerce_to_source_module(target_window, xp)``, and concluded the window re-uploads to
the device on every log-polar call. The upload would indeed be expensive if it happened:
measured on an RTX 4500 Ada, moving a float32 window host->device costs **2.1x to 4.2x a
full ``fft2`` over the same frame** (0.51ms vs 0.13ms at 1024 squared, 32.3ms vs 15.4ms at
8192 squared), and the function asks for two windows, which are the *same cached object*
whenever the two padded shapes agree.

It does not happen. ``xp`` in that function is always numpy, for two reasons that have to
hold together:

* the function opens by coercing both images to the host (``skimage.transform.warp_polar``
  is CPU-only), and
* ``pad_image_for_phase_correlation`` is array-module preserving -- numpy in, numpy out,
  *regardless of the active computation lib* -- so every array derived from those images
  stays on the host.

So ``_coerce_to_source_module(window, np)`` returns the identical object with no copy, and
the host-only cache is correct rather than a missed device cache. Closed ``wontfix``.

These tests pin the two premises rather than the conclusion, because the conclusion is
what a future change would silently break: move the log-polar FFTs to the device, or make
the padding helper follow the global lib, and the window uploads twice per call with no
test failing. If either assertion here starts failing, the right fix is a per-module window
cache keyed on the array module, not a coerce at the call site.
"""

from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import stos_brute
from nornir_imageregistration.hann_window_cache import HannWindowCache

try:
    import cupy
    _HAVE_CUPY = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover - depends on the host
    cupy = None
    _HAVE_CUPY = False


def _textured_pair(size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260831)
    target = rng.random((size, size), dtype=np.float32)
    source = np.roll(target, 3, axis=1)
    return target, source


class _ModuleSpy:
    """Records the array module each coerce targets, delegating to the real helper."""

    def __init__(self):
        self.real = stos_brute._coerce_to_source_module
        self.targets: list[str] = []

    def __call__(self, x, xp):
        self.targets.append(xp.__name__)
        return self.real(x, xp)


class TestTheWindowCoerceIsANoOp(unittest.TestCase):

    def test_coercing_a_cached_window_to_numpy_returns_the_same_object(self):
        window = HannWindowCache.GetOrCreate((128, 128))

        coerced = stos_brute._coerce_to_source_module(window, np)

        self.assertIs(coerced, window, 'a host window must not be copied on the way to a host FFT')
        self.assertFalse(coerced.flags.writeable, 'cache images are documented read-only')

    def test_two_equal_shapes_share_one_cached_window(self):
        """Why an upload would cost double: the two call sites hand back one object."""
        target_window = HannWindowCache.GetOrCreate((128, 128))
        source_window = HannWindowCache.GetOrCreate((128, 128))

        self.assertIs(target_window, source_window)


class TestThePaddingHelperPreservesTheArrayModule(unittest.TestCase):
    """The premise that keeps the log-polar frames on the host.

    ``pad_image_for_phase_correlation`` follows its *input*, not the active lib. If it ever
    starts following ``GetComputationModule()``, the log-polar frames become device arrays
    and the window upload in #96 becomes real.
    """

    def _pad(self, image):
        return nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
            image, min_overlap=0.5, image_median=0.5, image_stddev=0.1,
            new_height=128, new_width=128)

    def test_a_host_image_stays_on_the_host_under_every_lib(self):
        for lib in (nornir_imageregistration.ComputationLib.numpy,
                    nornir_imageregistration.ComputationLib.cupy):
            with self.subTest(lib=lib.name):
                nornir_imageregistration.SetActiveComputationLib(lib)
                padded = self._pad(np.zeros((64, 64), np.float32))
                self.assertIsInstance(padded, np.ndarray)

    @unittest.skipUnless(_HAVE_CUPY, 'requires a CUDA device')
    def test_a_device_image_stays_on_the_device(self):
        """The mirror of the above: the helper preserves, it does not force either way."""
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        padded = self._pad(cupy.zeros((64, 64), cupy.float32))

        self.assertIs(cupy.get_array_module(padded), cupy)


class TestTheLogPolarPathNeverAsksForADeviceWindow(unittest.TestCase):

    def setUp(self):
        self._lib = nornir_imageregistration.GetActiveComputationLib()
        self._real_coerce = stos_brute._coerce_to_source_module

    def tearDown(self):
        stos_brute._coerce_to_source_module = self._real_coerce
        nornir_imageregistration.SetActiveComputationLib(self._lib)

    def _run(self, source, target):
        return stos_brute._find_angle_and_scale_with_logpolar(
            source_image=source,
            target_image=target,
            source_stats=nornir_imageregistration.ImageStats.CalcStats(
                nornir_imageregistration.EnsureNumpyArray(source)),
            target_stats=nornir_imageregistration.ImageStats.CalcStats(
                nornir_imageregistration.EnsureNumpyArray(target)),
            min_overlap=0.5)

    def _targets_for(self, source, target) -> list[str]:
        spy = _ModuleSpy()
        stos_brute._coerce_to_source_module = spy
        try:
            self._run(source, target)
        finally:
            stos_brute._coerce_to_source_module = self._real_coerce
        return spy.targets

    def test_every_coerce_targets_numpy_with_host_inputs(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        target, source = _textured_pair()

        targets = self._targets_for(source, target)

        self.assertTrue(targets, 'the spy saw no coerce at all; the call sites moved')
        self.assertEqual({'numpy'}, set(targets))

    @unittest.skipUnless(_HAVE_CUPY, 'requires a CUDA device')
    def test_device_inputs_are_brought_to_the_host_and_stay_there(self):
        """Pyre and the GPU brute path arrive with device images; the windows still stay host."""
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        target, source = _textured_pair()

        targets = self._targets_for(cupy.asarray(source), cupy.asarray(target))

        self.assertEqual({'numpy'}, set(targets))

    @unittest.skipUnless(_HAVE_CUPY, 'requires a CUDA device')
    def test_the_angle_is_the_same_whichever_module_the_inputs_arrive_on(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        target, source = _textured_pair()

        host = self._run(source, target)
        device = self._run(cupy.asarray(source), cupy.asarray(target))

        self.assertAlmostEqual(host.angle, device.angle, places=6)
        self.assertAlmostEqual(host.scale, device.scale, places=6)


if __name__ == '__main__':
    unittest.main()
