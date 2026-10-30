"""The feature-score path picks its backend from the array, not from the import (#249).

`image_stats.py` binds three backend names once, at import time:

```python
try:
    import cupy as cp
    import cupyx.scipy as sp
    import cupy.fft as fftpack
except (ModuleNotFoundError, ImportError):
    import scipy as sp
    import numpy.fft as fftpack
```

So on any machine with CuPy installed, `sp` is `cupyx.scipy` and `fftpack` is `cupy.fft`
regardless of where the image being scored actually lives -- and
`ImageParamToImageArray` hands back a host array. Three failures followed, each hidden behind
the one before it:

| call | failure with a host array |
|---|---|
| `sp.ndimage.filters.gaussian_filter` | `AttributeError: module 'cupyx.scipy.ndimage' has no attribute 'filters'` |
| `fftpack.fft2` | `TypeError: The input array a must be a cupy.ndarray` |
| `numpy.percentile(score_list)` | (device arrays only) `TypeError: Implicit conversion to a NumPy array is not allowed` |

`cupyx.scipy.ndimage` has no `filters` submodule at all, and `scipy.ndimage.filters` is
deprecated and slated for removal in SciPy 2.0, so the host path was living on borrowed time
too.

Each call now dispatches on the array via `cp.get_array_module` /
`cupyx.scipy.get_array_module`, the idiom the Numpy-CuPy-compatibility rule requires and which
`core/_core.py:266` already uses. Measured after the fix: the two backends agree to 3.3e-08 and
6.5e-08 relative on 96x96 and 256x256 inputs, so `feature_score_threshold` means the same thing
either way.

Note the CuPy-specific failures only reproduce on a machine with CuPy installed; on a host-only
machine `sp` was already `scipy` and the old code worked. `test_the_deprecated_namespace_is_gone`
is the backend-independent guard.
"""

from __future__ import annotations

import unittest

import numpy as np
import scipy

import nornir_imageregistration
import nornir_imageregistration.image_stats as image_stats

try:
    import cupy
    import cupyx.scipy

    HAVE_CUPY = True
except (ModuleNotFoundError, ImportError):
    HAVE_CUPY = False

_needs_cupy = unittest.skipUnless(HAVE_CUPY, 'requires a CuPy build')


def _image(shape=(96, 96), seed=4):
    return np.random.default_rng(seed).random(shape).astype(np.float32)


class TestTheHostPathScores(unittest.TestCase):
    """The reported failure: any use_feature_score config aborted."""

    def test_a_numpy_image_produces_a_score(self):
        score = image_stats.__CalculateFeatureScoreSciPy__(_image())
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

    def test_it_scores_several_sizes(self):
        for shape in [(96, 96), (128, 128), (256, 256), (192, 96)]:
            with self.subTest(shape=shape):
                score = image_stats.__CalculateFeatureScoreSciPy__(_image(shape))
                self.assertIsInstance(score, float)
                self.assertGreater(score, 0.0)

    def test_the_power_spectral_density_scorer_takes_a_host_array(self):
        # This is the call that failed once .filters was dropped.
        value = image_stats.ScoreImageWithPowerSpectralDensity(_image())
        self.assertGreater(float(value), 0.0)

    def test_a_flat_image_scores_lower_than_a_noisy_one(self):
        flat = np.full((128, 128), 0.5, dtype=np.float32)
        noisy = _image((128, 128))
        self.assertLess(image_stats.__CalculateFeatureScoreSciPy__(flat),
                        image_stats.__CalculateFeatureScoreSciPy__(noisy))


class TestTheDeprecatedNamespaceIsGone(unittest.TestCase):
    """Backend-independent: the source must not reach through `.filters` any more."""

    @staticmethod
    def _source():
        import inspect
        return inspect.getsource(image_stats)

    def test_no_live_call_goes_through_ndimage_filters(self):
        live = [line for line in self._source().splitlines()
                if 'ndimage.filters' in line and not line.strip().startswith('#')]
        self.assertEqual([], live)

    def test_the_gaussian_filter_call_is_dispatched(self):
        source = self._source()
        self.assertIn('cupyx.scipy.get_array_module', source)

    def test_the_fft_call_is_dispatched(self):
        live = [line for line in self._source().splitlines()
                if 'fftpack.fft2' in line and not line.strip().startswith('#')]
        self.assertEqual([], live, 'fftpack is bound to cupy.fft at import time')


class TestThePremise(unittest.TestCase):
    """Why the old call could not work on both backends."""

    def test_scipy_still_has_the_deprecated_shim(self):
        self.assertTrue(hasattr(scipy.ndimage, 'filters'))
        self.assertTrue(hasattr(scipy.ndimage, 'gaussian_filter'))

    @_needs_cupy
    def test_cupyx_has_no_filters_submodule(self):
        self.assertFalse(hasattr(cupyx.scipy.ndimage, 'filters'))
        self.assertTrue(hasattr(cupyx.scipy.ndimage, 'gaussian_filter'),
                        'the function is there, only the submodule is missing')

    @_needs_cupy
    def test_dropping_filters_alone_would_not_have_been_enough(self):
        # The fix proposed on the issue. It still rejects the host array that
        # ImageParamToImageArray produces, so it only moved the error.
        with self.assertRaises(TypeError):
            cupyx.scipy.ndimage.gaussian_filter(_image(), sigma=2.5, radius=5)

    @_needs_cupy
    def test_radius_is_accepted_by_both_backends(self):
        host = _image()
        scipy.ndimage.gaussian_filter(host, sigma=2.5, radius=5)
        cupyx.scipy.ndimage.gaussian_filter(cupy.asarray(host), sigma=2.5, radius=5)

    def test_the_image_reaching_the_filter_is_a_host_array(self):
        out = nornir_imageregistration.ImageParamToImageArray(
            _image(), dtype=nornir_imageregistration.default_image_dtype())
        self.assertIsInstance(out, np.ndarray)


class TestDispatchFollowsTheArray(unittest.TestCase):
    """numpy in -> numpy out, cupy in -> cupy out."""

    @_needs_cupy
    def test_get_array_module_resolves_each_backend(self):
        self.assertIs(scipy, cupyx.scipy.get_array_module(_image()))
        self.assertIs(cupyx.scipy, cupyx.scipy.get_array_module(cupy.asarray(_image())))

    @_needs_cupy
    def test_a_device_image_scores(self):
        score = image_stats.__CalculateFeatureScoreSciPy__(cupy.asarray(_image()))
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

    @_needs_cupy
    def test_the_psd_scorer_keeps_the_input_backend(self):
        host = image_stats.ScoreImageWithPowerSpectralDensity(_image())
        device = image_stats.ScoreImageWithPowerSpectralDensity(cupy.asarray(_image()))
        self.assertNotIsInstance(host, cupy.ndarray)
        self.assertIsInstance(device, cupy.ndarray)


class TestTheBackendsAgree(unittest.TestCase):
    """So feature_score_threshold means the same thing either way."""

    @_needs_cupy
    def test_the_scores_match(self):
        # Measured at 3.3e-08 and 6.5e-08 relative.
        for shape in [(96, 96), (256, 256)]:
            with self.subTest(shape=shape):
                host = _image(shape)
                a = image_stats.__CalculateFeatureScoreSciPy__(host)
                b = image_stats.__CalculateFeatureScoreSciPy__(cupy.asarray(host))
                self.assertLess(abs(a - b) / max(abs(a), 1e-12), 1e-6)

    @_needs_cupy
    def test_the_filtered_images_match(self):
        host = _image()
        a = scipy.ndimage.gaussian_filter(host, sigma=2.5, radius=5)
        b = cupy.asnumpy(
            cupyx.scipy.ndimage.gaussian_filter(cupy.asarray(host), sigma=2.5, radius=5))
        np.testing.assert_allclose(a, b, atol=1e-5)

    @_needs_cupy
    def test_both_return_a_plain_float(self):
        host = _image()
        for label, arg in (('numpy', host), ('cupy', cupy.asarray(host))):
            with self.subTest(backend=label):
                self.assertIsInstance(
                    image_stats.__CalculateFeatureScoreSciPy__(arg), float)


if __name__ == '__main__':
    unittest.main()
