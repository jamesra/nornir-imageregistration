"""``ScaleImage`` on CuPy matches ``cupyx.scipy.ndimage.zoom`` and reuses kernels.

CuPyX's zoom kernel bakes in the output shape, and the StosBrute scale search asks for a new
shape at almost every scale. After the rotate fix, py-spy on srv (2026-11-01) still had 12% of
AlignSections time compiling kernels under ``ScaleImage``. ``_zoom_bucketed_cupy`` zooms into
an output rounded up to ``_SCALE_OUTPUT_BUCKET`` pixels and crops; the zoom factors are a
runtime argument, so the crop must be bitwise identical to the plain zoom.

Skipped without a CUDA device.
"""

from __future__ import annotations

import unittest

import nornir_imageregistration
import numpy as np
from nornir_imageregistration.core import _core

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupyx_ndimage
except ImportError:
    _HAVE_GPU = False
else:
    try:
        _HAVE_GPU = cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:  # CUDA runtime without a device
        _HAVE_GPU = False


def _order(scalar: float) -> int:
    """The interpolation order ScaleImage picks for *scalar*."""
    if scalar < 0.5:
        return 0
    return 1 if scalar < 1.0 else 3


@unittest.skipUnless(_HAVE_GPU, 'needs CuPy and a CUDA device')
class TestBucketedZoomMatchesCupyx(unittest.TestCase):

    def test_scale_image_bitwise_identical_to_zoom(self):
        rng = np.random.default_rng(20261101)
        for shape in ((818, 640), (777, 818), (301, 455)):
            for dtype in (np.float32, np.float16):
                image = cp.asarray(rng.random(shape).astype(dtype))
                for scalar in (0.3, 0.5, 0.77, 0.95, 1.03, 1.12, 1.3429, 2.0):
                    with self.subTest(shape=shape, dtype=dtype.__name__, scalar=scalar):
                        expected = cupyx_ndimage.zoom(image, zoom=scalar, order=_order(scalar))
                        got = nornir_imageregistration.ScaleImage(image, scalar)
                        self.assertEqual(expected.shape, got.shape)
                        self.assertEqual(expected.dtype, got.dtype)
                        self.assertTrue(got.flags.c_contiguous)
                        self.assertTrue(bool(cp.array_equal(expected, got, equal_nan=True)))

    def test_nearby_scales_share_a_bucket(self):
        # Measured 40 scales over 0.90-1.12 on 818x640: 38 compiles before, 2 after.
        bucket = _core._SCALE_OUTPUT_BUCKET
        exact = set()
        bucketed = set()
        for scalar in np.linspace(0.9, 1.12, 40):
            out = tuple(int(round(s * scalar)) for s in (818, 640))  # noqa: RUF046 NumPy float scalar
            exact.add(out)
            bucketed.add(tuple(-(-o // bucket) * bucket for o in out))
        self.assertLess(len(bucketed) * 4, len(exact),
                        f'{len(exact)} exact shapes fell into {len(bucketed)} buckets')


if __name__ == '__main__':
    unittest.main()
