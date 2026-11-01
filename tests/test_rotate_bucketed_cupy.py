"""The bucketed CuPy rotation matches ``cupyx.scipy.ndimage.rotate`` and reuses kernels.

CuPyX's ``affine_transform`` compiles its kernel with the output shape baked in. A
``reshape=True`` rotation has a different output shape for almost every angle, so StosBrute
compiled a new NVRTC kernel per angle: py-spy on srv (2026-11-01) put ~70% of AlignSections
time in ``Program.compile``, 47% of it under ``rotate_image``.
``_rotate_reshape_bucketed_cupy`` rotates into an output rounded up to
``_ROTATE_OUTPUT_BUCKET`` pixels and crops. Each output pixel depends only on the matrix,
offset and its own index, so the crop must be bitwise identical to the plain rotation.

Skipped without a CUDA device.
"""

from __future__ import annotations

import unittest

import numpy as np
from nornir_imageregistration import stos_brute

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


@unittest.skipUnless(_HAVE_GPU, 'needs CuPy and a CUDA device')
class TestBucketedRotationMatchesCupyx(unittest.TestCase):

    def test_bitwise_identical_to_rotate_reshape(self):
        rng = np.random.default_rng(20261101)
        for shape in ((818, 640), (777, 818), (301, 455)):
            for dtype in (np.float32, np.float16):
                image = cp.asarray(rng.random(shape).astype(dtype))
                for angle in (0.5, -3.0, 45.0, 90.0, 133.3, 180.0, -179.0, 271.0):
                    with self.subTest(shape=shape, dtype=dtype.__name__, angle=angle):
                        expected = cupyx_ndimage.rotate(image, axes=(1, 0), angle=angle, cval=np.nan)
                        got = stos_brute._rotate_reshape_bucketed_cupy(image, angle, cval=np.nan)
                        self.assertEqual(expected.shape, got.shape)
                        self.assertEqual(expected.dtype, got.dtype)
                        self.assertTrue(got.flags.c_contiguous)
                        self.assertTrue(bool(cp.array_equal(expected, got, equal_nan=True)))

    def test_nearby_angles_share_a_bucket(self):
        # The point of the change: small angle steps must not each get their own output
        # shape. Measured 80 angles over +-30 degrees: 39 compiles before, 6 after.
        image = cp.zeros((818, 640), dtype=cp.float32)
        bucket = stos_brute._ROTATE_OUTPUT_BUCKET
        exact = set()
        bucketed = set()
        for angle in np.arange(-30.0, 30.0, 0.75):
            out = cupyx_ndimage.rotate(image, axes=(1, 0), angle=float(angle), cval=np.nan)
            exact.add(out.shape)
            bucketed.add(tuple(-(-s // bucket) * bucket for s in out.shape))
        self.assertLess(len(bucketed) * 4, len(exact),
                        f'{len(exact)} exact shapes fell into {len(bucketed)} buckets')


if __name__ == '__main__':
    unittest.main()
