"""``return_valid_mask`` must not reallocate the output via ``xp.where`` (#111).

The canvas is ``xp.full(..., cval)`` before scatter, so unmapped pixels already hold
``cval``.  The old ``xp.where(valid_mask, outputImage, cval)`` built a second full
output array (peak ~2x the image) for a no-op fill.  Measured: exactly one
``np.where`` call on a bool canvas of the output shape during
``SourceImageToTargetSpace(..., return_valid_mask=True)`` before the fix; zero after.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import assemble


class TestReturnValidMaskSkipsWhereCopy(unittest.TestCase):

    def setUp(self):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        self.source = (np.random.default_rng(11).random((64, 64)) * 100).astype(np.float32)
        self.transform = nornir_imageregistration.transforms.Rigid(
            target_offset=(12.0, 9.0))
        self.cval = 7.0

    def _warp(self):
        return assemble.SourceImageToTargetSpace(
            self.transform, self.source,
            output_botleft=(0, 0), output_area=(64, 64),
            cval=self.cval, extrapolate=True, return_valid_mask=True)

    def test_where_is_not_called_for_the_cval_pass(self):
        real_where = np.where
        calls: list[tuple] = []

        def counting_where(*args, **kwargs):
            calls.append((getattr(args[0], 'dtype', None), getattr(args[0], 'shape', None)))
            return real_where(*args, **kwargs)

        with mock.patch.object(np, 'where', counting_where):
            image, mask = self._warp()

        bool_canvas_calls = [
            c for c in calls
            if c[0] == np.dtype(bool) and c[1] == (64, 64)
        ]
        self.assertEqual(
            [], bool_canvas_calls,
            msg=f'expected no bool-canvas where; saw {calls}')
        self.assertTrue(np.all(np.asarray(image)[~np.asarray(mask)] == self.cval))

    def test_output_matches_explicit_cval_fill_on_unmapped(self):
        image, mask = self._warp()
        image = np.asarray(image)
        mask = np.asarray(mask)
        self.assertTrue(np.all(image[~mask] == self.cval))
        self.assertGreater(float(mask.mean()), 0.5)
        self.assertLess(float(mask.mean()), 1.0)


if __name__ == '__main__':
    unittest.main()
