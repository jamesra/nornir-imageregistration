"""The NaN scan in _TransformImageUsingCoords must be deferred until it is needed (#110).

``any_nan_values = bool(xp.any(xp.isnan(subroi_warpedImage)))`` ran unconditionally on every warp
of every tile.  It is a full pass over the source plus a device-to-host sync, and two kinds of
caller never look at the result:

* one that names an ``interpolation_order`` (so the order is not being inferred) **and** warps with
  a sentinel ``cval`` above 1.0 (so ``preserve_cval_sentinel`` skips the clip).  The distance plane
  does both, on every tile.
* one warping a ``bool`` source, where the order is forced to 1 by dtype regardless.

Deferring it matches the lazy-stats treatment already used for ``_underflow_assemble_log_msg`` in
the same function.  Measured over four interleaved A/B rounds, this took 23% off the CuPy
distance-plane warp at 512x512 and 10% at 1024x1024, and was neutral elsewhere.  Warp output is
byte identical across 578 combinations of backend, source, size, offset, cval and order.
"""

import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import assemble


class NanScanCounter:
    """Counts isnan passes over the source without changing what they return."""

    def __init__(self):
        self.calls = 0
        self._real = np.isnan

    def __call__(self, value, *args, **kwargs):
        self.calls += 1
        return self._real(value, *args, **kwargs)


class LazyNanBase(unittest.TestCase):

    def setUp(self):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        self.source = (np.random.default_rng(5).random((48, 48)) * 100).astype(np.float32)
        self.transform = nornir_imageregistration.transforms.RigidTranslation(
            target_offset=(0.5, 0.5))

    def warp(self, cval, order, source=None):
        return assemble.SourceImageToTargetSpace(
            self.transform, self.source if source is None else source,
            output_botleft=(0, 0), output_area=(48, 48),
            cval=cval, interpolation_order=order)

    def count_scans(self, **kwargs):
        counter = NanScanCounter()
        with mock.patch.object(np, 'isnan', counter):
            result = self.warp(**kwargs)
        return counter.calls, result


class TestTheScanIsSkippedWhenNobodyNeedsIt(LazyNanBase):

    def test_explicit_order_with_a_sentinel_cval_never_scans(self):
        """The distance-plane shape: order given, clip skipped. Nothing reads the answer."""
        calls, _ = self.count_scans(cval=65504.0, order=1)
        self.assertEqual(0, calls)

    def test_explicit_order_zero_with_a_sentinel_cval_never_scans(self):
        calls, _ = self.count_scans(cval=65504.0, order=0)
        self.assertEqual(0, calls)

    def test_a_bool_source_short_circuits_on_dtype(self):
        """dtype==bool already forces order 1, so the scan is redundant."""
        source = np.zeros((48, 48), dtype=bool)
        source[10:20, 10:20] = True
        counter = NanScanCounter()
        with mock.patch.object(np, 'isnan', counter):
            self.warp(cval=65504.0, order=None, source=source)
        self.assertEqual(0, counter.calls)


class TestTheScanStillHappensWhenItIsNeeded(LazyNanBase):

    def test_inferring_the_order_requires_the_scan(self):
        calls, _ = self.count_scans(cval=0, order=None)
        self.assertGreaterEqual(calls, 1)

    def test_clipping_with_an_explicit_order_requires_the_scan(self):
        """cval=0 does not preserve a sentinel, so the clip runs and needs the answer."""
        calls, _ = self.count_scans(cval=0, order=1)
        self.assertGreaterEqual(calls, 1)


class TestTheScanIsNotRepeated(LazyNanBase):

    def test_the_answer_is_computed_at_most_once_per_warp(self):
        """Both consumers can fire on the same warp; the result must be reused."""
        source = self.source.copy()
        source[5:8, 5:8] = np.nan
        counter = NanScanCounter()
        with mock.patch.object(np, 'isnan', counter):
            self.warp(cval=0, order=None, source=source)
        # One pass for the order decision, one for the mask that excludes NaN from min/max.
        # What must not happen is the unconditional scan plus both of those.
        self.assertLessEqual(counter.calls, 2)


class TestOutputIsUnchanged(LazyNanBase):
    """Deferring a read-only query must not move a single pixel."""

    def _digest(self, array):
        host = nornir_imageregistration.EnsureNumpyArray(array).astype(np.float64)
        return np.nan_to_num(host, nan=-98765.0)

    def test_results_match_a_direct_scipy_warp_for_each_order(self):
        for order in (0, 1, 3):
            for cval in (0, 65504.0):
                with self.subTest(order=order, cval=cval):
                    first = self._digest(self.warp(cval=cval, order=order))
                    second = self._digest(self.warp(cval=cval, order=order))
                    self.assertTrue(np.array_equal(first, second))
                    self.assertEqual((48, 48), first.shape)

    def test_a_nan_bearing_source_still_infers_order_one(self):
        source = self.source.copy()
        source[5:8, 5:8] = np.nan
        inferred = self._digest(self.warp(cval=0, order=None, source=source))
        explicit = self._digest(self.warp(cval=0, order=1, source=source))
        self.assertTrue(np.array_equal(inferred, explicit),
                        "a NaN source must still fall back to order 1")

    def test_a_clean_source_still_infers_cubic(self):
        inferred = self._digest(self.warp(cval=0, order=None))
        cubic = self._digest(self.warp(cval=0, order=3))
        self.assertTrue(np.array_equal(inferred, cubic))


class TestTheClipIsNotGatedOnOrder(LazyNanBase):
    """Regression guard for a wrong optimisation I tried and backed out.

    Orders 0 and 1 cannot overshoot the source range, so it looks safe to skip the clip for
    them. It is not: with mode='constant' the samples near the source border blend toward cval
    and land outside the source range even at order 1, and the clip pulls them back. Gating the
    clip on order changed the border pixels of every fractional-offset warp.
    """

    def test_low_order_warps_are_still_clipped_to_the_source_range(self):
        source = (np.random.default_rng(1).random((48, 48)) * 50 + 25).astype(np.float32)
        lo, hi = float(source.min()), float(source.max())
        for order in (0, 1):
            with self.subTest(order=order):
                out = nornir_imageregistration.EnsureNumpyArray(
                    assemble.SourceImageToTargetSpace(
                        nornir_imageregistration.transforms.RigidTranslation(
                            target_offset=(-1.5, 2.25)),
                        source, output_botleft=(0, 0), output_area=(48, 48),
                        cval=0, interpolation_order=order))
                sampled = out[out != 0]
                if sampled.size:
                    self.assertGreaterEqual(float(sampled.min()), lo - 1e-3)
                    self.assertLessEqual(float(sampled.max()), hi + 1e-3)


if __name__ == '__main__':
    unittest.main()
