"""The CPU/GPU assemble comparison must actually assert, and must not cry wolf (review #237).

`CompareMosaicAsssembleAndTransformTile` and its `_GPU` sibling each computed a delta between
two assemble paths, showed a figure captioned "Unexpected high delta" whenever the *sum* of
absolute differences reached 0.65, and then left the real comparison commented out. So the
only surviving assertion was `ShowGrayscale(..., PassFail=True)`, which under NORNIR_HEADLESS
writes a PNG and returns success -- nothing compared the two paths numerically, while every
passing run produced an artifact claiming something was wrong.

After #241 (Parallel composites in submission order), IDoc 004 at 512x1024 measures:

                                  serial vs parallel      CPU vs GPU
    fraction of pixels differing  0                       ~7.2e-05 (38 px)
    mean |delta|                  0                       ~2.2e-08
    max |delta|                   0                       4.88e-04  (half float16 eps)
    mask pixels differing         0                       0

The pre-#241 seam profile (33 px at max 0.142) must now fail the assert. These tests
exercise `AssertAssembledImagesAgree` on synthetic arrays so they run in milliseconds.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from test_assemble_tiles import (
    _ASSEMBLE_MAX_ABS_DELTA,
    _ASSEMBLE_MAX_DIFFERING_FRACTION,
    _ASSEMBLE_MAX_MEAN_ABS_DELTA,
    AssertAssembledImagesAgree,
)

_SHOW = 'nornir_imageregistration.ShowGrayscale'
_SHAPE = (512, 1024)


def _seam_pair(differing: int = 33, largest: float = 0.14209):
    """Pre-#241 serial-vs-parallel seam profile (must fail tightened bounds)."""
    first = np.full(_SHAPE, 0.5, dtype=np.float16)
    second = first.copy()
    for i in range(differing):
        y = 406 + (i % 100)
        x = 1022 - (i * 7) % 240
        second[y, x] = np.float16(0.5 + largest if i == 0 else 0.5 + 0.042)
    delta = np.abs(second.astype(np.float64) - first.astype(np.float64))
    mask = np.ones(_SHAPE, dtype=bool)
    return delta, first, second, mask, mask.copy()


def _rounding_pair(differing: int = 38, quantum: float = 0.00048828125):
    """Post-#241 CPU vs GPU float16 rounding profile (must still pass)."""
    rng = np.random.default_rng(7)
    first = np.full(_SHAPE, 0.5, dtype=np.float16)
    second = first.copy()
    idx = rng.choice(first.size, size=differing, replace=False)
    flat = second.reshape(-1)
    flat[idx] = np.float16(0.5 + quantum)
    delta = np.abs(second.astype(np.float64) - first.astype(np.float64))
    mask = np.ones(_SHAPE, dtype=bool)
    return delta, first, second, mask, mask.copy()


class TestAgreementPasses(unittest.TestCase):

    def test_identical_images_pass(self):
        image = np.full(_SHAPE, 0.25, dtype=np.float16)
        mask = np.ones(_SHAPE, dtype=bool)
        delta = np.zeros(_SHAPE, dtype=np.float64)
        AssertAssembledImagesAgree(self, delta, image, image.copy(), mask, mask.copy(),
                                   label='identical', diagnostic_title='t')

    def test_post_241_cpu_gpu_rounding_profile_passes(self):
        delta, a, b, ma, mb = _rounding_pair()
        AssertAssembledImagesAgree(self, delta, a, b, ma, mb,
                                   label='rounding', diagnostic_title='t')

    def test_the_old_seam_sum_still_exceeds_065(self):
        """Pins the #237 false positive: the old seam sum exceeds 0.65."""
        delta, _, _, _, _ = _seam_pair()
        self.assertGreater(delta.sum(), 0.65)

    def test_single_quantum_float16_noise_everywhere_passes(self):
        """CPU vs GPU rounding: median difference was half an eps across scattered pixels."""
        delta, a, b, ma, mb = _rounding_pair(differing=60)
        AssertAssembledImagesAgree(self, delta, a, b, ma, mb,
                                   label='rounding', diagnostic_title='t')


class TestRealDivergenceNowFails(unittest.TestCase):
    """Each bound must be individually load-bearing, or it is decoration."""

    def setUp(self):
        # These cases fail on purpose, and a real ShowGrayscale would write a PNG into the
        # headless plot-artifact folder for each one -- five false positives for whoever
        # triages it, which is the complaint this issue is about.
        patcher = mock.patch(_SHOW, return_value=True)
        self.show = patcher.start()
        self.addCleanup(patcher.stop)

    def test_pre_241_seam_delta_now_fails(self):
        """#241 tightened max |delta|; the old 0.142 seam profile must not pass."""
        delta, a, b, ma, mb = _seam_pair()
        with self.assertRaises(AssertionError) as caught:
            AssertAssembledImagesAgree(self, delta, a, b, ma, mb,
                                       label='old-seam', diagnostic_title='t')
        self.assertIn('largest difference', str(caught.exception))

    def test_a_broad_small_shift_fails(self):
        """The case the old code could not catch: a slightly different picture everywhere."""
        first = np.full(_SHAPE, 0.5, dtype=np.float16)
        second = np.full(_SHAPE, 0.51, dtype=np.float16)
        delta = np.abs(second.astype(np.float64) - first.astype(np.float64))
        mask = np.ones(_SHAPE, dtype=bool)
        with self.assertRaises(AssertionError) as caught:
            AssertAssembledImagesAgree(self, delta, first, second, mask, mask.copy(),
                                       label='shifted', diagnostic_title='t')
        self.assertIn('pixels differ', str(caught.exception))

    def test_too_many_differing_pixels_fails_even_when_each_is_tiny(self):
        first = np.full(_SHAPE, 0.5, dtype=np.float16)
        second = first.copy()
        n_bad = int(_ASSEMBLE_MAX_DIFFERING_FRACTION * first.size) + 500
        flat = second.reshape(-1)
        flat[:n_bad] = np.float16(0.5 + 0.00048828125)
        delta = np.abs(second.astype(np.float64) - first.astype(np.float64))
        mask = np.ones(_SHAPE, dtype=bool)
        with self.assertRaises(AssertionError):
            AssertAssembledImagesAgree(self, delta, first, second, mask, mask.copy(),
                                       label='many', diagnostic_title='t')

    def test_one_pixel_far_out_of_range_fails(self):
        delta, a, b, ma, mb = _seam_pair(differing=1,
                                         largest=_ASSEMBLE_MAX_ABS_DELTA + 0.2)
        with self.assertRaises(AssertionError) as caught:
            AssertAssembledImagesAgree(self, delta, a, b, ma, mb,
                                       label='outlier', diagnostic_title='t')
        self.assertIn('largest difference', str(caught.exception))

    def test_a_mask_mismatch_fails_even_with_identical_pixels(self):
        """Previously not compared at all: right pixels, wrong coverage would have passed."""
        image = np.full(_SHAPE, 0.25, dtype=np.float16)
        delta = np.zeros(_SHAPE, dtype=np.float64)
        first_mask = np.ones(_SHAPE, dtype=bool)
        second_mask = first_mask.copy()
        second_mask[0, 0] = False
        with self.assertRaises(AssertionError) as caught:
            AssertAssembledImagesAgree(self, delta, image, image.copy(),
                                       first_mask, second_mask,
                                       label='mask', diagnostic_title='t')
        self.assertIn('mask pixel(s) differ', str(caught.exception))

    def test_the_failure_message_carries_the_numbers(self):
        first = np.full(_SHAPE, 0.5, dtype=np.float16)
        second = np.full(_SHAPE, 0.6, dtype=np.float16)
        delta = np.abs(second.astype(np.float64) - first.astype(np.float64))
        mask = np.ones(_SHAPE, dtype=bool)
        with self.assertRaises(AssertionError) as caught:
            AssertAssembledImagesAgree(self, delta, first, second, mask, mask.copy(),
                                       label='numbers', diagnostic_title='t')
        message = str(caught.exception)
        for token in ('differing=', 'mean=', 'max=', 'mask diff='):
            self.assertIn(token, message)


class TestTheDiagnosticOnlyAppearsWhenSomethingIsWrong(unittest.TestCase):
    """The old branch fired on every run, so its artifact carried no information."""

    def test_an_agreeing_pair_produces_no_figure(self):
        delta, a, b, ma, mb = _rounding_pair()
        with mock.patch(_SHOW, return_value=True) as show:
            AssertAssembledImagesAgree(self, delta, a, b, ma, mb,
                                       label='quiet', diagnostic_title='t')
        show.assert_not_called()

    def test_a_disagreeing_pair_produces_exactly_one_figure(self):
        first = np.full(_SHAPE, 0.5, dtype=np.float16)
        second = np.full(_SHAPE, 0.7, dtype=np.float16)
        delta = np.abs(second.astype(np.float64) - first.astype(np.float64))
        mask = np.ones(_SHAPE, dtype=bool)
        with mock.patch(_SHOW, return_value=True) as show:
            with self.assertRaises(AssertionError):
                AssertAssembledImagesAgree(self, delta, first, second, mask, mask.copy(),
                                           label='loud', diagnostic_title='the title')
        self.assertEqual(1, show.call_count)
        self.assertEqual('the title', show.call_args.kwargs['title'])

    def test_the_assertion_fires_even_if_the_figure_reports_success(self):
        """ShowGrayscale returning True was the whole of the old check."""
        first = np.full(_SHAPE, 0.5, dtype=np.float16)
        second = np.full(_SHAPE, 0.7, dtype=np.float16)
        delta = np.abs(second.astype(np.float64) - first.astype(np.float64))
        mask = np.ones(_SHAPE, dtype=bool)
        with mock.patch(_SHOW, return_value=True):
            with self.assertRaises(AssertionError):
                AssertAssembledImagesAgree(self, delta, first, second, mask, mask.copy(),
                                           label='loud', diagnostic_title='t')


class TestTheTolerancesKeepTheirMeasuredMargins(unittest.TestCase):
    """If someone nudges a constant, say what measurement it has to be re-derived from."""

    def test_the_fraction_bound_keeps_at_least_5x_margin(self):
        self.assertGreaterEqual(_ASSEMBLE_MAX_DIFFERING_FRACTION, 7.2e-05 * 5)

    def test_the_mean_bound_keeps_at_least_10x_margin(self):
        self.assertGreaterEqual(_ASSEMBLE_MAX_MEAN_ABS_DELTA, 2.2e-08 * 10)

    def test_the_max_bound_clears_measured_float16_rounding(self):
        self.assertGreater(_ASSEMBLE_MAX_ABS_DELTA, 4.88e-04)

    def test_the_max_bound_rejects_the_old_seam(self):
        self.assertLess(_ASSEMBLE_MAX_ABS_DELTA, 0.14209)

    def test_the_max_bound_is_still_a_fraction_of_full_range(self):
        """Images are on [0, 1]; a bound near 1 would assert nothing."""
        self.assertLessEqual(_ASSEMBLE_MAX_ABS_DELTA, 0.5)


if __name__ == '__main__':
    unittest.main()
