"""The CPU/GPU assemble comparison must actually assert, and must not cry wolf (review #237).

`CompareMosaicAsssembleAndTransformTile` and its `_GPU` sibling each computed a delta between
two assemble paths, showed a figure captioned "Unexpected high delta" whenever the *sum* of
absolute differences reached 0.65, and then left the real comparison commented out. So the
only surviving assertion was `ShowGrayscale(..., PassFail=True)`, which under NORNIR_HEADLESS
writes a PNG and returns success -- nothing compared the two paths numerically, while every
passing run produced an artifact claiming something was wrong.

Measured on the IDoc 004 fixture at 512x1024, both pairs genuinely agree:

                                  serial vs parallel      CPU vs GPU
    sum of |delta|                1.59363                 1.6051      <- both above 0.65
    fraction of pixels differing  6.3e-05 (33 px)         1.35e-04 (71 px)
    mean |delta|                  3.04e-06                3.06e-06
    max |delta|                   0.14209                 0.14209
    mask pixels differing         0                       0

These tests exercise `AssertAssembledImagesAgree` directly on synthetic arrays, so they run in
milliseconds and do not need the fixture. They cover the three things the old code got wrong:
it did not assert, it fired its diagnostic on agreement, and it ignored the masks entirely.
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


def _agreeing_pair(differing: int = 33, largest: float = 0.14209):
    """Reproduce the measured real-world profile: a few seam pixels, everything else equal.

    The differing pixels are placed on a diagonal because that is where they actually are --
    an interior seam between two overlapping tiles, not the mask boundary.
    """
    first = np.full(_SHAPE, 0.5, dtype=np.float16)
    second = first.copy()
    for i in range(differing):
        y = 406 + (i % 100)
        x = 1022 - (i * 7) % 240
        second[y, x] = np.float16(0.5 + largest if i == 0 else 0.5 + 0.042)
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

    def test_the_measured_real_world_delta_passes(self):
        delta, a, b, ma, mb = _agreeing_pair()
        AssertAssembledImagesAgree(self, delta, a, b, ma, mb,
                                   label='measured', diagnostic_title='t')

    def test_the_measured_delta_would_have_tripped_the_old_sum_bound(self):
        """Pins the false positive: these agreeing pairs exceed 0.65 on the sum."""
        delta, _, _, _, _ = _agreeing_pair()
        self.assertGreater(delta.sum(), 0.65)

    def test_single_quantum_float16_noise_everywhere_passes(self):
        """CPU vs GPU rounding: median difference was half an eps across scattered pixels."""
        rng = np.random.default_rng(7)
        first = np.full(_SHAPE, 0.5, dtype=np.float16)
        second = first.copy()
        idx = rng.choice(first.size, size=60, replace=False)
        flat = second.reshape(-1)
        flat[idx] = np.float16(0.5 + 0.00048828125)
        delta = np.abs(second.astype(np.float64) - first.astype(np.float64))
        mask = np.ones(_SHAPE, dtype=bool)
        AssertAssembledImagesAgree(self, delta, first, second, mask, mask.copy(),
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
        delta, a, b, ma, mb = _agreeing_pair(differing=1,
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
        delta, a, b, ma, mb = _agreeing_pair()
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
        self.assertGreaterEqual(_ASSEMBLE_MAX_DIFFERING_FRACTION, 1.35e-04 * 5)

    def test_the_mean_bound_keeps_at_least_10x_margin(self):
        self.assertGreaterEqual(_ASSEMBLE_MAX_MEAN_ABS_DELTA, 3.06e-06 * 10)

    def test_the_max_bound_clears_the_measured_seam_pixel(self):
        self.assertGreater(_ASSEMBLE_MAX_ABS_DELTA, 0.14209)

    def test_the_max_bound_is_still_a_fraction_of_full_range(self):
        """Images are on [0, 1]; a bound near 1 would assert nothing."""
        self.assertLessEqual(_ASSEMBLE_MAX_ABS_DELTA, 0.5)


if __name__ == '__main__':
    unittest.main()
