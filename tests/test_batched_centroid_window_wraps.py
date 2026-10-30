"""The centroid refinement window wraps at the edges instead of sliding inward.

``batched_find_peak`` refines the argmax with an intensity-weighted centre of mass over
a ``(2r+1)`` window. That window used to be kept in bounds by clamping its *centre*::

    cr = xp.clip(peak_r, r, h - 1 - r)

which does keep the samples valid, but slides the window off the peak whenever the peak
is within ``r`` of an edge. The peak then sits on the window rim and the centroid is
dragged toward the interior.

The correlation being refined is circular -- ``batched_image_phase_correlation`` is
``ifft2`` of a conjugate product, then ``fftshift`` -- so a peak on the first row has
its lobe continuing on the last. Wrapping the window is therefore not a workaround but
the correct neighbourhood. Measured against known sub-pixel shifts of a periodic lobe:

===========================  =========  =========
metric                       clamped    wrapped
===========================  =========  =========
border mean \\|error\\| (px)     0.3235     0.0303
border max \\|error\\| (px)      0.7247     0.0600
interior mean \\|error\\| (px)   0.0302     0.0302
sweep positions over 0.1px     15/512      0/512
r=3 max \\|error\\| (px)         1.1419     0.0361
===========================  =========  =========

Two details worth keeping in mind, both learned the hard way:

1. Truncating the window instead -- keeping it centred and dropping out-of-bounds
   samples -- only halves the error (max 0.5769px), because the baseline subtraction
   has fewer samples to work with and more often flattens to nothing.
2. The centroid must **not** be re-wrapped into ``[0, dim)`` afterwards. Doing so
   reports a peak just before row 0 as ``h - 0.14`` rather than ``-0.14``. Both name
   the same circular position, but the wrapped one flips the sign of the reported
   offset at the seam, turning a ``+16.0`` shift into ``-15.86`` on a 32px cell and
   breaking agreement with the serial path.

Interior peaks are unaffected: weighting relative offsets and adding the peak back is
algebraically what weighting absolute indices did when no wrapping occurred.

See review issue #89.
"""

from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.batched_phase_correlation import batched_find_peak

_N = 64
_CENTRE = _N / 2.0


def _circular_lobe(shift_row: float, shift_col: float, sigma: float = 0.9,
                   n: int = _N) -> np.ndarray:
    """A periodic Gaussian lobe, as a circular correlation actually produces."""
    rows = np.arange(n)
    cols = np.arange(n)
    d_row = np.minimum(np.abs(rows - shift_row), n - np.abs(rows - shift_row))
    d_col = np.minimum(np.abs(cols - shift_col), n - np.abs(cols - shift_col))
    with np.errstate(under='ignore'):
        return np.exp(-(d_row[:, None] ** 2 + d_col[None, :] ** 2)
                      / (2.0 * sigma ** 2)).astype(np.float64)


def _circular_error(got: float, want: float, n: int = _N) -> float:
    """Signed error on a ring, so the seam does not register as a huge miss."""
    delta = (got - want) % n
    return delta - n if delta > n / 2 else delta


def _centroid_errors(shifts, centroid_radius: int = 1, sigma: float = 0.9):
    stack = np.stack([_circular_lobe(s, 32.0, sigma) for s in shifts])
    peaks, _weights, _ratios = batched_find_peak(
        stack, centroid_radius=centroid_radius)
    return np.array([abs(_circular_error(_CENTRE - float(peak[0]), shift))
                     for shift, peak in zip(shifts, peaks)])


# Rows within r=1 of an edge, where clamping used to slide the window.
_BORDER_SHIFTS = [0.0, 0.25, 0.5, 0.75, 63.0, 63.25, 63.5, 63.75]
_INTERIOR_SHIFTS = [20.0, 24.25, 31.5, 32.0, 32.5, 40.75]


class TestBorderPeaksAreAsAccurateAsInteriorOnes(unittest.TestCase):
    """The fix, stated as the property that matters."""

    def test_border_error_matches_interior_error(self):
        border = _centroid_errors(_BORDER_SHIFTS).mean()
        interior = _centroid_errors(_INTERIOR_SHIFTS).mean()

        self.assertLess(
            border, interior * 1.5,
            f'border mean error {border:.4f} should be comparable to interior '
            f'{interior:.4f}; clamping made it 0.3235 against 0.0302')

    def test_no_sub_pixel_position_is_off_by_more_than_a_tenth_of_a_pixel(self):
        errors = _centroid_errors(np.arange(0.0, 64.0, 0.125))

        over = int((errors > 0.1).sum())
        self.assertEqual(
            over, 0,
            f'{over} of {errors.size} positions exceed 0.1px (clamping left 15), '
            f'worst {errors.max():.4f}')

    def test_a_peak_on_the_first_row_is_not_pushed_inward(self):
        """The single clearest case: the lobe is exactly on row 0."""
        peaks, _weights, _ratios = batched_find_peak(
            _circular_lobe(0.0, 32.0)[None, :, :], centroid_radius=1)

        centroid_row = _CENTRE - float(peaks[0][0])

        self.assertAlmostEqual(
            _circular_error(centroid_row, 0.0), 0.0, delta=0.01,
            msg='clamping reported this 0.36px inward')

    def test_a_corner_peak_is_handled_on_both_axes(self):
        peaks, _weights, _ratios = batched_find_peak(
            _circular_lobe(0.0, 0.0)[None, :, :], centroid_radius=1)

        row = _circular_error(_CENTRE - float(peaks[0][0]), 0.0)
        col = _circular_error(_CENTRE - float(peaks[0][1]), 0.0)

        self.assertAlmostEqual(row, 0.0, delta=0.01)
        self.assertAlmostEqual(col, 0.0, delta=0.01)

    def test_accuracy_improves_with_radius_rather_than_degrading(self):
        """Clamping got *worse* with radius, since a wider window slides further."""
        shifts = np.arange(0.0, 64.0, 0.25)
        worst = [_centroid_errors(shifts, centroid_radius=r, sigma=1.4).max()
                 for r in (1, 2, 3)]

        for radius, value in zip((1, 2, 3), worst):
            with self.subTest(radius=radius):
                self.assertLess(value, 0.2,
                                f'r={radius} worst error {value:.4f}; clamping gave '
                                f'0.86, 1.13 and 1.14 for r=1,2,3')


class TestInteriorPeaksAreUnchanged(unittest.TestCase):
    """The change must be surgical, since this feeds every refine measurement."""

    def test_interior_accuracy_is_the_documented_baseline(self):
        errors = _centroid_errors(_INTERIOR_SHIFTS)

        self.assertAlmostEqual(errors.mean(), 0.0302, delta=0.005)
        self.assertAlmostEqual(errors.max(), 0.0600, delta=0.005)

    def test_an_exactly_centred_peak_reports_zero_offset(self):
        peaks, _weights, _ratios = batched_find_peak(
            _circular_lobe(32.0, 32.0)[None, :, :], centroid_radius=1)

        np.testing.assert_allclose(np.asarray(peaks[0], dtype=float),
                                   [0.0, 0.0], atol=1e-9)


class TestTheCentroidIsNotRewrapped(unittest.TestCase):
    """Guards the sign convention the serial path and callers depend on."""

    def test_a_lobe_just_before_row_zero_reports_a_positive_offset(self):
        """Re-wrapping turned +16.0 into -15.86 on a 32px cell."""
        # Lobe centre just below row 0, so the argmax lands on row 0.
        stack = _circular_lobe(_N - 0.25, 32.0)[None, :, :]

        peaks, _weights, _ratios = batched_find_peak(stack, centroid_radius=1)
        offset_row = float(peaks[0][0])

        self.assertGreater(
            offset_row, _CENTRE - 1.0,
            'the offset should stay just under +N/2, not flip to about -N/2')

    def test_the_offset_is_continuous_across_the_seam(self):
        """No jump of a full dimension as the true peak crosses row 0."""
        shifts = [63.5, 63.75, 63.9, 0.0, 0.1, 0.25, 0.5]
        stack = np.stack([_circular_lobe(s, 32.0) for s in shifts])

        peaks, _weights, _ratios = batched_find_peak(stack, centroid_radius=1)
        offsets = [float(peak[0]) for peak in peaks]

        for earlier, later in zip(offsets, offsets[1:]):
            self.assertLess(
                abs(later - earlier), _N / 2.0,
                f'offset jumped by a whole dimension: {offsets}')


class TestExistingBehaviourIsPreserved(unittest.TestCase):
    """Paths the rewrite touched but must not have altered."""

    def test_a_flat_image_falls_back_without_error(self):
        stack = np.zeros((2, _N, _N), dtype=np.float64)

        peaks, weights, ratios = batched_find_peak(stack, centroid_radius=1)

        self.assertEqual(peaks.shape, (2, 2))
        np.testing.assert_array_equal(np.asarray(weights, dtype=float), [0.0, 0.0])
        np.testing.assert_array_equal(np.asarray(ratios, dtype=float), [0.0, 0.0])

    def test_an_overlap_mask_still_confines_the_argmax(self):
        """A strong peak outside the mask must be ignored in favour of one inside."""
        surface = _circular_lobe(2.0, 2.0) + 0.4 * _circular_lobe(32.0, 32.0)
        mask = np.zeros((_N, _N), dtype=bool)
        mask[24:40, 24:40] = True

        peaks, _weights, _ratios = batched_find_peak(
            surface[None, :, :], overlap_mask=mask, centroid_radius=1)

        self.assertAlmostEqual(float(peaks[0][0]), 0.0, delta=0.5)
        self.assertAlmostEqual(float(peaks[0][1]), 0.0, delta=0.5)

    def test_the_batch_is_processed_independently(self):
        shifts = [0.0, 32.0, 63.5, 20.25]
        stack = np.stack([_circular_lobe(s, 32.0) for s in shifts])

        together, _w, _r = batched_find_peak(stack, centroid_radius=1)
        apart = [batched_find_peak(_circular_lobe(s, 32.0)[None, :, :],
                                   centroid_radius=1)[0][0] for s in shifts]

        for index, (batched, single) in enumerate(zip(together, apart)):
            with self.subTest(index=index):
                np.testing.assert_allclose(np.asarray(batched, dtype=float),
                                           np.asarray(single, dtype=float), atol=1e-9)

    def test_a_non_stack_input_is_rejected(self):
        with self.assertRaises(ValueError):
            batched_find_peak(np.zeros((_N, _N)))


if __name__ == '__main__':
    unittest.main()
