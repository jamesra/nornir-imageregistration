"""Smooth FFT frame sizing -- review #234.

The brute sweep used to round its correlation frame up to a power of two, which turned a
6000px requirement into 8192 and paid 1.86x the area for nothing. It now rounds up to the
nearest **even** 5-smooth size instead.

Two properties carry the whole change and both are pinned here:

* the result is never larger than the power of two, so frame memory cannot regress;
* the result is always **even**, which is a correctness constraint rather than a preference --
  ``find_peak`` uses true-half centring while ``fftshift`` puts the zero-shift sample at
  ``(n-1)/2`` for odd n, so an odd frame biases every measured offset by exactly +0.5px.
"""

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration.core import _core
from nornir_imageregistration.stos_brute import (
    _fixed_correlation_shape,
    _rotated_aabb_shape,
)


def _is_five_smooth(n: int) -> bool:
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


SIZES = [3, 4, 17, 100, 513, 1000, 1024, 1025, 2049, 4097, 6000, 6006, 6145, 8000, 8193]


class TestTheSizeRuleItself:
    @pytest.mark.parametrize('val', SIZES)
    def test_the_result_is_at_least_the_request(self, val):
        assert nornir_imageregistration.NextSmoothFFTSize(val) >= val

    @pytest.mark.parametrize('val', SIZES)
    def test_the_result_is_even(self, val):
        """Odd frames carry a half-pixel offset bias; see TestWhyEven."""
        assert nornir_imageregistration.NextSmoothFFTSize(val) % 2 == 0

    @pytest.mark.parametrize('val', SIZES)
    def test_the_result_is_five_smooth(self, val):
        assert _is_five_smooth(nornir_imageregistration.NextSmoothFFTSize(val))

    @pytest.mark.parametrize('val', SIZES)
    def test_it_never_exceeds_the_power_of_two(self, val):
        """Powers of two are themselves even and 5-smooth, so this is a guarantee, not luck.

        It is what makes the change safe for memory: the frame can only shrink.
        """
        po2 = int(nornir_imageregistration.NearestPowerOfTwo(val))
        assert nornir_imageregistration.NextSmoothFFTSize(val) <= po2

    @pytest.mark.parametrize('val', SIZES)
    def test_it_is_the_smallest_such_size(self, val):
        got = nornir_imageregistration.NextSmoothFFTSize(val)
        for candidate in range(int(val) if int(val) % 2 == 0 else int(val) + 1, got, 2):
            assert not _is_five_smooth(candidate), (
                f'{candidate} is an even 5-smooth size below the returned {got}')

    def test_the_measured_case_from_the_issue(self):
        """The ds32 pair needs 6000, which is itself even and 5-smooth (2**4 * 3 * 5**3)."""
        assert nornir_imageregistration.NextSmoothFFTSize(6000) == 6000
        assert int(nornir_imageregistration.NearestPowerOfTwo(6000)) == 8192

    def test_a_request_just_above_skips_the_odd_candidate(self):
        """6075 (3**5 * 5**2) is the smallest 5-smooth size >= 6006 but is odd, so 6144 wins."""
        assert _is_five_smooth(6075) and 6075 % 2 == 1
        assert nornir_imageregistration.NextSmoothFFTSize(6006) == 6144

    def test_it_is_monotonic(self):
        previous = 0
        for val in range(2, 4000):
            got = nornir_imageregistration.NextSmoothFFTSize(val)
            assert got >= previous
            previous = got

    def test_tiny_requests_do_not_return_one(self):
        """A frame of 1 would be odd; the floor is 2."""
        for val in (0, 1, 2):
            assert nornir_imageregistration.NextSmoothFFTSize(val) == 2


class TestTheOverlapWrapper:
    @pytest.mark.parametrize('overlap', [0.0, 0.05, 0.25, 0.5, 0.75, 1.0])
    @pytest.mark.parametrize('val', [1000, 4183, 6000])
    def test_it_rounds_the_overlap_adjusted_dimension(self, val, overlap):
        expected = nornir_imageregistration.NextSmoothFFTSize(
            _core.DimensionWithOverlap(val, overlap))
        assert nornir_imageregistration.SmoothFFTSizeWithOverlap(val, overlap) == expected

    @pytest.mark.parametrize('overlap', [None, -1.0, 2.0])
    def test_out_of_range_overlap_is_clamped_like_the_power_of_two_helper(self, overlap):
        """Both helpers share `_clamped_overlap`, so they cannot drift apart."""
        smooth = nornir_imageregistration.SmoothFFTSizeWithOverlap(4183, overlap)
        po2 = nornir_imageregistration.NearestPowerOfTwoWithOverlap(4183, overlap)
        assert smooth <= po2
        assert smooth % 2 == 0

    @pytest.mark.parametrize('val', [512, 1000, 4183, 6000, 7000])
    @pytest.mark.parametrize('overlap', [0.0, 0.5, 1.0])
    def test_it_never_exceeds_the_power_of_two_wrapper(self, val, overlap):
        assert (nornir_imageregistration.SmoothFFTSizeWithOverlap(val, overlap)
                <= nornir_imageregistration.NearestPowerOfTwoWithOverlap(val, overlap))


class TestWhyEven:
    """Pins the reason the size rule excludes odd candidates.

    ``find_peak`` computes ``shape / 2.0 - peak_center_of_mass``. ``fftshift`` puts the
    zero-shift sample at ``n/2`` for even n but ``(n-1)/2`` for odd n, so the true-half
    centring is off by exactly half a sample on an odd axis. Power-of-two frames are always
    even, which is why this has never surfaced in production.
    """

    @staticmethod
    def _measure(n: int, shift: tuple[int, int]) -> tuple[float, float]:
        rng = np.random.default_rng(1234)
        base = rng.random((n, n)).astype(np.float32)
        moved = np.roll(np.roll(base, shift[0], axis=0), shift[1], axis=1)
        cross = np.fft.fft2(base) * np.conj(np.fft.fft2(moved))
        mag = np.abs(cross)
        mag[mag == 0] = 1.0
        corr = np.fft.fftshift(np.real(np.fft.ifft2(cross / mag)))
        record = nornir_imageregistration.phasecorrelation.find_peak(corr)
        return tuple(float(v) for v in record.scaled_offset)

    @pytest.mark.parametrize('n', [64, 128, 256])
    @pytest.mark.parametrize('shift', [(0, 0), (7, -5)])
    def test_an_even_frame_recovers_the_shift_exactly(self, n, shift):
        got = self._measure(n, shift)
        assert got == pytest.approx(shift, abs=1e-6)

    @pytest.mark.parametrize('n', [65, 129, 255])
    @pytest.mark.parametrize('shift', [(0, 0), (7, -5)])
    def test_an_odd_frame_is_biased_by_exactly_half_a_pixel(self, n, shift):
        got = self._measure(n, shift)
        assert got == pytest.approx((shift[0] + 0.5, shift[1] + 0.5), abs=1e-6)

    def test_so_the_size_rule_never_produces_an_odd_frame(self):
        for val in range(2, 3000):
            assert nornir_imageregistration.NextSmoothFFTSize(val) % 2 == 0


class TestTheSweepFrame:
    """`_fixed_correlation_shape` must still cover every rotated source AABB."""

    ANGLES = [float(a) for a in range(0, 360, 2)]

    @pytest.mark.parametrize('shape', [(96, 96), (512, 400), (4183, 4309)])
    def test_the_frame_covers_every_rotated_aabb(self, shape):
        frame = _fixed_correlation_shape(shape, shape, self.ANGLES, 0.5)
        for angle in self.ANGLES:
            rh, rw = _rotated_aabb_shape(shape[0], shape[1], angle)
            assert rh <= frame[0]
            assert rw <= frame[1]

    @pytest.mark.parametrize('shape', [(96, 96), (512, 400), (4183, 4309)])
    def test_the_frame_is_even_and_smooth(self, shape):
        frame = _fixed_correlation_shape(shape, shape, self.ANGLES, 0.5)
        for dim in frame:
            assert dim % 2 == 0
            assert _is_five_smooth(dim)

    @pytest.mark.parametrize('shape', [(96, 96), (512, 400), (4183, 4309)])
    def test_the_frame_never_grew(self, shape):
        """Against the power-of-two rule this replaces, computed the same way."""
        max_h, max_w = shape
        for angle in self.ANGLES:
            rh, rw = _rotated_aabb_shape(shape[0], shape[1], angle)
            max_h, max_w = max(max_h, rh), max(max_w, rw)
        old = (int(nornir_imageregistration.NearestPowerOfTwoWithOverlap(max_h, 0.5)),
               int(nornir_imageregistration.NearestPowerOfTwoWithOverlap(max_w, 0.5)))
        frame = _fixed_correlation_shape(shape, shape, self.ANGLES, 0.5)
        assert frame[0] <= old[0]
        assert frame[1] <= old[1]

    def test_the_real_pair_shrinks_as_measured(self):
        """4183x4309 vs 4184x4299: 8192 -> 6000, the 1.86x area saving from the issue."""
        frame = _fixed_correlation_shape((4183, 4309), (4184, 4299), self.ANGLES, 0.5)
        assert frame == (6000, 6000)
        assert (8192 * 8192) / (frame[0] * frame[1]) == pytest.approx(1.864, abs=0.01)

    def test_a_narrow_angle_range_still_gives_a_smaller_frame(self):
        """Kept from the pre-existing contract in test_fixed_angle_fft_reuse."""
        shape = (512, 400)
        narrow = _fixed_correlation_shape(shape, shape, [-2.0, 0.0, 2.0], 0.5)
        full = _fixed_correlation_shape(shape, shape, self.ANGLES, 0.5)
        assert narrow[0] < full[0] or narrow[1] < full[1]
