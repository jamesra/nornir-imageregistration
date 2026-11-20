"""Masked registration: band-pass on valid pixels, then masked normalized cross-correlation.

Registration used to replace masked pixels (and padding, and rotation corners) with random noise and
then run a partly whitened phase correlation. Near a tissue edge the window is then partly noise: the
true peak weakens, edge and noise peaks can win, and the result changes from run to run. Here the
masked pixels are left out instead:

* :func:`masked_bandpass` is a difference of two normalized convolutions (Knutsson & Westin 1993):
  each low-pass is ``G*(I*M) / G*M``, so the mask edge and the image border do not leak into the
  filtered image. It plays the role the 0.65 spectral whitening had (suppress the low frequencies that
  dominate a plain NCC, and the pixel noise).
* :func:`masked_ncc_surface` is Padfield's masked NCC (IEEE TIP 21(5), 2012): the Pearson correlation
  at every shift, over only the pixels valid in both images, from six forward FFTs.

Both work on a single ``(h, w)`` image or a ``(N, h, w)`` stack, on NumPy or CuPy. Shifts follow the
:class:`AlignmentRecord` convention used by :func:`phasecorrelation.find_offset`: the offset moves the
moving image's centre onto the fixed image's centre. See ``docs/publication_notes.md``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.fft
from numpy.typing import NDArray

import nornir_imageregistration

try:
    import cupy as cp
    import cupyx
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx

from nornir_imageregistration.phasecorrelation import find_peak

# Normalized convolution: where the blurred mask falls below this fraction of its full value the local
# estimate rests on too few valid pixels, and the filtered pixel is treated as masked.
BANDPASS_MIN_SUPPORT: float = 0.1

# A shift's overlap must be at least this fraction of the smaller image's valid pixel count.
DEFAULT_MIN_OVERLAP_FRACTION: float = 0.25

# Variance below this fraction of the image's own variance (per overlap pixel) is flat: no NCC there.
_FLAT_VARIANCE_FRACTION: float = 1e-6


def default_bandpass_sigmas(shape: tuple[int, ...]) -> tuple[float, float]:
    """Low and high band-pass sigmas for a window of *shape*: pixel noise, and 1/16 of the short side."""
    short_side = float(min(shape[-2], shape[-1]))
    return 1.0, max(4.0, short_side / 16.0)


def _ndimage(array):
    """SciPy or CuPyX ndimage, following *array*."""
    return cupyx.scipy.get_array_module(array).ndimage


def _valid_lowpass(image_times_mask, mask_float, sigma: float):
    """Normalized-convolution low-pass: ``G*(I*M) / G*M``, plus the support ``G*M``."""
    nd = _ndimage(image_times_mask)
    sig = (0.0, sigma, sigma) if image_times_mask.ndim == 3 else (sigma, sigma)
    blurred = nd.gaussian_filter(image_times_mask, sigma=sig, mode='constant', cval=0.0)
    support = nd.gaussian_filter(mask_float, sigma=sig, mode='constant', cval=0.0)
    xp = cp.get_array_module(image_times_mask)
    lowpass = xp.where(support > 0, blurred / xp.maximum(support, 1e-12), 0.0)
    return lowpass, support


def masked_bandpass(image: NDArray, mask: NDArray[np.bool_] | None,
                    sigma_lo: float, sigma_hi: float) -> tuple[NDArray, NDArray[np.bool_]]:
    """Band-pass *image* using only its valid pixels.

    :param image: ``(h, w)`` or ``(N, h, w)``.
    :param mask: True where the pixel is valid; None means every finite pixel is valid.
    :param sigma_lo: Gaussian sigma of the noise low-pass (pixels).
    :param sigma_hi: Gaussian sigma of the background that is subtracted (pixels).
    :return: ``(filtered, valid)``: zero and False where the pixel was masked or the high-sigma
        estimate has less than ``BANDPASS_MIN_SUPPORT`` support.
    """
    xp = cp.get_array_module(image)
    image = xp.asarray(image, dtype=xp.float64)
    finite = xp.isfinite(image)
    valid = finite if mask is None else (xp.asarray(mask, dtype=bool) & finite)
    mask_float = valid.astype(xp.float64)
    image_times_mask = xp.where(valid, image, 0.0)

    low, _ = _valid_lowpass(image_times_mask, mask_float, sigma_lo)
    background, support = _valid_lowpass(image_times_mask, mask_float, sigma_hi)
    valid = valid & (support >= BANDPASS_MIN_SUPPORT)
    filtered = xp.where(valid, low - background, 0.0)
    return filtered, valid


def _flat_tolerance(values, valid, xp):
    """Per-image variance scale, broadcastable against a surface, for the flat-overlap test."""
    axes = (-2, -1)
    count = xp.maximum(valid.sum(axis=axes, keepdims=True), 1)
    mean = values.sum(axis=axes, keepdims=True) / count
    var = (xp.where(valid, values - mean, 0.0) ** 2).sum(axis=axes, keepdims=True) / count
    # Floor at tiny / fraction so the product is never subnormal (float32 underflow warnings).
    return xp.maximum(var, float(np.finfo(var.dtype).tiny) / _FLAT_VARIANCE_FRACTION) * _FLAT_VARIANCE_FRACTION


def masked_ncc_surface(fixed: NDArray, fixed_mask: NDArray[np.bool_] | None,
                       moving: NDArray, moving_mask: NDArray[np.bool_] | None,
                       min_overlap_fraction: float = DEFAULT_MIN_OVERLAP_FRACTION,
                       ) -> tuple[NDArray, NDArray]:
    """Masked NCC (Padfield 2012) of *moving* against *fixed* at every integer shift.

    :param fixed: ``(h1, w1)`` or ``(N, h1, w1)``.
    :param moving: ``(h2, w2)`` or ``(N, h2, w2)``.
    :param fixed_mask: True = valid. None = every finite pixel.
    :param min_overlap_fraction: Shifts whose overlap has fewer valid pixels than this fraction of
        the smaller image's valid count are invalid.
    :return: ``(ncc, overlap)``, both of shape ``(..., h1 + h2 - 1, w1 + w2 - 1)`` with the
        ``fftshift`` layout of a phase-correlation surface: index ``(h // 2, w // 2)`` is the moving
        origin on the fixed origin, and :func:`phasecorrelation.find_peak` returns the moving
        origin's position in fixed coordinates. Invalid shifts are NaN in *ncc*.
    """
    xp = cp.get_array_module(fixed)
    fixed = xp.asarray(fixed, dtype=xp.float64)
    moving = xp.asarray(moving, dtype=xp.float64)
    size = _correlation_size(fixed.shape, moving.shape)
    fixed_side = _SideSpectra.create(fixed, fixed_mask, size, xp.float64)
    moving_side = _SideSpectra.create(moving, moving_mask, size, xp.float64)
    return _ncc_from_spectra(fixed_side, moving_side, size, min_overlap_fraction, xp)


def _correlation_size(fixed_shape: tuple[int, ...], moving_shape: tuple[int, ...]) -> tuple[int, int]:
    """FFT size for a linear (not circular) correlation of the two images."""
    return (scipy.fft.next_fast_len(fixed_shape[-2] + moving_shape[-2] - 1, real=True),
            scipy.fft.next_fast_len(fixed_shape[-1] + moving_shape[-1] - 1, real=True))


def _rfft2(a, size: tuple[int, int], xp):
    """Real 2-D FFT over the last two axes, zero-padded to *size*; SciPy on all cores for NumPy."""
    if xp is np:
        return scipy.fft.rfft2(a, s=size, axes=(-2, -1), workers=-1)
    return xp.fft.rfft2(a, s=size, axes=(-2, -1))


def _irfft2(a_hat, size: tuple[int, int], xp):
    """Inverse of :func:`_rfft2`."""
    if xp is np:
        return scipy.fft.irfft2(a_hat, s=size, axes=(-2, -1), workers=-1)
    return xp.fft.irfft2(a_hat, s=size, axes=(-2, -1))


class _SideSpectra(NamedTuple):
    """One image's part of Padfield's masked NCC: spectra of I*M, I^2*M and M, and its gate inputs."""
    shape: tuple[int, ...]
    values_hat: NDArray
    squares_hat: NDArray
    mask_hat: NDArray
    valid_count: NDArray
    flat_tolerance: NDArray

    @staticmethod
    def create(image: NDArray, mask: NDArray[np.bool_] | None, size: tuple[int, int], dtype) -> _SideSpectra:
        """Spectra of *image* (valid where *mask* and finite), zero-padded to *size*, computed in *dtype*."""
        xp = cp.get_array_module(image)
        image = xp.asarray(image, dtype=dtype)
        valid = xp.isfinite(image) if mask is None else (xp.asarray(mask, dtype=bool) & xp.isfinite(image))
        values = xp.where(valid, image, 0.0).astype(dtype, copy=False)
        return _SideSpectra(tuple(image.shape), _rfft2(values, size, xp), _rfft2(values * values, size, xp),
                            _rfft2(valid.astype(dtype), size, xp), valid.sum(axis=(-2, -1), keepdims=True),
                            _flat_tolerance(values, valid, xp))


def _ncc_from_spectra(fixed: _SideSpectra, moving: _SideSpectra, size: tuple[int, int],
                      min_overlap_fraction: float, xp) -> tuple[NDArray, NDArray]:
    """The masked NCC surface and overlap count from both sides' spectra; see :func:`masked_ncc_surface`."""
    def corr(a_hat, b_hat):
        # sum_x a(x) b(x + k): the conjugate sits on the fixed side, as in fft_phase_correlation.
        return _irfft2(xp.conj(a_hat) * b_hat, size, xp)

    overlap = xp.rint(corr(fixed.mask_hat, moving.mask_hat))
    sum_f = corr(fixed.values_hat, moving.mask_hat)
    sum_f2 = corr(fixed.squares_hat, moving.mask_hat)
    sum_g = corr(fixed.mask_hat, moving.values_hat)
    sum_g2 = corr(fixed.mask_hat, moving.squares_hat)
    sum_fg = corr(fixed.values_hat, moving.values_hat)

    safe_n = xp.maximum(overlap, 1.0)
    var_f = sum_f2 - sum_f * sum_f / safe_n
    var_g = sum_g2 - sum_g * sum_g / safe_n
    cov = sum_fg - sum_f * sum_g / safe_n

    min_valid = xp.minimum(fixed.valid_count, moving.valid_count)
    min_overlap = xp.maximum(min_overlap_fraction * min_valid, 2.0)
    ok = ((overlap >= min_overlap)
          & (var_f > fixed.flat_tolerance * overlap)
          & (var_g > moving.flat_tolerance * overlap))
    tiny = float(np.finfo(var_f.dtype).tiny)  # 1e-300 is 0 in float32
    ncc = xp.where(ok, cov / xp.sqrt(xp.maximum(var_f * var_g, tiny)), xp.nan)
    ncc = xp.clip(ncc, -1.0, 1.0)

    # Keep only shifts with any overlap, k in [-(h1-1), h2-1] per axis, laid out so that index
    # (h // 2, w // 2) is k = 0. find_peak reports centre - index, which is -k: the moving origin's
    # position in fixed coordinates.
    ncc = _centre_crop(ncc, fixed.shape, moving.shape, size, xp)
    overlap = _centre_crop(overlap, fixed.shape, moving.shape, size, xp)
    return ncc, overlap


def _centre_crop(surface, fixed_shape, moving_shape, size, xp):
    """Gather the linear-correlation shifts into an fftshift-style array of shape h1+h2-1, w1+w2-1."""
    out_shape = (fixed_shape[-2] + moving_shape[-2] - 1, fixed_shape[-1] + moving_shape[-1] - 1)
    rows = (xp.arange(out_shape[0]) - out_shape[0] // 2) % size[0]
    cols = (xp.arange(out_shape[1]) - out_shape[1] // 2) % size[1]
    k_lo = (-(fixed_shape[-2] - 1), -(fixed_shape[-1] - 1))
    k_hi = (moving_shape[-2] - 1, moving_shape[-1] - 1)
    row_k = xp.arange(out_shape[0]) - out_shape[0] // 2
    col_k = xp.arange(out_shape[1]) - out_shape[1] // 2
    gathered = surface[..., rows[:, None], cols[None, :]]
    in_range = (((row_k >= k_lo[0]) & (row_k <= k_hi[0]))[:, None]
                & ((col_k >= k_lo[1]) & (col_k <= k_hi[1]))[None, :])
    fill = xp.nan if np.issubdtype(surface.dtype, np.floating) else 0
    return xp.where(in_range, gathered, fill)


class MaskedOffsetResult(NamedTuple):
    record: nornir_imageregistration.AlignmentRecord
    ncc: float  # raw masked NCC at the peak sample, NaN when there is no valid shift
    overlap: int  # valid pixels in the overlap at the peak sample


def significance_surface(ncc, overlap, xp):
    """Fisher z of each shift's NCC scaled by sqrt(overlap): how far the match stands above chance.

    Raw NCC over a small overlap has a wide null distribution, so its maximum over all shifts drifts
    to the smallest allowed overlaps (on RC3 294-296, Space peaks landed ~110 px out at ~55% overlap
    on points whose true offset was ~0). ``atanh(r) * sqrt(n)`` is proportional to the z-score of r
    against r = 0 for any fixed correlation length, so its argmax prefers the match that is least
    likely by chance; the per-image correlation length is a constant factor and does not move the
    argmax. Invalid shifts are NaN.
    """
    r = xp.clip(ncc, -0.999999, 0.999999)
    return xp.arctanh(r) * xp.sqrt(xp.maximum(overlap, 0.0))


def surface_for_peak_finding(score, xp):
    """Scale a significance surface to [0, 1] for find_peak; invalid and negative shifts become 0.

    Only valid shifts are eligible. ``weight`` (peak / mean) and ``peak_ratio`` (peak / second peak)
    are then ratios of significance.
    """
    valid = xp.isfinite(score)
    positive = xp.where(valid, xp.maximum(score, 0.0), 0.0)
    top = float(positive.max()) if positive.size else 0.0
    if top <= 0.0:
        return positive, valid & (positive > 0)
    return positive / top, valid


def masked_find_offset(fixed: NDArray, fixed_mask: NDArray[np.bool_] | None,
                       moving: NDArray, moving_mask: NDArray[np.bool_] | None,
                       *,
                       bandpass_sigmas: tuple[float, float] | None = None,
                       min_overlap_fraction: float = DEFAULT_MIN_OVERLAP_FRACTION,
                       ) -> MaskedOffsetResult:
    """Masked counterpart of :func:`phasecorrelation.find_offset` for one ``(h, w)`` pair.

    Band-passes both images on their valid pixels, computes the masked NCC surface and finds the peak
    of its significance (:func:`significance_surface`) with the same :func:`find_peak`. The returned
    peak is the centre-to-centre offset of *moving* over *fixed*, as from ``find_offset``; ``ncc`` is
    the raw masked NCC at that shift.
    """
    xp = cp.get_array_module(fixed)
    if bandpass_sigmas is None:
        bandpass_sigmas = default_bandpass_sigmas(fixed.shape)
    f, f_valid = masked_bandpass(fixed, fixed_mask, *bandpass_sigmas)
    g, g_valid = masked_bandpass(moving, moving_mask, *bandpass_sigmas)
    ncc, overlap = masked_ncc_surface(f, f_valid, g, g_valid, min_overlap_fraction)
    return _offset_from_surface(ncc, overlap, fixed.shape, moving.shape, xp)


def _offset_from_surface(ncc: NDArray, overlap: NDArray, fixed_shape: tuple[int, ...],
                         moving_shape: tuple[int, ...], xp) -> MaskedOffsetResult:
    """Peak of a masked NCC surface's significance, as the centre-to-centre offset of moving over fixed."""
    surface, eligible = surface_for_peak_finding(significance_surface(ncc, overlap, xp), xp)

    if not bool(eligible.any()):
        record = nornir_imageregistration.AlignmentRecord(peak=(0.0, 0.0), weight=0.0, peak_ratio=0.0)
        return MaskedOffsetResult(record, float('nan'), 0)

    peak = find_peak(surface, eligible, allow_in_place=False)
    origin = np.asarray(peak.scaled_offset, dtype=np.float64)
    offset = origin + (np.asarray(moving_shape[-2:], dtype=np.float64)
                       - np.asarray(fixed_shape[-2:], dtype=np.float64)) / 2.0

    centre = np.asarray(surface.shape, dtype=np.int64) // 2
    row, col = (int(v) for v in np.clip(np.rint(centre - origin), 0, np.asarray(surface.shape) - 1))
    peak_ncc = float(ncc[row, col])
    peak_overlap = int(float(overlap[row, col]))

    record = nornir_imageregistration.AlignmentRecord(
        peak=offset, weight=peak.peak_strength, peak_ratio=float(peak.peak_ratio))
    return MaskedOffsetResult(record, peak_ncc, peak_overlap)


class MaskedNccSweep:
    """Masked NCC of one band-passed fixed image against many moving images of bounded size.

    StosBrute scores the same target against the source at every candidate angle and scale. The FFT
    size is fixed by *max_moving_shape*, so the fixed image's three spectra are computed once and each
    moving image costs three forward and six inverse FFTs. Both inputs must already be band-passed
    (:func:`masked_bandpass`); masks are True where valid. NumPy or CuPy, following *fixed*.

    *dtype* float32 halves memory and, on consumer GPUs, is many times faster than float64. The
    variances are differences of sums, so float32 is safe only on band-passed (near zero-mean) input;
    ``tests/test_masked_correlation.py`` checks the peak against the float64 surface.
    """

    def __init__(self, fixed: NDArray, fixed_valid: NDArray[np.bool_], max_moving_shape: tuple[int, int],
                 min_overlap_fraction: float = DEFAULT_MIN_OVERLAP_FRACTION, dtype=np.float32) -> None:
        self._xp = cp.get_array_module(fixed)
        self._dtype = dtype
        self._min_overlap_fraction = float(min_overlap_fraction)
        self._max_moving_shape = (int(max_moving_shape[0]), int(max_moving_shape[1]))
        self._size = _correlation_size(fixed.shape, self._max_moving_shape)
        self._fixed = _SideSpectra.create(fixed, fixed_valid, self._size, dtype)

    def find_offset(self, moving: NDArray, moving_valid: NDArray[np.bool_]) -> MaskedOffsetResult:
        """Offset of *moving* over the fixed image, as :func:`masked_find_offset` returns it."""
        if moving.shape[-2] > self._max_moving_shape[0] or moving.shape[-1] > self._max_moving_shape[1]:
            raise ValueError(f"moving image {moving.shape} is larger than the sweep's {self._max_moving_shape}")
        moving_side = _SideSpectra.create(moving, moving_valid, self._size, self._dtype)
        ncc, overlap = _ncc_from_spectra(self._fixed, moving_side, self._size, self._min_overlap_fraction, self._xp)
        return _offset_from_surface(ncc, overlap, self._fixed.shape, tuple(moving.shape), self._xp)
