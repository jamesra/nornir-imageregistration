"""
Batched phase correlation primitives for grid refinement (prototype).

This module provides vectorized, batched analogs of the per-image functions in
``phasecorrelation.py``, intended for the mosaic grid-refinement vertex loop
where many small equal-sized cells are correlated against neighbors.

The serial path issues one tiny FFT + one connected-component ``find_peak`` per
mesh vertex; on a GPU each of those is launch/synchronization bound. These
batched primitives stack ``N`` cells into a single ``(N, h, w)`` array so the
FFTs run as one batched transform and peak detection runs as vectorized array
ops, with a single host transfer for the whole batch.

Peak detection here is a vectorized argmax + local center-of-mass refinement,
which differs from the connected-component ``find_peak``. The batched path is
opt-in and validated against the C++ golden mosaic before any production use.

The same code runs on NumPy and CuPy: the array module is resolved from the
input arrays with ``cupy.get_array_module`` (numpy in -> numpy out).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.peak_uniqueness import (
    DEFAULT_PEAK_RATIO_EXCLUSION_RADIUS,
    batched_masked_peak_ratios,
)

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp


_DEFAULT_CORRELATION_COEFFICIENT = 0.65


def _correlation_work_dtype(*dtypes) -> np.dtype:
    """Precision the batched correlation runs in: the inputs', but at least float32.

    Both FFT backends honour single precision -- ``numpy.fft`` and ``cupy.fft`` each
    return ``complex64`` for a ``float32`` input -- so forcing ``float64`` doubled the
    transform workspace for the ``float32`` stacks callers actually pass. The floor
    keeps integer and ``float16`` inputs from running the transform at a precision that
    would lose the correlation peak.

    CuPy dtypes are NumPy dtypes, so this needs no array module.
    """
    result = np.dtype(np.float32)
    for dtype in dtypes:
        result = np.promote_types(result, dtype)
    return result


def batched_image_phase_correlation(targets: NDArray[np.floating],
                                    sources: NDArray[np.floating],
                                    correlation_coefficient: Optional[float] = None
                                    ) -> NDArray[np.floating]:
    """Batched phase correlation of two ``(N, h, w)`` image stacks.

    Mirrors ``phasecorrelation.image_phase_correlation`` +
    ``fft_phase_correlation`` but over a leading batch axis. Each image is
    mean-subtracted before the FFT and the cross-power spectrum is normalized
    by ``|.|**correlation_coefficient`` (0.65 by default; 1.0 == pure phase
    correlation).

    :param targets: ``(N, h, w)`` target (fixed) images.
    :param sources: ``(N, h, w)`` source (moving) images, same shape.
    Runs at the inputs' precision, floored at float32, so a float32 stack keeps a
    complex64 transform instead of paying double the FFT workspace. The returned
    correlation carries that same precision.

    :param correlation_coefficient: Cross-power normalization exponent in
        ``[0, 1]``; defaults to 0.65.
    :return: ``(N, h, w)`` real correlation images on the same array module, in the
        working precision (the inputs' dtype promoted to at least float32).
    """
    xp = cp.get_array_module(targets)
    if targets.shape != sources.shape:
        raise ValueError("targets and sources must have identical shapes")
    if targets.ndim != 3:
        raise ValueError("targets and sources must be (N, h, w) stacks")

    cc = (_DEFAULT_CORRELATION_COEFFICIENT
          if correlation_coefficient is None else float(correlation_coefficient))
    if cc < 0 or cc > 1:
        raise ValueError("correlation_coefficient must be between 0 and 1")

    # Work in the caller's precision, floored at float32, rather than forcing float64.
    # Serial image_phase_correlation has always used the native dtype; this side
    # unconditionally upcast, so a float32 cell stack paid double the FFT workspace and
    # a conversion the serial path never made. The float32 floor keeps integer and
    # float16 inputs off a lossy FFT.
    work_dtype = _correlation_work_dtype(targets.dtype, sources.dtype)
    targets = xp.asarray(targets, dtype=work_dtype)
    sources = xp.asarray(sources, dtype=work_dtype)

    target_mean = targets.mean(axis=(-2, -1), keepdims=True)
    source_mean = sources.mean(axis=(-2, -1), keepdims=True)

    target_fft = xp.fft.fft2(targets - target_mean, axes=(-2, -1))
    source_fft = xp.fft.fft2(sources - source_mean, axes=(-2, -1))

    conj = xp.conjugate(target_fft)
    conj *= source_fft
    del target_fft, source_fft

    abs_conj = xp.absolute(conj)
    # Only normalize entries above a small threshold (matches fft_phase_correlation);
    # below-threshold entries are divided by 1.0 (left unchanged).
    # The one literal must match the working precision: a float64 scalar here promoted
    # denom back to float64 and undid the saving for a float32 stack.
    denom = xp.where(abs_conj > 1e-5, xp.power(abs_conj, cc),
                     xp.asarray(1.0, dtype=abs_conj.dtype))
    conj /= denom
    del abs_conj, denom

    correlation = xp.real(xp.fft.ifft2(conj, axes=(-2, -1)))
    del conj
    return correlation


def batched_find_peak(images: NDArray[np.floating],
                      overlap_mask: Optional[NDArray[np.bool_]] = None,
                      centroid_radius: int = 1,
                      peak_ratio_exclusion_radius: int = DEFAULT_PEAK_RATIO_EXCLUSION_RADIUS,
                      ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Vectorized peak finder over a ``(N, h, w)`` correlation-image stack.

    For each image: take the masked argmax, then refine to sub-pixel accuracy
    with an intensity-weighted center of mass over a ``(2r+1)`` window centered
    on the argmax (baseline removed by subtracting the window minimum). The
    offset is reported as ``(shape/2) - peak_coord`` to match
    ``phasecorrelation.find_peak``'s sign convention.

    The window wraps at the image edges, because the correlation this refines is
    circular. Serial ``find_peak`` does not wrap: it takes the center of mass of a
    thresholded connected component, and a lobe straddling the wrap splits into two
    components there. That divergence is inherent to the two algorithms rather than a
    tuning choice, and it only shows up for peaks within ``r`` of an edge.

    This is a vectorizable substitute for the connected-component
    ``find_peak`` (label / center_of_mass / sum_labels), which cannot batch.

    :param images: ``(N, h, w)`` correlation images, expected normalized to
        ``[0, 1]`` per image.
    :param overlap_mask: Optional ``(h, w)`` boolean mask of eligible peak
        locations, shared across the batch.
    :param centroid_radius: Half-width of the centroid refinement window.
    :param peak_ratio_exclusion_radius: Half-width cleared around the primary
        peak before measuring uniqueness (primary / 2nd peak).
    :return: ``(peaks, weights, peak_ratios)`` where ``peaks`` is ``(N, 2)``
        ``(dy, dx)``, ``weights`` is ``(N,)`` signal-to-noise analog, and
        ``peak_ratios`` is ``(N,)`` uniqueness. All on the input module.
    """
    xp = cp.get_array_module(images)
    if images.ndim != 3:
        raise ValueError("images must be a (N, h, w) stack")
    n, h, w = images.shape
    r = int(centroid_radius)

    if overlap_mask is not None:
        xp_mask = cp.get_array_module(overlap_mask)
        if xp_mask is not xp:
            overlap_mask = xp.asarray(overlap_mask)
        mask2d = overlap_mask.astype(xp.bool_)
        # Argmax should ignore non-overlap pixels.
        search = xp.where(mask2d[None, :, :], images, xp.asarray(-xp.inf, dtype=images.dtype))
        # Keep the count on-device: float(xp.count_nonzero(...)) forces a host sync and
        # can raise CuPy implicit-conversion errors under some backends.
        mask_count = xp.asarray(xp.count_nonzero(mask2d), dtype=images.dtype)
        mean_pixel = (
            (images * mask2d[None, :, :]).sum(axis=(-2, -1))
            / xp.maximum(mask_count, xp.asarray(1.0, dtype=images.dtype))
        )
    else:
        search = images
        mean_pixel = images.mean(axis=(-2, -1))
        mask2d = None

    flat = search.reshape(n, -1)
    idx = xp.argmax(flat, axis=1)
    peak_r = (idx // w).astype(xp.int64)
    peak_c = (idx % w).astype(xp.int64)

    images_flat = images.reshape(n, -1)
    n_arange = xp.arange(n)
    peak_val = images_flat[n_arange, idx]

    # Wrap the centroid window instead of clamping its centre. The correlation is
    # circular (ifft2, then fftshift), so a peak on the first row continues on the last
    # and the wrapped neighbourhood is the true one. Clamping kept the window in bounds
    # but slid it off the peak, leaving the peak on the window edge and dragging the
    # centroid inward: measured against known sub-pixel shifts, border peaks were off by
    # up to 0.72px, and truncating the window instead still left 0.58px. Wrapping holds
    # them to 0.06px.
    offsets = xp.arange(-r, r + 1)
    # Separate float view for the centroid weighting below: the integer form is needed
    # for the index arithmetic, but multiplying it into the window would promote a
    # float32 correlation back to float64 and undo the single-precision workspace.
    offsets_weight = offsets.astype(images.dtype)

    row_idx = (peak_r[:, None, None] + offsets[None, :, None]) % h   # (N, 2r+1, 1)
    col_idx = (peak_c[:, None, None] + offsets[None, None, :]) % w   # (N, 1, 2r+1)
    n_idx = n_arange[:, None, None]

    # Fancy indexing broadcasts these three itself, so the explicit broadcast_to the
    # absolute-index form needed is gone along with it.
    window = images[n_idx, row_idx, col_idx]                 # (N, 2r+1, 2r+1)
    # Remove baseline so the centroid is dominated by the peak, not the DC level.
    window = window - window.min(axis=(-2, -1), keepdims=True)
    wsum = window.sum(axis=(-2, -1))

    # Weight the *relative* offsets, not absolute indices: a wrapped index would
    # otherwise pull the mean clear across the image. Adding the result back to the peak
    # is algebraically what the absolute form computed whenever no wrapping occurred, so
    # interior peaks are unchanged (agreement to 1e-14, and better conditioned since the
    # large common term is no longer summed and divided out).
    # Left to broadcast in the multiply rather than materialized with broadcast_to:
    # these are (1, 2r+1, 1) and (1, 1, 2r+1) against an (N, 2r+1, 2r+1) window, and
    # expanding them cost 42% on a 64-cell batch, where fixed overhead dominates.
    weighted_r = (window * offsets_weight[None, :, None]).sum(axis=(-2, -1))
    weighted_c = (window * offsets_weight[None, None, :]).sum(axis=(-2, -1))
    valid = wsum > 0
    safe_wsum = xp.where(valid, wsum, 1.0)
    # Left deliberately un-wrapped, so a peak on row 0 whose lobe sits just before it
    # reports -0.14 rather than h-0.14. Both name the same circular position, but only
    # the un-wrapped one stays continuous across the seam, and the reported offset keeps
    # the sign the serial path and every caller already expect. Re-wrapping here turned
    # a +16.0 shift into -15.86 on a 32px cell.
    centroid_r = peak_r.astype(images.dtype) + xp.where(
        valid, weighted_r / safe_wsum, 0.0)
    centroid_c = peak_c.astype(images.dtype) + xp.where(
        valid, weighted_c / safe_wsum, 0.0)

    center_r = xp.asarray(h, dtype=images.dtype) / xp.asarray(2.0, dtype=images.dtype)
    center_c = xp.asarray(w, dtype=images.dtype) / xp.asarray(2.0, dtype=images.dtype)
    offset_r = center_r - centroid_r
    offset_c = center_c - centroid_c
    peaks = xp.stack((offset_r, offset_c), axis=1)

    # Signal-to-noise analog matching find_peak: peak / mean over overlap.
    safe_mean = xp.where(mean_pixel != 0, mean_pixel, xp.asarray(1.0, dtype=images.dtype))
    weights = xp.where((mean_pixel > 0) & (peak_val > 0), peak_val / safe_mean,
                       xp.asarray(0.0, dtype=images.dtype))
    peak_ratios = batched_masked_peak_ratios(
        images,
        peak_r,
        peak_c,
        exclusion_radius=peak_ratio_exclusion_radius,
        overlap_mask=mask2d,
        primary_values=peak_val,
    )
    peak_ratios = xp.where(weights > 0, peak_ratios, xp.asarray(0.0, dtype=peak_ratios.dtype))
    return peaks, weights, peak_ratios


def batched_find_offset(fixed_cells: NDArray[np.floating],
                        moving_cells: NDArray[np.floating],
                        cell_shape: NDArray[np.integer],
                        min_overlap: float = 0.25,
                        max_overlap: float = 1.0,
                        correlation_coefficient: Optional[float] = None,
                        centroid_radius: int = 1,
                        peak_ratio_exclusion_radius: int = DEFAULT_PEAK_RATIO_EXCLUSION_RADIUS,
                        min_std: Optional[float] = None,
                        ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Batched analog of ``phasecorrelation.find_offset`` for equal-sized cells.

    Normalizes each cell to ``[0, 1]`` (matching ``_phase_correlate_refinement_cell``),
    runs batched phase correlation, fft-shifts, per-image normalizes, applies the
    shared overlap mask, and runs the vectorized peak finder.

    Degenerate cells yield a zero-weight result, mirroring the serial
    ``is_alignable_cell`` guard: constant intensity, all-zero, and intensity std
    below the low-content floor are all rejected. The std floor matters because
    each cell is normalized by its own span, which amplifies micro-contrast into
    full-range noise and would otherwise produce a confident-looking peak.

    :param fixed_cells: ``(N, h, w)`` fixed (target) cells.
    :param moving_cells: ``(N, h, w)`` moving (source) cells, same shape.
    :param cell_shape: ``(h, w)`` cell shape used to build the overlap mask.
    :param min_overlap: Minimum overlap fraction for the mask.
    :param max_overlap: Maximum overlap fraction for the mask.
    :param correlation_coefficient: See ``batched_image_phase_correlation``.
    :param centroid_radius: Centroid refinement window half-width.
    :param peak_ratio_exclusion_radius: See ``batched_find_peak``.
    :param min_std: Minimum per-cell intensity std, matching
        ``is_alignable_cell(min_std=...)``. ``None`` reads
        ``NORNIR_REFINE_LOW_CONTENT_STD_MIN``.
    :return: ``(peaks (N,2), weights (N,), peak_ratios (N,))`` on the input module.
    """
    xp = cp.get_array_module(fixed_cells)
    if fixed_cells.shape != moving_cells.shape:
        raise ValueError("fixed_cells and moving_cells must have identical shapes")
    if fixed_cells.ndim != 3:
        raise ValueError("cells must be (N, h, w) stacks")

    # Match the caller's precision (floored at float32) rather than forcing float64;
    # see _correlation_work_dtype. Everything downstream keys off ``.dtype``, so the
    # normalize, correlate, and peak stages all follow.
    work_dtype = _correlation_work_dtype(fixed_cells.dtype, moving_cells.dtype)
    fixed = xp.asarray(fixed_cells, dtype=work_dtype)
    moving = xp.asarray(moving_cells, dtype=work_dtype)

    # Imported lazily: refine_shared.cell_validity lives under a package whose
    # __init__ imports cell_measurement, which imports this module.
    from nornir_imageregistration.refine_shared.cell_validity import (
        low_content_std_min_threshold)

    std_threshold = float(min_std if min_std is not None else low_content_std_min_threshold())

    # For any sample with range ``span`` over ``n`` points, std >= span / sqrt(2n):
    # the least-spread arrangement puts one point at each extreme. So a large
    # enough span guarantees the floor is met, and the std reduction can be
    # skipped for those cells. Real cells span most of their range, so this
    # normally clears the whole batch and keeps the gate close to free.
    n_pixels = int(fixed.shape[-1]) * int(fixed.shape[-2])
    span_implies_content = std_threshold * float(np.sqrt(2.0 * n_pixels))

    def _normalize(stack: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
        amin = stack.min(axis=(-2, -1), keepdims=True)
        amax = stack.max(axis=(-2, -1), keepdims=True)
        span = amax - amin
        span_flat = span.reshape(-1)
        valid = (span_flat > 0) & (amax.reshape(-1) != 0)
        if std_threshold > 0.0:
            # Serial rejects low-content cells on intensity std; without this the
            # batched path normalizes near-flat cells up to full range and returns
            # a noise peak with a plausible weight.
            content = span_flat >= span_implies_content
            if not bool(xp.all(content | ~valid)):
                # Some cell is ambiguous, so pay for the exact reduction. Cells
                # holding NaN already failed the span test above, so plain std
                # matches cell_intensity_std for everything still in play.
                content = content | (stack.std(axis=(-2, -1)).reshape(-1) >= std_threshold)
            valid = valid & content
        safe_span = xp.where(span > 0, span, xp.asarray(1.0, dtype=stack.dtype))
        return (stack - amin) / safe_span, valid

    norm_fixed, valid_fixed = _normalize(fixed)
    norm_moving, valid_moving = _normalize(moving)
    valid = valid_fixed & valid_moving

    correlation = batched_image_phase_correlation(
        norm_fixed, norm_moving, correlation_coefficient=correlation_coefficient)
    correlation = xp.fft.fftshift(correlation, axes=(-2, -1))

    # Per-image normalize to [0, 1] with a flat-response guard (matches find_offset).
    correlation -= correlation.min(axis=(-2, -1), keepdims=True)
    cmax = correlation.max(axis=(-2, -1), keepdims=True)
    safe_cmax = xp.where(cmax > 0, cmax, xp.asarray(1.0, dtype=correlation.dtype))
    correlation = correlation / safe_cmax
    finite_max = xp.isfinite(cmax.reshape(-1)) & (cmax.reshape(-1) > 0)
    valid = valid & finite_max

    corr_shape = correlation.shape[-2:]
    overlap_mask = nornir_imageregistration.overlapmasking.GetOverlapMaskOnDevice(
        tuple(int(v) for v in np.asarray(cell_shape).reshape(-1)[:2]),
        tuple(int(v) for v in np.asarray(cell_shape).reshape(-1)[:2]),
        corr_shape,
        min_overlap,
        max_overlap,
        xp=xp)

    peaks, weights, peak_ratios = batched_find_peak(
        correlation,
        overlap_mask,
        centroid_radius=centroid_radius,
        peak_ratio_exclusion_radius=peak_ratio_exclusion_radius)

    zero = xp.asarray(0.0, dtype=weights.dtype)
    weights = xp.where(valid, weights, zero)
    peak_ratios = xp.where(valid, peak_ratios, zero)
    # Zero the offset too, matching find_offset, which blanks the whole correlation
    # image when its max is non-finite. Leaving the argmax in place returned the
    # centre of the correlation surface -- a plausible-looking shift of half the cell
    # -- for a cell with no usable signal. Callers that gate on weight were unharmed,
    # but the offset is the primary return value and should not carry a number the
    # weight says is meaningless.
    peaks = xp.where(valid[:, None], peaks, xp.asarray(0.0, dtype=peaks.dtype))
    return peaks, weights, peak_ratios
