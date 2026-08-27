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
    :param correlation_coefficient: Cross-power normalization exponent in
        ``[0, 1]``; defaults to 0.65.
    :return: ``(N, h, w)`` real correlation images on the same array module.
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

    targets = xp.asarray(targets, dtype=xp.float64)
    sources = xp.asarray(sources, dtype=xp.float64)

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
    denom = xp.where(abs_conj > 1e-5, xp.power(abs_conj, cc), xp.float64(1.0))
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

    # Clamp the centroid window fully inside the image bounds.
    cr = xp.clip(peak_r, r, h - 1 - r)
    cc_ = xp.clip(peak_c, r, w - 1 - r)
    offsets = xp.arange(-r, r + 1)

    row_idx = cr[:, None, None] + offsets[None, :, None]      # (N, 2r+1, 1)
    col_idx = cc_[:, None, None] + offsets[None, None, :]     # (N, 1, 2r+1)
    row_idx_b = xp.broadcast_to(row_idx, (n, 2 * r + 1, 2 * r + 1))
    col_idx_b = xp.broadcast_to(col_idx, (n, 2 * r + 1, 2 * r + 1))
    n_idx = n_arange[:, None, None]

    window = images[n_idx, row_idx_b, col_idx_b]             # (N, 2r+1, 2r+1)
    # Remove baseline so the centroid is dominated by the peak, not the DC level.
    window = window - window.min(axis=(-2, -1), keepdims=True)
    wsum = window.sum(axis=(-2, -1))

    weighted_r = (window * row_idx_b).sum(axis=(-2, -1))
    weighted_c = (window * col_idx_b).sum(axis=(-2, -1))
    valid = wsum > 0
    centroid_r = xp.where(valid, weighted_r / xp.where(valid, wsum, 1.0), peak_r.astype(images.dtype))
    centroid_c = xp.where(valid, weighted_c / xp.where(valid, wsum, 1.0), peak_c.astype(images.dtype))

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
                        ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Batched analog of ``phasecorrelation.find_offset`` for equal-sized cells.

    Normalizes each cell to ``[0, 1]`` (matching ``_phase_correlate_refinement_cell``),
    runs batched phase correlation, fft-shifts, per-image normalizes, applies the
    shared overlap mask, and runs the vectorized peak finder. Degenerate cells
    (constant intensity) yield a zero-weight result, mirroring the serial guard.

    :param fixed_cells: ``(N, h, w)`` fixed (target) cells.
    :param moving_cells: ``(N, h, w)`` moving (source) cells, same shape.
    :param cell_shape: ``(h, w)`` cell shape used to build the overlap mask.
    :param min_overlap: Minimum overlap fraction for the mask.
    :param max_overlap: Maximum overlap fraction for the mask.
    :param correlation_coefficient: See ``batched_image_phase_correlation``.
    :param centroid_radius: Centroid refinement window half-width.
    :param peak_ratio_exclusion_radius: See ``batched_find_peak``.
    :return: ``(peaks (N,2), weights (N,), peak_ratios (N,))`` on the input module.
    """
    xp = cp.get_array_module(fixed_cells)
    if fixed_cells.shape != moving_cells.shape:
        raise ValueError("fixed_cells and moving_cells must have identical shapes")
    if fixed_cells.ndim != 3:
        raise ValueError("cells must be (N, h, w) stacks")

    fixed = xp.asarray(fixed_cells, dtype=xp.float64)
    moving = xp.asarray(moving_cells, dtype=xp.float64)

    def _normalize(stack: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
        amin = stack.min(axis=(-2, -1), keepdims=True)
        amax = stack.max(axis=(-2, -1), keepdims=True)
        span = amax - amin
        valid = (span.reshape(-1) > 0) & (amax.reshape(-1) != 0)
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
    return peaks, weights, peak_ratios
