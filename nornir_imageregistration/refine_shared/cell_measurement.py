"""Shared cell measurement (serial and batched translation phase correlation)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
import nornir_imageregistration.batched_phase_correlation
import nornir_imageregistration.phasecorrelation
from nornir_imageregistration.refine_shared.cell_validity import is_alignable_cell
from nornir_imageregistration.refine_shared.gpu_batch_budget import batched_fft_cell_chunk_size

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp


def normalize_cell(cell: NDArray[np.floating]) -> NDArray[np.floating]:
    """Normalize a cell to ``[0, 1]`` (min-subtracted, max-scaled)."""
    xp = cp.get_array_module(cell)
    cell = xp.asarray(cell, dtype=np.float64)
    amin = cell.min()
    amax = cell.max()
    normalized = cell - amin
    if amax != amin:
        normalized = normalized / (amax - amin)
    return normalized


def measure_translation_cell(
        cell_a: NDArray[np.floating],
        cell_b: NDArray[np.floating],
        subregion_shape: NDArray[np.integer] | None = None,
        *,
        min_overlap: float = 0.25,
        max_overlap: float = 1.0) -> nornir_imageregistration.AlignmentRecord:
    """Phase-correlate one equal-sized refinement cell (translation only).

    Matches legacy mosaic ``refine_one_point_fft`` / ``_phase_correlate_refinement_cell``
    semantics: normalize each cell to ``[0, 1]``, FFT raw equal-size cells with no
    random-noise padding, return an ``AlignmentRecord``. Degenerate cells yield
    zero weight.
    """
    xp = cp.get_array_module(cell_a)
    cell_a = xp.asarray(cell_a, dtype=np.float64)
    cell_b = xp.asarray(cell_b, dtype=np.float64)
    if subregion_shape is None:
        subregion_shape = np.asarray(cell_a.shape, dtype=np.int64)
    else:
        subregion_shape = np.asarray(subregion_shape, dtype=np.int64)

    if not is_alignable_cell(cell_a) or not is_alignable_cell(cell_b):
        return nornir_imageregistration.AlignmentRecord(
            peak=np.zeros(2, dtype=np.float64), weight=0.0)

    return nornir_imageregistration.phasecorrelation.find_offset(
        normalize_cell(cell_a),
        normalize_cell(cell_b),
        min_overlap=min_overlap,
        max_overlap=max_overlap,
        target_shape=subregion_shape,
        source_shape=subregion_shape,
        fft_required=True)


def measure_translation_cells_batched(
        fixed_cells: NDArray[np.floating],
        moving_cells: NDArray[np.floating],
        cell_shape: NDArray[np.integer],
        *,
        min_overlap: float = 0.25,
        max_overlap: float = 1.0,
        correlation_coefficient: Optional[float] = None,
        centroid_radius: int = 1
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Batched translation measurement over ``(N, h, w)`` cell stacks.

    Thin wrapper around ``batched_phase_correlation.batched_find_offset`` so mosaic
    and STOS callers share one entry point. Large batches are chunked to cap GPU
    FFT workspace (see ``NORNIR_REFINE_BATCHED_FFT_CELLS``).

    :return: ``(peaks, weights, peak_ratios)`` on the input array module.
    """
    num_cells = int(fixed_cells.shape[0])
    chunk_size = batched_fft_cell_chunk_size(fixed_cells.shape[1:])
    if num_cells <= chunk_size:
        return nornir_imageregistration.batched_phase_correlation.batched_find_offset(
            fixed_cells,
            moving_cells,
            cell_shape,
            min_overlap=min_overlap,
            max_overlap=max_overlap,
            correlation_coefficient=correlation_coefficient,
            centroid_radius=centroid_radius)

    xp = cp.get_array_module(fixed_cells)
    peak_chunks: list[NDArray[np.floating]] = []
    weight_chunks: list[NDArray[np.floating]] = []
    ratio_chunks: list[NDArray[np.floating]] = []
    for start in range(0, num_cells, chunk_size):
        stop = min(num_cells, start + chunk_size)
        peaks, weights, peak_ratios = nornir_imageregistration.batched_phase_correlation.batched_find_offset(
            fixed_cells[start:stop],
            moving_cells[start:stop],
            cell_shape,
            min_overlap=min_overlap,
            max_overlap=max_overlap,
            correlation_coefficient=correlation_coefficient,
            centroid_radius=centroid_radius)
        peak_chunks.append(peaks)
        weight_chunks.append(weights)
        ratio_chunks.append(peak_ratios)
    return (
        xp.concatenate(peak_chunks, axis=0),
        xp.concatenate(weight_chunks, axis=0),
        xp.concatenate(ratio_chunks, axis=0),
    )
