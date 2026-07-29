"""VRAM-aware batch limits for grid-refine GPU paths."""

from __future__ import annotations

import os

import numpy as np

import nornir_imageregistration

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp

# Fraction of currently *free* device memory to budget for one batched refine step.
# Matches the contrast-conversion helper in ``core._core`` (40% of free VRAM).
_REFINE_BATCH_VRAM_FRACTION: float = 0.40

# Reserve headroom for full section images, overlap-mask cache, and driver overhead.
_REFINE_BATCH_HEADROOM_BYTES: int = 512 * 1024 * 1024

# Measured FFT workspace per 128×128 cell from Grid16 refine logs (~1 MiB/cell).
_FFT_PEAK_BYTES_PER_CELL_128: int = 128 * 128 * 16 * 4

# Conservative bytes per map_coordinates sample (output + coord intermediates).
_ROI_BYTES_PER_SAMPLE: int = 12

_CPU_FFT_CELL_CHUNK: int = 1024
_CPU_ROI_SAMPLE_BUDGET: int = 16_000_000
_MAX_FFT_CELL_CHUNK: int = 16_384
_MAX_ROI_SAMPLE_BUDGET: int = 128_000_000
_MIN_FFT_CELL_CHUNK: int = 256


def cuda_memory_info() -> tuple[int | None, int | None]:
    """Return ``(free_bytes, total_bytes)`` for the active CUDA device, or ``(None, None)``."""
    if not nornir_imageregistration.UsingCupy():
        return None, None
    mem_get_info = getattr(getattr(cp, 'cuda', None), 'runtime', None)
    if mem_get_info is None:
        return None, None
    mem_get_info = getattr(mem_get_info, 'memGetInfo', None)
    if mem_get_info is None:
        return None, None
    try:
        free_bytes, total_bytes = mem_get_info()
        return int(free_bytes), int(total_bytes)
    except Exception:
        return None, None


def _fft_peak_bytes_per_cell(cell_h: int, cell_w: int) -> int:
    """Estimate batched FFT workspace bytes for one ``(h, w)`` cell."""
    pixels = max(1, int(cell_h) * int(cell_w))
    baseline_pixels = 128 * 128
    return max(1, int(_FFT_PEAK_BYTES_PER_CELL_128 * pixels / baseline_pixels))


def batched_fft_cell_chunk_size(cell_shape: np.ndarray | tuple[int, ...] | list[int]) -> int:
    """Max cells per batched FFT launch, tuned from free VRAM unless overridden."""
    raw = os.environ.get('NORNIR_REFINE_BATCHED_FFT_CELLS', '').strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass

    shape = np.asarray(cell_shape, dtype=np.int64).reshape(-1)
    cell_h = int(shape[0]) if shape.size > 0 else 128
    cell_w = int(shape[1]) if shape.size > 1 else cell_h

    free_bytes, _ = cuda_memory_info()
    if free_bytes is None:
        return _CPU_FFT_CELL_CHUNK

    bytes_per_cell = _fft_peak_bytes_per_cell(cell_h, cell_w)
    budget_bytes = max(0, int(free_bytes * _REFINE_BATCH_VRAM_FRACTION) - _REFINE_BATCH_HEADROOM_BYTES)
    if budget_bytes <= 0:
        return _MIN_FFT_CELL_CHUNK
    chunk = max(_MIN_FFT_CELL_CHUNK, budget_bytes // bytes_per_cell)
    return min(chunk, _MAX_FFT_CELL_CHUNK)


def batched_roi_sample_budget(cell_h: int, cell_w: int) -> int:
    """Max ``cells * H * W`` samples per batched ROI ``map_coordinates`` launch."""
    raw = os.environ.get('NORNIR_REFINE_BATCHED_ROI_SAMPLES', '').strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass

    samples_per_cell = max(1, int(cell_h) * int(cell_w))
    free_bytes, _ = cuda_memory_info()
    if free_bytes is None:
        return _CPU_ROI_SAMPLE_BUDGET

    budget_bytes = max(0, int(free_bytes * _REFINE_BATCH_VRAM_FRACTION) - _REFINE_BATCH_HEADROOM_BYTES)
    if budget_bytes <= 0:
        return samples_per_cell
    sample_budget = max(samples_per_cell, budget_bytes // _ROI_BYTES_PER_SAMPLE)
    return min(sample_budget, _MAX_ROI_SAMPLE_BUDGET)
