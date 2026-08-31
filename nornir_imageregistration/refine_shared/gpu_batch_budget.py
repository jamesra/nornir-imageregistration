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

# Measured FFT workspace per 128×128 cell from Grid16 refine logs (~1 MiB/cell). The
# ``16`` is ``complex128``, so this models a float64 cell.
#
# Deliberately left float64-modelled even though #92 made the correlation run at the
# caller's precision and callers pass float32. Making it precision-aware doubles the
# chunk, and measurement on a 22 GiB card (8192 cells of 128px, float32, median of 7)
# says larger chunks are *slower*, not faster:
#
#   chunk  256 (32 launches) 0.174s, peak  176 MiB
#   chunk 2048 ( 4 launches) 0.241s, peak 1152 MiB
#   chunk 8195 ( 1 launch  ) 0.282s, peak 4608 MiB
#
# Launch overhead is trivial next to the bandwidth cost of a larger working set, so the
# over-estimate buys headroom for free rather than costing throughput. Chunk size does
# not affect the measurements themselves, only time and peak memory. Raising this — or
# scaling it down by precision — needs a fresh sweep first. See review issue #227.
_FFT_PEAK_BYTES_PER_CELL_128: int = 128 * 128 * 16 * 4

# Modelled FFT workspace per launch that measured fastest. The optimum is a *working-set
# size*, not a cell count. Sweeping chunk size at three cell sizes on an RTX 4500 Ada
# (float32, median of 7, results identical at every chunk) puts the best chunk at:
#
#   cell  64px -> chunk 1024   0.0353s  (worst 0.3001s, spread 8.50x)
#   cell 128px -> chunk  256   0.1250s  (worst 0.2400s, spread 2.32x)
#   cell 256px -> chunk   64   0.1326s  (worst 0.2614s, spread 1.97x)
#
# A 16x spread in cell count, but the same 16 MiB of cell input and the same 145 MiB peak
# workspace every time -- so it is a cache/bandwidth effect, and a byte target is what has
# any chance of transferring to other hardware. In the units ``_fft_peak_bytes_per_cell``
# models, all three optima are 256 MiB, which is why the target is expressed that way
# rather than as cells.
#
# This also corrects #228, which read the trend as monotonically favouring smaller chunks
# because it only swept down to 256 at 128px, right at the optimum. There is an interior
# optimum and undershooting is worse than overshooting: at 64px cells chunk 64 is 8.5x
# slower than chunk 1024. A fixed *cell* ceiling would therefore be actively harmful --
# the 256 that #228 suggested as a workaround is 4x too large at 256px cells and 4x too
# small at 64px.
#
# One GPU, so the VRAM budget is deliberately kept as the upper bound rather than replaced:
# this only ever lowers the chunk. Override with NORNIR_REFINE_BATCHED_FFT_CELLS. Re-sweep
# before changing the constant. See review #228.
_FFT_PREFERRED_WORKSPACE_BYTES: int = 256 * 1024 * 1024

# Conservative bytes per map_coordinates sample (output + coord intermediates).
_ROI_BYTES_PER_SAMPLE: int = 12

_CPU_FFT_CELL_CHUNK: int = 1024
# Same peak as 1024 cells of 128×128 (~1 GiB). Scale CPU chunks by cell area
# from this budget so a 4096² stack cannot request 58 GiB in one fft2.
_CPU_FFT_BUDGET_BYTES: int = _CPU_FFT_CELL_CHUNK * _FFT_PEAK_BYTES_PER_CELL_128
_CPU_ROI_SAMPLE_BUDGET: int = 16_000_000
_MAX_FFT_CELL_CHUNK: int = 16_384
_MAX_ROI_SAMPLE_BUDGET: int = 128_000_000


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

    bytes_per_cell = _fft_peak_bytes_per_cell(cell_h, cell_w)
    free_bytes, _ = cuda_memory_info()
    if free_bytes is None:
        budget_bytes = _CPU_FFT_BUDGET_BYTES
    else:
        budget_bytes = max(
            0, int(free_bytes * _REFINE_BATCH_VRAM_FRACTION) - _REFINE_BATCH_HEADROOM_BYTES)
    if budget_bytes <= 0:
        return 1
    # Throughput target first, VRAM as the ceiling it already was. Filling the batch to
    # whatever fits measured 2.3-8.5x slower than the preferred working set, depending on
    # cell size, so "as large as fits" was maximising the wrong quantity.
    budget_bytes = min(budget_bytes, _FFT_PREFERRED_WORKSPACE_BYTES)
    chunk = max(1, budget_bytes // bytes_per_cell)
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
