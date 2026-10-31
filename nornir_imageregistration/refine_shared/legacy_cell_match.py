"""Port of the legacy ir-refine-grid per-vertex translation estimator (match_one_pair).

The 2022 RC3 grids came from the SVN ir-tools build (``code/the/itk/fft_common.cxx``,
``grid_common.hxx``, ``common.hxx``). For each mesh-vertex cell it:

1. Low-pass filters both cell spectra (r 0.5, s 0.1), forms the Girod-Kuo normalized
   cross-power spectrum, low-pass filters that (r 0.4, s 0.1), and inverts it to a PDF.
   ``fft_data_t::apply_lp_filter`` stores ``sy`` where it means ``sy * sy``, so the filter
   is ``d^2 = sx^2 + sy`` with ``sy`` signed; reproduced here because the 2022 build had it.
2. Blanks PDF pixels whose four mirrored displacements all fall outside the overlap range.
3. Thresholds the PDF at the CDF bin holding ``1 - fraction`` of its pixels, clusters the
   pixels above it (8-connected, periodic), and takes each cluster's centre of mass; the
   cluster value is its mean remapped height.
4. Drops clusters below half the best value; more than 10 left rejects the vertex.
5. Scores the zero shift and the four wrap permutations of each peak by -NCC of the moving
   cell against the whole fixed tile (nearest-neighbour), keeping a peak only while the
   moving cell overlaps the fixed tile, and the fixed cell, by 25-100%.

The vertex shift is minus the winning translation. Tile masks are not used by the legacy
tool (``use_std_mask`` off), so a pixel is valid when it lies inside a tile's buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration import cp

_LP_FILTER_R: float = 0.5
_LP_FILTER_S: float = 0.1
_PDF_FILTER_R_SCALE: float = 0.8
_OVERLAP_MIN: float = 0.25
_OVERLAP_MAX: float = 1.0
_MAX_PEAKS: int = 10
_NEGLIGIBLE_RATIO: float = 2.0
_HISTOGRAM_BINS: int = 4096
# Candidates scored per launch: 256 cells of 128x128 hold ~100 MB of float64 working copies.
_SCORE_CHUNK_CELLS: int = 256
# The SVN source's apply_lp_filter adds sy where it means sy * sy. Whether the binary that
# built the 2022 grids carried that bug is settled against the legacy tool's output.
LOWPASS_SIGNED_Y_BUG: bool = True


@lru_cache(maxsize=16)
def _lowpass(shape: tuple[int, int], r: float, s: float, signed_y_bug: bool = True) -> NDArray[np.float64]:
    """Legacy ``apply_lp_filter`` gain as a ``(rows, cols)`` array (row = ITK y)."""
    ny, nx = shape
    hx, hy = nx // 2, ny // 2
    r0, r1 = r - s, r + s
    sx = 2.0 * (((np.arange(nx) + hx) % nx) - hx) / nx
    sy = 2.0 * (((np.arange(ny) + hy) % ny) - hy) / ny
    d2 = sx[None, :] ** 2 + (sy[:, None] if signed_y_bug else sy[:, None] ** 2)
    gain = np.where(d2 < r0 * r0, 1.0, 0.0)
    band = (d2 >= r0 * r0) & (d2 <= r1 * r1)
    d = np.sqrt(np.where(band, d2, 0.0))
    gain = np.where(band, (1.0 + np.cos(np.pi * (d - r0) / (r1 - r0))) / 2.0, gain)
    return gain


def _overlap_percent(size: tuple[int, int], offset: tuple[float, float]) -> float:
    """Legacy ``OverlapPercent``: product of per-axis overlap fractions, 0 when disjoint."""
    x_overlap = size[0] - abs(offset[0])
    y_overlap = size[1] - abs(offset[1])
    if x_overlap < 0 or y_overlap < 0:
        return 0.0
    return (x_overlap / size[0]) * (y_overlap / size[1])


@lru_cache(maxsize=16)
def _overlap_zone(shape: tuple[int, int], overlap_min: float, overlap_max: float) -> tuple[NDArray[np.bool_], float]:
    """Return the legacy PDF blanking mask ``(rows, cols)`` and its overlap-zone pixel count."""
    ny, nx = shape
    size = (nx, ny)
    blank = np.zeros((ny, nx), dtype=bool)
    area = 0
    for iy in range(ny // 2 + 1):
        for ix in range(nx // 2 + 1):
            if any(overlap_min <= _overlap_percent(size, pt) <= overlap_max
                   for pt in ((ix, iy), (nx - ix, iy), (ix, ny - iy), (nx - ix, ny - iy))):
                area += 4
                continue
            for x, y in ((ix, iy), (nx - 1 - ix, iy), (ix, ny - 1 - iy), (nx - 1 - ix, ny - 1 - iy)):
                blank[y, x] = True
    return blank, float(area)


def legacy_pdf(fixed_cells: NDArray[np.floating], moving_cells: NDArray[np.floating]) -> NDArray[np.floating]:
    """Legacy displacement PDF for ``(N, h, w)`` cell stacks, blanked outside the overlap zone.

    Accepts NumPy or CuPy; follows ``cp.get_array_module(fixed_cells)``.
    """
    xp = cp.get_array_module(fixed_cells)
    shape = (int(fixed_cells.shape[-2]), int(fixed_cells.shape[-1]))
    lp = xp.asarray(_lowpass(shape, _LP_FILTER_R, _LP_FILTER_S, LOWPASS_SIGNED_Y_BUG), dtype=np.float32)
    lp_pdf = xp.asarray(_lowpass(shape, _LP_FILTER_R * _PDF_FILTER_R_SCALE, _LP_FILTER_S, LOWPASS_SIGNED_Y_BUG),
                        dtype=np.float32)
    f0 = xp.fft.fft2(xp.asarray(fixed_cells, dtype=np.float32), axes=(-2, -1)) * lp
    f1 = xp.fft.fft2(xp.asarray(moving_cells, dtype=np.float32), axes=(-2, -1)) * lp
    p10 = f1 * xp.conj(f0)
    del f0, f1
    p = p10 / (xp.abs(p10) + np.float32(1e-8))
    del p10
    p *= lp_pdf
    pdf = xp.real(xp.fft.ifft2(p, axes=(-2, -1)))
    del p
    blank, _ = _overlap_zone(shape, _OVERLAP_MIN, _OVERLAP_MAX)
    pdf_min = pdf.min(axis=(-2, -1), keepdims=True)
    return xp.where(xp.asarray(blank)[None, :, :], pdf_min, pdf)


@dataclass
class LegacyPeaks:
    """Per-cell peak lists from ``legacy_find_peaks`` (host arrays, peaks sorted best first)."""
    cell: NDArray[np.int64]       # owning cell of each peak
    x: NDArray[np.float64]        # centre of mass, PDF column (ITK x)
    y: NDArray[np.float64]        # centre of mass, PDF row (ITK y)
    rejected: NDArray[np.bool_]   # per cell: more than _MAX_PEAKS peaks survived


def legacy_find_peaks(pdf: NDArray[np.floating]) -> LegacyPeaks:
    """Legacy ``find_maxima_cm`` + ``reject_negligible_maxima`` over a ``(N, h, w)`` PDF stack.

    Host-only: SciPy connected components. Accepts NumPy or CuPy input (one transfer).
    """
    pdf = np.asarray(nornir_imageregistration.EnsureNumpyArray(pdf), dtype=np.float64)
    n, h, w = pdf.shape
    _, area = _overlap_zone((h, w), _OVERLAP_MIN, _OVERLAP_MAX)
    fraction = min(64.0 / area, max(5.0 / area, 1e-2))
    target = (1.0 - fraction) * float(h * w)

    flat = pdf.reshape(n, -1)
    v_min = flat.min(axis=1)
    v_max = flat.max(axis=1)
    v_rng = v_max - v_min
    has_peaks = np.isfinite(v_rng) & (v_rng > 0)
    safe_rng = np.where(has_peaks, v_rng, 1.0)
    bins = ((flat - v_min[:, None]) / safe_rng[:, None] * (_HISTOGRAM_BINS - 1)).astype(np.int64)
    # First bin i >= 1 whose CDF reaches the target count: the ceil(target)-th smallest bin.
    kth = max(0, int(np.ceil(target)) - 1)
    first_bin = np.maximum(1, np.partition(bins, kth, axis=1)[:, kth])
    clip_min = v_min + first_bin / float(_HISTOGRAM_BINS - 1) * v_rng
    background = clip_min - v_rng * 1e-3
    above = (pdf >= clip_min[:, None, None]) & has_peaks[:, None, None]
    with np.errstate(divide='ignore', invalid='ignore'):   # flat PDFs have no pixels above the clip
        remapped = (pdf - background[:, None, None]) / (v_max - background)[:, None, None]

    # 8-connected clusters within each cell, then merged across the periodic boundary.
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1] = True
    labels, num = scipy.ndimage.label(above, structure=structure)
    empty = LegacyPeaks(np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(n, bool))
    if num == 0:
        return empty
    pairs = []
    for d in (-1, 0, 1):
        pairs.append((labels[:, 0, :], np.roll(labels[:, h - 1, :], -d, axis=1)))
        pairs.append((labels[:, :, 0], np.roll(labels[:, :, w - 1], -d, axis=1)))
    a = np.concatenate([p[0].ravel() for p in pairs])
    b = np.concatenate([p[1].ravel() for p in pairs])
    both = (a > 0) & (b > 0)
    graph = scipy.sparse.coo_matrix((np.ones(int(both.sum())), (a[both], b[both])), shape=(num + 1, num + 1))
    _, component = scipy.sparse.csgraph.connected_components(graph, directed=False)

    cell_idx, rows, cols = np.nonzero(labels)
    cluster = component[labels[cell_idx, rows, cols]]
    # Unwrap each pixel to within half a period of its cluster's first pixel.
    order = np.argsort(cluster, kind='stable')
    first = np.zeros(component.max() + 1, dtype=np.int64)
    uniq, first_pos = np.unique(cluster[order], return_index=True)
    first[uniq] = order[first_pos]
    ref_r, ref_c = rows[first[cluster]], cols[first[cluster]]
    x = cols + w * np.round((ref_c - cols) / w)
    y = rows + h * np.round((ref_r - rows) / h)
    m = remapped[cell_idx, rows, cols]

    count = np.bincount(cluster, minlength=first.size).astype(np.float64)
    mt = np.bincount(cluster, weights=m, minlength=first.size)
    mx = np.bincount(cluster, weights=m * x, minlength=first.size)
    my = np.bincount(cluster, weights=m * y, minlength=first.size)
    live = np.flatnonzero(count > 0)
    peak_cell = cell_idx[first[live]]
    value = mt[live] / count[live]
    peak_x = mx[live] / mt[live]
    peak_y = my[live] / mt[live]

    # Best first within each cell; keep peaks within a factor of two of the cell's best.
    order = np.lexsort((-value, peak_cell))
    peak_cell, value, peak_x, peak_y = peak_cell[order], value[order], peak_x[order], peak_y[order]
    best = np.full(n, 0.0)
    np.maximum.at(best, peak_cell, value)
    keep = ~(best[peak_cell] / value > _NEGLIGIBLE_RATIO)
    peak_cell, peak_x, peak_y = peak_cell[keep], peak_x[keep], peak_y[keep]
    rejected = np.bincount(peak_cell, minlength=n) > _MAX_PEAKS
    return LegacyPeaks(peak_cell, peak_x, peak_y, rejected)


def _ncc_metric(fixed: NDArray[np.floating], moving: NDArray[np.floating],
                valid: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Legacy ``my_metric_mt``: -NCC over valid pixels per ``(C, h, w)`` slice; +inf when empty."""
    xp = cp.get_array_module(fixed)
    a = xp.where(valid, fixed, 0).astype(np.float64)
    b = xp.where(valid, moving, 0).astype(np.float64)
    n = valid.sum(axis=(-2, -1)).astype(np.float64)
    sa, sb = a.sum(axis=(-2, -1)), b.sum(axis=(-2, -1))
    safe_n = xp.where(n > 0, n, 1.0)
    ab = (a * b).sum(axis=(-2, -1)) - sa * sb / safe_n
    aa = (a * a).sum(axis=(-2, -1)) - sa * sa / safe_n
    bb = (b * b).sum(axis=(-2, -1)) - sb * sb / safe_n
    with np.errstate(divide='ignore', invalid='ignore'):
        metric = -ab / xp.sqrt(aa * bb)
    metric = xp.where(n > 0, metric, np.inf)
    return nornir_imageregistration.EnsureNumpyArray(metric).astype(np.float64)


def _axis_overlap(lo: NDArray[np.floating], hi: NDArray[np.floating], size: float) -> NDArray[np.floating]:
    """Length of ``[lo, hi)`` clipped to ``[0, size)``, as legacy ``overlap_ratio`` clamps."""
    return np.clip(hi, 0.0, size) - np.clip(lo, 0.0, size)


@dataclass
class LegacyCellBatch:
    """Cells of one batch with the geometry the candidate scoring needs (host arrays).

    ``fixed_tiles[k]`` and ``fixed_starts[k]`` locate cell ``k`` in its fixed tile, so a
    candidate translation can be scored against tile pixels beyond the cell.
    """
    fixed_tile_index: NDArray[np.int64]
    fixed_starts: NDArray[np.int64]
    moving_inside: NDArray[np.bool_]   # (N, h, w): moving cell pixel inside the moving tile


def legacy_choose_shifts(peaks: LegacyPeaks,
                         batch: LegacyCellBatch,
                         fixed_tiles: list[NDArray[np.floating]],
                         moving_cells: NDArray[np.floating],
                         gather) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Score every candidate translation and return per-cell ``(shift (N, 2) as (y, x), ok)``.

    ``gather(tile, starts, cell_shape) -> (stack, inside)`` copies windows of a fixed tile.
    The shift is minus the winning translation, in working-resolution pixels.
    """
    n, h, w = (int(v) for v in moving_cells.shape)
    cell_shape = np.asarray((h, w), dtype=np.int64)
    num_cells = n

    # Candidate list: zero shift first, then each peak's permutations in legacy order.
    cand_cell = [np.arange(num_cells)]
    cand_t = [np.zeros((num_cells, 2))]
    cand_peak = [np.full(num_cells, -1)]
    for dx_wrap, dy_wrap in ((0, 0), (w, 0), (0, h), (w, h)):
        cand_cell.append(peaks.cell)
        cand_t.append(np.column_stack((peaks.y - dy_wrap, peaks.x - dx_wrap)))
        cand_peak.append(np.arange(peaks.cell.size))
    cells = np.concatenate(cand_cell)
    t = np.concatenate(cand_t)
    peak_id = np.concatenate(cand_peak)

    # Tile overlap: the fixed tile bbox mapped by +t against the moving cell, over the cell area.
    starts = batch.fixed_starts[cells].astype(np.float64)
    tile_shape = np.asarray([fixed_tiles[i].shape[-2:] for i in batch.fixed_tile_index], dtype=np.float64)[cells]
    oy = _axis_overlap(t[:, 0] - starts[:, 0], tile_shape[:, 0] - starts[:, 0] + t[:, 0], h)
    ox = _axis_overlap(t[:, 1] - starts[:, 1], tile_shape[:, 1] - starts[:, 1] + t[:, 1], w)
    tile_ratio = oy * ox / float(h * w)
    is_zero = peak_id < 0
    passes = np.where(is_zero, (tile_ratio >= 0.0) & (tile_ratio <= 1.0),
                      (tile_ratio >= _OVERLAP_MIN) & (tile_ratio <= _OVERLAP_MAX))
    # Cell overlap, checked on the running best after each peak (fixed and moving cells coincide).
    cell_ratio = (np.clip(h - np.abs(t[:, 0]), 0, h) * np.clip(w - np.abs(t[:, 1]), 0, w)) / float(h * w)
    cell_ok = (cell_ratio >= _OVERLAP_MIN) & (cell_ratio <= _OVERLAP_MAX)

    metric = np.full(cells.size, np.inf)
    scored = np.flatnonzero(passes & ~peaks.rejected[cells])
    if scored.size:
        # Nearest-neighbour sampling: moving index j = q - start + t  ->  q = j + start - floor(t + 0.5).
        shift_int = np.floor(t[scored] + 0.5).astype(np.int64)
        xp = cp.get_array_module(moving_cells)
        # Bounded chunks: each candidate costs a few float64 cell copies, and a batch can hold
        # thousands of candidates; on a GPU with little free memory WDDM pages instead of failing.
        chunk = _SCORE_CHUNK_CELLS
        for tile_index in np.unique(batch.fixed_tile_index[cells[scored]]):
            tile_sel = scored[batch.fixed_tile_index[cells[scored]] == tile_index]
            for begin in range(0, tile_sel.size, chunk):
                sel = tile_sel[begin:begin + chunk]
                sub = np.searchsorted(scored, sel)
                window_starts = batch.fixed_starts[cells[sel]] - shift_int[sub]
                fixed_stack, fixed_inside = gather(fixed_tiles[tile_index], window_starts, cell_shape)
                moving = moving_cells[xp.asarray(cells[sel])]
                valid = fixed_inside & xp.asarray(batch.moving_inside[cells[sel]])
                metric[sel] = _ncc_metric(fixed_stack, moving, valid)
                del fixed_stack, fixed_inside, moving, valid

    # Legacy selection, in closed form. Within a peak, estimate_displacement keeps the first
    # strictly-best passing permutation; match_one_pair then reverts it if the cells overlap
    # by less than 25%, so the peak contributes nothing. Across the zero shift and the
    # peaks (in order), the first strict minimum wins.
    metric = np.where(np.isnan(metric) | ~passes, np.inf, metric)
    num_peaks = peaks.cell.size
    zero_metric = metric[:num_cells]
    best_metric = zero_metric.copy()
    best_t = np.zeros((num_cells, 2))
    if num_peaks:
        perm_metric = metric[num_cells:].reshape(4, num_peaks).T
        perm_t = t[num_cells:].reshape(4, num_peaks, 2).transpose(1, 0, 2)
        perm_cell_ok = cell_ok[num_cells:].reshape(4, num_peaks).T
        winner = np.argmin(perm_metric, axis=1)
        rows = np.arange(num_peaks)
        winner_metric = perm_metric[rows, winner]
        winner_t = perm_t[rows, winner]
        winner_metric = np.where(perm_cell_ok[rows, winner], winner_metric, np.inf)
        # First minimum per cell among the peak winners (peaks are already in legacy order).
        order = np.lexsort((rows, winner_metric, peaks.cell))
        first = np.ones(order.size, dtype=bool)
        first[1:] = peaks.cell[order[1:]] != peaks.cell[order[:-1]]
        top = order[first]
        top_cell = peaks.cell[top]
        better = winner_metric[top] < best_metric[top_cell]
        best_metric[top_cell[better]] = winner_metric[top][better]
        best_t[top_cell[better]] = winner_t[top][better]
    ok = np.isfinite(best_metric) & ~peaks.rejected
    return -best_t, ok
