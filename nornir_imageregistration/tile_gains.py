"""
Per-tile brightness gains from the overlaps of a mosaic's tiles.

The beam's brightness drifts during a capture (a warming filament makes the first tiles of a pass brighter than the
last, and a pass that resumes later can start at a different brightness: RC2's "white stripe"). Overlapping tiles show
the same tissue, so the ratio of their brightness over the overlap measures the ratio of their gains. One robust
least-squares solve over all overlaps of a section gives a gain per tile; dividing a tile's counts by its gain removes
the drift. Design: ``.cursor/plans/crush_optimized_pipeline.plan.md`` (brightness-drift, decided 2026-11-01).

Model, per overlap of tile ``i`` (left or top) with tile ``j`` (right or bottom)::

    log m_j - log m_i = A_j - A_i + c_dir

``m`` is a robust statistic (median) of the overlap strip in each tile, ``A = log gain`` and ``c_dir`` is one shading
constant per overlap direction (left-right, top-bottom). Without the shading constants the solve reads edge shading
(vignetting) as gain and compounds it along rows and columns (RC3 0800 trial: gains 0.1-80). A shading constant and a
gain ramp across columns give identical overlap ratios; the solve fixes that mode by convention: the correction adds
no linear brightness gradient across the section (see :func:`solve_tile_gains`).

Guards:

- strips with too little tissue (blank, saturated or flat) are skipped;
- the solve is iteratively reweighted: overlaps whose ratio disagrees with the rest (a misplaced tile compares
  unrelated tissue) are dropped;
- a weak temporal prior ties tiles adjacent in capture order, so tiles without usable overlaps (resin, section edge)
  take their gain from their capture-time neighbours instead of staying at 1;
- overlaps with a very large ratio (a dim tile against a bright one) count for less in the solve, and a rejected overlap
  is blamed on the tile with fewer agreeing overlaps, so a dying beam's dim tiles do not drag their bright neighbours off;
- a dim tile, or block of tiles, left without consensus overlaps takes its level from its own overlaps (RC1 120);
- the section's median gain is anchored at 1.

Strip positions come from the mosaic's tile placements (any ``.mosaic``: ``Stage``, ``Translated_*`` or ``Grid_*``;
for warped tiles the placement is the bounding box), so the strips tolerate the usual few hundred pixels of stage
error. Gains are measured on counts: the caller's reader must return values proportional to detector counts (crushed
codes inflated with ``noise_crush.uncrush``; legacy 8-bit tiles mapped back through the filter's contrast cutoffs).
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import os
import shutil
import struct
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
from numpy.typing import NDArray

#: A tile with no overlap that agrees with the consensus takes its gain from its own overlaps (the median over the
#: neighbours' gains) only when that estimate is a clearly dim tile: a misplaced tile compares unrelated tissue and
#: gives ratios near 1, which must not be believed.
DIM_FALLBACK_MAX_GAIN = 0.3

#: Directions of an overlap: tile j to the right of tile i, or below it.
LEFT_RIGHT = 'LR'
TOP_BOTTOM = 'TB'


@dataclasses.dataclass(frozen=True)
class TilePlacement:
    """Where a tile sits in mosaic space, full-resolution pixels: origin (Y, X) and size (height, width)."""
    y: float
    x: float
    height: float
    width: float


@dataclasses.dataclass(frozen=True)
class OverlapMeasurement:
    """Brightness of the shared strip of two neighbouring tiles."""
    tile_i: int  # left or top tile
    tile_j: int  # right or bottom tile
    direction: str  # LEFT_RIGHT or TOP_BOTTOM
    stat_i: float  # robust statistic of the strip in tile i (counts)
    stat_j: float
    pixels: int  # pixels that contributed to the statistic in each strip (the smaller of the two)

    @property
    def log_ratio(self) -> float:
        return float(np.log(self.stat_j) - np.log(self.stat_i))


@dataclasses.dataclass
class TileGainResult:
    """Gains per tile and how well they are determined."""
    gains: dict[int, float]
    shading: dict[str, float]  # log-ratio constant per overlap direction
    pairs_used: int
    pairs_rejected: int
    pairs_skipped: int  # strips without enough tissue
    residual_mad: float  # median absolute residual of the used pairs, log units (0.01 = 1%)
    tiles_without_pairs: list[int]
    measurements: list[OverlapMeasurement] = dataclasses.field(default_factory=list)
    #: Tiles without consensus overlaps whose gain was taken from their own overlaps (see ``DIM_FALLBACK_MAX_GAIN``).
    estimated_tiles: list[int] = dataclasses.field(default_factory=list)

    def checksum(self) -> str:
        """Stable digest of the gain set (rounded to 1e-5), for provenance and staleness checks."""
        return gains_checksum(self.gains)


def gains_checksum(gains: Mapping[int, float]) -> str:
    """Stable digest of a gain set (rounded to 1e-5), for provenance and staleness checks."""
    text = ';'.join(f'{t}:{gains[t]:.5f}' for t in sorted(gains))
    return hashlib.md5(text.encode('ascii')).hexdigest()


def neighbour_pairs(placements: Mapping[int, TilePlacement], min_side_overlap: float = 0.5,
                    max_overlap: float = 0.6) -> list[tuple[int, int, str]]:
    """
    Pairs of tiles that are left-right or top-bottom neighbours.

    A pair qualifies when the tiles share at least ``min_side_overlap`` of their extent along the shared edge and
    overlap by less than ``max_overlap`` of a tile across it (larger overlaps are piles, not neighbours).
    """
    ids = sorted(placements)
    if not ids:
        return []
    origin = np.array([[placements[t].y, placements[t].x] for t in ids])
    size = np.array([[placements[t].height, placements[t].width] for t in ids])
    reach = float(size.max()) * 1.5
    order = np.argsort(origin[:, 1])
    xs = origin[order, 1]
    pairs: list[tuple[int, int, str]] = []
    for a_pos, a in enumerate(order):
        hi = np.searchsorted(xs, origin[a, 1] + reach, side='right')
        lo = np.searchsorted(xs, origin[a, 1] - reach, side='left')
        for b in order[lo:hi]:
            if b <= a:
                continue
            dy, dx = origin[b] - origin[a]
            oy = min(origin[a, 0] + size[a, 0], origin[b, 0] + size[b, 0]) - max(origin[a, 0], origin[b, 0])
            ox = min(origin[a, 1] + size[a, 1], origin[b, 1] + size[b, 1]) - max(origin[a, 1], origin[b, 1])
            if oy <= 0 or ox <= 0:
                continue
            if abs(dx) >= abs(dy):  # left-right
                if oy < min_side_overlap * min(size[a, 0], size[b, 0]) or ox > max_overlap * min(size[a, 1], size[b, 1]):
                    continue
                i, j = (a, b) if dx >= 0 else (b, a)
                pairs.append((ids[i], ids[j], LEFT_RIGHT))
            else:
                if ox < min_side_overlap * min(size[a, 1], size[b, 1]) or oy > max_overlap * min(size[a, 0], size[b, 0]):
                    continue
                i, j = (a, b) if dy >= 0 else (b, a)
                pairs.append((ids[i], ids[j], TOP_BOTTOM))
    return pairs


def _strip_slices(p_i: TilePlacement, p_j: TilePlacement, downsample: float, inset: float
                  ) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
    """Slices of the shared region in each tile's own (downsampled) pixels, inset along both axes."""
    y0, y1 = max(p_i.y, p_j.y), min(p_i.y + p_i.height, p_j.y + p_j.height)
    x0, x1 = max(p_i.x, p_j.x), min(p_i.x + p_i.width, p_j.x + p_j.width)
    iy, ix = (y1 - y0) * inset, (x1 - x0) * inset
    y0, y1, x0, x1 = y0 + iy, y1 - iy, x0 + ix, x1 - ix
    if y1 - y0 < 2 * downsample or x1 - x0 < 2 * downsample:
        return None

    def local(p: TilePlacement) -> tuple[slice, slice]:
        return (slice(int(np.floor((y0 - p.y) / downsample)), int(np.ceil((y1 - p.y) / downsample))),
                slice(int(np.floor((x0 - p.x) / downsample)), int(np.ceil((x1 - p.x) / downsample))))

    return local(p_i), local(p_j)


def strip_statistic(strip: NDArray, low: float, high: float, min_valid_fraction: float,
                    min_texture: float) -> tuple[float, int] | None:
    """
    Median of the strip's tissue pixels, or None when the strip has too little tissue.

    Tissue pixels lie strictly between ``low`` and ``high`` (blank background and saturation excluded). A strip is
    rejected when fewer than ``min_valid_fraction`` of its pixels are tissue, or when its relative spread
    ((p90 - p10) / median) is below ``min_texture`` (flat resin, a blank tear).
    """
    values = np.asarray(strip, dtype=np.float64).ravel()
    valid = values[(values > low) & (values < high)]
    if valid.size < max(16, min_valid_fraction * values.size):
        return None
    p10, median, p90 = np.percentile(valid, (10, 50, 90))
    if median <= 0 or (p90 - p10) / median < min_texture:
        return None
    return float(median), int(valid.size)


def measure_overlaps(placements: Mapping[int, TilePlacement], read_tile: Callable[[int], NDArray], downsample: float,
                     low: float, high: float, min_valid_fraction: float = 0.5, min_texture: float = 0.02,
                     inset: float = 0.1, pairs: Sequence[tuple[int, int, str]] | None = None,
                     max_workers: int = 8) -> tuple[list[OverlapMeasurement], int]:
    """
    Measure every neighbour overlap. ``read_tile(id)`` returns the tile at ``downsample`` in counts.

    Each tile is read once, by a bounded thread pool (reads are I/O bound over the NAS), and reduced at once to the
    statistics of its strips, so memory holds only the tiles in flight, never the section. Returns the measurements and
    the number of pairs skipped for lack of tissue.
    """
    if pairs is None:
        pairs = neighbour_pairs(placements)
    jobs: dict[int, list[tuple[int, int, tuple[slice, slice]]]] = {}  # tile -> (pair index, side, slices)
    geometry_skipped: set[int] = set()
    for k, (i, j, _) in enumerate(pairs):
        slices = _strip_slices(placements[i], placements[j], downsample, inset)
        if slices is None:
            geometry_skipped.add(k)
            continue
        jobs.setdefault(i, []).append((k, 0, slices[0]))
        jobs.setdefault(j, []).append((k, 1, slices[1]))

    def tile_stats(tile: int) -> list[tuple[int, int, tuple[float, int] | None]]:
        image = np.asarray(read_tile(tile))
        return [(k, side, strip_statistic(image[sl], low, high, min_valid_fraction, min_texture))
                for k, side, sl in jobs[tile]]

    stats: dict[tuple[int, int], tuple[float, int] | None] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for result in pool.map(tile_stats, sorted(jobs)):
            for k, side, value in result:
                stats[(k, side)] = value

    out: list[OverlapMeasurement] = []
    skipped = len(geometry_skipped)
    for k, (i, j, direction) in enumerate(pairs):
        if k in geometry_skipped:
            continue
        si, sj = stats.get((k, 0)), stats.get((k, 1))
        if si is None or sj is None:
            skipped += 1
            continue
        out.append(OverlapMeasurement(i, j, direction, si[0], sj[0], min(si[1], sj[1])))
    return out, skipped


def estimate_shading(measurements: Sequence[OverlapMeasurement]) -> dict[str, float]:
    """
    Edge shading per direction from the tiles themselves (a flat-field estimate), not from the gain solve.

    For a left-right overlap the model constant is ``c_LR = S_left - S_right``: the log brightness of a tile's left
    edge strip minus its right edge strip. Overlap ratios alone cannot separate it from a gain that ramps linearly
    across columns (drift in a column-by-column capture looks exactly like that), so it is measured inside tiles:
    every tile that has both a left and a right strip gives ``log(left) - log(right)``, and the median over the section
    averages out the tissue. Same for top-bottom. Returns 0 for a direction with no such tiles.
    """
    edges: dict[tuple[int, str, str], float] = {}  # (tile, direction, side) -> log statistic
    for m in measurements:
        first, second = ('right', 'left') if m.direction == LEFT_RIGHT else ('bottom', 'top')
        edges[(m.tile_i, m.direction, first)] = float(np.log(m.stat_i))
        edges[(m.tile_j, m.direction, second)] = float(np.log(m.stat_j))
    shading = {}
    for direction, near, far in ((LEFT_RIGHT, 'left', 'right'), (TOP_BOTTOM, 'top', 'bottom')):
        tiles = {t for (t, d, _) in edges if d == direction}
        diffs = [edges[(t, direction, near)] - edges[(t, direction, far)] for t in tiles
                 if (t, direction, near) in edges and (t, direction, far) in edges]
        shading[direction] = float(np.median(diffs)) if diffs else 0.0
    return shading


def _largest_component(n: int, a: NDArray, b: NDArray) -> NDArray:
    """Indices of the tiles in the largest connected set of the graph with edges ``a[k] - b[k]`` over *n* tiles."""
    graph = scipy.sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    count, label = scipy.sparse.csgraph.connected_components(graph, directed=False)
    return np.flatnonzero(label == int(np.argmax(np.bincount(label, minlength=count))))


def _place_detached_blocks(x: NDArray, n: int, ids: Sequence[int], i_idx: NDArray, j_idx: NDArray, c_idx: NDArray,
                           d: NDArray, weight: NDArray, temporal: NDArray) -> list[int]:
    """
    Set the level of each set of tiles that the surviving overlaps do not connect to the main body, in place in ``x``.

    The robust solve leaves such a set at a level only the weak capture-order ties hold: free to drift, or (a few tiles
    whose overlaps all disagreed with the consensus) left at its neighbours' gain. Its level is better known from the
    overlaps that were rejected: a dim set (RC1 120: the tail of a dying beam) disagrees with the consensus because it
    is dim, and each of those overlaps still measures how dim. The set is moved by the median shift that makes its
    overlaps with the placed tiles agree, accepted when at least two overlaps agree to within ``_BLOCK_SPREAD`` (log
    units), and, for a single tile, only when that tile is clearly dim (``DIM_FALLBACK_MAX_GAIN``: a misplaced tile
    compares unrelated tissue and gives ratios near 1). A multi-tile set that cannot be placed that way is moved to its
    capture-order neighbours' median level. Returns the ids of the tiles that were moved by their overlaps.
    """
    used = weight > 0
    count, label = scipy.sparse.csgraph.connected_components(
        scipy.sparse.coo_matrix((np.ones(int(used.sum())), (i_idx[used], j_idx[used])), shape=(n, n)), directed=False)
    sizes = np.bincount(label, minlength=count)
    placed = label == int(np.argmax(sizes))
    touching: list[list[int]] = [[] for _ in range(n)]
    for k in range(len(d)):
        touching[int(i_idx[k])].append(k)
        touching[int(j_idx[k])].append(k)
    moved: list[int] = []
    pending = [int(c) for c in np.argsort(-sizes, kind='stable') if not placed[label == c].any()]
    progress = True
    while pending and progress:
        progress = False
        for c in list(pending):
            members = np.flatnonzero(label == c)
            shifts: list[float] = []
            for t in members:
                for k in touching[int(t)]:
                    i, j = int(i_idx[k]), int(j_idx[k])
                    other = j if i == t else i
                    if not placed[other] or label[other] == c:
                        continue
                    # d = x_j - x_i + shading: the shift that makes this overlap agree
                    shifts.append(float(x[other] - x[t] + (d[k] - x[c_idx[k]] if other == i else x[c_idx[k]] - d[k])))
            if len(shifts) < 2:
                continue
            shift = float(np.median(shifts))
            spread = float(np.median(np.abs(np.asarray(shifts) - shift)))
            level = float(np.median(x[members])) + shift
            if spread > _BLOCK_SPREAD or (len(members) == 1 and np.exp(level - np.median(x[:n][placed])) >= DIM_FALLBACK_MAX_GAIN):
                continue
            x[members] += shift
            placed[members] = True
            moved.extend(ids[int(t)] for t in members)
            pending.remove(c)
            progress = True
    for c in pending:
        members = np.flatnonzero(label == c)
        if len(members) < 2:
            continue
        inside = set(members.tolist())
        neighbours = [int(b if a in inside else a) for a, b in temporal.tolist() if (a in inside) != (b in inside)]
        neighbours = [t for t in neighbours if placed[t]]
        target = float(np.median(x[neighbours])) if neighbours else float(np.median(x[:n][placed]))
        x[members] += target - float(np.median(x[members]))
    return moved


#: Log ratio at which an overlap's weight in the solve falls to 1/sqrt(2): about 7x brighter on one side.
_LEVERAGE_LOG_RATIO = 2.0

#: Largest median disagreement (log units) among a detached set's own overlaps for its level to be taken from them.
_BLOCK_SPREAD = 0.5


def solve_tile_gains(measurements: Sequence[OverlapMeasurement], tile_ids: Iterable[int],
                     capture_order: Sequence[int] | None = None, temporal_weight: float = 0.01,
                     iterations: int = 12, reject_sigmas: float = 4.0, min_scale: float = 0.01,
                     temporal_scale: float = 0.02, skipped: int = 0,
                     positions: Mapping[int, tuple[float, float]] | None = None, trend_degree: int = 2) -> TileGainResult:
    """
    Solve log gains and the two shading constants from overlap log-ratios (robust, iteratively reweighted).

    A shading constant per direction and a gain ramp across columns (or rows) give identical overlap ratios, so the
    data cannot choose between them: that mode needs a convention. With ``positions`` (tile centres, (Y, X)) the
    convention is that **the correction leaves the section's smooth large-scale brightness as captured**: the log gains
    have no trend of degree ``trend_degree`` over the tile centres (1: planar, the minimum that fixes the mode; 2, the
    default: also quadratic). The gains correct tile-to-tile differences and seams (a resumed pass, RC2's white
    stripe); the large-scale level is left to display contrast. Why degree 2: chaining overlap ratios across ~30
    columns integrates any small systematic bias in the ratios (RC2 1145: the left-right ratio drifts 0.2% per
    column) into a bowl that is not in the image (clean sections came out 15-20% brighter in the centre with degree 1).
    A step near the centre projects slightly onto the planar term, leaving about slope x pitch of it (~1% on a
    32-column RC2 section). Alternatives tried and rejected (RC2 1140-1190, 2026-11-05): a flat-field estimate of the
    constants inside the tiles is biased by large-scale tissue gradients; capture-time continuity spreads a resumed
    pass's step into a ramp. Without ``positions`` the flat-field estimate (:func:`estimate_shading`) is used.

    ``capture_order`` lists tile ids in capture order. Consecutive tiles are tied with ``temporal_weight`` (a weak
    prior; overlaps decide wherever they exist), reweighted with a redescending (Cauchy) weight of scale
    ``temporal_scale`` so a pass that resumes at another brightness stays a step. The ties give tiles without usable
    overlaps (blank resin, section edge) and separate blocks a gain from their capture-time neighbours; without
    ``capture_order`` such tiles get gain 1. Overlaps whose residual exceeds ``reject_sigmas`` robust sigmas (at least
    ``min_scale``) are dropped and the solve repeated; a tile that loses half or more of its overlaps (at least two)
    loses all of them and takes its gain from the capture-time ties. The median gain of the tiles with usable overlaps
    is anchored at 1.
    """
    ids = sorted(set(tile_ids))
    index = {t: k for k, t in enumerate(ids)}
    n = len(ids)
    directions = [LEFT_RIGHT, TOP_BOTTOM]
    meas = [m for m in measurements if m.tile_i in index and m.tile_j in index]
    d = np.array([m.log_ratio for m in meas], dtype=np.float64)
    i_idx = np.array([index[m.tile_i] for m in meas], dtype=np.int64)
    j_idx = np.array([index[m.tile_j] for m in meas], dtype=np.int64)
    c_idx = np.array([n + directions.index(m.direction) for m in meas], dtype=np.int64)
    weight = np.ones(len(meas))
    unknowns = n + len(directions)

    temporal = np.zeros((0, 2), dtype=np.int64)
    if capture_order is not None:
        ordered = [index[t] for t in capture_order if t in index]
        temporal = np.array(list(itertools.pairwise(ordered)), dtype=np.int64).reshape(-1, 2)
    temporal_w = np.ones(len(temporal))

    # Convention rows for the shading/ramp mode (weight 100: exact in practice).
    if positions is not None:
        centre = np.array([positions[t] for t in ids], dtype=np.float64)
        centre -= centre.mean(axis=0)
        span = np.maximum(np.abs(centre).max(axis=0), 1.0)
        u, v = centre[:, 1] / span[1], centre[:, 0] / span[0]
        trend_terms = [u, v]
        if trend_degree >= 2:
            trend_terms += [u * u - np.mean(u * u), v * v - np.mean(v * v), u * v]
        mode_rows = [np.concatenate([term, [0.0, 0.0]]) for term in trend_terms]
        mode_rhs = [0.0] * len(mode_rows)
    else:
        flat_field = estimate_shading(meas)
        mode_rows = [np.eye(unknowns)[n + k] for k in range(len(directions))]
        mode_rhs = [flat_field[direction] for direction in directions]

    x = np.zeros(unknowns)
    for _ in range(max(1, iterations)):
        active = np.flatnonzero(weight > 0)
        n_pair, n_temp = len(active), len(temporal)
        tw = temporal_weight * temporal_w
        rows = [np.repeat(np.arange(n_pair), 3), np.repeat(np.arange(n_temp) + n_pair, 2)]
        cols = [np.stack([j_idx[active], i_idx[active], c_idx[active]], axis=1).ravel(), temporal[:, ::-1].ravel()]
        # A large log-ratio (a dim tile against a bright one) is weighed down: the dim tile's strips then vary far
        # more from side to side than a single gain explains, and a few such overlaps must not drag its bright
        # neighbours away from their own agreeing overlaps (RC1 120: neighbours of a dying beam came out at 2.4).
        leverage = 1.0 / np.sqrt(1.0 + (d[active] / _LEVERAGE_LOG_RATIO) ** 2)
        vals = [np.tile([1.0, -1.0, 1.0], n_pair) * np.repeat(leverage, 3), np.stack([tw, -tw], axis=1).ravel()]
        rhs = [d[active] * leverage, np.zeros(n_temp)]
        r = n_pair + n_temp
        # Gauge (sum of log gains 0; the median anchor is applied afterwards) and the mode convention, both over the
        # tiles that have overlaps only: a row that includes a tile without overlaps is met by moving that tile
        # instead (RC2 1226: such a tile came out at gain 4e-16). A tiny ridge keeps untouched tiles at 0.
        # Solved directly: n is a few thousand at most.
        covered = np.zeros(unknowns)
        covered[np.unique(np.concatenate([i_idx[active], j_idx[active]]))] = 1.0
        gauge = covered.copy()
        covered[n:] = 1.0
        dense_rows = [gauge] + [10.0 * row * covered for row in mode_rows]
        dense_rhs = [0.0] + [10.0 * v for v in mode_rhs]
        for row, value in zip(dense_rows, dense_rhs):
            nz = np.flatnonzero(row)
            rows.append(np.full(len(nz), r))
            cols.append(nz)
            vals.append(row[nz])
            rhs.append(np.array([value]))
            r += 1
        rows.append(r + np.arange(n))
        cols.append(np.arange(n))
        vals.append(np.full(n, 1e-4))
        rhs.append(np.zeros(n))
        r += n
        A = scipy.sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                    shape=(r, unknowns))
        x = scipy.sparse.linalg.spsolve((A.T @ A).tocsc(), A.T @ np.concatenate(rhs))

        residual = d - (x[j_idx] - x[i_idx] + x[c_idx]) if len(meas) else np.zeros(0)
        if not len(active):
            break
        scale = max(min_scale, 1.4826 * float(np.median(np.abs(residual[active]))))
        new_weight = (np.abs(residual) <= reject_sigmas * scale).astype(np.float64)
        # A tile with half or more of its overlaps rejected (at least two) is itself suspect (misplaced, foreign
        # tissue): drop all its overlaps, or the ones that slipped under the threshold keep bending its neighbours.
        # A rejected overlap is blamed on the tile with fewer agreeing overlaps: a tile whose other neighbours agree
        # is not suspect because a dim or misplaced neighbour disagrees (RC1 120: bright tiles beside a dying beam lost
        # all their good overlaps and were left at a gain of 2.4).
        both = np.concatenate([i_idx, j_idx])
        rejected = np.bincount(both, weights=np.tile(1.0 - new_weight, 2), minlength=n)
        total = np.bincount(both, minlength=n)
        agreeing = total - rejected
        pair_rejected = 1.0 - new_weight
        blamed = (np.bincount(i_idx, weights=pair_rejected * (agreeing[i_idx] <= agreeing[j_idx]), minlength=n) +
                  np.bincount(j_idx, weights=pair_rejected * (agreeing[j_idx] <= agreeing[i_idx]), minlength=n))
        suspect = (blamed >= 2) & (blamed >= total / 2)
        new_weight[suspect[i_idx] | suspect[j_idx]] = 0.0
        steps = x[temporal[:, 1]] - x[temporal[:, 0]]
        new_temporal_w = 1.0 / np.sqrt(1.0 + (steps / temporal_scale) ** 2)
        if np.array_equal(new_weight, weight) and np.allclose(new_temporal_w, temporal_w, atol=1e-3):
            break
        weight, temporal_w = new_weight, new_temporal_w

    used = weight > 0
    with_pairs = set(i_idx[used].tolist()) | set(j_idx[used].tolist())
    anchor_set = sorted(with_pairs) or list(range(n))
    estimated: list[int] = []
    if with_pairs:
        estimated = _place_detached_blocks(x, n, ids, i_idx, j_idx, c_idx, d, weight, temporal)
        main = _largest_component(n, i_idx[used], j_idx[used])
        anchor = float(np.median(x[:n][main]))
    else:
        anchor = float(np.median(x[:n][anchor_set]))
    log_gain = x[:n] - anchor
    residual = d - (x[j_idx] - x[i_idx] + x[c_idx]) if len(meas) else np.zeros(0)
    return TileGainResult(
        gains={t: float(np.exp(log_gain[index[t]])) for t in ids},
        shading={direction: float(x[n + k]) for k, direction in enumerate(directions)},
        pairs_used=int(used.sum()),
        pairs_rejected=int((~used).sum()),
        pairs_skipped=int(skipped),
        residual_mad=float(np.median(np.abs(residual[used]))) if used.any() else float('nan'),
        tiles_without_pairs=[t for t in ids if index[t] not in with_pairs],
        measurements=list(meas),
        estimated_tiles=estimated)


def compute_tile_gains(placements: Mapping[int, TilePlacement], read_tile: Callable[[int], NDArray], downsample: float,
                       low: float, high: float, capture_order: Sequence[int] | None = None,
                       **measure_kwargs) -> TileGainResult:
    """Measure the overlaps of a mosaic's tiles and solve their gains (see the module docstring)."""
    measurements, skipped = measure_overlaps(placements, read_tile, downsample, low, high, **measure_kwargs)
    centres = {t: (p.y + p.height / 2, p.x + p.width / 2) for t, p in placements.items()}
    return solve_tile_gains(measurements, placements.keys(),
                            capture_order=capture_order if capture_order is not None else sorted(placements),
                            skipped=skipped, positions=centres)


#: Chunk key shared with noise_crush (``nornir_crush`` tEXt chunk of crushed tiles).
CRUSH_PNG_TEXT_KEY = 'nornir_crush'
_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def _find_crush_chunk(f, path: str) -> tuple[int, int, dict] | None:
    """
    Offset, data length and parsed JSON of the ``nornir_crush`` chunk of the open PNG ``f``, or None without one.

    Only the chunks before the first ``IDAT`` are read: a few hundred bytes per tile, not its pixel data.
    """
    if f.read(len(_PNG_SIGNATURE)) != _PNG_SIGNATURE:
        raise ValueError(f'Not a PNG file: {path}')
    pos = len(_PNG_SIGNATURE)
    while True:
        header = f.read(8)
        if len(header) < 8:
            return None
        length, chunk_type = struct.unpack('>I4s', header)
        if chunk_type == b'IDAT':
            return None
        if chunk_type == b'tEXt':
            body = f.read(length)
            f.seek(4, os.SEEK_CUR)
            key, _, text = body.partition(b'\0')
            if key == CRUSH_PNG_TEXT_KEY.encode():
                return pos, length, json.loads(text.decode('latin-1'))
        else:
            f.seek(length + 4, os.SEEK_CUR)
        pos += 12 + length


def read_gain_png(path: str) -> float | None:
    """The gain in a tile's ``nornir_crush`` chunk (header-only read), or None when it has no chunk or no gain yet."""
    with open(path, 'rb') as f:
        found = _find_crush_chunk(f, path)
    if found is None or 'gain' not in found[2]:
        return None
    return float(found[2]['gain'])


def stamp_gain_png(path: str, gain: float) -> bool:
    """
    Write a tile's gain into its ``nornir_crush`` chunk without decoding its pixels.

    Adds ``gain``, ``effective_step`` (``step / gain``) and ``effective_offset`` (``(step / 2 + offset) / gain``) to the
    chunk's JSON, so ``codes * effective_step + effective_offset`` are gain-corrected counts. Every other chunk is
    copied byte for byte and the file is replaced atomically. Returns False (file untouched) when the PNG has no
    ``nornir_crush`` chunk or already carries this gain. Only the chunks before the first ``IDAT`` are read to decide
    that, so a rerun with unchanged gains reads a few hundred bytes per tile, not its pixel data. A changed gain still
    copies the rest of the file once (streamed). Host-only: file I/O.
    """
    with open(path, 'rb') as f:
        found = _find_crush_chunk(f, path)
        if found is None:
            return False
        pos, length, values = found
        rounded = round(float(gain), 6)
        if values.get('gain') == rounded:
            return False
        values['gain'] = rounded
        values['effective_step'] = values['step'] / rounded
        values['effective_offset'] = (values['step'] / 2.0 + values['offset']) / rounded
        text = b'tEXt' + CRUSH_PNG_TEXT_KEY.encode('latin-1') + b'\0' + \
            json.dumps(values, sort_keys=True).encode('latin-1')
        chunk = struct.pack('>I', len(text) - 4) + text + struct.pack('>I', zlib.crc32(text) & 0xFFFFFFFF)
        temp_path = path + '.gainstamp'
        with open(temp_path, 'wb') as out:
            f.seek(0)
            out.write(f.read(pos))
            out.write(chunk)
            f.seek(pos + 12 + length)
            shutil.copyfileobj(f, out, 1 << 20)
    # Replace after the source is closed: Windows cannot replace an open file.
    os.replace(temp_path, path)
    return True
