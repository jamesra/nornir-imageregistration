"""Translate-step report: tile groups, failed pairs, manual offsets used, and a join check.

Translate assembles the mosaic from tile-pair offsets. When stage positions are wrong by more than
the pair search covers, the pairs across a join fail, or are measured at a false peak and kept, and
a block of tiles is attached in the wrong place: on RC3 1637 a 300-tile block was left at its
stage offset, its texture not continuing across the join, and nothing said so. This module records
how the layout was assembled and checks from the images that the final mosaic holds together, so a
bad join is flagged for review instead of shipping silently.

The join check is host-only: it reads tile images from disk and computes NCC with NumPy.

An overlap "matches" when its NCC is significantly above zero. Neighbouring pixels are correlated,
so the test uses the effective sample size of Clifford, Richardson & Hemon (1989, Biometrics 45:123)
rather than the pixel count, and the overlaps of a section are tested together at a false discovery
rate (Benjamini & Hochberg 1995, J. R. Stat. Soc. B 57:289). See docs/publication_notes.md.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterable
from typing import Any

import nornir_pools
import numpy as np
from numpy.typing import NDArray
from scipy import stats

import nornir_imageregistration
from nornir_imageregistration.settings.mosaic_tile_offset import TileOffset

REPORT_VERSION = 2

# Groups smaller than this are reported but never flagged. A blank resin tile or a corner tile
# with too little texture routinely ends up alone; flagging those would bury the joins that matter.
MIN_FLAGGED_GROUP_TILES = 3

# False discovery rate for calling overlaps matched: the conventional 5%, applied over all of a
# section's overlaps (Benjamini-Hochberg). This replaced "NCC at least half the section median"
# (2026-11-19), which left false seams in good sections: RC2 0400 and 0800 each had 6-9 unmatched
# overlaps whose tiles shared no matched neighbour; with the significance test they have none.
MATCH_FDR = 0.05

# Overlap crops are block-averaged by this factor before scoring. NCC only has to tell
# continuing texture from unrelated texture, and the reduction keeps every crop of a large
# section in memory at once (1637 at downsample 4: ~6,000 crops, ~150 MB).
SCORE_DOWNSAMPLE = 4

# Overlaps smaller than this many pixels after reduction are not scored.
MIN_SCORED_PIXELS = 64

# A join is flagged only on at least this many scored overlaps; one weak overlap between two
# small groups is as likely to be low texture as a misplacement.
MIN_FLAGGED_JOIN_PAIRS = 2

# A matched group of at least this many tiles, besides the largest, is a block that does not hold
# together with the rest even when no overlap joins them directly (RC3 1637: the two halves are
# separated by a strip of poorly matched tiles). RC3 0800 has none: 849 of 853 tiles in one group.
MIN_SECONDARY_BLOCK_TILES = 10


@dataclasses.dataclass
class FailedPair:
    """A tile pair whose offset could not be measured on the last pass that tried it."""
    A: int
    B: int
    reason: str


@dataclasses.dataclass
class JoinCheck:
    """Overlaps between two matched groups: by construction none of them match."""
    groups: tuple[int, int]  # indices into TranslateReport.matched_groups
    group_sizes: tuple[int, int]
    pairs: int
    scored_pairs: int
    ncc_median: float | None  # None when every overlap was blank on one side
    flagged: bool


@dataclasses.dataclass
class SeamCheck:
    """Unmatched overlaps inside one matched group whose tiles share no matched neighbour.

    Where tiles are placed right, two overlapping tiles nearly always both match some third tile,
    even when their own overlap is too bland to score: every unmatched overlap in RC2 0400 and 0800
    (700 of them) has such a neighbour. A row or column shifted against the rest breaks that along
    a line: RC1_Crush 4's column misjoin left 8 overlaps without one.
    """
    tiles: list[int]
    pairs: list[tuple[int, int]]
    ncc_median: float
    flagged: bool


@dataclasses.dataclass
class TranslateReport:
    """Everything translate knows about how a section's layout was assembled.

    groups are translate's own: tiles linked by measured or known offsets, each group attached to
    the rest by stage offsets. matched_groups come from the images on the final layout: tiles
    linked by overlaps whose texture continues. A large block attached by overlaps that do not
    match is its own matched group however translate grouped it.
    """
    tile_count: int = 0
    image_to_source_space_scale: float = 1.0
    passes: int = 0
    groups: list[list[int]] = dataclasses.field(default_factory=list)
    failed_pairs: list[FailedPair] = dataclasses.field(default_factory=list)
    known_offsets: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    known_offsets_skipped: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    scored_pairs: int = 0
    median_ncc: float | None = None
    matched_groups: list[list[int]] = dataclasses.field(default_factory=list)
    joins: list[JoinCheck] = dataclasses.field(default_factory=list)
    seams: list[SeamCheck] = dataclasses.field(default_factory=list)
    review_reasons: list[str] = dataclasses.field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return len(self.review_reasons) > 0

    @property
    def secondary_blocks(self) -> list[list[int]]:
        """Matched groups of MIN_SECONDARY_BLOCK_TILES or more, other than the largest, largest first."""
        return [g for g in self.matched_groups[1:] if len(g) >= MIN_SECONDARY_BLOCK_TILES]

    def set_known_offsets(self, used: Iterable[TileOffset], skipped: Iterable[tuple[TileOffset, str]]) -> None:
        """Record the manual offsets translate applied, and the ones it could not, with why."""
        self.known_offsets = [_tile_offset_to_dict(o) for o in used]
        self.known_offsets_skipped = [dict(_tile_offset_to_dict(o), reason=reason) for o, reason in skipped]

    def check_joins(self, tileset: nornir_imageregistration.mosaic_tileset.MosaicTileset,
                    min_overlap: float | None = None,
                    exclude_diagonal_overlaps: bool = True,
                    score_downsample: int = SCORE_DOWNSAMPLE) -> None:
        """Check on the final tile positions that the mosaic holds together, and set review_reasons.

        Scores every overlap by NCC, links tiles through overlaps that match, and flags any two
        sizeable matched groups that overlap only where the texture does not continue. Also flags
        a translate group that overlaps no other tile, since only its stage position placed it.

        :param tileset: The translated tileset; tiles must carry image paths at the translate level.
        """
        overlaps = list(nornir_imageregistration.tile_overlap.IterateTileOverlaps(
            tileset, image_to_source_space_scale=tileset.image_to_source_space_scale,
            min_overlap=min_overlap, exclude_diagonal_overlaps=exclude_diagonal_overlaps))
        scores, p_values = _score_overlaps(overlaps, score_downsample)
        finite = scores[np.isfinite(scores)]
        self.scored_pairs = int(finite.size)
        self.median_ncc = float(np.median(finite)) if finite.size else None
        self.joins = []
        self.seams = []
        self.review_reasons = []

        matched = _benjamini_hochberg(p_values, MATCH_FDR)
        tile_ids = sorted(int(i) for i in tileset)
        self.matched_groups = _matched_groups(tile_ids, overlaps, matched)
        tile_to_group = {tile_id: i for i, group in enumerate(self.matched_groups) for tile_id in group}

        cross: dict[tuple[int, int], list[float]] = {}
        for overlap, score in zip(overlaps, scores):
            group_a, group_b = tile_to_group[int(overlap.A.ID)], tile_to_group[int(overlap.B.ID)]
            if group_a != group_b:
                cross.setdefault((min(group_a, group_b), max(group_a, group_b)), []).append(float(score))

        for (group_a, group_b), join_scores in sorted(cross.items()):
            join_finite = [s for s in join_scores if math.isfinite(s)]
            median = float(np.median(join_finite)) if join_finite else None
            sizes = (len(self.matched_groups[group_a]), len(self.matched_groups[group_b]))
            flagged = len(join_finite) >= MIN_FLAGGED_JOIN_PAIRS and min(sizes) >= MIN_FLAGGED_GROUP_TILES
            self.joins.append(JoinCheck(groups=(group_a, group_b), group_sizes=sizes, pairs=len(join_scores),
                                        scored_pairs=len(join_finite), ncc_median=median, flagged=flagged))

        self.seams = _find_seams(overlaps, scores, matched, tile_to_group)

        if self.secondary_blocks:
            self.review_reasons.append(
                f"Besides its main block of {len(self.matched_groups[0])} tiles, the mosaic has blocks of "
                f"{', '.join(str(len(g)) for g in self.secondary_blocks)} tiles that no matching overlap "
                f"connects to it.")

        flagged_joins = sorted((j for j in self.joins if j.flagged), key=lambda j: -min(j.group_sizes))
        if flagged_joins:
            blocks = sorted({g for j in flagged_joins for g in j.groups})
            worst = flagged_joins[0]
            self.review_reasons.append(
                f"The mosaic splits into blocks that overlap only where their texture does not continue "
                f"(block sizes {', '.join(str(len(self.matched_groups[g])) for g in blocks)} tiles). Largest "
                f"join: {worst.group_sizes[0]} and {worst.group_sizes[1]} tiles, median NCC "
                f"{worst.ncc_median:.3f} across {worst.scored_pairs} overlaps, against {self.median_ncc:.3f} "
                f"for the section. A block is probably misplaced; see joins in the report.")

        flagged_seams = sorted((s for s in self.seams if s.flagged), key=lambda s: -len(s.pairs))
        if flagged_seams:
            worst = flagged_seams[0]
            examples = ', '.join(f"{a}-{b}" for a, b in worst.pairs[:6])
            self.review_reasons.append(
                f"{sum(len(s.pairs) for s in flagged_seams)} overlaps along "
                f"{len(flagged_seams)} line(s) do not match although both of their tiles match the tiles "
                f"around them, so a row or column of tiles is probably shifted against its neighbours "
                f"(largest: {len(worst.pairs)} overlaps, median NCC {worst.ncc_median:.3f}, e.g. {examples}). "
                f"See seams in the report.")

        if len(self.groups) > 1:
            # The largest group is the reference the others were placed against; "overlaps no
            # other tile" only means something for the rest (RC3 0633: an 848-tile main group was
            # flagged because the one stray tile beside it did not overlap it).
            largest = max(self.groups, key=len)
            for group in self.groups:
                if group is largest:
                    continue
                members = set(group)
                touches_others = any((int(o.A.ID) in members) != (int(o.B.ID) in members) for o in overlaps)
                if len(group) >= MIN_FLAGGED_GROUP_TILES and not touches_others:
                    self.review_reasons.append(
                        f"A group of {len(group)} tiles overlaps no other tile, so only its stage "
                        f"position placed it and nothing can check it.")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data['version'] = REPORT_VERSION
        data['needs_review'] = self.needs_review
        return data

    def save(self, path: str) -> None:
        """Write the report as JSON."""
        with open(path, 'w', encoding='utf-8') as report_file:
            json.dump(self.to_dict(), report_file, indent=1)


def load_known_offsets(path: str) -> list[TileOffset]:
    """Return the manual offsets a saved report says translate applied."""
    with open(path, encoding='utf-8') as report_file:
        data = json.load(report_file)
    return [TileOffset(A=int(o['A']), B=int(o['B']), Y=float(o['Y']), X=float(o['X']), Comment=o.get('comment'))
            for o in data.get('known_offsets', [])]


def _tile_offset_to_dict(offset: TileOffset) -> dict[str, Any]:
    return {'A': int(offset.A), 'B': int(offset.B), 'Y': float(offset.Y), 'X': float(offset.X),
            'comment': offset.Comment}


def _union_find_groups(tile_ids: Iterable[int], links: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Connected components of *tile_ids* under *links*, largest first."""
    parent = {tile_id: tile_id for tile_id in tile_ids}

    def root(tile_id: int) -> int:
        while parent[tile_id] != tile_id:
            parent[tile_id] = parent[parent[tile_id]]
            tile_id = parent[tile_id]
        return tile_id

    for a, b in links:
        parent[root(a)] = root(b)

    groups: dict[int, list[int]] = {}
    for tile_id in parent:
        groups.setdefault(root(tile_id), []).append(tile_id)
    return sorted((sorted(g) for g in groups.values()), key=lambda group: (-len(group), group[0]))


def _matched_groups(tile_ids: list[int], overlaps: list[Any], matched: NDArray[np.bool_]) -> list[list[int]]:
    """Connected components of tiles linked by matching overlaps, largest first."""
    return _union_find_groups(tile_ids, ((int(o.A.ID), int(o.B.ID)) for o, m in zip(overlaps, matched) if m))


def _find_seams(overlaps: list[Any], scores: NDArray[np.floating], matched: NDArray[np.bool_],
                tile_to_group: dict[int, int]) -> list[SeamCheck]:
    """Group the scored, unmatched overlaps inside a matched group whose tiles share no matched neighbour.

    Overlaps that share a tile form one seam. A seam is flagged with MIN_FLAGGED_JOIN_PAIRS or more
    overlaps, for the same reason as a join: one bland overlap is as likely to be low texture.
    """
    neighbours: dict[int, set[int]] = {}
    for overlap, m in zip(overlaps, matched):
        if m:
            a, b = int(overlap.A.ID), int(overlap.B.ID)
            neighbours.setdefault(a, set()).add(b)
            neighbours.setdefault(b, set()).add(a)

    seam_pairs: dict[tuple[int, int], float] = {}
    for overlap, score, m in zip(overlaps, scores, matched):
        a, b = int(overlap.A.ID), int(overlap.B.ID)
        if m or not math.isfinite(score) or tile_to_group[a] != tile_to_group[b]:
            continue
        around_a, around_b = neighbours.get(a, set()), neighbours.get(b, set())
        if around_a and around_b and not (around_a & around_b):
            seam_pairs[(min(a, b), max(a, b))] = float(score)

    seam_tiles = {t for pair in seam_pairs for t in pair}
    seams = []
    for tiles in _union_find_groups(seam_tiles, seam_pairs):
        members = set(tiles)
        pairs = sorted(p for p in seam_pairs if p[0] in members)
        seams.append(SeamCheck(tiles=tiles, pairs=pairs,
                               ncc_median=float(np.median([seam_pairs[p] for p in pairs])),
                               flagged=len(pairs) >= MIN_FLAGGED_JOIN_PAIRS))
    return seams


def _benjamini_hochberg(p_values: NDArray[np.floating], fdr: float) -> NDArray[np.bool_]:
    """True for the p-values called significant at false discovery rate *fdr*; NaN never is."""
    called = np.zeros(p_values.shape, dtype=bool)
    tested = np.flatnonzero(np.isfinite(p_values))
    if tested.size == 0:
        return called
    order = tested[np.argsort(p_values[tested])]
    below = p_values[order] <= fdr * np.arange(1, order.size + 1) / order.size
    if below.any():
        called[order[:np.flatnonzero(below)[-1] + 1]] = True
    return called


def _score_overlaps(overlaps: list[Any], downsample: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """NCC of each overlap's two image regions and its one-sided p-value against unrelated texture.

    Both are NaN where either side has no texture. Each tile image is read once, on the thread pool,
    and cut into the reduced crops of all its overlaps; the scores are then computed from the crops.
    """
    if not overlaps:
        return np.zeros(0), np.zeros(0)

    regions: dict[str, list[tuple[int, int, nornir_imageregistration.Rectangle]]] = {}
    for i, overlap in enumerate(overlaps):
        regions.setdefault(overlap.A.ImagePath, []).append((i, 0, overlap.scaled_overlapping_source_rect_A))
        regions.setdefault(overlap.B.ImagePath, []).append((i, 1, overlap.scaled_overlapping_source_rect_B))

    pool = nornir_pools.GetGlobalMultithreadingPool()
    tasks = [pool.add_task(f"Overlap crops {path}", _tile_crops, path, tile_regions, downsample)
             for path, tile_regions in regions.items()]
    crops: list[list[NDArray[np.floating] | None]] = [[None, None] for _ in overlaps]
    for task in tasks:
        for i, side, crop in task.wait_return():
            crops[i][side] = crop

    results = np.array([_ncc_significance(a, b) for a, b in crops], dtype=np.float64).reshape(-1, 2)
    return results[:, 0], results[:, 1]


def _tile_crops(path: str, regions: list[tuple[int, int, nornir_imageregistration.Rectangle]],
                downsample: int) -> list[tuple[int, int, NDArray[np.floating]]]:
    image = nornir_imageregistration.LoadImage(path, dtype=np.float32, backend='numpy')
    out = []
    for i, side, rect in regions:
        y0 = max(0, int(np.rint(rect.BottomLeft[0])))
        x0 = max(0, int(np.rint(rect.BottomLeft[1])))
        y1 = min(image.shape[0], y0 + int(np.rint(rect.Height)))
        x1 = min(image.shape[1], x0 + int(np.rint(rect.Width)))
        out.append((i, side, _block_mean(np.asarray(image[y0:y1, x0:x1]), downsample)))
    return out


def _block_mean(image: NDArray[np.floating], factor: int) -> NDArray[np.floating]:
    if factor <= 1:
        return image.astype(np.float32, copy=False)
    height, width = image.shape[0] // factor, image.shape[1] // factor
    trimmed = image[:height * factor, :width * factor]
    return trimmed.reshape(height, factor, width, factor).mean(axis=(1, 3), dtype=np.float32)


def _autocovariance_sums(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum of x[i] * x[i + k] for every lag k, from a zero-padded FFT (so lags do not wrap)."""
    shape = (2 * x.shape[0], 2 * x.shape[1])
    spectrum = np.fft.rfft2(x, shape)
    return np.fft.irfft2(spectrum * np.conj(spectrum), shape)


def _ncc_significance(a: NDArray[np.floating] | None, b: NDArray[np.floating] | None) -> tuple[float, float]:
    """NCC of two crops and its one-sided p-value under "unrelated texture"; (NaN, NaN) when unscorable.

    Clifford, Richardson & Hemon (1989): with both images autocorrelated, the variance of r under
    independence is sum_k C_a(k) C_b(k) / N_k divided by (N var_a var_b)^2, with C the autocovariance
    sums and N_k the pixel pairs at lag k; the effective sample size is 1 + 1 / var(r). r is tested
    with a t statistic on Neff - 2 degrees of freedom.
    """
    if a is None or b is None:
        return float('nan'), float('nan')
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    if height * width < MIN_SCORED_PIXELS:
        return float('nan'), float('nan')

    centred_a = a[:height, :width].astype(np.float64) - a[:height, :width].mean()
    centred_b = b[:height, :width].astype(np.float64) - b[:height, :width].mean()
    sum_aa, sum_bb = float((centred_a * centred_a).sum()), float((centred_b * centred_b).sum())
    if sum_aa <= 0 or sum_bb <= 0:
        return float('nan'), float('nan')
    r = float((centred_a * centred_b).sum() / math.sqrt(sum_aa * sum_bb))

    lag_counts = np.rint(_autocovariance_sums(np.ones_like(centred_a)))
    has_pairs = lag_counts > 0
    products = _autocovariance_sums(centred_a) * _autocovariance_sums(centred_b)
    var_r = float(np.sum(products[has_pairs] / lag_counts[has_pairs])) / (sum_aa * sum_bb)
    if not var_r > 0:
        return r, float('nan')
    dof = max(1.0 / var_r - 1.0, 1.0)  # Neff - 2
    t = r * math.sqrt(dof / max(1.0 - r * r, 1e-12))
    return r, float(stats.t.sf(t, dof))
