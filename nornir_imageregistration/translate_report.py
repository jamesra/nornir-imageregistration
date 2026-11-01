"""Translate-step report: tile groups, failed pairs, manual offsets used, and a join check.

Translate assembles the mosaic from tile-pair offsets. When stage positions are wrong by more than
the pair search covers, the pairs across a join fail, or are measured at a false peak and kept, and
a block of tiles is attached in the wrong place: on RC3 1637 a 300-tile block was left at its
stage offset, its texture not continuing across the join, and nothing said so. This module records
how the layout was assembled and checks from the images that the final mosaic holds together, so a
bad join is flagged for review instead of shipping silently.

The join check is host-only: it reads tile images from disk and computes NCC with NumPy.
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

import nornir_imageregistration
from nornir_imageregistration.settings.mosaic_tile_offset import TileOffset

REPORT_VERSION = 1

# Groups smaller than this are reported but never flagged. A blank resin tile or a corner tile
# with too little texture routinely ends up alone; flagging those would bury the joins that matter.
MIN_FLAGGED_GROUP_TILES = 3

# An overlap "matches" when its NCC is at least this fraction of the section's median overlap NCC.
# RC3 0800 (2026-11-01): median 0.75, one matched component of 849 of 853 tiles. RC3 1637: median
# ~0.50, and the stage-offset misjoin scores 0.03 across the join.
MATCH_NCC_RATIO = 0.5

# Overlap crops are block-averaged by this factor before scoring. NCC only has to tell
# continuing texture from unrelated texture, and the reduction keeps every crop of a large
# section in memory at once (1637 at downsample 4: ~6,000 crops, ~150 MB).
SCORE_DOWNSAMPLE = 4

# Overlaps smaller than this many pixels after reduction are not scored.
MIN_SCORED_PIXELS = 64

# A join is flagged only on at least this many scored overlaps; one weak overlap between two
# small groups is as likely to be low texture as a misplacement.
MIN_FLAGGED_JOIN_PAIRS = 2


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
    review_reasons: list[str] = dataclasses.field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return len(self.review_reasons) > 0

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
        scores = _score_overlaps(overlaps, score_downsample)
        finite = scores[np.isfinite(scores)]
        self.scored_pairs = int(finite.size)
        self.median_ncc = float(np.median(finite)) if finite.size else None
        self.joins = []
        self.review_reasons = []

        tile_ids = sorted(int(i) for i in tileset)
        self.matched_groups = _matched_groups(tile_ids, overlaps, scores, self.median_ncc)
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

        if len(self.groups) > 1:
            for group in self.groups:
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


def _matched_groups(tile_ids: list[int], overlaps: list[Any], scores: NDArray[np.floating],
                    median_ncc: float | None) -> list[list[int]]:
    """Connected components of tiles linked by matching overlaps, largest first."""
    parent = {tile_id: tile_id for tile_id in tile_ids}

    def root(tile_id: int) -> int:
        while parent[tile_id] != tile_id:
            parent[tile_id] = parent[parent[tile_id]]
            tile_id = parent[tile_id]
        return tile_id

    if median_ncc is not None and median_ncc > 0:
        threshold = MATCH_NCC_RATIO * median_ncc
        for overlap, score in zip(overlaps, scores):
            if math.isfinite(score) and score >= threshold:
                parent[root(int(overlap.A.ID))] = root(int(overlap.B.ID))

    groups: dict[int, list[int]] = {}
    for tile_id in tile_ids:
        groups.setdefault(root(tile_id), []).append(tile_id)
    return sorted(groups.values(), key=lambda group: (-len(group), group[0]))


def _score_overlaps(overlaps: list[Any], downsample: int) -> NDArray[np.floating]:
    """NCC of each overlap's two image regions, NaN where either side has no texture.

    Each tile image is read once, on the thread pool, and cut into the reduced crops of all its
    overlaps; the NCCs are then computed from the crops.
    """
    if not overlaps:
        return np.zeros(0)

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

    return np.array([_ncc(a, b) for a, b in crops], dtype=np.float64)


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


def _ncc(a: NDArray[np.floating] | None, b: NDArray[np.floating] | None) -> float:
    if a is None or b is None:
        return float('nan')
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    if height * width < MIN_SCORED_PIXELS:
        return float('nan')

    centred_a = a[:height, :width].astype(np.float64) - a[:height, :width].mean()
    centred_b = b[:height, :width].astype(np.float64) - b[:height, :width].mean()
    denominator = math.sqrt(float((centred_a * centred_a).sum()) * float((centred_b * centred_b).sum()))
    return float((centred_a * centred_b).sum() / denominator) if denominator > 0 else float('nan')
