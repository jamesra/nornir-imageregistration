"""Map points from one mosaic's coordinate frame to another's through the tiles both mosaics contain.

When a section's mosaic is rebuilt (tiles re-pruned, re-translated, re-refined) every tile keeps its pixels but
moves. Annotations are stored in mosaic coordinates, so each must follow the tissue: a point at old mosaic
position p, lying in tile t, is at tile pixel s = old_t.InverseTransform(p), and at new_t.Transform(s) in the new
mosaic. No tissue matching is involved; the tile images are the same in both.

Where tiles overlap, a point can lie in several tiles, and the old mosaic displays only one of them: Assemble
composites with the tile whose centre is closest. The same rule picks the tile here, so an annotation follows the
tile it was drawn on.

Transform direction in this library: ``Transform`` maps tile (mapped) pixels to mosaic (fixed) space,
``InverseTransform`` maps mosaic points to tile pixels. Points are (Y, X), full-resolution pixels.
"""
from __future__ import annotations

import dataclasses
import enum

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration

DEFAULT_TILE_SHAPE = (4080.0, 4080.0)


class RemapStatus(enum.IntEnum):
    """How a point was mapped."""
    MAPPED = 0  # inside the image of the tile nearest its centre among the shared tiles that contain it
    EXTRAPOLATED = 1  # inside a shared tile's outline, but off its image (warp edge); mapped through the nearest-centre tile
    NO_SHARED_TILE = 2  # no tile present in both mosaics covers the point; new position is NaN


@dataclasses.dataclass
class RemapResult:
    """Result of remapping N points. Arrays have one row per input point, in input order."""
    new_points: NDArray[np.floating]  # (N, 2) (Y, X); NaN where status is NO_SHARED_TILE
    tile_names: list[str | None]
    status: NDArray[np.integer]
    shift: NDArray[np.floating]  # distance moved, pixels; NaN where unmapped
    #: Where a second shared tile also contains the point, how far apart the two tiles put it in the NEW mosaic.
    #: Zero when the overlapping tiles still agree about the tissue; large where the old mosaic misregistered the
    #: overlap (an annotation drawn there is on tissue the old mosaic showed twice, at two places). NaN if only one tile.
    overlap_disagreement: NDArray[np.floating] = dataclasses.field(default_factory=lambda: np.empty(0))

    def summary(self) -> dict[str, float | int]:
        mapped = self.status != RemapStatus.NO_SHARED_TILE
        moved = self.shift[mapped]
        return {
            "points": len(self.status),
            "mapped": int((self.status == RemapStatus.MAPPED).sum()),
            "extrapolated": int((self.status == RemapStatus.EXTRAPOLATED).sum()),
            "no_shared_tile": int((~mapped).sum()),
            "shift_median": float(np.median(moved)) if moved.size else float("nan"),
            "shift_p95": float(np.percentile(moved, 95)) if moved.size else float("nan"),
            "shift_max": float(moved.max()) if moved.size else float("nan"),
            "in_overlap": int(np.isfinite(self.overlap_disagreement).sum()),
            "overlap_disagree_gt5px": int((self.overlap_disagreement > 5.0).sum()),
            "overlap_disagree_p95": (float(np.nanpercentile(self.overlap_disagreement, 95))
                                     if np.isfinite(self.overlap_disagreement).any() else float("nan")),
        }


def _transforms(mosaic: nornir_imageregistration.Mosaic | str) -> dict:
    if isinstance(mosaic, str):
        mosaic = nornir_imageregistration.Mosaic.LoadFromMosaicFile(mosaic)
    return mosaic.ImageToTransform


class MosaicPointRemapper:
    """Maps points old mosaic -> new mosaic through the tiles both contain."""

    def __init__(self, old: nornir_imageregistration.Mosaic | str, new: nornir_imageregistration.Mosaic | str,
                 tile_shape: tuple[float, float] | None = None, margin: float = 2.0):
        """
        :param tile_shape: (height, width) of the tile images in the pixel units of the transforms. Default: read
            from the old transform's mapped bounding box where it has one, else 4080 x 4080.
        :param margin: pixels of slack when testing whether a mapped point is inside a tile image.
        """
        old_t, new_t = _transforms(old), _transforms(new)
        self.names = sorted(set(old_t) & set(new_t))
        if not self.names:
            raise ValueError("The two mosaics share no tile")
        self._old = [old_t[n] for n in self.names]
        self._new = [new_t[n] for n in self.names]
        self._margin = float(margin)
        self._shapes = np.array([self._shape_of(t, tile_shape) for t in self._old], dtype=np.float64)
        # Outline of every tile in the OLD mosaic: its image corners and edge midpoints mapped to mosaic space.
        self._lo = np.empty((len(self.names), 2))
        self._hi = np.empty((len(self.names), 2))
        self._centre = np.empty((len(self.names), 2))
        for i, transform in enumerate(self._old):
            h, w = self._shapes[i]
            probe = np.array([[0, 0], [0, w / 2], [0, w], [h / 2, 0], [h / 2, w / 2], [h / 2, w], [h, 0], [h, w / 2], [h, w]],
                             dtype=np.float64)
            mapped = np.asarray(transform.Transform(probe), dtype=np.float64)
            self._lo[i], self._hi[i] = mapped.min(axis=0), mapped.max(axis=0)
            self._centre[i] = mapped[4]

    @staticmethod
    def _shape_of(transform, tile_shape: tuple[float, float] | None) -> tuple[float, float]:
        if tile_shape is not None:
            return float(tile_shape[0]), float(tile_shape[1])
        try:
            box = transform.MappedBoundingBox  # translation-only transforms carry no bounding box
        except (AttributeError, NotImplementedError, ValueError):
            return DEFAULT_TILE_SHAPE
        height, width = float(box.MaxY - box.MinY), float(box.MaxX - box.MinX)
        if height > 0 and width > 0:
            return height, width
        return DEFAULT_TILE_SHAPE

    def remap(self, points: NDArray[np.floating]) -> RemapResult:
        """Map (N, 2) points in (Y, X) order from the old mosaic's frame to the new mosaic's."""
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        count = len(points)
        new_points = np.full((count, 2), np.nan)
        tile_index = np.full(count, -1, dtype=np.int64)
        status = np.full(count, int(RemapStatus.NO_SHARED_TILE), dtype=np.int8)

        # Candidates: every shared tile whose old outline contains the point, nearest centre first.
        candidates: list[NDArray[np.integer]] = []
        for p in points:
            inside = np.flatnonzero((p >= self._lo).all(axis=1) & (p <= self._hi).all(axis=1))
            candidates.append(inside[np.argsort(np.linalg.norm(self._centre[inside] - p, axis=1))])

        unresolved = np.array([i for i in range(count) if len(candidates[i])], dtype=np.int64)
        fallback: dict[int, int] = {i: int(candidates[i][0]) for i in unresolved}
        rank = 0
        while len(unresolved):
            by_tile: dict[int, list[int]] = {}
            for i in unresolved:
                if rank < len(candidates[i]):
                    by_tile.setdefault(int(candidates[i][rank]), []).append(int(i))
            if not by_tile:
                break
            accepted: set[int] = set()
            for tile, rows in by_tile.items():
                rows_array = np.array(rows)
                pixel = np.asarray(self._old[tile].InverseTransform(points[rows_array]), dtype=np.float64)
                h, w = self._shapes[tile]
                ok = ((pixel >= -self._margin).all(axis=1) & (pixel[:, 0] <= h + self._margin) & (pixel[:, 1] <= w + self._margin))
                if ok.any():
                    new_points[rows_array[ok]] = np.asarray(self._new[tile].Transform(pixel[ok]), dtype=np.float64)
                    tile_index[rows_array[ok]] = tile
                    status[rows_array[ok]] = int(RemapStatus.MAPPED)
                    accepted.update(int(r) for r in rows_array[ok])
            unresolved = np.array([i for i in unresolved if int(i) not in accepted and rank + 1 < len(candidates[i])], dtype=np.int64)
            rank += 1

        # Inside an outline but off every tile image: extrapolate through the nearest-centre tile.
        for i, tile in fallback.items():
            if status[i] == int(RemapStatus.NO_SHARED_TILE):
                pixel = np.asarray(self._old[tile].InverseTransform(points[i:i + 1]), dtype=np.float64)
                new_points[i] = np.asarray(self._new[tile].Transform(pixel), dtype=np.float64)[0]
                tile_index[i] = tile
                status[i] = int(RemapStatus.EXTRAPOLATED)

        shift = np.linalg.norm(new_points - points, axis=1)
        return RemapResult(new_points=new_points,
                           tile_names=[self.names[t] if t >= 0 else None for t in tile_index],
                           status=status, shift=shift,
                           overlap_disagreement=self._overlap_disagreement(points, new_points, tile_index, candidates))

    def _overlap_disagreement(self, points: NDArray[np.floating], new_points: NDArray[np.floating],
                              tile_index: NDArray[np.integer], candidates: list[NDArray[np.integer]]) -> NDArray[np.floating]:
        """Distance between the new position through the chosen tile and through the next tile that holds the point."""
        out = np.full(len(points), np.nan)
        by_tile: dict[int, list[int]] = {}
        for i, options in enumerate(candidates):
            if tile_index[i] < 0:
                continue
            others = [int(t) for t in options if int(t) != int(tile_index[i])]
            if others:
                by_tile.setdefault(others[0], []).append(i)
        for tile, rows in by_tile.items():
            rows_array = np.array(rows)
            pixel = np.asarray(self._old[tile].InverseTransform(points[rows_array]), dtype=np.float64)
            h, w = self._shapes[tile]
            ok = ((pixel >= -self._margin).all(axis=1) & (pixel[:, 0] <= h + self._margin) & (pixel[:, 1] <= w + self._margin))
            if ok.any():
                alt = np.asarray(self._new[tile].Transform(pixel[ok]), dtype=np.float64)
                out[rows_array[ok]] = np.linalg.norm(alt - new_points[rows_array[ok]], axis=1)
        return out
