"""Document why mosaic refine recomputes neighbor pairs each pass (#185)."""

from __future__ import annotations

import time
import unittest

import numpy as np

import nornir_imageregistration as nir
from nornir_imageregistration.local_distortion_correction import _grid_refine_neighbors


class _BoxTile:
    def __init__(self, tid: int, origin: tuple[float, float], size=(512.0, 512.0)) -> None:
        self.ID = tid
        self.FixedBoundingBox = nir.Rectangle.CreateFromPointAndArea(
            np.asarray(origin, dtype=np.float64),
            np.asarray(size, dtype=np.float64),
        )


class TestGridRefineNeighborsCost(unittest.TestCase):
    """O(N^2) neighbor scan is intentional; cost stays small at section scale."""

    def test_moving_boxes_change_neighbor_sets(self) -> None:
        """Caching across passes would be wrong when a tile slides into overlap."""
        a = _BoxTile(0, (0.0, 0.0))
        b = _BoxTile(1, (600.0, 0.0))  # no overlap
        before = _grid_refine_neighbors([a, b])
        self.assertEqual(before[0], [])
        b.FixedBoundingBox = nir.Rectangle.CreateFromPointAndArea(
            np.asarray((400.0, 0.0), dtype=np.float64),
            np.asarray((512.0, 512.0), dtype=np.float64),
        )
        after = _grid_refine_neighbors([a, b])
        self.assertEqual([t.ID for t in after[0]], [1])

    def test_section_scale_neighbor_scan_is_sub_20ms(self) -> None:
        cols = 12
        tiles = [
            _BoxTile(i, (float(divmod(i, cols)[0]) * 480.0, float(divmod(i, cols)[1]) * 480.0))
            for i in range(130)
        ]
        t0 = time.perf_counter()
        for _ in range(20):
            _grid_refine_neighbors(tiles)
        per_call_ms = (time.perf_counter() - t0) / 20 * 1000.0
        self.assertLess(per_call_ms, 20.0, f"neighbor scan {per_call_ms:.2f} ms at 130 tiles")


if __name__ == "__main__":
    unittest.main()
