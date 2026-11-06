"""Tests for mapping points between two mosaics through the tiles they share (annotation rescue for repaired mosaics)."""
from __future__ import annotations

import unittest

import nornir_imageregistration
import numpy as np
from nornir_imageregistration.mosaic_remap import MosaicPointRemapper, RemapStatus
from nornir_imageregistration.transforms.rigid import RigidNoRotation

TILE = (100.0, 100.0)


def _mosaic(offsets: dict[str, tuple[float, float]]) -> nornir_imageregistration.Mosaic:
    """Translation-only tiles of 100 x 100 px, each moved to the (Y, X) offset given."""
    return nornir_imageregistration.Mosaic({name: RigidNoRotation(target_offset=offset) for name, offset in offsets.items()})


class TestMosaicPointRemapper(unittest.TestCase):
    """A point must follow the tile it lies in, and tile pixels are the same in both mosaics."""

    def setUp(self) -> None:
        # Old: A and B overlap by 10 px (A covers x 0-100, B covers x 90-190). New: they moved independently.
        self.old = _mosaic({"000.png": (0.0, 0.0), "001.png": (0.0, 90.0), "002.png": (200.0, 0.0)})
        self.new = _mosaic({"000.png": (10.0, 5.0), "001.png": (7.0, 200.0)})  # tile 2 is not in the new mosaic
        self.remapper = MosaicPointRemapper(self.old, self.new, tile_shape=TILE)

    def test_a_point_in_one_tile_moves_with_that_tile(self) -> None:
        result = self.remapper.remap(np.array([[50.0, 20.0], [50.0, 150.0]]))
        np.testing.assert_allclose(result.new_points[0], [60.0, 25.0], atol=1e-3)   # tile 0 moved by (10, 5)
        np.testing.assert_allclose(result.new_points[1], [57.0, 260.0], atol=1e-3)  # tile 1 pixel (50, 60) now at (7, 200) + that
        self.assertEqual(result.tile_names, ["000.png", "001.png"])
        self.assertTrue(np.all(result.status == RemapStatus.MAPPED))

    def test_a_point_in_an_overlap_follows_the_tile_whose_centre_is_closer(self) -> None:
        # x = 92 is 42 px from tile 0's centre (x 50) and 48 px from tile 1's (x 140); x = 98 is the reverse.
        near_a, near_b = self.remapper.remap(np.array([[50.0, 92.0], [50.0, 98.0]])).tile_names
        self.assertEqual((near_a, near_b), ("000.png", "001.png"))
        result = self.remapper.remap(np.array([[50.0, 92.0], [50.0, 98.0]]))
        np.testing.assert_allclose(result.new_points[0], [60.0, 97.0], atol=1e-3)
        np.testing.assert_allclose(result.new_points[1], [57.0, 208.0], atol=1e-3)

    def test_overlap_disagreement_measures_how_far_apart_overlapping_tiles_put_the_tissue(self) -> None:
        result = self.remapper.remap(np.array([[50.0, 92.0], [50.0, 20.0]]))
        # Through tile 0 the point goes to (60, 97); through tile 1 (pixel (50, 2)) it goes to (57, 202).
        self.assertAlmostEqual(float(result.overlap_disagreement[0]), float(np.hypot(3.0, 105.0)), places=2)
        self.assertTrue(np.isnan(result.overlap_disagreement[1]))  # only tile 0 holds this point
        self.assertEqual(result.summary()["in_overlap"], 1)

    def test_overlapping_tiles_that_moved_together_agree(self) -> None:
        old = _mosaic({"000.png": (0.0, 0.0), "001.png": (0.0, 90.0)})
        new = _mosaic({"000.png": (4.0, 4.0), "001.png": (4.0, 94.0)})  # both moved by (4, 4): still consistent
        result = MosaicPointRemapper(old, new, tile_shape=TILE).remap(np.array([[50.0, 95.0]]))
        self.assertAlmostEqual(float(result.overlap_disagreement[0]), 0.0, places=3)

    def test_a_point_outside_every_shared_tile_is_reported_not_guessed(self) -> None:
        result = self.remapper.remap(np.array([[500.0, 500.0], [250.0, 50.0]]))  # nowhere / only in the tile missing from new
        self.assertTrue(np.all(result.status == RemapStatus.NO_SHARED_TILE))
        self.assertTrue(np.isnan(result.new_points).all())
        self.assertEqual(result.tile_names, [None, None])

    def test_mapping_forward_then_back_returns_the_original_point(self) -> None:
        points = np.array([[10.0, 10.0], [50.0, 20.0], [50.0, 150.0], [90.0, 180.0]])
        forward = self.remapper.remap(points)
        back = MosaicPointRemapper(self.new, self.old, tile_shape=TILE).remap(forward.new_points)
        np.testing.assert_allclose(back.new_points, points, atol=1e-3)

    def test_summary_counts_every_status(self) -> None:
        result = self.remapper.remap(np.array([[50.0, 20.0], [500.0, 500.0], [50.0, 150.0]]))
        summary = result.summary()
        self.assertEqual((summary["points"], summary["mapped"], summary["no_shared_tile"]), (3, 2, 1))
        self.assertAlmostEqual(summary["shift_max"], float(np.hypot(7.0, 110.0)), places=2)

    def test_mosaics_with_no_shared_tile_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MosaicPointRemapper(_mosaic({"000.png": (0.0, 0.0)}), _mosaic({"001.png": (0.0, 0.0)}), tile_shape=TILE)


if __name__ == "__main__":
    unittest.main()
