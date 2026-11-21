"""Masked registration stage 5: mosaic translate measures tile pairs by band-pass + masked NCC.

``__tile_offset_remote`` used to noise-fill the overlap crops outside the tile and over extrema,
phase-correlate them and weight the peak by peak/mean times the valid fraction. On RC1 section 4
that weight was 0.38-0.44 for seam pairs with no match anywhere in their overlap (masked NCC
0.12-0.18) against 0.6-2.2 for real matches, so a whole tile column followed the seam pairs
600-750 px out. Now the crops carry a validity mask instead of noise, the record holds the peak's
masked NCC, and ``_FindTileOffsets`` keeps a pair only when that NCC is at least
``MATCH_NCC_RATIO`` times the section median.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import scipy.ndimage

import nornir_imageregistration as nir
from nornir_imageregistration import arrange_mosaic

_TILE = 256


def _texture(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = scipy.ndimage.gaussian_filter(rng.standard_normal(shape), 2.0)
    image -= image.min()
    return (image / image.max()).astype(np.float32)


def _pair(true_offset: tuple[int, int], stage_offset: tuple[int, int], seed: int = 3):
    """Tiles A and B cut from one texture with B at *true_offset* from A; rects from *stage_offset*."""
    texture = _texture((_TILE + 64, 2 * _TILE + 64), seed)
    a = texture[16:16 + _TILE, 16:16 + _TILE]
    by, bx = 16 + true_offset[0], 16 + true_offset[1]
    b = texture[by:by + _TILE, bx:bx + _TILE]
    sy, sx = stage_offset
    rect_a = nir.Rectangle.CreateFromBounds((max(sy, 0), sx, min(_TILE, _TILE + sy), _TILE))
    rect_b = nir.Rectangle.CreateFromBounds((max(-sy, 0), 0, min(_TILE, _TILE - sy), _TILE - sx))
    return a, b, rect_a, rect_b


def _measure(a: np.ndarray, b: np.ndarray, rect_a, rect_b, stage_offset) -> nir.AlignmentRecord:
    tile_offset = getattr(arrange_mosaic, '__tile_offset_remote')
    images = {'a.png': a, 'b.png': b}
    with mock.patch.object(nir, 'LoadImage', side_effect=lambda path, dtype=None: images[path].copy()):
        return tile_offset('a.png', 'b.png', rect_a, rect_b, np.asarray(stage_offset, dtype=float),
                           excess_scalar=3.0, mask_extrema=False)


class TestTileOffsetIsMaskedNcc(unittest.TestCase):

    def test_it_recovers_the_true_offset_when_the_stage_is_off(self) -> None:
        a, b, rect_a, rect_b = _pair(true_offset=(9, 200), stage_offset=(0, 190))
        record = _measure(a, b, rect_a, rect_b, (0, 190))
        np.testing.assert_allclose(record.peak, (9, 200), atol=1.0)
        self.assertGreater(record.masked_ncc, 0.9)
        self.assertGreater(record.weight, 0)

    def test_it_is_deterministic(self) -> None:
        """No noise fill, so the same pair gives the same record every time (it did not before)."""
        a, b, rect_a, rect_b = _pair(true_offset=(-5, 196), stage_offset=(0, 190))
        first = _measure(a, b, rect_a, rect_b, (0, 190))
        second = _measure(a, b, rect_a, rect_b, (0, 190))
        np.testing.assert_array_equal(first.peak, second.peak)
        self.assertEqual(first.weight, second.weight)

    def test_unrelated_tiles_have_a_low_ncc(self) -> None:
        a, _, rect_a, rect_b = _pair(true_offset=(0, 200), stage_offset=(0, 190), seed=3)
        _, unrelated, _, _ = _pair(true_offset=(0, 200), stage_offset=(0, 190), seed=11)
        record = _measure(a, unrelated, rect_a, rect_b, (0, 190))
        self.assertLess(record.masked_ncc, 0.4)

    def test_a_flat_region_has_weight_zero_at_the_stage_offset(self) -> None:
        a, _, rect_a, rect_b = _pair(true_offset=(0, 200), stage_offset=(0, 190))
        flat = np.full_like(a, 0.5)
        record = _measure(a, flat, rect_a, rect_b, (0, 190))
        self.assertEqual(0, record.weight)
        np.testing.assert_allclose(record.peak, (0, 190))

    def test_the_window_is_trimmed_to_the_tile(self) -> None:
        """Pixels past the tile edge would be invalid, so they are not carried at all."""
        image = np.arange(100, dtype=np.float32).reshape(10, 10)
        rect = nir.Rectangle.CreateFromBounds((0, 6, 10, 10))  # the rightmost 4 columns
        side = arrange_mosaic._overlap_region_with_validity(image, rect, 3.0, mask_extrema=False)
        self.assertEqual((30, 12), side.window_shape)  # 3x the overlap, centred on it
        self.assertEqual((10, 8), side.image.shape)  # rows 0-9 and columns 2-9 lie inside the tile
        np.testing.assert_array_equal(image[:, 2:], side.image)
        self.assertTrue(side.valid.all())
        np.testing.assert_allclose((0.0, -2.0), side.centre_shift)

def _overlap(a_id: int, b_id: int):
    rect = nir.Rectangle.CreateFromBounds((0, 0, 8, 8))
    tile = lambda i: SimpleNamespace(ID=i, ImagePath=f'{i}.png',
                                     FixedBoundingBox=SimpleNamespace(Center=np.array((0.0, 10.0 * i))))
    return SimpleNamespace(ID=(a_id, b_id), A=tile(a_id), B=tile(b_id),
                           scaled_overlapping_source_rect_A=rect, scaled_overlapping_source_rect_B=rect,
                           scaled_offset=np.array((0.0, 10.0)), normalized_feature_scores=None)


class TestPairsThatDoNotMatchAreDropped(unittest.TestCase):

    def _run(self, ncc_by_pair: dict[tuple[int, int], float], weight_zero=frozenset()):
        def fake(path_a, path_b, *args, **kwargs):
            key = (int(path_a.split('.')[0]), int(path_b.split('.')[0]))
            record = nir.AlignmentRecord(peak=(0.0, 10.0), weight=0.0 if key in weight_zero else 50.0)
            record.masked_ncc = ncc_by_pair[key]
            return record

        overlaps = [_overlap(a, b) for a, b in ncc_by_pair]
        failed: dict[tuple[int, int], str] = {}
        # The serial pool runs the fake in-process (the shared pool pickles its tasks).
        with mock.patch.object(arrange_mosaic, '__tile_offset_remote', side_effect=fake), \
                mock.patch.object(arrange_mosaic.nornir_pools, 'GetGlobalMultithreadingPool',
                                  arrange_mosaic.nornir_pools.GetGlobalSerialPool):
            layout = arrange_mosaic._FindTileOffsets(overlaps, excess_scalar=3.0, image_to_source_space_scale=1.0,
                                                    failed_pairs=failed)
        return layout, failed

    def test_a_seam_pair_far_below_the_section_median_is_not_linked(self) -> None:
        ncc = {(1, 2): 0.90, (2, 3): 0.88, (3, 4): 0.92, (4, 5): 0.86, (5, 6): 0.15}
        layout, failed = self._run(ncc)
        self.assertFalse(layout.nodes[5].ContainsOffset(6))
        self.assertIn('No match', failed[(5, 6)])
        for a, b in [(1, 2), (2, 3), (3, 4), (4, 5)]:
            self.assertTrue(layout.nodes[a].ContainsOffset(b))
            self.assertNotIn((a, b), failed)

    def test_the_threshold_is_relative_to_the_section(self) -> None:
        """A low-contrast section whose matches all sit near 0.3 keeps them."""
        ncc = {(1, 2): 0.30, (2, 3): 0.28, (3, 4): 0.33, (4, 5): 0.26}
        layout, failed = self._run(ncc)
        self.assertEqual({}, failed)
        self.assertTrue(all(layout.nodes[a].ContainsOffset(b) for a, b in ncc))

    def test_a_weight_zero_pair_leaves_no_spring(self) -> None:
        ncc = {(1, 2): 0.90, (2, 3): float('nan')}
        layout, failed = self._run(ncc, weight_zero={(2, 3)})
        self.assertFalse(layout.nodes[2].ContainsOffset(3))
        self.assertIn((2, 3), failed)


if __name__ == '__main__':
    unittest.main()
