"""Known (manual) offsets in translate, and the translate report's join check.

RC3 1637 (2026-10-31): a 300-tile block's stage positions were ~5,500 px off, every pair across
the join failed, and translate attached the block by its stage offset with no warning. The
manual-offsets CSV that could have fixed it was read into ``TranslateSettings.known_offsets``
but nothing in translate ever used it.

These tests cut tiles from one smooth texture, so overlapping tiles really match, then:

- displace one column of tiles far enough that its overlaps vanish, and check a single known
  offset reattaches it where its texture continues;
- score joins on correct and misplaced layouts, and check only the misplaced one is flagged;
- round-trip the offsets file and its content checksum.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from typing import ClassVar
from unittest import mock

import nornir_imageregistration
import nornir_imageregistration.transforms.factory as tfactory
import numpy as np
import scipy.ndimage
from nornir_imageregistration import arrange_mosaic, translate_report
from nornir_imageregistration.settings import (
    LoadMosaicOffsets,
    MosaicOffsetsChecksum,
    SaveMosaicOffsets,
    TileOffset,
)
from nornir_imageregistration.translate_report import (
    TranslateReport,
    load_known_offsets,
)
from PIL import Image

_TILE = 128
_STEP = 96  # 32 px (25%) overlap between neighbours
_GRID = 3
_FAR = 4000.0  # displacement that leaves the moved column with no overlap at all


def _settings(known_offsets=None):
    return nornir_imageregistration.settings.TranslateSettings(
        min_overlap=0.05, max_relax_iterations=50, max_translate_iterations=2,
        feature_score_threshold=None, use_feature_score=False, known_offsets=known_offsets)


def _true_position(tile_id: int) -> np.ndarray:
    row, col = divmod(tile_id, _GRID)
    return np.array([row * _STEP, col * _STEP], dtype=np.float64)


class _TextureTiles(unittest.TestCase):
    """A 3x3 grid of tiles cut from one texture; tile ID = row * 3 + column."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='test_known_offsets_')
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        size = _STEP * (_GRID - 1) + _TILE
        texture = scipy.ndimage.gaussian_filter(np.random.default_rng(7).random((size, size)), 2.0)
        texture = (texture - texture.min()) / (texture.max() - texture.min())
        self.paths = []
        for tile_id in range(_GRID * _GRID):
            y, x = _true_position(tile_id).astype(int)
            path = os.path.join(self.tmpdir, f'Tile{tile_id:05d}.png')
            Image.fromarray((texture[y:y + _TILE, x:x + _TILE] * 255).astype(np.uint8)).save(path)
            self.paths.append(path)

    def tileset(self, positions):
        shape = np.array((_TILE, _TILE))
        transforms = [tfactory.CreateRigidTransform(target_image_shape=shape, source_image_shape=shape, rangle=0,
                                                    warped_offset=np.asarray(p, dtype=np.float64))
                      for p in positions]
        return nornir_imageregistration.mosaic_tileset.Create(transforms=transforms, imagepaths=self.paths,
                                                              image_to_source_space_scale=1.0)

    @staticmethod
    def right_column_moved(displacement):
        return [_true_position(i) + (displacement if i % _GRID == _GRID - 1 else 0) for i in range(_GRID * _GRID)]

    @staticmethod
    def relative_error(layout) -> float:
        """Largest disagreement with the true layout once both are pinned at tile 0."""
        error = 0.0
        for tile_id, node in layout.nodes.items():
            expected = _true_position(tile_id) - _true_position(0)
            measured = np.asarray(node.Position) - np.asarray(layout.nodes[0].Position)
            error = max(error, float(np.max(np.abs(measured - expected))))
        return error


class TestKnownOffsetsJoinGroups(_TextureTiles):

    def test_without_a_known_offset_the_moved_column_stays_at_its_stage_offset(self):
        report = TranslateReport()
        layout, _ = arrange_mosaic.TranslateTiles2(self.tileset(self.right_column_moved((_FAR, 0))), _settings(),
                                                   report=report)
        self.assertGreater(self.relative_error(layout), _FAR / 2)
        self.assertEqual([6, 3], [len(g) for g in report.groups])

    def test_one_known_offset_reattaches_the_column_where_its_texture_continues(self):
        # Tile 1 (row 0, column 1) to tile 2 (row 0, column 2): one step right.
        known = [TileOffset(A=1, B=2, Y=0.0, X=float(_STEP))]
        report = TranslateReport()
        layout, _ = arrange_mosaic.TranslateTiles2(self.tileset(self.right_column_moved((_FAR, 0))),
                                                   _settings(known), report=report)
        self.assertLess(self.relative_error(layout), 2.0)
        self.assertEqual([9], [len(g) for g in report.groups])
        self.assertEqual([{'A': 1, 'B': 2, 'Y': 0.0, 'X': float(_STEP), 'comment': None}], report.known_offsets)

    def test_a_known_offset_written_b_first_means_the_same_thing(self):
        known = [[2, 1, 0.0, -float(_STEP)]]  # a plain row, as settings JSON would carry it
        layout, _ = arrange_mosaic.TranslateTiles2(self.tileset(self.right_column_moved((_FAR, 0))),
                                                   _settings(known))
        self.assertLess(self.relative_error(layout), 2.0)

    def test_a_known_offset_naming_a_missing_tile_is_reported_and_skipped(self):
        report = TranslateReport()
        arrange_mosaic.TranslateTiles2(self.tileset([_true_position(i) for i in range(9)]),
                                       _settings([TileOffset(A=1, B=99, Y=0.0, X=0.0)]), report=report)
        self.assertEqual([], report.known_offsets)
        self.assertEqual(1, len(report.known_offsets_skipped))
        self.assertIn('99', report.known_offsets_skipped[0]['reason'])

    def test_known_offsets_leave_an_already_connected_layout_alone(self):
        positions = [_true_position(i) for i in range(9)]
        without, _ = arrange_mosaic.TranslateTiles2(self.tileset(positions), _settings())
        with_known, _ = arrange_mosaic.TranslateTiles2(self.tileset(positions),
                                                       _settings([TileOffset(A=1, B=2, Y=0.0, X=float(_STEP))]))
        self.assertLess(self.relative_error(without), 2.0)
        self.assertLess(self.relative_error(with_known), 2.0)


class TestJoinCheck(_TextureTiles):
    """The check works from the images, whatever groups translate reports."""
    COLUMN_SPLIT: ClassVar[list[list[int]]] = [[0, 1, 3, 4, 6, 7], [2, 5, 8]]

    def check(self, positions, translate_groups=None) -> TranslateReport:
        report = TranslateReport(groups=[list(g) for g in (translate_groups or [list(range(9))])])
        report.check_joins(self.tileset(positions), min_overlap=0.05)
        return report

    def test_a_correct_layout_holds_together(self):
        report = self.check([_true_position(i) for i in range(9)], self.COLUMN_SPLIT)
        self.assertEqual([list(range(9))], report.matched_groups)
        self.assertEqual([], report.joins)
        self.assertFalse(report.needs_review)
        assert report.median_ncc is not None
        self.assertGreater(report.median_ncc, 0.8)

    def test_a_block_placed_wrongly_but_still_overlapping_is_flagged(self):
        # Shifted by a third of a tile: the column still overlaps its neighbours, but the texture
        # no longer continues. Translate calls it one group, as it did on RC3 1637.
        report = self.check(self.right_column_moved((_TILE // 3, 0)))
        self.assertEqual(self.COLUMN_SPLIT, report.matched_groups)
        self.assertEqual(1, len(report.joins))
        self.assertTrue(report.joins[0].flagged)
        self.assertTrue(report.needs_review)
        self.assertIn('does not continue', report.review_reasons[0])

    def test_a_translate_group_that_overlaps_nothing_is_flagged(self):
        report = self.check(self.right_column_moved((_FAR, 0)), self.COLUMN_SPLIT)
        self.assertEqual([], report.joins)
        self.assertTrue(report.needs_review)
        self.assertIn('overlaps no other tile', report.review_reasons[0])

    def test_small_groups_are_reported_but_not_flagged(self):
        # A lone tile split off (a blank resin tile, typically) is not worth a review.
        positions = [_true_position(i) for i in range(9)]
        positions[8] = positions[8] + (_TILE // 3, 0)
        report = self.check(positions)
        self.assertEqual([list(range(8)), [8]], report.matched_groups)
        self.assertEqual(1, len(report.joins))
        self.assertFalse(report.joins[0].flagged)
        self.assertFalse(report.needs_review)

    def test_a_secondary_block_is_flagged_even_without_a_direct_join(self):
        # RC3 1637: the two halves meet only through a strip of poorly matched tiles, so no
        # overlap joins them directly. Far apart here, so there is no join at all.
        with mock.patch.object(translate_report, 'MIN_SECONDARY_BLOCK_TILES', 3):
            report = self.check(self.right_column_moved((_FAR, 0)))
            self.assertEqual([[2, 5, 8]], report.secondary_blocks)
        self.assertTrue(any('Besides its main block of 6 tiles' in r for r in report.review_reasons))

    def test_the_main_translate_group_is_not_flagged_for_a_stray_tile(self):
        # RC3 0633: translate's 848-tile main group "overlapped no other tile" because the one
        # stray tile beside it did not overlap it.
        positions = [_true_position(i) for i in range(9)]
        positions[8] = positions[8] + (_FAR, 0)
        report = self.check(positions, [list(range(8)), [8]])
        self.assertFalse(report.needs_review, report.review_reasons)


class TestArrangeTilesWithTranslateFillsTheReport(_TextureTiles):

    def test_the_report_names_the_misjoin(self):
        report = TranslateReport()
        self.tileset(self.right_column_moved((_FAR, 0))).ArrangeTilesWithTranslate(_settings(), report=report)
        self.assertTrue(report.needs_review)
        self.assertEqual(9, report.tile_count)

    def test_a_clean_section_needs_no_review(self):
        report = TranslateReport()
        self.tileset([_true_position(i) for i in range(9)]).ArrangeTilesWithTranslate(_settings(), report=report)
        self.assertFalse(report.needs_review)
        self.assertEqual([9], [len(g) for g in report.groups])


class TestOffsetsFile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='test_offsets_file_')
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def test_swapping_the_ids_negates_the_offset(self):
        offset = TileOffset(A=5, B=2, Y=10.0, X=-3.0)
        self.assertEqual((2, 5), offset.ID)
        self.assertEqual((-10.0, 3.0), offset.Offset)

    def test_the_file_round_trips_and_its_header_names_y_first(self):
        path = os.path.join(self.tmpdir, 'offsets.csv')
        offsets = [TileOffset(A=1, B=2, Y=12.5, X=-4.0, Comment='seam'), TileOffset(A=7, B=3, Y=1.0, X=2.0)]
        SaveMosaicOffsets(offsets, path)
        with open(path) as f:
            self.assertEqual(['A', 'B', 'Y', 'X', 'Comment'], f.readline().split())
        loaded = LoadMosaicOffsets(path)
        self.assertEqual(sorted(offsets), loaded)

    def test_a_row_written_b_first_loads_as_the_same_offset(self):
        path = os.path.join(self.tmpdir, 'offsets.csv')
        with open(path, 'w') as f:
            f.write('A B Y X Comment\n7 3 1.0 2.0\n')
        self.assertEqual([TileOffset(A=3, B=7, Y=-1.0, X=-2.0)], LoadMosaicOffsets(path))

    def test_the_checksum_follows_content_not_comments_or_order(self):
        a = [TileOffset(A=1, B=2, Y=12.5, X=-4.0, Comment='x'), TileOffset(A=3, B=4, Y=1.0, X=2.0)]
        b = [TileOffset(A=3, B=4, Y=1.0, X=2.0), TileOffset(A=1, B=2, Y=12.5, X=-4.0)]
        c = [TileOffset(A=1, B=2, Y=12.5, X=-4.5), TileOffset(A=3, B=4, Y=1.0, X=2.0)]
        self.assertEqual(MosaicOffsetsChecksum(a), MosaicOffsetsChecksum(b))
        self.assertNotEqual(MosaicOffsetsChecksum(a), MosaicOffsetsChecksum(c))
        self.assertEqual(MosaicOffsetsChecksum([]), MosaicOffsetsChecksum(None))

    def test_a_saved_report_restores_the_offsets_it_used(self):
        report = TranslateReport()
        used = [TileOffset(A=1, B=2, Y=12.5, X=-4.0, Comment='seam')]
        report.set_known_offsets(used, [])
        path = os.path.join(self.tmpdir, 'report.json')
        report.save(path)
        self.assertEqual(used, load_known_offsets(path))


if __name__ == '__main__':
    unittest.main()
