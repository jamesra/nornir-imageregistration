"""TranslateTiles2 when no tile overlap qualifies (#128).

``relaxed_layout`` is initialised to None and only assigned inside the pass loop. If the very
first ``GenerateTileOverlaps`` finds nothing to align, the loop breaks before assigning it, and
the ``relaxed_layout.TranslateToZeroOrigin()`` after the loop raised:

    AttributeError: 'NoneType' object has no attribute 'TranslateToZeroOrigin'

which says nothing about the tiles. Reproduced against the real function with tiles written to
disk. All three of these crashed at ``arrange_mosaic.py:246``:

| case | tiles |
|------|-------|
| two tiles far apart | (0,0), (10000,10000) |
| three tiles far apart | (0,0), (10000,0), (0,10000) |
| **two tiles touching edge to edge** | (0,0), (0,64) for 64px tiles |

The third is the one that makes this more than a corner case: adjacency is not overlap, so a
perfectly ordinary edge-to-edge pair took the whole section down. A properly overlapping pair
worked fine, before and after.

A second path existed too: ``MergeDisconnectedLayoutsWithOffsets`` returns None for an empty
layout list, which would have made the three dereferences *inside* the loop fail. Both now
route to the same fallback.

The fallback places every tile at its reported stage position, matching what the single-tile
branch already does with its one node, and logs why. Tile transforms are left untouched, since
no alignment was measured.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import nornir_imageregistration
import nornir_imageregistration.arrange_mosaic as arrange_mosaic
import nornir_imageregistration.transforms.factory as tfactory

_TILE = 64


def _settings(min_overlap=0.05):
    return nornir_imageregistration.settings.TranslateSettings(
        min_overlap=min_overlap,
        max_relax_iterations=5,
        max_translate_iterations=2,
        feature_score_threshold=None,
        use_feature_score=False)


class _TilesOnDisk(unittest.TestCase):
    """Writes real tiles, because TranslateTiles2 loads them."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='test128_')
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def tileset(self, offsets):
        paths = []
        transforms = []
        for i, offset in enumerate(offsets):
            path = os.path.join(self.tmpdir, f'Tile{i:05d}.png')
            rng = np.random.default_rng(i + 1)
            Image.fromarray((rng.random((_TILE, _TILE)) * 255).astype(np.uint8)).save(path)
            paths.append(path)
            transforms.append(tfactory.CreateRigidTransform(
                target_image_shape=(_TILE, _TILE),
                source_image_shape=(_TILE, _TILE),
                rangle=0,
                warped_offset=np.asarray(offset, dtype=np.float64)))

        return nornir_imageregistration.mosaic_tileset.Create(
            transforms=transforms, imagepaths=paths, image_to_source_space_scale=1.0)

    @staticmethod
    def positions(layout):
        return {ID: np.asarray(node.Position) for ID, node in layout.nodes.items()}


class TestANonOverlappingMosaicStillReturns(_TilesOnDisk):
    """The crash, one case per geometry that reproduced it."""

    def test_two_tiles_far_apart(self):
        layout, tileset = arrange_mosaic.TranslateTiles2(
            self.tileset([(0, 0), (10000, 10000)]), _settings())
        self.assertEqual(2, len(layout.nodes))

    def test_two_tiles_touching_edge_to_edge(self):
        # Adjacency is not overlap. This is an ordinary tile pair, not a degenerate one.
        layout, _ = arrange_mosaic.TranslateTiles2(
            self.tileset([(0, 0), (0, _TILE)]), _settings())
        self.assertEqual(2, len(layout.nodes))

    def test_three_tiles_far_apart(self):
        layout, _ = arrange_mosaic.TranslateTiles2(
            self.tileset([(0, 0), (10000, 0), (0, 10000)]), _settings())
        self.assertEqual(3, len(layout.nodes))

    def test_min_overlap_above_what_the_mosaic_has(self):
        # A half-overlapping pair rejected by an unreachable min_overlap takes the same path.
        layout, _ = arrange_mosaic.TranslateTiles2(
            self.tileset([(0, 0), (0, _TILE // 2)]), _settings(min_overlap=0.99))
        self.assertEqual(2, len(layout.nodes))

    def test_every_tile_gets_a_node(self):
        for count in (2, 3, 5):
            with self.subTest(tiles=count):
                offsets = [(0, i * 10000) for i in range(count)]
                layout, _ = arrange_mosaic.TranslateTiles2(
                    self.tileset(offsets), _settings())
                self.assertEqual(count, len(layout.nodes),
                                 'no tile should be dropped by the fallback')


class TestTheFallbackKeepsTheStageArrangement(_TilesOnDisk):
    """A layout that discarded the stage positions would be worse than the crash."""

    def test_the_relative_offsets_are_preserved(self):
        offsets = [(0, 0), (10000, 10000)]
        layout, _ = arrange_mosaic.TranslateTiles2(self.tileset(offsets), _settings())
        positions = self.positions(layout)

        measured = positions[1] - positions[0]
        np.testing.assert_allclose(np.asarray(offsets[1]) - np.asarray(offsets[0]),
                                   measured, atol=1e-6)

    def test_an_edge_to_edge_pair_stays_one_tile_apart(self):
        layout, _ = arrange_mosaic.TranslateTiles2(
            self.tileset([(0, 0), (0, _TILE)]), _settings())
        positions = self.positions(layout)
        np.testing.assert_allclose([0, _TILE], positions[1] - positions[0], atol=1e-6)

    def test_the_layout_is_translated_to_a_zero_origin(self):
        layout, _ = arrange_mosaic.TranslateTiles2(
            self.tileset([(0, 0), (10000, 10000)]), _settings())
        stacked = np.vstack(list(self.positions(layout).values()))
        np.testing.assert_allclose([0, 0], stacked.min(axis=0), atol=1e-6)

    def test_a_three_tile_l_shape_keeps_its_shape(self):
        offsets = [(0, 0), (10000, 0), (0, 10000)]
        layout, _ = arrange_mosaic.TranslateTiles2(self.tileset(offsets), _settings())
        positions = self.positions(layout)
        np.testing.assert_allclose([10000, 0], positions[1] - positions[0], atol=1e-6)
        np.testing.assert_allclose([0, 10000], positions[2] - positions[0], atol=1e-6)


class TestItSaysWhyNothingWasAligned(_TilesOnDisk):
    """The AttributeError explained nothing; the fallback should."""

    def test_it_reports_the_fallback(self):
        with mock.patch('nornir_shared.prettyoutput.LogErr') as logged:
            arrange_mosaic.TranslateTiles2(
                self.tileset([(0, 0), (10000, 10000)]), _settings())

        message = ' '.join(str(call) for call in logged.call_args_list)
        self.assertIn('stage coordinates', message)
        self.assertIn('min_overlap', message,
                      'the likely misconfiguration should be named')

    def test_it_says_nothing_when_the_tiles_do_overlap(self):
        with mock.patch('nornir_shared.prettyoutput.LogErr') as logged:
            arrange_mosaic.TranslateTiles2(
                self.tileset([(0, 0), (0, _TILE // 2)]), _settings())

        message = ' '.join(str(call) for call in logged.call_args_list)
        self.assertNotIn('stage coordinates', message,
                         'a healthy mosaic should not report the fallback')


class TestTheOverlappingCaseIsUnchanged(_TilesOnDisk):
    """The fix must not touch the path that already worked."""

    def test_a_half_overlapping_pair_is_aligned(self):
        layout, _ = arrange_mosaic.TranslateTiles2(
            self.tileset([(0, 0), (0, _TILE // 2)]), _settings())
        self.assertEqual(2, len(layout.nodes))

        positions = self.positions(layout)
        offset = positions[1] - positions[0]
        # Measured before the fix as [49.0, 28.5]; assert it is still a real alignment
        # rather than the stage position (0, 32) the fallback would have produced.
        self.assertFalse(np.allclose([0, _TILE // 2], offset, atol=1e-6),
                         'this pair should be aligned, not fed to the fallback')

    def test_a_single_tile_still_short_circuits(self):
        layout, _ = arrange_mosaic.TranslateTiles2(self.tileset([(0, 0)]), _settings())
        self.assertEqual(1, len(layout.nodes))
        np.testing.assert_allclose([0, 0], self.positions(layout)[0], atol=1e-6)


class TestTheEmptyMergePathIsGuarded(_TilesOnDisk):
    """MergeDisconnectedLayoutsWithOffsets returns None for an empty list."""

    def test_merge_returns_none_for_an_empty_list(self):
        self.assertIsNone(
            nornir_imageregistration.layout.MergeDisconnectedLayoutsWithOffsets([], {}))

    def test_an_empty_merge_result_falls_back_instead_of_crashing(self):
        # Force the in-loop None on a mosaic that does overlap, so the break is the only
        # thing standing between it and three AttributeErrors.
        with mock.patch.object(nornir_imageregistration.layout,
                               'MergeDisconnectedLayoutsWithOffsets', return_value=None):
            layout, _ = arrange_mosaic.TranslateTiles2(
                self.tileset([(0, 0), (0, _TILE // 2)]), _settings())

        self.assertEqual(2, len(layout.nodes))
        np.testing.assert_allclose([0, _TILE // 2],
                                   self.positions(layout)[1] - self.positions(layout)[0],
                                   atol=1e-6)


if __name__ == '__main__':
    unittest.main()
