"""Tests for tileset pyramid lineage helpers."""

from __future__ import annotations

import os
import tempfile
import unittest

from PIL import Image

from nornir_imageregistration import tileset_functions


def _write_tile(path: str) -> None:
    Image.new('L', (64, 64), 128).save(path)


class TestTilesetLineageHelpers(unittest.TestCase):
    """Parent-coordinate derivation for sparse tileset pyramids."""

    def test_find_missing_lineage_parent_tiles(self) -> None:
        prefix = 'Leveled_'
        postfix = '.png'
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as dest_dir:
            for x, y in ((0, 0), (2, 0), (0, 2)):
                _write_tile(os.path.join(source_dir, f'{prefix}X{x:03d}_Y{y:03d}{postfix}'))

            _write_tile(os.path.join(dest_dir, f'{prefix}X000_Y000{postfix}'))

            missing = tileset_functions.find_missing_lineage_parent_tiles(
                source_dir, dest_dir, prefix, postfix)

            self.assertEqual(missing, [(0, 1), (1, 0)])


if __name__ == '__main__':
    unittest.main()
