"""Tests for MissingTilesetInputError on single-tile pyramid paths."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from PIL import Image

from nornir_imageregistration import tileset_functions
from nornir_imageregistration.exceptions import MissingTilesetInputError


def _write_tile(path: str, width: int = 64, height: int = 64) -> None:
    """Write a small grayscale PNG tile for merge tests."""
    Image.new('L', (width, height), 128).save(path)


class TestTilesetMissingPaths(unittest.TestCase):
    """Try-on-access missing path handling for single-tile pyramid functions."""

    def test_partial_missing_inputs_still_produces_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            top_left = os.path.join(temp_dir, 'tl.png')
            _write_tile(top_left)
            output_path = os.path.join(temp_dir, 'tile.png')

            tileset_functions.CreateOneTilesetTileWithPillow(
                (64, 64),
                TopLeft=top_left,
                TopRight=os.path.join(temp_dir, 'tr.png'),
                BottomLeft=os.path.join(temp_dir, 'bl.png'),
                BottomRight=os.path.join(temp_dir, 'br.png'),
                OutputFileFullPath=output_path,
            )

            self.assertTrue(os.path.isfile(output_path))

    def test_missing_output_parent_dir_raises_on_save(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            top_left = os.path.join(temp_dir, 'tl.png')
            top_right = os.path.join(temp_dir, 'tr.png')
            bottom_left = os.path.join(temp_dir, 'bl.png')
            bottom_right = os.path.join(temp_dir, 'br.png')
            for path in (top_left, top_right, bottom_left, bottom_right):
                _write_tile(path)

            output_path = os.path.join(temp_dir, 'missing_parent', 'tile.png')

            with self.assertRaises(MissingTilesetInputError) as ctx:
                tileset_functions.CreateOneTilesetTileWithPillow(
                    (64, 64),
                    TopLeft=top_left,
                    TopRight=top_right,
                    BottomLeft=bottom_left,
                    BottomRight=bottom_right,
                    OutputFileFullPath=output_path,
                )

            self.assertIn(os.path.dirname(output_path), ctx.exception.missing_paths)

    def test_missing_dest_parent_on_copy_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = os.path.join(temp_dir, 'work')
            os.makedirs(work_dir, exist_ok=True)
            for name in ('tl.png', 'tr.png', 'bl.png', 'br.png'):
                _write_tile(os.path.join(work_dir, name))

            dest_path = os.path.join(temp_dir, 'missing_dest', 'tile.png')

            def fake_pillow(_tile_dims, _tl, _tr, _bl, _br, output_path, **_kwargs) -> None:
                with open(output_path, 'wb') as handle:
                    handle.write(b'png')

            with mock.patch.object(tileset_functions, 'CreateOneTilesetTileWithPillow', side_effect=fake_pillow):
                with self.assertRaises(MissingTilesetInputError) as ctx:
                    tileset_functions.CreateOneTilesetTileWithPillowOverNetwork(
                        (64, 64),
                        TopLeft=os.path.join(work_dir, 'tl.png'),
                        TopRight=os.path.join(work_dir, 'tr.png'),
                        BottomLeft=os.path.join(work_dir, 'bl.png'),
                        BottomRight=os.path.join(work_dir, 'br.png'),
                        OutputFileFullPath=dest_path,
                        temp_input_dir=None,
                        output_level_temp_dir=work_dir,
                    )

            self.assertIn(os.path.dirname(dest_path), ctx.exception.missing_paths)


if __name__ == '__main__':
    unittest.main()
