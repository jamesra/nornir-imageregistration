"""Tests for skipping pyramid cells when all inputs are absent."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from nornir_imageregistration import tileset_functions


class TestCreateOneTilesetTileAllInputsMissing(unittest.TestCase):
    """When all four source tiles are absent, skip without writing output."""

    def test_skips_when_all_inputs_missing_over_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'out', 'tile.png')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with mock.patch.object(tileset_functions.shutil, 'copyfile') as copy_mock:
                tileset_functions.CreateOneTilesetTileWithPillowOverNetwork(
                    (64, 64),
                    TopLeft=os.path.join(temp_dir, 'tl.png'),
                    TopRight=os.path.join(temp_dir, 'tr.png'),
                    BottomLeft=os.path.join(temp_dir, 'bl.png'),
                    BottomRight=os.path.join(temp_dir, 'br.png'),
                    OutputFileFullPath=output_path,
                    temp_input_dir=None,
                    output_level_temp_dir=temp_dir,
                )

            copy_mock.assert_not_called()
            self.assertFalse(os.path.exists(output_path))

    def test_skips_when_all_inputs_missing_direct(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'tile.png')

            tileset_functions.CreateOneTilesetTileWithPillow(
                (64, 64),
                TopLeft=os.path.join(temp_dir, 'tl.png'),
                TopRight=os.path.join(temp_dir, 'tr.png'),
                BottomLeft=os.path.join(temp_dir, 'bl.png'),
                BottomRight=os.path.join(temp_dir, 'br.png'),
                OutputFileFullPath=output_path,
            )

            self.assertFalse(os.path.exists(output_path))


if __name__ == '__main__':
    unittest.main()
