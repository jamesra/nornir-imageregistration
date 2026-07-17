"""Tests for MissingTilesetInputError when all pyramid inputs are absent."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from nornir_imageregistration import tileset_functions
from nornir_imageregistration.exceptions import MissingTilesetInputError


class TestCreateOneTilesetTileAllInputsMissing(unittest.TestCase):
    """When all four source tiles are absent, raise before network copy."""

    def test_raises_when_all_inputs_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'out', 'tile.png')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with mock.patch.object(tileset_functions.shutil, 'copyfile') as copy_mock:
                with self.assertRaises(MissingTilesetInputError):
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


if __name__ == '__main__':
    unittest.main()
