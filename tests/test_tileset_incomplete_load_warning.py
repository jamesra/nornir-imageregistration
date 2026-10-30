"""Regression for #187: warn when existing pyramid sources fail to load."""
from __future__ import annotations

import logging
import os
import tempfile
import unittest

from PIL import Image

from nornir_imageregistration import tileset_functions


def _write_tile(path: str, width: int = 64, height: int = 64) -> None:
    Image.new('L', (width, height), 128).save(path)


class TestPyramidIncompleteLoadWarning(unittest.TestCase):
    def test_warns_when_existing_quadrant_fails_to_decode(self) -> None:
        """#187: 3-of-4 load with a corrupt existing file must not stay silent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            top_left = os.path.join(temp_dir, 'tl.png')
            top_right = os.path.join(temp_dir, 'tr.png')
            bottom_left = os.path.join(temp_dir, 'bl.png')
            bottom_right = os.path.join(temp_dir, 'br.png')
            for path in (top_left, top_right, bottom_left):
                _write_tile(path)
            with open(bottom_right, 'wb') as handle:
                handle.write(b'not-a-png')
            output_path = os.path.join(temp_dir, 'tile.png')

            with self.assertLogs(
                    'nornir_imageregistration.tileset_functions', level='WARNING') as captured:
                tileset_functions.CreateOneTilesetTileWithPillow(
                    (64, 64),
                    TopLeft=top_left,
                    TopRight=top_right,
                    BottomLeft=bottom_left,
                    BottomRight=bottom_right,
                    OutputFileFullPath=output_path,
                )

            self.assertTrue(os.path.isfile(output_path))
            joined = '\n'.join(captured.output)
            self.assertIn('incomplete', joined.lower())
            self.assertIn('br.png', joined)

    def test_missing_quadrant_still_silent(self) -> None:
        """Absent edge tiles remain an expected case without a failure warning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            top_left = os.path.join(temp_dir, 'tl.png')
            _write_tile(top_left)
            output_path = os.path.join(temp_dir, 'tile.png')

            with self.assertNoLogs(
                    'nornir_imageregistration.tileset_functions', level='WARNING'):
                tileset_functions.CreateOneTilesetTileWithPillow(
                    (64, 64),
                    TopLeft=top_left,
                    TopRight=os.path.join(temp_dir, 'tr.png'),
                    BottomLeft=os.path.join(temp_dir, 'bl.png'),
                    BottomRight=os.path.join(temp_dir, 'br.png'),
                    OutputFileFullPath=output_path,
                )

            self.assertTrue(os.path.isfile(output_path))


if __name__ == '__main__':
    unittest.main()
