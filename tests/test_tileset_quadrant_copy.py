"""Regression for #189: avoid frombytes(tobytes()) on pyramid quadrant load."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from PIL import Image

from nornir_imageregistration import tileset_functions


class TestPyramidQuadrantCopy(unittest.TestCase):
    def test_load_does_not_call_tobytes(self) -> None:
        """#189: img.copy() keeps pixels without a full tobytes intermediate."""
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for name in ('tl', 'tr', 'bl', 'br'):
                path = os.path.join(temp_dir, f'{name}.png')
                Image.new('L', (32, 32), 100).save(path)
                paths.append(path)
            output_path = os.path.join(temp_dir, 'out.png')

            with mock.patch.object(
                    Image.Image, 'tobytes', side_effect=AssertionError('tobytes')):
                tileset_functions.CreateOneTilesetTileWithPillow(
                    (32, 32), *paths, output_path)

            self.assertTrue(os.path.isfile(output_path))
            with Image.open(output_path) as out:
                self.assertEqual(out.size, (32, 32))


if __name__ == '__main__':
    unittest.main()
