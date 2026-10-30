"""Regression test for thread-safe pyramid tile merge."""

from __future__ import annotations

import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from nornir_imageregistration import tileset_functions


def _write_tile(path: str, fill: int) -> None:
    Image.new('L', (64, 64), fill).save(path)


class TestCreateOneTilesetTileConcurrentMerge(unittest.TestCase):
    """Parallel loads must not lose quadrants when compositing."""

    def test_concurrent_merge_preserves_all_quadrants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fills = {
                'tl.png': 40,
                'tr.png': 80,
                'bl.png': 120,
                'br.png': 200,
            }
            for name, fill in fills.items():
                _write_tile(os.path.join(temp_dir, name), fill)

            output_path = os.path.join(temp_dir, 'out.png')

            def merge_once() -> None:
                if os.path.exists(output_path):
                    os.remove(output_path)
                with ThreadPoolExecutor(max_workers=4) as executor:
                    tileset_functions.CreateOneTilesetTileWithPillow(
                        (64, 64),
                        TopLeft=os.path.join(temp_dir, 'tl.png'),
                        TopRight=os.path.join(temp_dir, 'tr.png'),
                        BottomLeft=os.path.join(temp_dir, 'bl.png'),
                        BottomRight=os.path.join(temp_dir, 'br.png'),
                        OutputFileFullPath=output_path,
                        executor=executor,
                    )
                with Image.open(output_path) as merged:
                    self.assertEqual(merged.size, (64, 64))
                    # Each quadrant should retain non-zero signal from its source fill.
                    self.assertGreater(merged.crop((0, 0, 32, 32)).getextrema()[1], 0)
                    self.assertGreater(merged.crop((32, 0, 64, 32)).getextrema()[1], 0)
                    self.assertGreater(merged.crop((0, 32, 32, 64)).getextrema()[1], 0)
                    self.assertGreater(merged.crop((32, 32, 64, 64)).getextrema()[1], 0)

            for _ in range(50):
                merge_once()


if __name__ == '__main__':
    unittest.main()
