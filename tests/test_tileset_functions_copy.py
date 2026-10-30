"""Tests that pyramid tile copies complete before returning."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from nornir_imageregistration import tileset_functions
import nornir_shared.files


class TestCreateOneTilesetTileWithPillowOverNetworkCopy(unittest.TestCase):
    """Network tile builder must not leave copyfile submits in flight."""

    def test_copyfile_completes_before_return(self) -> None:
        """copyfile is invoked and finished before the function returns."""
        copy_started = False
        copy_finished = False
        pillow_called = False

        def fake_copyfile(src: str, dst: str) -> None:
            nonlocal copy_started, copy_finished
            copy_started = True
            self.assertTrue(pillow_called, 'copyfile should run after pillow tile build')
            copy_finished = True

        def fake_pillow(_tile_dims, _tl, _tr, _bl, _br, output_path, **_kwargs) -> None:
            nonlocal pillow_called
            pillow_called = True
            self.assertFalse(copy_started, 'copyfile should not start before pillow tile build')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as handle:
                handle.write(b'png')

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'out', 'tile.png')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with mock.patch.object(tileset_functions, 'CreateOneTilesetTileWithPillow', side_effect=fake_pillow), \
                    mock.patch.object(nornir_shared.files, 'copy_file', side_effect=fake_copyfile):
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

        self.assertTrue(copy_finished)


if __name__ == '__main__':
    unittest.main()
