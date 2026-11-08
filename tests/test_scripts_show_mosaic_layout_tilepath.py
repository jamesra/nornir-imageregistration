"""nornir-show-mosaic-layout must check the tile folder itself, not its parent (nornir-review#279).

``ValidateArgs`` tested ``os.path.exists(os.path.dirname(tile_path))``: with a bare mosaic name and a
relative ``-t`` (``-i in.mosaic -t 001``) that is ``exists('')`` and the script exited 2 although the
folder exists, while a missing tile folder whose parent exists was accepted.
"""
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

from hypothesis import example, given, settings
from hypothesis import strategies as st

import nornir_imageregistration.scripts.nornir_show_mosaic_layout as show_mosaic_layout

_dir_names = st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789_', min_size=1, max_size=20)


class _Stop(Exception):
    """Raised by the tileset stub to end Execute once the tile folder is known."""


class TestShowMosaicLayoutTilePath(unittest.TestCase):
    def setUp(self):
        self.addCleanup(os.chdir, os.getcwd())

    @settings(max_examples=25, deadline=None)
    @given(tile_dir_name=_dir_names)
    @example(tile_dir_name='001')
    @example(tile_dir_name='tiles')
    def test_relative_tile_folder_next_to_bare_mosaic_name(self, tile_dir_name: str):
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                with open('in.mosaic', 'w'):
                    pass
                os.makedirs(tile_dir_name, exist_ok=True)
                argv = ['-i', 'in.mosaic', '-o', 'out.svg', '-t', tile_dir_name, '-d', '1']
                with mock.patch.object(show_mosaic_layout.nornir_imageregistration.Mosaic, 'LoadFromMosaicFile'), \
                        mock.patch.object(show_mosaic_layout.nornir_imageregistration.mosaic_tileset,
                                          'CreateFromMosaic', side_effect=_Stop) as create, \
                        contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(_Stop):
                    show_mosaic_layout.Execute(argv)
                self.assertTrue(os.path.isdir(create.call_args.kwargs['image_folder']))
            finally:
                os.chdir(os.path.dirname(tmp))

    def test_tile_folder_relative_to_mosaic_in_subfolder(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                os.makedirs(os.path.join('sub', 'tiles'))
                with open(os.path.join('sub', 'in.mosaic'), 'w'):
                    pass
                args, _ = show_mosaic_layout.ParseArgs(['-i', os.path.join('sub', 'in.mosaic'), '-o', 'out.svg',
                                                        '-t', 'tiles'])
                show_mosaic_layout.ValidateArgs(args)
            finally:
                os.chdir(os.path.dirname(tmp))

    def test_missing_tile_folder_exits_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            mosaic_path = os.path.join(tmp, 'in.mosaic')
            with open(mosaic_path, 'w'):
                pass
            argv = ['-i', mosaic_path, '-o', os.path.join(tmp, 'out.svg'), '-t', os.path.join(tmp, 'absent')]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as cm:
                show_mosaic_layout.ValidateArgs(show_mosaic_layout.ParseArgs(argv)[0])
            self.assertEqual(cm.exception.code, 2)

    def test_tile_path_naming_a_file_exits_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            mosaic_path = os.path.join(tmp, 'in.mosaic')
            with open(mosaic_path, 'w'):
                pass
            argv = ['-i', mosaic_path, '-o', os.path.join(tmp, 'out.svg'), '-t', mosaic_path]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as cm:
                show_mosaic_layout.ValidateArgs(show_mosaic_layout.ParseArgs(argv)[0])
            self.assertEqual(cm.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
