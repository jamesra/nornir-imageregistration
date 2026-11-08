"""nornir-assemble-tiles must accept a tile directory and run without one (nornir-review#279).

``-tilepath`` was declared ``type=float, default=1.0`` (the bug nornir-translatemosaic had): a
directory path failed argparse ("invalid float value") and omitting it made ``ValidateArgs``
call ``os.path.exists(1.0)``, which raises ``TypeError``, so the script could not run at all.
"""
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st

import nornir_imageregistration.scripts.nornir_assemble_tiles as assemble_tiles

_dir_names = st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789_', min_size=1, max_size=20)


class _Stop(Exception):
    """Raised by the tileset stub to end Execute once the tile folder is known."""


class TestAssembleTilesTilePath(unittest.TestCase):
    def _image_folder_used(self, tmp: str, tile_dir_name: str | None) -> str:
        """Run Execute on an empty mosaic file and return the image_folder handed to CreateFromMosaic."""
        mosaic_path = os.path.join(tmp, 'in.mosaic')
        with open(mosaic_path, 'w'):
            pass
        argv = ['-i', mosaic_path, '-o', os.path.join(tmp, 'out.png')]
        if tile_dir_name is not None:
            tile_dir = os.path.join(tmp, tile_dir_name)
            os.makedirs(tile_dir, exist_ok=True)
            argv += ['-tilepath', tile_dir]

        with mock.patch.object(assemble_tiles.nornir_imageregistration.Mosaic, 'LoadFromMosaicFile'), \
                mock.patch.object(assemble_tiles.nornir_imageregistration.mosaic_tileset, 'CreateFromMosaic',
                                  side_effect=_Stop) as create, \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(_Stop):
            assemble_tiles.Execute(argv)
        return create.call_args.kwargs['image_folder']

    @settings(max_examples=25, deadline=None)
    @given(tile_dir_name=st.one_of(st.none(), _dir_names))
    @example(tile_dir_name=None)
    @example(tile_dir_name='tiles')
    def test_tile_folder_is_given_directory_or_mosaic_directory(self, tile_dir_name: str | None):
        with tempfile.TemporaryDirectory() as tmp:
            expected = tmp if tile_dir_name is None else os.path.join(tmp, tile_dir_name)
            self.assertEqual(self._image_folder_used(tmp, tile_dir_name), expected)

    def test_bare_mosaic_filename_uses_working_directory(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(tmp.name)
        with open('in.mosaic', 'w'):
            pass
        args, _ = assemble_tiles.ParseArgs(['-i', 'in.mosaic', '-o', 'out.png'])
        assemble_tiles.ValidateArgs(args)
        self.assertEqual(os.path.normcase(args.tilepath), os.path.normcase(os.getcwd()))

    def test_missing_tile_folder_exits_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            mosaic_path = os.path.join(tmp, 'in.mosaic')
            with open(mosaic_path, 'w'):
                pass
            argv = ['-i', mosaic_path, '-o', os.path.join(tmp, 'out.png'), '-p', os.path.join(tmp, 'absent')]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as cm:
                assemble_tiles.ValidateArgs(assemble_tiles.ParseArgs(argv)[0])
            self.assertEqual(cm.exception.code, 2)

    def test_short_option_parses_as_path(self):
        args, _ = assemble_tiles.ParseArgs(['-i', 'in.mosaic', '-o', 'out.png', '-p', os.path.join('a', 'b')])
        self.assertEqual(args.tilepath, os.path.join('a', 'b'))

    def test_tilepath_defaults_to_none(self):
        args, _ = assemble_tiles.ParseArgs(['-i', 'in.mosaic', '-o', 'out.png'])
        self.assertIsNone(args.tilepath)

    def test_saves_assembled_image_not_image_mask_tuple(self):
        """AssembleImage returns (image, mask); SaveImage must get the image array alone."""
        image = np.full((4, 6), 0.5, dtype=np.float32)
        mask = np.ones((4, 6), dtype=bool)
        tileset = mock.Mock()
        tileset.AssembleImage.return_value = (image, mask)
        with tempfile.TemporaryDirectory() as tmp:
            mosaic_path = os.path.join(tmp, 'in.mosaic')
            with open(mosaic_path, 'w'):
                pass
            output_path = os.path.join(tmp, 'out.png')
            with mock.patch.object(assemble_tiles.nornir_imageregistration.Mosaic, 'LoadFromMosaicFile'), \
                    mock.patch.object(assemble_tiles.nornir_imageregistration.mosaic_tileset, 'CreateFromMosaic',
                                      return_value=tileset), \
                    mock.patch.object(assemble_tiles.nornir_imageregistration, 'SaveImage') as save, \
                    contextlib.redirect_stdout(io.StringIO()):
                assemble_tiles.Execute(['-i', mosaic_path, '-o', output_path])
        save.assert_called_once()
        saved_path, saved_image = save.call_args.args
        self.assertEqual(saved_path, output_path)
        self.assertIs(saved_image, image)


if __name__ == '__main__':
    unittest.main()
