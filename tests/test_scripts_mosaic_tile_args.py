"""nornir-assemble, nornir-assemble-tiles and nornir-translatemosaic validate -i/-o/-tilepath alike.

All three read a .mosaic plus its tile folder and share ``_cli_paths.validate_mosaic_tile_args``: a missing
input mosaic exits 2 before anything is created, the output folder is created next, a missing tile path
becomes the mosaic's directory, and a tile path that does not exist exits 2.
"""
import contextlib
import io
import os
import tempfile
import unittest
from argparse import Namespace
from types import ModuleType

from hypothesis import example, given, settings
from hypothesis import strategies as st

import nornir_imageregistration.scripts.nornir_assemble as assemble
import nornir_imageregistration.scripts.nornir_assemble_tiles as assemble_tiles
import nornir_imageregistration.scripts.nornir_translatemosaic as translatemosaic

_SCRIPTS: tuple[ModuleType, ...] = (assemble, assemble_tiles, translatemosaic)

_dir_parts = st.lists(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789_', min_size=1, max_size=12),
                      min_size=0, max_size=3)


class TestMosaicTileArgs(unittest.TestCase):
    def _check(self, script: ModuleType, input_exists: bool, tile_case: str, out_parts: list[str]) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mosaic = os.path.join(tmp, 'in.mosaic')
            if input_exists:
                with open(mosaic, 'w'):
                    pass
            missing_tiles = os.path.join(tmp, 'absent')
            tilepath = None
            if tile_case == 'existing':
                tilepath = os.path.join(tmp, 'tiles')
                os.makedirs(tilepath)
            elif tile_case == 'missing':
                tilepath = missing_tiles
            out = os.path.join(tmp, *out_parts, 'out.png')
            out_dir = os.path.dirname(out)
            args = Namespace(inputpath=mosaic, outputpath=out, tilepath=tilepath)

            expected_error = None
            if not input_exists:
                expected_error = 'Input mosaic file not found: ' + mosaic
            elif tile_case == 'missing':
                expected_error = 'Tile path not found: ' + missing_tiles

            if expected_error is None:
                script.ValidateArgs(args)
                self.assertEqual(args.tilepath, tmp if tile_case == 'default' else tilepath)
            else:
                with self.assertLogs(level='ERROR') as logs, contextlib.redirect_stdout(io.StringIO()), \
                        self.assertRaises(SystemExit) as raised:
                    script.ValidateArgs(args)
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual([r.getMessage() for r in logs.records], [expected_error])

            if out_parts:
                self.assertEqual(os.path.isdir(out_dir), input_exists)
            self.assertFalse(os.path.exists(out))

    @settings(max_examples=60, deadline=None)
    @given(script=st.sampled_from(_SCRIPTS), input_exists=st.booleans(),
           tile_case=st.sampled_from(['default', 'existing', 'missing']), out_parts=_dir_parts)
    @example(script=assemble, input_exists=False, tile_case='existing', out_parts=['new'])
    @example(script=assemble_tiles, input_exists=True, tile_case='missing', out_parts=['new', 'deeper'])
    @example(script=translatemosaic, input_exists=True, tile_case='default', out_parts=[])
    @example(script=translatemosaic, input_exists=False, tile_case='default', out_parts=['new'])
    def test_scripts_validate_mosaic_tile_args_alike(self, script: ModuleType, input_exists: bool, tile_case: str,
                                                     out_parts: list[str]):
        self._check(script, input_exists, tile_case, out_parts)


if __name__ == '__main__':
    unittest.main()
