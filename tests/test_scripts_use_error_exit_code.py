"""CLI scripts must exit with a non-zero status when argument validation fails.

A bare ``sys.exit()`` exits with status 0, so shell chains and build wrappers that
check the exit code treated a missing input file as success (nornir-review#278).
"""
import os
import tempfile
import unittest
from argparse import Namespace

from hypothesis import example, given, settings
from hypothesis import strategies as st

import nornir_imageregistration.scripts.nornir_translatemosaic as translatemosaic

_missing_names = st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789_', min_size=1, max_size=20)


class TestTranslateMosaicUseErrorExitCode(unittest.TestCase):
    def assertExitsWithError(self, func, *args):
        with self.assertRaises(SystemExit) as raised:
            func(*args)
        self.assertNotIn(raised.exception.code, (None, 0))

    def test_on_use_error_exits_non_zero(self):
        self.assertExitsWithError(translatemosaic.OnUseError, 'bad arguments')

    @settings(max_examples=25, deadline=None)
    @given(name=_missing_names)
    @example(name='missing')
    def test_missing_input_mosaic_exits_non_zero(self, name: str):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, name + '.mosaic')
            output = os.path.join(tmp, 'out.mosaic')
            self.assertExitsWithError(translatemosaic.Execute, ['-i', missing, '-o', output])

    def test_missing_tile_path_exits_non_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            mosaic = os.path.join(tmp, 'in.mosaic')
            with open(mosaic, 'w'):
                pass
            args = Namespace(inputpath=mosaic, outputpath=os.path.join(tmp, 'out.mosaic'),
                             tilepath=os.path.join(tmp, 'no_such_tiles'))
            self.assertExitsWithError(translatemosaic.ValidateArgs, args)


if __name__ == '__main__':
    unittest.main()
