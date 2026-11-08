"""CLI scripts must accept a bare output filename (no directory part).

``os.path.dirname('out.stos')`` is ``''``; the scripts used to call ``os.makedirs('')``
on it, which raises ``FileNotFoundError`` before any work is done.
"""
import os
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

from hypothesis import example, given, settings
from hypothesis import strategies as st

import nornir_imageregistration.scripts.nornir_addtransforms as addtransforms
import nornir_imageregistration.scripts.nornir_assemble as assemble
import nornir_imageregistration.scripts.nornir_assemble_tiles as assemble_tiles
import nornir_imageregistration.scripts.nornir_rotate_translate as rotate_translate
import nornir_imageregistration.scripts.nornir_scaletransform as scaletransform
import nornir_imageregistration.scripts.nornir_slicetomosaic as slicetomosaic
import nornir_imageregistration.scripts.nornir_translatemosaic as translatemosaic

_dir_parts = st.lists(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789_', min_size=1, max_size=12),
                      min_size=0, max_size=3)


class _Stop(Exception):
    """Raised by a mock to end rotate_translate.Execute right after the directory step."""


class TestScriptsBareOutputPath(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._old_cwd = os.getcwd()
        self.addCleanup(os.chdir, self._old_cwd)
        os.chdir(self._tmp.name)
        for name in ('a.stos', 'b.stos', 'in.mosaic'):
            with open(name, 'w'):
                pass

    def test_addtransforms_bare_output(self):
        addtransforms.ValidateArgs(Namespace(fixedpath='a.stos', warpedpath='b.stos', outputpath='out.stos'))

    def test_scaletransform_bare_output(self):
        scaletransform.ValidateArgs(Namespace(inputpath='a.stos', outputpath='out.stos', scalefactor=2.0))

    def test_assemble_tiles_bare_output(self):
        assemble_tiles.ValidateArgs(Namespace(inputpath='in.mosaic', outputpath='out.png', tilepath=None))

    @settings(max_examples=25, deadline=None)
    @given(parts=_dir_parts)
    @example(parts=[])
    @example(parts=['new', 'deeper'])
    def test_assemble_creates_relative_output_directory(self, parts: list[str]):
        with tempfile.TemporaryDirectory() as example_dir:
            os.chdir(example_dir)
            try:
                with open('in.mosaic', 'w'):
                    pass
                out = os.path.join(*parts, 'out.png')
                assemble.ValidateArgs(Namespace(inputpath='in.mosaic', outputpath=out, tilepath=None))
                self.assertTrue(os.path.isdir(os.path.dirname(os.path.abspath(out))))
                assemble.ValidateArgs(Namespace(inputpath='in.mosaic', outputpath=out, tilepath=None))
            finally:
                os.chdir(self._tmp.name)

    @settings(max_examples=25, deadline=None)
    @given(parts=_dir_parts)
    @example(parts=[])
    @example(parts=['new', 'deeper'])
    def test_translatemosaic_creates_relative_output_directory(self, parts: list[str]):
        with tempfile.TemporaryDirectory() as example_dir:
            os.chdir(example_dir)
            try:
                with open('in.mosaic', 'w'):
                    pass
                out = os.path.join(*parts, 'out.mosaic')
                translatemosaic.ValidateArgs(Namespace(inputpath='in.mosaic', outputpath=out, tilepath=example_dir))
                self.assertTrue(os.path.isdir(os.path.dirname(os.path.abspath(out))))
                self.assertFalse(os.path.exists(out))
                translatemosaic.ValidateArgs(Namespace(inputpath='in.mosaic', outputpath=out, tilepath=example_dir))
            finally:
                os.chdir(self._tmp.name)

    def test_slicetomosaic_bare_output(self):
        slicetomosaic.ValidateArgs(Namespace(inputpath='a.stos', outputpath='out.stos',
                                             fixedimagepath=None, warpedimagepath=None))

    def test_rotate_translate_bare_output(self):
        with mock.patch.object(rotate_translate, 'StosOverrideArgs'), \
                mock.patch.object(rotate_translate.sb, 'SliceToSliceRigidRegistration', side_effect=_Stop), \
                self.assertRaises(_Stop):
            rotate_translate.Execute(['-f', 'a.png', '-w', 'b.png', '-o', 'out.stos'])

    def test_nested_output_directory_created_and_existing_tolerated(self):
        out = os.path.join('new', 'deeper', 'out.stos')
        args = Namespace(fixedpath='a.stos', warpedpath='b.stos', outputpath=out)
        addtransforms.ValidateArgs(args)
        self.assertTrue(os.path.isdir(os.path.join('new', 'deeper')))
        addtransforms.ValidateArgs(args)


if __name__ == '__main__':
    unittest.main()
