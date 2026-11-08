"""Script-level tests for nornir-assemble (nornir_imageregistration.scripts.nornir_assemble).

Covers what the CLI does on HEAD and is correct: the validation exits, ``-scale`` parsing, the tile
folder and scale handed to ``CreateFromMosaic``, and which arrays reach ``SaveImage`` (the script
unpacks the ``(image, mask)`` tuple from ``AssembleImage``; nornir-assemble-tiles did not).

``-tilepath`` was declared ``type=str, default=True`` (nornir-review#279): omitting it checked
``os.path.exists(True)`` (file descriptor 1) and handed ``image_folder=True`` to
``CreateFromMosaic``, whose ``os.path.join(True, ...)`` raised ``TypeError``. It now defaults to
``None`` and falls back to the mosaic's folder, as translatemosaic and assemble-tiles do.

Not pinned here, because the script is wrong today and a category 13 change must fix it first:
a bare output filename makes ``ValidateArgs`` call ``os.makedirs('')`` (nornir-review#280).
"""
import contextlib
import io
import os
import tempfile
import unittest
from collections.abc import Iterator
from unittest import mock

import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st

import nornir_imageregistration
import nornir_imageregistration.scripts.nornir_assemble as assemble

_dir_names = st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789_', min_size=1, max_size=20)
_scales = st.floats(min_value=0.01, max_value=64.0, allow_nan=False, allow_infinity=False)


class _Run:
    """What one stubbed ``Execute`` call handed to the library functions."""

    def __init__(self) -> None:
        self.image = np.full((4, 5), 7, dtype=np.uint8)
        self.mask = np.ones((4, 5), dtype=bool)
        self.tileset = mock.MagicMock()
        self.tileset.AssembleImage.return_value = (self.image, self.mask)
        self.create = mock.MagicMock(return_value=self.tileset)
        self.save = mock.MagicMock()
        self.set_lib = mock.MagicMock()


@contextlib.contextmanager
def _stubbed_library() -> Iterator[_Run]:
    run = _Run()
    with mock.patch.object(nornir_imageregistration.Mosaic, 'LoadFromMosaicFile'), \
            mock.patch.object(nornir_imageregistration.mosaic_tileset, 'CreateFromMosaic', run.create), \
            mock.patch.object(nornir_imageregistration.core, 'SaveImage', run.save), \
            mock.patch.object(nornir_imageregistration, 'SetActiveComputationLib', run.set_lib), \
            contextlib.redirect_stdout(io.StringIO()):
        yield run


class TestAssembleScript(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.mosaic_path = os.path.join(self.tmp, 'in.mosaic')
        with open(self.mosaic_path, 'w'):
            pass
        self.tile_dir = os.path.join(self.tmp, 'tiles')
        os.makedirs(self.tile_dir)

    def _argv(self, *extra: str, output: str | None = None) -> list[str]:
        return ['-i', self.mosaic_path, '-o', output or os.path.join(self.tmp, 'out.png'), *extra]

    def test_missing_tile_folder_exits_with_error(self):
        for option in ('-p', '-tilepath'):
            with self.subTest(option=option), contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaises(SystemExit) as raised:
                assemble.ValidateArgs(assemble.ParseArgs(self._argv(option, os.path.join(self.tmp, 'absent')))[0])
            self.assertEqual(raised.exception.code, 2)

    def test_missing_input_mosaic_exits_with_error(self):
        argv = ['-i', os.path.join(self.tmp, 'absent.mosaic'), '-o', os.path.join(self.tmp, 'out.png'), '-p', self.tile_dir]
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
            assemble.Execute(argv)
        self.assertEqual(raised.exception.code, 2)

    def test_existing_tile_folder_passes_validation(self):
        assemble.ValidateArgs(assemble.ParseArgs(self._argv('-p', self.tile_dir))[0])

    def test_missing_required_options_exit_with_error(self):
        for argv in ([], ['-i', self.mosaic_path], ['-o', os.path.join(self.tmp, 'out.png')]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as raised:
                assemble.ParseArgs(argv)
            self.assertEqual(raised.exception.code, 2)

    @settings(max_examples=25, deadline=None)
    @given(scale=_scales)
    @example(scale=1.0)
    def test_scale_parses_as_float_under_both_spellings(self, scale: float):
        for option in ('-s', '-scale'):
            args, _ = assemble.ParseArgs(['-i', 'in.mosaic', '-o', 'out.png', option, repr(scale)])
            self.assertEqual(args.scalar, scale)

    def test_scale_defaults_to_one(self):
        args, _ = assemble.ParseArgs(['-i', 'in.mosaic', '-o', 'out.png'])
        self.assertEqual(args.scalar, 1.0)

    @settings(max_examples=25, deadline=None)
    @given(scale=_scales)
    @example(scale=1.0)
    @example(scale=0.25)
    def test_scale_sets_source_space_scale_and_target_space_scale(self, scale: float):
        with _stubbed_library() as run:
            assemble.Execute(self._argv('-p', self.tile_dir, '-s', repr(scale)))
        self.assertEqual(run.create.call_args.kwargs['image_to_source_space_scale'], 1.0 / scale)
        self.assertEqual(run.tileset.AssembleImage.call_args.kwargs['target_space_scale'], scale)

    @settings(max_examples=25, deadline=None)
    @given(tile_dir_name=st.one_of(st.none(), _dir_names))
    @example(tile_dir_name=None)
    @example(tile_dir_name='tiles')
    def test_tile_folder_is_given_directory_or_mosaic_directory(self, tile_dir_name: str | None):
        if tile_dir_name is None:
            extra: tuple[str, ...] = ()
            expected = self.tmp
        else:
            expected = os.path.join(self.tmp, 'tiles_' + tile_dir_name)
            os.makedirs(expected, exist_ok=True)
            extra = ('-p', expected)
        with _stubbed_library() as run:
            assemble.Execute(self._argv(*extra))
        self.assertEqual(run.create.call_args.args[1], expected)

    def test_tilepath_defaults_to_none(self):
        args, _ = assemble.ParseArgs(['-i', 'in.mosaic', '-o', 'out.png'])
        self.assertIsNone(args.tilepath)

    def test_bare_mosaic_filename_uses_working_directory(self):
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.tmp)
        args, _ = assemble.ParseArgs(['-i', 'in.mosaic', '-o', os.path.join(self.tmp, 'out.png')])
        assemble.ValidateArgs(args)
        self.assertEqual(os.path.normcase(args.tilepath), os.path.normcase(os.getcwd()))

    def test_image_and_mask_are_saved_not_the_tuple(self):
        out = os.path.join(self.tmp, 'out.png')
        with _stubbed_library() as run:
            assemble.Execute(self._argv('-p', self.tile_dir, output=out))
        self.assertEqual(run.tileset.TranslateToZeroOrigin.call_count, 1)
        self.assertEqual(run.save.call_count, 2)
        (image_call, mask_call) = run.save.call_args_list
        self.assertEqual(image_call.args[0], out)
        self.assertIs(image_call.args[1], run.image)
        self.assertEqual(image_call.kwargs, {'bpp': 8})
        self.assertEqual(mask_call.args[0], os.path.join(self.tmp, 'out_mask.png'))
        self.assertIs(mask_call.args[1], run.mask)

    def test_output_without_extension_is_saved_as_png(self):
        with _stubbed_library() as run:
            assemble.Execute(self._argv('-p', self.tile_dir, output=os.path.join(self.tmp, 'result')))
        self.assertEqual(run.save.call_args_list[0].args[0], os.path.join(self.tmp, 'result.png'))

    def test_output_directory_gets_a_png_named_after_the_mosaic(self):
        out_dir = os.path.join(self.tmp, 'results')
        os.makedirs(out_dir)
        with _stubbed_library() as run:
            assemble.Execute(self._argv('-p', self.tile_dir, output=out_dir))
        self.assertEqual(run.save.call_args_list[0].args[0], os.path.join(out_dir, 'in.png'))

    def test_cuda_flag_selects_the_computation_library(self):
        for extra, expected in (((), nornir_imageregistration.ComputationLib.numpy),
                                (('-cuda',), nornir_imageregistration.ComputationLib.cupy),
                                (('-c',), nornir_imageregistration.ComputationLib.cupy)):
            with self.subTest(extra=extra), _stubbed_library() as run:
                assemble.Execute(self._argv('-p', self.tile_dir, *extra))
            run.set_lib.assert_called_once_with(expected)


if __name__ == '__main__':
    unittest.main()
