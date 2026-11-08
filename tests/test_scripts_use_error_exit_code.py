"""CLI scripts must exit with a non-zero status when argument validation fails.

A bare ``sys.exit()`` exits with status 0, so shell chains and build wrappers that
check the exit code treated a missing input file as success (nornir-review#278).
"""
import ast
import contextlib
import importlib
import io
import os
import tempfile
import unittest
from argparse import Namespace
from collections.abc import Callable
from types import ModuleType

from hypothesis import example, given, settings
from hypothesis import strategies as st

import nornir_imageregistration.scripts
import nornir_imageregistration.scripts.nornir_translatemosaic as translatemosaic

_missing_names = st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789_', min_size=1, max_size=20)

_SCRIPTS_DIR = os.path.dirname(nornir_imageregistration.scripts.__file__)


def _missing_input_argv(missing: str, tmp: str) -> list[str]:
    return ['-i', missing, '-o', os.path.join(tmp, 'out')]


# Script module -> (logger its OnUseError writes to, argv builder naming a missing input file).
_USE_ERROR_SCRIPTS: dict[str, tuple[str, Callable[[str, str], list[str]]]] = {
    'nornir_addtransforms': ('AddTransforms',
                             lambda missing, tmp: ['-o', os.path.join(tmp, 'out.stos'), '-f', missing, '-w', missing]),
    'nornir_assemble': ('nornir-assemble', _missing_input_argv),
    'nornir_assemble_tiles': ('nornir-assemble-tiles', _missing_input_argv),
    'nornir_registerimages': ('RegisterImages', lambda missing, tmp: ['-f', missing, '-w', missing]),
    'nornir_scaletransform': ('AddTransforms', lambda missing, tmp: _missing_input_argv(missing, tmp) + ['-s', '2']),
    'nornir_show_mosaic_layout': ('nornir-assemble', lambda missing, tmp: _missing_input_argv(missing, tmp) + ['-t', tmp]),
    'nornir_slicetomosaic': ('Ir-Stom', _missing_input_argv),
    'nornir_translatemosaic': ('nornir-assemble', _missing_input_argv),
}


def _load_script(name: str) -> ModuleType:
    return importlib.import_module('nornir_imageregistration.scripts.' + name)


def _script_trees() -> dict[str, ast.Module]:
    trees = {}
    for filename in sorted(os.listdir(_SCRIPTS_DIR)):
        if filename.endswith('.py'):
            with open(os.path.join(_SCRIPTS_DIR, filename), encoding='utf-8') as f:
                trees[filename[:-3]] = ast.parse(f.read(), filename=filename)
    return trees


def _is_sys_exit_without_error_status(node: ast.Call) -> bool:
    if not (isinstance(node.func, ast.Attribute)
            and node.func.attr == 'exit' and isinstance(node.func.value, ast.Name) and node.func.value.id == 'sys'):
        return False
    if not node.args:
        return True
    status = node.args[0]
    return isinstance(status, ast.Constant) and status.value in (None, 0)


class TestScriptsUseErrorExitCode(unittest.TestCase):
    def assertExitsWithError(self, func, *args):
        with self.assertRaises(SystemExit) as raised:
            func(*args)
        self.assertNotIn(raised.exception.code, (None, 0))

    def test_on_use_error_prints_usage_logs_and_exits_2(self):
        for name, (logger_name, _) in _USE_ERROR_SCRIPTS.items():
            with self.subTest(script=name):
                stdout = io.StringIO()
                with (self.assertLogs(logger_name, level='ERROR') as logs,
                      contextlib.redirect_stdout(stdout),
                      self.assertRaises(SystemExit) as raised):
                    _load_script(name).OnUseError('bad arguments')
                self.assertEqual(raised.exception.code, 2)
                self.assertIn('usage:', stdout.getvalue())
                self.assertEqual([r.getMessage() for r in logs.records], ['bad arguments'])

    @settings(max_examples=40, deadline=None)
    @given(name=st.sampled_from(sorted(_USE_ERROR_SCRIPTS)), missing_name=_missing_names)
    @example(name='nornir_translatemosaic', missing_name='missing')
    @example(name='nornir_addtransforms', missing_name='missing')
    @example(name='nornir_registerimages', missing_name='missing')
    def test_missing_input_exits_non_zero(self, name: str, missing_name: str):
        build_argv = _USE_ERROR_SCRIPTS[name][1]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            missing = os.path.join(tmp, missing_name + '.missing')
            self.assertExitsWithError(_load_script(name).Execute, build_argv(missing, tmp))

    def test_missing_tile_path_exits_non_zero(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            mosaic = os.path.join(tmp, 'in.mosaic')
            with open(mosaic, 'w'):
                pass
            args = Namespace(inputpath=mosaic, outputpath=os.path.join(tmp, 'out.mosaic'),
                             tilepath=os.path.join(tmp, 'no_such_tiles'))
            self.assertExitsWithError(translatemosaic.ValidateArgs, args)

    def test_no_script_exits_with_success_status_on_error(self):
        offenders = [f'{name}.py:{node.lineno}'
                     for name, tree in _script_trees().items()
                     for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and _is_sys_exit_without_error_status(node)]
        self.assertEqual(offenders, [])

    def test_every_on_use_error_script_is_covered(self):
        defining = {name for name, tree in _script_trees().items()
                    if any(isinstance(node, ast.FunctionDef) and node.name == 'OnUseError' for node in tree.body)}
        self.assertEqual(defining, set(_USE_ERROR_SCRIPTS))


if __name__ == '__main__':
    unittest.main()
