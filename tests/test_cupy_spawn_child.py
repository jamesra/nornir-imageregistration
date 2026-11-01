"""CuPy in spawned multiprocessing children: the ``allow_in_child`` opt-in.

Children refuse CuPy by default. ``SetActiveComputationLib(cupy, allow_in_child=True)``
in a spawn-context worker activates it (with the CUDA preload the main process does);
forked children stay refused.
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing
import sys
import textwrap

import nornir_imageregistration
import numpy as np
import pytest

# Child-side code lives in a module written to tmp_path: spawn children re-import the
# function by module name, which a pytest-collected test module does not reliably allow.
_CHILD_MODULE = textwrap.dedent(
    '''
    from __future__ import annotations

    import numpy as np


    def signal() -> np.ndarray:
        """Deterministic 2D test signal."""
        y, x = np.mgrid[0:64, 0:96]
        return (np.sin(x / 5.0) * np.cos(y / 7.0) + (x * y % 13) / 13.0).astype(np.float64)


    def checksum(spectrum) -> float:
        """Order-independent summary of an FFT result."""
        return float(np.abs(np.asarray(spectrum)).sum())


    def fft_in_child(allow: bool) -> dict:
        """Activate CuPy (optionally opting in), run an FFT, report backend and checksum."""
        import multiprocessing

        import nornir_imageregistration as nir

        report = {"is_child": multiprocessing.parent_process() is not None,
                  "using_cupy_at_import": nir.UsingCupy()}
        try:
            nir.SetActiveComputationLib(nir.ComputationLib.cupy, allow_in_child=allow)
        except RuntimeError as e:
            report["error"] = str(e)
            return report
        xp = nir.GetComputationModule()
        spectrum = xp.fft.fft2(xp.asarray(signal()))
        report["using_cupy"] = nir.UsingCupy()
        report["array_module"] = type(spectrum).__module__.split(".")[0]
        if report["array_module"] == "cupy":
            spectrum = spectrum.get()
        report["checksum"] = checksum(spectrum)
        return report
    '''
)


@pytest.fixture(scope='module')
def child_module(tmp_path_factory: pytest.TempPathFactory):
    """Import path for the child helper module; spawn children inherit ``sys.path``."""
    folder = tmp_path_factory.mktemp('cupy_spawn_child')
    (folder / 'cupy_spawn_child_helpers.py').write_text(_CHILD_MODULE, encoding='utf-8')
    sys.path.insert(0, str(folder))
    try:
        import cupy_spawn_child_helpers  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
        yield cupy_spawn_child_helpers
    finally:
        sys.path.remove(str(folder))
        sys.modules.pop('cupy_spawn_child_helpers', None)


def _run_in(context_name: str, func, *args):
    """Run ``func(*args)`` in one worker of a ``context_name`` pool and return its result."""
    context = multiprocessing.get_context(context_name)
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        return pool.submit(func, *args).result(timeout=300)


def _require_gpu() -> None:
    if not nornir_imageregistration.HasCupy():
        pytest.skip('CuPy not installed')
    previous = nornir_imageregistration.GetActiveComputationLib()
    try:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        if not nornir_imageregistration.UsingCupy():
            pytest.skip('CuPy runtime probe failed (no usable GPU)')
    finally:
        nornir_imageregistration.SetActiveComputationLib(previous)


def test_spawned_child_runs_cupy_fft_with_opt_in(child_module) -> None:
    """A spawned child that opts in runs the FFT on CuPy and matches the NumPy result."""
    _require_gpu()
    report = _run_in('spawn', child_module.fft_in_child, True)
    assert report['is_child'] is True
    assert report['using_cupy_at_import'] is False, 'children still start on NumPy'
    assert 'error' not in report, report
    assert report['using_cupy'] is True
    assert report['array_module'] == 'cupy'
    expected = child_module.checksum(np.fft.fft2(child_module.signal()))
    assert report['checksum'] == pytest.approx(expected, rel=1e-9)


def test_spawned_child_without_opt_in_is_refused(child_module) -> None:
    """Default behaviour is unchanged: CuPy in a child raises without the opt-in."""
    if not nornir_imageregistration.HasCupy():
        pytest.skip('CuPy not installed')
    report = _run_in('spawn', child_module.fft_in_child, False)
    assert report['is_child'] is True
    assert report['using_cupy_at_import'] is False
    assert 'allow_in_child' in report['error']


@pytest.mark.skipif(sys.platform == 'win32', reason='fork start method is not available on Windows')
def test_forked_child_is_refused_even_with_opt_in(child_module) -> None:
    """A forked child may inherit CUDA state, so the opt-in does not apply there."""
    if not nornir_imageregistration.HasCupy():
        pytest.skip('CuPy not installed')
    report = _run_in('fork', child_module.fft_in_child, True)
    assert 'forked child' in report['error']


def test_opt_in_flag_is_inert_in_main_process() -> None:
    """``allow_in_child`` changes nothing in the main process."""
    previous = nornir_imageregistration.GetActiveComputationLib()
    try:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy,
                                                         allow_in_child=True)
        assert not nornir_imageregistration.UsingCupy()
    finally:
        nornir_imageregistration.SetActiveComputationLib(previous)

