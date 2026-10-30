"""
Pytest: enable headless figure output before any test imports nornir_imageregistration.

``nornir_imageregistration`` calls ``matplotlib.use("Agg" if is_headless() else "qtAgg")``
at import time. Set ``NORNIR_HEADLESS`` here so pytest loads this module before
collecting tests, avoiding GUI windows and writing PNG artifacts instead.

This file lives at the package root (not under ``tests/``) so pytest loads it before
collecting modules under ``tests/``.

Override for interactive debugging: ``NORNIR_HEADLESS=0 pytest ...``
"""

from __future__ import annotations

import os

os.environ.setdefault("NORNIR_HEADLESS", "1")

import pytest

# How long to let a child exit on its own after pools are closed, before calling it
# leaked. Graceful multiprocessing teardown is well under a second; this is slack for a
# loaded machine.
_WORKER_EXIT_GRACE_SECONDS = 10.0

# Pool shutdown runs on a timer because it can hang -- wait_completion busy-spins while
# a task is registered active, so a task whose callback never fires blocks it forever.
# A guard against leaks must not itself become the hang it is reporting.
_POOL_SHUTDOWN_TIMEOUT_SECONDS = 60.0

_LEAK_CHECK_OPT_OUT = "NORNIR_ALLOW_LEAKED_WORKERS"


@pytest.fixture(autouse=True)
def _deterministic_padding_noise():
    """Give every test the same padding noise, whatever order it runs in.

    ``pad_image_for_phase_correlation`` fills padding with noise so phase
    correlation has no hard edge to lock onto. That draw is now reproducible per
    run, but the generator still advances between calls, so a test's noise would
    otherwise depend on how many tests ran before it. Reseeding here makes a test
    behave the same alone as in a suite -- the rotating failures in
    ``test_SliceToSliceBrute`` were this.
    """
    import nornir_imageregistration

    nornir_imageregistration.seed_random_data()
    yield


def _describe(proc) -> str:
    """One-line identification of a surviving process, best effort."""
    import psutil

    try:
        cmdline = ' '.join(proc.cmdline()[:4])
    except (psutil.Error, OSError):
        cmdline = '<unavailable>'
    try:
        name = proc.name()
    except (psutil.Error, OSError):
        name = '<unknown>'
    return f'pid {proc.pid} ({name}): {cmdline}'


def _close_pools_with_timeout(timeout: float) -> str | None:
    """Force pools down off-thread. Returns a message if it did not finish in time.

    ``FastClosePools`` terminates process workers rather than waiting for them, which is
    what a test session wants -- a stuck task should not keep workers alive into the next
    run.
    """
    import threading

    outcome: list[str] = []

    def run() -> None:
        try:
            import nornir_pools
        except ImportError:
            return
        try:
            nornir_pools.FastClosePools()
        except Exception as exc:  # a broken pool must not mask the leak report
            outcome.append(f'{type(exc).__name__}: {exc}')

    worker = threading.Thread(target=run, name='nornir-pool-teardown', daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return f'pool shutdown did not finish within {timeout:.0f}s'
    return f'pool shutdown raised {outcome[0]}' if outcome else None


def reap_leaked_workers(grace: float = _WORKER_EXIT_GRACE_SECONDS,
                        shutdown_timeout: float = _POOL_SHUTDOWN_TIMEOUT_SECONDS
                        ) -> tuple[list[str], str | None]:
    """Close pools, then kill and describe any child process still running.

    Returns ``(leaked_descriptions, shutdown_problem)``. Survivors are killed rather than
    only reported: a leak that outlives the session contends for the GPU and silently
    corrupts the next run's timings, which is the damage worth preventing.

    Only descendants of this process are considered, so a developer's unrelated
    interpreters are never touched.
    """
    import time

    import psutil

    shutdown_problem = _close_pools_with_timeout(shutdown_timeout)

    me = psutil.Process()
    deadline = time.monotonic() + grace
    survivors: list[psutil.Process] = []
    while True:
        try:
            survivors = me.children(recursive=True)
        except psutil.Error:
            survivors = []
        if not survivors or time.monotonic() >= deadline:
            break
        time.sleep(0.25)

    leaked = [_describe(p) for p in survivors]
    for proc in survivors:
        try:
            proc.kill()
        except (psutil.Error, OSError):
            pass
    if survivors:
        psutil.wait_procs(survivors, timeout=5)
    return leaked, shutdown_problem


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001 - pytest hook signature
    """Fail the session if it leaked worker processes, and clean them up either way.

    Leaked pool workers used to survive pytest, accumulate across runs, and contend for
    the GPU. That is worse than a plain resource leak: it produced plausible but wrong
    performance numbers (one measurement was off by two orders of magnitude) because the
    contention was invisible at the time. Failing here turns a silent leak into a visible
    one, and killing the survivors stops it poisoning the next run.

    Set ``NORNIR_ALLOW_LEAKED_WORKERS=1`` to reduce this to a printed warning.
    """
    if os.environ.get(_LEAK_CHECK_OPT_OUT, '').strip() not in ('', '0'):
        return

    try:
        leaked, shutdown_problem = reap_leaked_workers()
    except ImportError:
        return  # psutil missing: no guard rather than a collection error

    if not leaked and shutdown_problem is None:
        return

    reporter = session.config.pluginmanager.get_plugin('terminalreporter')

    def report(line: str) -> None:
        if reporter is not None:
            reporter.write_line(line)
        else:
            print(line)

    report('')
    if shutdown_problem is not None:
        report(f'POOL TEARDOWN: {shutdown_problem}')
    if leaked:
        # Counted as OS processes, not logical workers: a venv's Scripts/python.exe is a
        # launcher that spawns the real interpreter, so one leaked child shows up twice.
        report(f'LEAKED {len(leaked)} child process(es) after pool shutdown '
               f'(killed; they would have contended with the next run):')
        for line in leaked[:16]:
            report(f'  {line}')
        if len(leaked) > 16:
            report(f'  ... and {len(leaked) - 16} more')
    report('')

    if session.exitstatus == 0:
        session.exitstatus = 1
