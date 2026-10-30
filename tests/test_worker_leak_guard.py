"""The session leak guard has to actually catch a leak, not just always pass.

Leaked pool workers survived pytest, accumulated across runs and contended for the GPU.
That is worse than a plain resource leak: the contention is invisible, so it produced
plausible but wrong performance numbers -- one measurement on review issue #213 was off
by two orders of magnitude and was used to set that issue's severity. ``conftest`` closes
pools at session end and fails the run if any child survives.

A guard nothing exercises is worth very little, so these tests plant a deliberate stray
child and assert it is both reported and killed. See review issue #217.
"""

from __future__ import annotations

import subprocess
import sys
import time
import unittest

import psutil

from conftest import reap_leaked_workers


def _spawn_stray() -> subprocess.Popen:
    """A child that outlives its parent unless something kills it."""
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if psutil.pid_exists(proc.pid):
            return proc
        time.sleep(0.05)
    raise AssertionError('stray child never started')


class TestTheGuardCatchesAStray(unittest.TestCase):

    def setUp(self):
        self.stray: subprocess.Popen | None = None

    def tearDown(self):
        if self.stray is not None and self.stray.poll() is None:
            self.stray.kill()
            self.stray.wait(timeout=30)

    def test_a_surviving_child_is_reported(self):
        self.stray = _spawn_stray()

        # grace=0: the stray sleeps for 600s, so waiting longer only slows the test.
        leaked, problem = reap_leaked_workers(grace=0.0)

        self.assertTrue(any(str(self.stray.pid) in line for line in leaked),
                        f'stray {self.stray.pid} missing from {leaked}')
        self.assertIsNone(problem, 'pool shutdown should not have been a problem here')

    def test_a_surviving_child_is_killed_not_merely_counted(self):
        """Reporting alone would still let the stray poison the next run."""
        self.stray = _spawn_stray()
        pid = self.stray.pid

        reap_leaked_workers(grace=0.0)

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.stray.poll() is not None:
                break
            time.sleep(0.1)
        self.assertIsNotNone(self.stray.poll(), f'stray {pid} survived the reaper')

    def test_the_description_identifies_the_process(self):
        """A bare count is not actionable; the report must name what leaked."""
        self.stray = _spawn_stray()

        leaked, _ = reap_leaked_workers(grace=0.0)

        line = next(ln for ln in leaked if str(self.stray.pid) in ln)
        self.assertIn('python', line.lower())

    def test_several_strays_are_all_caught(self):
        strays = [_spawn_stray() for _ in range(3)]
        try:
            pids = {p.pid for p in strays}

            leaked, _ = reap_leaked_workers(grace=0.0)

            for pid in pids:
                self.assertTrue(any(str(pid) in ln for ln in leaked),
                                f'{pid} not reported')
        finally:
            for p in strays:
                if p.poll() is None:
                    p.kill()
                    p.wait(timeout=30)

    def test_the_count_is_os_processes_not_logical_children(self):
        """A venv ``Scripts/python.exe`` is a launcher that spawns the real interpreter,
        so one stray can appear twice. Documented so the count is not read as a bug."""
        self.stray = _spawn_stray()

        leaked, _ = reap_leaked_workers(grace=0.0)

        self.assertGreaterEqual(len(leaked), 1)


class TestTheGuardIsQuietWhenNothingLeaked(unittest.TestCase):
    """A guard that cries wolf gets disabled, so the clean case must be clean."""

    def test_no_children_means_nothing_reported(self):
        me = psutil.Process()
        for child in me.children(recursive=True):
            try:
                child.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(me.children(recursive=True), timeout=10)

        leaked, problem = reap_leaked_workers(grace=0.0)

        self.assertEqual(leaked, [])
        self.assertIsNone(problem)

    def test_it_only_looks_at_our_own_descendants(self):
        """It must never touch a developer's unrelated interpreters."""
        import inspect

        source = inspect.getsource(reap_leaked_workers)

        self.assertIn('children(recursive=True)', source)
        self.assertNotIn('process_iter', source,
                         'a global process sweep would kill unrelated work')


class TestPoolShutdownCannotWedgeTheSession(unittest.TestCase):
    """The teardown it performs is exactly the one known to hang, so it must be bounded."""

    def test_a_hanging_shutdown_is_reported_rather_than_waited_on(self):
        import conftest

        original = conftest._close_pools_with_timeout
        try:
            conftest._close_pools_with_timeout = (
                lambda timeout: 'pool shutdown did not finish within 60s')
            leaked, problem = reap_leaked_workers(grace=0.0)
        finally:
            conftest._close_pools_with_timeout = original

        self.assertIsNotNone(problem)
        self.assertIn('did not finish', problem)

    def test_the_shutdown_runs_off_thread_with_a_join_timeout(self):
        import inspect

        from conftest import _close_pools_with_timeout

        source = inspect.getsource(_close_pools_with_timeout)

        self.assertIn('join(timeout)', source)
        self.assertIn('daemon=True', source)
        self.assertIn('FastClosePools', source,
                      'a graceful close can block on a stuck task')


if __name__ == '__main__':
    unittest.main()
