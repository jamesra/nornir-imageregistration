"""Subprocess and fork-pool tests for computational library startup.

Covers fresh-interpreter ``init_computational_library`` (as build does) and fork-pool
worker inheritance for each requested/effective backend combination.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import textwrap
import unittest

import nornir_imageregistration
import nornir_pools
from nornir_imageregistration.computational_lib import (
    NORNIR_COMPUTATIONAL_LIBRARY_ENV,
    NORNIR_COMPUTATIONAL_LIBRARY_REQUESTED_ENV,
    NORNIR_POOL_WORKER_ENV,
    ComputationLib,
    SetActiveComputationLib,
)

try:
    from nornir_buildmanager.build import init_computational_library
except ImportError:  # pragma: no cover - optional in minimal installs
    init_computational_library = None  # type: ignore[assignment,misc]


def _fork_pool_worker_snapshot() -> dict[str, object]:
    """Read backend state inside a fork-pool worker (must be picklable)."""
    import nornir_imageregistration as nir

    return {
        "using_cupy": nir.UsingCupy(),
        "effective_env": os.environ.get(NORNIR_COMPUTATIONAL_LIBRARY_ENV),
        "requested_env": os.environ.get(NORNIR_COMPUTATIONAL_LIBRARY_REQUESTED_ENV),
        "pool_worker": os.environ.get(NORNIR_POOL_WORKER_ENV),
    }


def _fresh_subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a child Python that should not inherit parent Nornir backend state."""
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("NORNIR_"):
            del env[key]
    if extra:
        env.update(extra)
    return env


def _run_init_in_subprocess(requested: str, extra_env: dict[str, str] | None = None) -> dict:
    """Run ``init_computational_library`` in a fresh interpreter; return parsed JSON."""
    if init_computational_library is None:
        raise unittest.SkipTest("nornir_buildmanager is not installed")

    script = textwrap.dedent(
        f"""
        import argparse
        import json
        import os

        from nornir_buildmanager.build import init_computational_library
        import nornir_imageregistration

        args = argparse.Namespace(computational_library={requested!r})
        init_computational_library(args)

        print(json.dumps({{
            "requested": os.environ.get({NORNIR_COMPUTATIONAL_LIBRARY_REQUESTED_ENV!r}),
            "effective_env": os.environ.get({NORNIR_COMPUTATIONAL_LIBRARY_ENV!r}),
            "using_cupy": nornir_imageregistration.UsingCupy(),
            "has_cupy": nornir_imageregistration.HasCupy(),
            "cuda_path": os.environ.get("CUDA_PATH"),
        }}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=_fresh_subprocess_env(extra_env),
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"subprocess init({requested!r}) failed (exit {completed.returncode}):\n"
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
        )
    line = completed.stdout.strip().splitlines()[-1]
    return json.loads(line)


def _run_pool_worker_after_parent(
    parent_setup,
    pool_label: str = "test",
) -> dict[str, object]:
    """Configure the parent backend, fork one pool worker, and snapshot its state."""
    del pool_label  # reserved for debugging failed matrix cases
    parent_setup()
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(
        processes=1,
        initializer=nornir_pools.init_pool_process,
        initargs=(None, logging.WARNING),
    ) as pool:
        return pool.apply(_fork_pool_worker_snapshot)


class TestFreshSubprocessInit(unittest.TestCase):
    """``init_computational_library`` in an isolated subprocess (build entry path)."""

    def test_numpy_subprocess(self):
        result = _run_init_in_subprocess("numpy")
        self.assertEqual(result["requested"], "numpy")
        self.assertEqual(result["effective_env"], "numpy")
        self.assertFalse(result["using_cupy"])

    def test_cupy_subprocess(self):
        if not nornir_imageregistration.HasCupy():
            self.skipTest("CuPy not installed")

        result = _run_init_in_subprocess("cupy")
        self.assertEqual(result["requested"], "cupy")
        self.assertTrue(result["has_cupy"])
        # Effective env must match probe outcome (no stale pre-probe cupy env).
        self.assertEqual(result["using_cupy"], result["effective_env"] == "cupy")
        if result["using_cupy"]:
            self.assertEqual(result["effective_env"], "cupy")
            self.assertTrue(result["cuda_path"])

    def test_detect_subprocess(self):
        result = _run_init_in_subprocess("detect")
        if result["has_cupy"] and result["using_cupy"]:
            self.assertEqual(result["requested"], "cupy")
            self.assertEqual(result["effective_env"], "cupy")
        else:
            self.assertEqual(result["requested"], "numpy")
            self.assertEqual(result["effective_env"], "numpy")
            self.assertFalse(result["using_cupy"])


class TestForkPoolWorkerCombinations(unittest.TestCase):
    """Fork-pool workers after each parent backend configuration."""

    def _parent_numpy(self):
        os.environ[NORNIR_COMPUTATIONAL_LIBRARY_REQUESTED_ENV] = "numpy"
        SetActiveComputationLib(ComputationLib.numpy)

    def _parent_cupy(self):
        if init_computational_library is None:
            raise unittest.SkipTest("nornir_buildmanager is not installed")
        init_computational_library(argparse.Namespace(computational_library="cupy"))

    def _parent_cupy_requested_numpy_effective(self):
        """Simulate CuPy probe failure: requested GPU, effective CPU in parent."""
        os.environ[NORNIR_COMPUTATIONAL_LIBRARY_REQUESTED_ENV] = "cupy"
        SetActiveComputationLib(ComputationLib.numpy)

    def test_main_process_init_pool_process_is_noop(self):
        if not nornir_imageregistration.HasCupy():
            self.skipTest("CuPy not installed")

        SetActiveComputationLib(ComputationLib.cupy)
        if not nornir_imageregistration.UsingCupy():
            self.skipTest("CuPy runtime probe failed in this environment")

        prev_effective = os.environ.get(NORNIR_COMPUTATIONAL_LIBRARY_ENV)
        prev_pool_worker = os.environ.get(NORNIR_POOL_WORKER_ENV)
        try:
            nornir_pools.init_pool_process()
            self.assertTrue(nornir_imageregistration.UsingCupy())
            self.assertEqual(os.environ.get(NORNIR_COMPUTATIONAL_LIBRARY_ENV), prev_effective)
            self.assertEqual(os.environ.get(NORNIR_POOL_WORKER_ENV), prev_pool_worker)
        finally:
            os.environ.pop(NORNIR_POOL_WORKER_ENV, None)

    def test_fork_worker_after_parent_numpy(self):
        worker = _run_pool_worker_after_parent(self._parent_numpy, "test-fork-numpy")
        self.assertFalse(worker["using_cupy"])
        self.assertEqual(worker["requested_env"], "numpy")
        self.assertEqual(worker["effective_env"], "numpy")
        self.assertEqual(worker["pool_worker"], "1")

    def test_fork_worker_after_parent_cupy(self):
        if not nornir_imageregistration.HasCupy():
            self.skipTest("CuPy not installed")
        self._parent_cupy()
        if not nornir_imageregistration.UsingCupy():
            self.skipTest("CuPy runtime probe failed in this environment")

        worker = _run_pool_worker_after_parent(lambda: None, "test-fork-cupy")
        self.assertFalse(worker["using_cupy"])
        self.assertEqual(worker["requested_env"], "cupy")
        self.assertEqual(worker["effective_env"], "cupy")
        self.assertEqual(worker["pool_worker"], "1")
        self.assertTrue(nornir_imageregistration.UsingCupy())

    def test_fork_worker_after_cupy_probe_fallback(self):
        if not nornir_imageregistration.HasCupy():
            self.skipTest("CuPy not installed")

        worker = _run_pool_worker_after_parent(
            self._parent_cupy_requested_numpy_effective,
            "test-fork-cupy-fallback",
        )
        self.assertFalse(worker["using_cupy"])
        self.assertEqual(worker["requested_env"], "cupy")
        self.assertEqual(worker["effective_env"], "numpy")
        self.assertEqual(worker["pool_worker"], "1")

    def test_parent_cupy_survives_pool_start(self):
        """Main process must stay on CuPy after fork pool workers are spawned."""
        if not nornir_imageregistration.HasCupy():
            self.skipTest("CuPy not installed")
        if init_computational_library is None:
            self.skipTest("nornir_buildmanager is not installed")

        init_computational_library(argparse.Namespace(computational_library="cupy"))
        if not nornir_imageregistration.UsingCupy():
            self.skipTest("CuPy runtime probe failed in this environment")

        worker = _run_pool_worker_after_parent(lambda: None, "parent-cupy-survives")
        self.assertTrue(nornir_imageregistration.UsingCupy())
        self.assertEqual(os.environ.get(NORNIR_COMPUTATIONAL_LIBRARY_ENV), "cupy")
        self.assertFalse(worker["using_cupy"])


class TestSubprocessLaunchMatrix(unittest.TestCase):
    """Parametric matrix: requested backend x launch style."""

    LAUNCH_STYLES = ("fresh_subprocess", "fork_pool_worker")

    def test_all_combinations(self):
        if init_computational_library is None:
            self.skipTest("nornir_buildmanager is not installed")

        for requested in ("numpy", "cupy", "detect"):
            for launch in self.LAUNCH_STYLES:
                with self.subTest(requested=requested, launch=launch):
                    if launch == "fresh_subprocess":
                        result = _run_init_in_subprocess(requested)
                        self.assertEqual(
                            result["using_cupy"],
                            result["effective_env"] == "cupy",
                        )
                        if requested == "numpy":
                            self.assertFalse(result["using_cupy"])
                        continue

                    # fork_pool_worker launch
                    if requested == "numpy":
                        os.environ[NORNIR_COMPUTATIONAL_LIBRARY_REQUESTED_ENV] = "numpy"
                        SetActiveComputationLib(ComputationLib.numpy)
                        expected_effective = "numpy"
                    elif requested == "cupy":
                        if not nornir_imageregistration.HasCupy():
                            self.skipTest("CuPy not installed")
                        init_computational_library(
                            argparse.Namespace(computational_library="cupy")
                        )
                        expected_effective = (
                            "cupy"
                            if nornir_imageregistration.UsingCupy()
                            else "numpy"
                        )
                    else:
                        init_computational_library(
                            argparse.Namespace(computational_library="detect")
                        )
                        expected_effective = (
                            "cupy"
                            if nornir_imageregistration.UsingCupy()
                            else "numpy"
                        )

                    worker = _run_pool_worker_after_parent(
                        lambda: None,
                        f"matrix-{requested}-{launch}",
                    )
                    self.assertFalse(worker["using_cupy"])
                    self.assertEqual(worker["pool_worker"], "1")
                    self.assertEqual(worker["effective_env"], expected_effective)
                    if requested == "detect":
                        resolved_requested = (
                            "cupy" if nornir_imageregistration.HasCupy() else "numpy"
                        )
                    else:
                        resolved_requested = requested
                    self.assertEqual(worker["requested_env"], resolved_requested)


class TestConfigureForkPoolWorkerSubprocess(unittest.TestCase):
    """``ConfigureForkPoolWorker`` in a child process with inherited env vars."""

    def test_worker_subprocess_inherits_requested_cupy_env(self):
        if not nornir_imageregistration.HasCupy():
            self.skipTest("CuPy not installed")

        script = textwrap.dedent(
            f"""
            import json
            import os
            from unittest import mock
            import multiprocessing

            os.environ[{NORNIR_COMPUTATIONAL_LIBRARY_REQUESTED_ENV!r}] = "cupy"
            os.environ[{NORNIR_COMPUTATIONAL_LIBRARY_ENV!r}] = "cupy"

            import nornir_imageregistration.computational_lib as cl

            with mock.patch.object(multiprocessing, "parent_process", return_value=object()):
                cl.ConfigureForkPoolWorker()

            print(json.dumps({{
                "using_cupy": cl.UsingCupy(),
                "effective_env": os.environ.get({NORNIR_COMPUTATIONAL_LIBRARY_ENV!r}),
                "pool_worker": os.environ.get({NORNIR_POOL_WORKER_ENV!r}),
            }}))
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=_fresh_subprocess_env(),
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertFalse(result["using_cupy"])
        self.assertEqual(result["effective_env"], "cupy")
        self.assertEqual(result["pool_worker"], "1")


if __name__ == "__main__":
    unittest.main()
