"""CI parity gate: CPU batched vs GPU batched grid refine on section 0690."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import nornir_imageregistration

from grid_seam_metrics import require_grid_refine_input_section_fixture

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VERIFY_SCRIPT = _REPO_ROOT / "scripts" / "verify_cpu_vs_batched.py"


class TestBatchedRefineParity(unittest.TestCase):
    """Automated CPU batched vs GPU batched refine parity (no serial CPU in CI)."""

    def test_cpu_batched_vs_gpu_batched_parity(self) -> None:
        """CPU batched and GPU batched refine must match within working-res tolerance."""
        require_grid_refine_input_section_fixture()
        if not nornir_imageregistration.HasCupy():
            self.skipTest("CuPy unavailable; GPU batched leg cannot run")

        env = dict(os.environ)
        env.setdefault("NORNIR_HEADLESS", "1")
        result = subprocess.run(
            [sys.executable, str(_VERIFY_SCRIPT), "--batched-vs-batched"],
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        self.assertEqual(
            result.returncode,
            0,
            msg=(
                "CPU batched vs GPU batched parity failed "
                f"(exit {result.returncode}). See subprocess output above."))


if __name__ == "__main__":
    unittest.main()
