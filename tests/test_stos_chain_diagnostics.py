"""Tests for SliceToVolume chain boundary diagnostics."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from stos_chain_diagnostics import diagnose_interface, find_stos


class TestStosChainDiagnostics(unittest.TestCase):
    def test_rpc3_453_452_interface_if_available(self) -> None:
        """Run the RPC3 repro when the volume is mounted."""
        volume_root = Path("/storage4/RPC3/TEM")
        if not find_stos(volume_root, "Grid16", "453-452"):
            self.skipTest("RPC3 Grid16 453-452 transform is not available")

        report = diagnose_interface(
            volume_root,
            mapped_section=453,
            control_section=452,
            center_section=450,
            downsample=16,
        )

        self.assertTrue(report.level_comparison is not None)
        self.assertTrue(report.level_comparison["jump_present_in_stv16"])
        self.assertEqual(report.primary_cause, "linear_blend_accumulation")
        self.assertLess(
            next(item.max_residual for item in report.experiments if item.name == "no_linear_blend"),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
