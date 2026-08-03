"""Unit tests for refine cell-history store / plot helpers."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from nornir_imageregistration.refine_shared.cell_history import (
    CellPassHistoryStore,
    write_cell_history_plots,
)
from nornir_imageregistration.refine_shared.pass_diagnostics import PassDiagnosticRow


def _row(
        *,
        grid_row: int,
        grid_col: int,
        travel: float,
        source_x: float,
        peak_ratio: float = 1.4,
        zncc: float = float('nan'),
        locked: bool = False,
        role: int = 1,
        lock_candidate: bool = False,
) -> PassDiagnosticRow:
    return PassDiagnosticRow(
        grid_row=grid_row,
        grid_col=grid_col,
        source_y=0.0,
        source_x=source_x,
        target_y=0.0,
        target_x=source_x,
        weight=12.0,
        peak_y=0.0,
        peak_x=travel,
        travel=travel,
        mesh_included=True,
        travel_dropped=False,
        below_weight_cutoff=False,
        locked=locked,
        unlocked_this_pass=False,
        candidate_stable_count=0,
        residual_to_mesh=0.0,
        discontinuity=False,
        smoothed_peak_y=float('nan'),
        smoothed_peak_x=float('nan'),
        raw_vs_smooth_delta=float('nan'),
        peak_ratio=peak_ratio,
        role=role,
        reject_reason=0,
        zncc=zncc,
        lock_candidate=lock_candidate,
        source_content=float('nan'),
    )


class TestCellPassHistory(unittest.TestCase):
    """Lifetime NPZ + SavePlots helpers."""

    def test_append_and_write_npz(self) -> None:
        store = CellPassHistoryStore()
        store.append_pass(1, [
            _row(grid_row=0, grid_col=0, travel=16.0, source_x=1.0),
            _row(grid_row=0, grid_col=1, travel=0.1, source_x=100.0, zncc=0.75, locked=True),
        ])
        store.append_pass(2, [
            _row(grid_row=0, grid_col=0, travel=12.0, source_x=1.0),
            _row(grid_row=0, grid_col=1, travel=0.0, source_x=100.0, zncc=0.8, locked=True),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = store.write_npz(tmp)
            self.assertIsNotNone(path)
            assert path is not None
            data = np.load(path)
            self.assertEqual(int(data['pass'].shape[0]), 4)
            plots = write_cell_history_plots(tmp, store.as_arrays(), pair_label='unit')
            self.assertIn('half_aggregates', plots)
            self.assertTrue(os.path.isfile(plots['half_aggregates']))


if __name__ == '__main__':
    unittest.main()
