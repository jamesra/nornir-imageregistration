"""Per-pass STOS refine diagnostics (tables + heatmaps)."""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.refine_shared.runtime_config import get_runtime_config


@dataclass(frozen=True)
class PassDiagnosticRow:
    """One grid cell's measurement / gate state for a refine pass."""

    grid_row: int
    grid_col: int
    source_y: float
    source_x: float
    target_y: float
    target_x: float
    weight: float
    peak_y: float
    peak_x: float
    travel: float
    mesh_included: bool
    travel_dropped: bool
    below_weight_cutoff: bool
    locked: bool
    unlocked_this_pass: bool
    candidate_stable_count: int
    residual_to_mesh: float
    discontinuity: bool
    smoothed_peak_y: float
    smoothed_peak_x: float
    raw_vs_smooth_delta: float
    peak_ratio: float
    role: int = -1
    reject_reason: int = 0
    zncc: float = float('nan')
    lock_candidate: bool = False
    source_content: float = float('nan')


def pass_diagnostics_enabled(save_plots: bool) -> bool:
    """True when SavePlots is on or ``NORNIR_REFINE_PASS_DIAGNOSTICS=1``.

    Reads the cached config; refine entry points refresh once per pass.
    """
    if save_plots:
        return True
    return get_runtime_config().pass_diagnostics


def build_pass_diagnostic_rows(
        alignment_points: Sequence,
        finalized: Mapping[tuple[int, int], object],
        included_ids: set[tuple[int, int]],
        travel_dropped_ids: set[tuple[int, int]],
        unlocked_ids: set[tuple[int, int]],
        transform_cutoff: float,
        finalize_candidates: Mapping[tuple[int, int], object] | None,
        transform,
        discontinuity_ids: set[tuple[int, int]] | None = None,
        smoothed_by_id: Mapping[tuple[int, int], object] | None = None,
        role_by_id: Mapping[tuple[int, int], int] | None = None,
        reject_reason_by_id: Mapping[tuple[int, int], int] | None = None,
        zncc_by_id: Mapping[tuple[int, int], float] | None = None,
        lock_candidate_ids: set[tuple[int, int]] | None = None,
        source_content_by_id: Mapping[tuple[int, int], float] | None = None,
) -> list[PassDiagnosticRow]:
    """Assemble per-cell diagnostic rows for one refine pass."""
    discontinuity_ids = discontinuity_ids or set()
    smoothed_by_id = smoothed_by_id or {}
    finalize_candidates = finalize_candidates or {}
    role_by_id = role_by_id or {}
    reject_reason_by_id = reject_reason_by_id or {}
    zncc_by_id = zncc_by_id or {}
    lock_candidate_ids = lock_candidate_ids or set()
    source_content_by_id = source_content_by_id or {}
    rows: list[PassDiagnosticRow] = []

    # Include locked cells that may not appear in this pass's free measurements.
    records_by_id: dict[tuple[int, int], object] = {}
    for rec in alignment_points:
        records_by_id[tuple(int(v) for v in rec.ID)] = rec
    for key, rec in finalized.items():
        records_by_id[(int(key[0]), int(key[1]))] = rec

    source_list: list[NDArray[np.float64]] = []
    keys: list[tuple[int, int]] = []
    for key in sorted(records_by_id.keys()):
        rec = records_by_id[key]
        keys.append(key)
        source_list.append(np.asarray(rec.SourcePoint, dtype=np.float64).reshape(2))

    residuals = np.zeros(len(keys), dtype=np.float64)
    if transform is not None and source_list:
        sources = np.asarray(source_list, dtype=np.float64)
        # Host diagnostics boundary: Transform may return CuPy; never np.asarray(cupy).
        mapped = transform.Transform(sources)
        if hasattr(mapped, 'get'):
            predicted = np.asarray(mapped.get(), dtype=np.float64).reshape(-1, 2)
        else:
            predicted = np.asarray(mapped, dtype=np.float64).reshape(-1, 2)
        adjusted = np.asarray(
            [np.asarray(records_by_id[k].AdjustedTargetPoint, dtype=np.float64).reshape(2)
             for k in keys],
            dtype=np.float64)
        residuals = np.linalg.norm(adjusted - predicted, axis=1)

    for i, key in enumerate(keys):
        rec = records_by_id[key]
        peak = np.asarray(rec.peak, dtype=np.float64).reshape(2)
        travel = float(np.linalg.norm(peak))
        weight = float(rec.weight)
        source = np.asarray(rec.SourcePoint, dtype=np.float64).reshape(2)
        target = np.asarray(rec.TargetPoint, dtype=np.float64).reshape(2)
        smooth_rec = smoothed_by_id.get(key)
        if smooth_rec is not None:
            smooth_peak = np.asarray(smooth_rec.peak, dtype=np.float64).reshape(2)
            delta = float(np.linalg.norm(peak - smooth_peak))
        else:
            smooth_peak = np.asarray((np.nan, np.nan), dtype=np.float64)
            delta = float('nan')
        cand = finalize_candidates.get(key)
        stable = int(getattr(cand, 'consecutive_stable', 0)) if cand is not None else 0
        raw_ratio = getattr(rec, 'peak_ratio', None)
        peak_ratio = float(raw_ratio) if raw_ratio is not None else float('nan')
        zncc_val = zncc_by_id.get(key, float('nan'))
        src_content = source_content_by_id.get(key, float('nan'))
        rows.append(PassDiagnosticRow(
            grid_row=int(key[0]),
            grid_col=int(key[1]),
            source_y=float(source[0]),
            source_x=float(source[1]),
            target_y=float(target[0]),
            target_x=float(target[1]),
            weight=weight,
            peak_y=float(peak[0]),
            peak_x=float(peak[1]),
            travel=travel,
            mesh_included=key in included_ids,
            travel_dropped=key in travel_dropped_ids,
            below_weight_cutoff=weight < float(transform_cutoff),
            locked=key in finalized,
            unlocked_this_pass=key in unlocked_ids,
            candidate_stable_count=stable,
            residual_to_mesh=float(residuals[i]),
            discontinuity=key in discontinuity_ids,
            smoothed_peak_y=float(smooth_peak[0]),
            smoothed_peak_x=float(smooth_peak[1]),
            raw_vs_smooth_delta=delta,
            peak_ratio=peak_ratio,
            role=int(role_by_id.get(key, -1)),
            reject_reason=int(reject_reason_by_id.get(key, 0)),
            zncc=float(zncc_val) if zncc_val is not None else float('nan'),
            lock_candidate=key in lock_candidate_ids,
            source_content=float(src_content) if src_content is not None else float('nan'),
        ))
    return rows


def write_pass_diagnostics(
        output_dir: str,
        pass_index: int,
        rows: Sequence[PassDiagnosticRow],
        *,
        pair_label: str = 'stos',
        write_heatmaps: bool = False,
) -> dict[str, str | float]:
    """Write NPZ/CSV (and optional heatmaps) for one pass.

    Heatmaps are opt-in via ``write_heatmaps`` (typically ``SavePlots`` only).
    ``NORNIR_REFINE_PASS_DIAGNOSTICS`` should use tables without heatmaps.

    Returns written paths plus ``tables_s`` / ``heatmaps_s`` wall seconds.
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = f'refine_pass{pass_index:02d}_diagnostics'
    npz_path = os.path.join(output_dir, f'{stem}.npz')
    csv_path = os.path.join(output_dir, f'{stem}.csv')

    t_tables = time.perf_counter()
    arrays = {
        'grid_row': np.asarray([r.grid_row for r in rows], dtype=np.int64),
        'grid_col': np.asarray([r.grid_col for r in rows], dtype=np.int64),
        'source_y': np.asarray([r.source_y for r in rows], dtype=np.float64),
        'source_x': np.asarray([r.source_x for r in rows], dtype=np.float64),
        'target_y': np.asarray([r.target_y for r in rows], dtype=np.float64),
        'target_x': np.asarray([r.target_x for r in rows], dtype=np.float64),
        'weight': np.asarray([r.weight for r in rows], dtype=np.float64),
        'peak_y': np.asarray([r.peak_y for r in rows], dtype=np.float64),
        'peak_x': np.asarray([r.peak_x for r in rows], dtype=np.float64),
        'travel': np.asarray([r.travel for r in rows], dtype=np.float64),
        'mesh_included': np.asarray([r.mesh_included for r in rows], dtype=bool),
        'travel_dropped': np.asarray([r.travel_dropped for r in rows], dtype=bool),
        'below_weight_cutoff': np.asarray([r.below_weight_cutoff for r in rows], dtype=bool),
        'locked': np.asarray([r.locked for r in rows], dtype=bool),
        'unlocked_this_pass': np.asarray([r.unlocked_this_pass for r in rows], dtype=bool),
        'candidate_stable_count': np.asarray([r.candidate_stable_count for r in rows], dtype=np.int64),
        'residual_to_mesh': np.asarray([r.residual_to_mesh for r in rows], dtype=np.float64),
        'discontinuity': np.asarray([r.discontinuity for r in rows], dtype=bool),
        'smoothed_peak_y': np.asarray([r.smoothed_peak_y for r in rows], dtype=np.float64),
        'smoothed_peak_x': np.asarray([r.smoothed_peak_x for r in rows], dtype=np.float64),
        'raw_vs_smooth_delta': np.asarray([r.raw_vs_smooth_delta for r in rows], dtype=np.float64),
        'peak_ratio': np.asarray([r.peak_ratio for r in rows], dtype=np.float64),
        'role': np.asarray([r.role for r in rows], dtype=np.int64),
        'reject_reason': np.asarray([r.reject_reason for r in rows], dtype=np.int64),
        'zncc': np.asarray([r.zncc for r in rows], dtype=np.float64),
        'lock_candidate': np.asarray([r.lock_candidate for r in rows], dtype=bool),
        'source_content': np.asarray([r.source_content for r in rows], dtype=np.float64),
    }
    np.savez_compressed(npz_path, **arrays)

    fieldnames = list(arrays.keys())
    with open(csv_path, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(rows)):
            writer.writerow({name: arrays[name][i] for name in fieldnames})
    tables_s = time.perf_counter() - t_tables

    written: dict[str, str | float] = {
        'npz': npz_path,
        'csv': csv_path,
        'tables_s': tables_s,
        'heatmaps_s': 0.0,
    }
    if write_heatmaps and rows:
        t_heat = time.perf_counter()
        written.update(_write_heatmaps(output_dir, pass_index, rows, pair_label=pair_label))
        written['heatmaps_s'] = time.perf_counter() - t_heat
    return written


def _sparse_grid(rows: Sequence[PassDiagnosticRow], values: NDArray[np.floating]) -> NDArray[np.float64]:
    """Scatter cell values onto a dense (row, col) grid filled with NaN."""
    max_r = max(r.grid_row for r in rows)
    max_c = max(r.grid_col for r in rows)
    grid = np.full((max_r + 1, max_c + 1), np.nan, dtype=np.float64)
    for r, value in zip(rows, values):
        grid[r.grid_row, r.grid_col] = float(value)
    return grid


def _write_heatmaps(
        output_dir: str,
        pass_index: int,
        rows: Sequence[PassDiagnosticRow],
        *,
        pair_label: str,
) -> dict[str, str]:
    """Write one PNG per map under ``output_dir`` (single render each)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    maps = {
        'weight': np.asarray([r.weight for r in rows], dtype=np.float64),
        'travel': np.asarray([r.travel for r in rows], dtype=np.float64),
        'residual': np.asarray([r.residual_to_mesh for r in rows], dtype=np.float64),
        'locked': np.asarray([1.0 if r.locked else 0.0 for r in rows], dtype=np.float64),
        'raw_vs_smooth_delta': np.asarray([r.raw_vs_smooth_delta for r in rows], dtype=np.float64),
        'discontinuity': np.asarray([1.0 if r.discontinuity else 0.0 for r in rows], dtype=np.float64),
        'peak_ratio': np.asarray([r.peak_ratio for r in rows], dtype=np.float64),
        'role': np.asarray([float(r.role) for r in rows], dtype=np.float64),
        'zncc': np.asarray([r.zncc for r in rows], dtype=np.float64),
    }
    written: dict[str, str] = {}
    for name, values in maps.items():
        if name in ('raw_vs_smooth_delta', 'peak_ratio', 'zncc') and not np.any(np.isfinite(values)):
            continue
        if name == 'role' and not np.any(values >= 0):
            continue
        grid = _sparse_grid(rows, values)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid, origin='lower', aspect='equal', interpolation='nearest')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        title = f'{pair_label} pass {pass_index} {name}'
        ax.set_title(title)
        ax.set_xlabel('grid col')
        ax.set_ylabel('grid row')
        out_path = os.path.join(output_dir, f'refine_pass{pass_index:02d}_{name}.png')
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        written[name] = out_path
    return written
