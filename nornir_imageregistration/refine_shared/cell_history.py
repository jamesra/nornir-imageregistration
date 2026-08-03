"""Per-cell refine-lifetime history for PASS_DIAGNOSTICS / SavePlots.

Observability only — does not brand Roles or change gate behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.refine_shared.pass_diagnostics import PassDiagnosticRow


@dataclass
class CellPassHistoryStore:
    """Append-only store of per-cell per-pass travel / role / scores."""

    grid_row: list[int] = field(default_factory=list)
    grid_col: list[int] = field(default_factory=list)
    pass_index: list[int] = field(default_factory=list)
    travel: list[float] = field(default_factory=list)
    locked: list[bool] = field(default_factory=list)
    role: list[int] = field(default_factory=list)
    lock_candidate: list[bool] = field(default_factory=list)
    peak_ratio: list[float] = field(default_factory=list)
    zncc: list[float] = field(default_factory=list)
    source_x: list[float] = field(default_factory=list)
    source_y: list[float] = field(default_factory=list)

    def append_pass(self, pass_index: int, rows: Sequence[PassDiagnosticRow]) -> None:
        """Append one pass of diagnostic rows into the lifetime store."""
        for row in rows:
            self.grid_row.append(int(row.grid_row))
            self.grid_col.append(int(row.grid_col))
            self.pass_index.append(int(pass_index))
            self.travel.append(float(row.travel))
            self.locked.append(bool(row.locked))
            self.role.append(int(row.role))
            self.lock_candidate.append(bool(row.lock_candidate))
            self.peak_ratio.append(float(row.peak_ratio))
            self.zncc.append(float(row.zncc))
            self.source_x.append(float(row.source_x))
            self.source_y.append(float(row.source_y))

    def as_arrays(self) -> dict[str, NDArray]:
        """Return numpy arrays suitable for ``np.savez_compressed``."""
        return {
            'grid_row': np.asarray(self.grid_row, dtype=np.int64),
            'grid_col': np.asarray(self.grid_col, dtype=np.int64),
            'pass': np.asarray(self.pass_index, dtype=np.int64),
            'travel': np.asarray(self.travel, dtype=np.float64),
            'locked': np.asarray(self.locked, dtype=bool),
            'role': np.asarray(self.role, dtype=np.int64),
            'lock_candidate': np.asarray(self.lock_candidate, dtype=bool),
            'peak_ratio': np.asarray(self.peak_ratio, dtype=np.float64),
            'zncc': np.asarray(self.zncc, dtype=np.float64),
            'source_x': np.asarray(self.source_x, dtype=np.float64),
            'source_y': np.asarray(self.source_y, dtype=np.float64),
        }

    def write_npz(self, output_dir: str, *, stem: str = 'refine_cell_history') -> str | None:
        """Write lifetime NPZ under *output_dir*; return path or None if empty."""
        if not self.pass_index:
            return None
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'{stem}.npz')
        np.savez_compressed(path, **self.as_arrays())
        return path


def _cell_series(arrays: Mapping[str, NDArray], key: tuple[int, int]) -> dict[str, NDArray]:
    """Extract sorted-by-pass series for one grid cell."""
    mask = (arrays['grid_row'] == key[0]) & (arrays['grid_col'] == key[1])
    order = np.argsort(arrays['pass'][mask])
    return {name: np.asarray(arr[mask])[order] for name, arr in arrays.items()}


def _select_sample_cells(
        arrays: Mapping[str, NDArray],
        *,
        max_per_group: int = 7,
        travel_eps: float = 0.5,
) -> list[tuple[int, int]]:
    """Pick up to ~20 cells: cold identity lockers, hot travelers, healthy locks."""
    rows = arrays['grid_row']
    cols = arrays['grid_col']
    passes = arrays['pass']
    travel = arrays['travel']
    locked = arrays['locked']
    source_x = arrays['source_x']
    mid = float(np.median(source_x)) if source_x.size else 0.0
    max_pass = int(passes.max()) if passes.size else 0

    keys = sorted({(int(r), int(c)) for r, c in zip(rows, cols)})
    cold_id: list[tuple[int, int]] = []
    hot_free: list[tuple[int, int]] = []
    healthy: list[tuple[int, int]] = []

    for key in keys:
        series = _cell_series(arrays, key)
        t = series['travel']
        lck = series['locked']
        sx = float(series['source_x'][-1]) if series['source_x'].size else mid
        last_t = float(t[-1]) if t.size else float('nan')
        ever_locked = bool(np.any(lck))
        max_t = float(np.nanmax(t)) if t.size else 0.0
        if sx > mid and ever_locked and last_t < float(travel_eps):
            cold_id.append(key)
        elif sx <= mid and max_t > float(travel_eps) * 4.0 and not ever_locked:
            hot_free.append(key)
        elif ever_locked and max_t < float(travel_eps) * 2.0 and sx <= mid:
            healthy.append(key)

    selected: list[tuple[int, int]] = []
    for group in (cold_id, hot_free, healthy):
        selected.extend(group[:max_per_group])
    # Fill remaining slots from any remaining keys if groups were sparse.
    if len(selected) < 3 and keys:
        for key in keys:
            if key not in selected:
                selected.append(key)
            if len(selected) >= min(20, max_per_group * 3):
                break
    return selected[: max_per_group * 3]


def write_cell_history_plots(
        output_dir: str,
        arrays: Mapping[str, NDArray],
        *,
        pair_label: str = 'stos',
        travel_eps: float = 0.5,
) -> dict[str, str]:
    """Write half-aggregate and sample-cell polyline PNGs (SavePlots only).

    Two-row layout: row A = travel vs pass; row B = peak_ratio / zncc at
    synthetic X positions N+1 / N+2 (dual-scale honesty).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    written: dict[str, str] = {}
    if arrays.get('pass') is None or arrays['pass'].size == 0:
        return written

    passes = arrays['pass']
    max_pass = int(passes.max())
    pass_axis = np.arange(1, max_pass + 1, dtype=np.float64)
    score_x_pr = float(max_pass + 1)
    score_x_zncc = float(max_pass + 2)

    # --- Half aggregates ---
    mid = float(np.median(arrays['source_x']))
    fig, (ax_t, ax_s) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    for name, mask_fn, color in (
            ('low_x', lambda sx: sx <= mid, '#1f77b4'),
            ('high_x', lambda sx: sx > mid, '#d62728'),
    ):
        med_travel = []
        med_pr = []
        med_zncc = []
        for p in pass_axis:
            m = (passes == int(p)) & mask_fn(arrays['source_x'])
            med_travel.append(float(np.nanmedian(arrays['travel'][m])) if np.any(m) else float('nan'))
            pr = arrays['peak_ratio'][m]
            zn = arrays['zncc'][m]
            med_pr.append(float(np.nanmedian(pr[np.isfinite(pr)])) if np.any(np.isfinite(pr)) else float('nan'))
            med_zncc.append(float(np.nanmedian(zn[np.isfinite(zn)])) if np.any(np.isfinite(zn)) else float('nan'))
        ax_t.plot(pass_axis, med_travel, '-o', color=color, label=f'{name} travel med', markersize=4)
        pr_arr = np.asarray(med_pr, dtype=np.float64)
        zn_arr = np.asarray(med_zncc, dtype=np.float64)
        if np.any(np.isfinite(pr_arr)):
            ax_s.plot([score_x_pr], [float(np.nanmedian(pr_arr))], 's', color=color,
                      label=f'{name} peak_ratio')
        if np.any(np.isfinite(zn_arr)):
            ax_s.plot([score_x_zncc], [float(np.nanmedian(zn_arr))], '^', color=color,
                      label=f'{name} zncc')

    ax_t.set_ylabel('travel (px)')
    ax_t.set_title(f'{pair_label} half-aggregate cell history')
    ax_t.legend(loc='best', fontsize=8)
    ax_t.grid(True, alpha=0.3)
    ax_s.set_ylabel('score')
    ax_s.set_xlabel('pass')
    ax_s.set_xticks(list(pass_axis) + [score_x_pr, score_x_zncc])
    ax_s.set_xticklabels([str(int(p)) for p in pass_axis] + ['PC', 'ZNCC'])
    ax_s.legend(loc='best', fontsize=8)
    ax_s.grid(True, alpha=0.3)
    half_path = os.path.join(output_dir, 'refine_cell_history_half_aggregates.png')
    fig.savefig(half_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    written['half_aggregates'] = half_path

    # --- Sample cell small multiples ---
    samples = _select_sample_cells(arrays, travel_eps=travel_eps)
    if not samples:
        return written

    n = len(samples)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(3.2 * ncols, 2.6 * nrows * 2), squeeze=False)
    for idx, key in enumerate(samples):
        col = idx % ncols
        row_block = (idx // ncols) * 2
        ax_t = axes[row_block][col]
        ax_s = axes[row_block + 1][col]
        series = _cell_series(arrays, key)
        p = series['pass'].astype(np.float64)
        ax_t.plot(p, series['travel'], '-o', markersize=3, color='#333333')
        if np.any(series['locked']):
            lock_p = p[series['locked']]
            lock_t = series['travel'][series['locked']]
            ax_t.scatter(lock_p, lock_t, marker='x', color='#d62728', s=36, zorder=3)
        ax_t.set_title(f'cell {key[0]},{key[1]}', fontsize=8)
        ax_t.set_ylabel('travel', fontsize=7)
        ax_t.tick_params(labelsize=6)
        ax_t.grid(True, alpha=0.25)

        pr = series['peak_ratio']
        zn = series['zncc']
        # Use last finite score across passes for the synthetic markers.
        pr_last = float(pr[np.isfinite(pr)][-1]) if np.any(np.isfinite(pr)) else float('nan')
        zn_last = float(zn[np.isfinite(zn)][-1]) if np.any(np.isfinite(zn)) else float('nan')
        ax_s.plot([score_x_pr], [pr_last], 's', color='#1f77b4')
        ax_s.plot([score_x_zncc], [zn_last], '^', color='#2ca02c')
        ax_s.set_xlim(0.5, score_x_zncc + 0.5)
        ax_s.set_xticks([score_x_pr, score_x_zncc])
        ax_s.set_xticklabels(['PC', 'ZNCC'], fontsize=6)
        ax_s.set_ylabel('score', fontsize=7)
        ax_s.tick_params(labelsize=6)
        ax_s.grid(True, alpha=0.25)

    # Hide unused axes.
    for idx in range(n, nrows * ncols):
        col = idx % ncols
        row_block = (idx // ncols) * 2
        axes[row_block][col].axis('off')
        axes[row_block + 1][col].axis('off')

    fig.suptitle(f'{pair_label} sample cell history', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    sample_path = os.path.join(output_dir, 'refine_cell_history_samples.png')
    fig.savefig(sample_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    written['samples'] = sample_path
    return written
