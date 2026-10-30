"""A/B refine of assessment fixtures (baseline vs trusted-mesh)."""

from __future__ import annotations

import csv
import html
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.files.stosfile import StosFile
from nornir_imageregistration.refine_assessment.catalog import Catalog, RunsDb
from nornir_imageregistration.refine_assessment.verdict import (
    ScoreVector,
    Verdict,
    compare_to_best,
)
from nornir_imageregistration.stos_quality import compute_pair_zncc


@dataclass(frozen=True)
class AbResult:
    """One fixture A/B comparison."""

    fixture_dir: str
    relative_dir: str
    baseline_run_id: int | None
    flagged_run_id: int | None
    verdict: str
    baseline_scores: dict[str, float]
    flagged_scores: dict[str, float]
    wall_s_baseline: float
    wall_s_flagged: float


def _git_sha(repo: Path) -> str | None:
    head = repo / '.git' / 'HEAD'
    if not head.is_file():
        return None
    text = head.read_text(encoding='utf-8').strip()
    if text.startswith('ref:'):
        ref = repo / '.git' / text.split(' ', 1)[1].strip()
        if ref.is_file():
            return ref.read_text(encoding='utf-8').strip()[:12]
        return None
    return text[:12]


def _score_output(
        stos_path: Path,
        *,
        quality_flag: bool = False,
        unique_frac_series: Sequence[float] | None = None,
        lock_frac: float | None = None,
) -> ScoreVector:
    """Build a ScoreVector from an output .stos (pair ZNCC + optional refine fields)."""
    values: dict[str, float] = {}
    try:
        result = compute_pair_zncc(str(stos_path), max_side=1024)
        values['pair_zncc'] = float(result.pair_zncc)
    except Exception:
        values['pair_zncc'] = float('nan')
    if lock_frac is not None:
        values['lock_frac'] = float(lock_frac)
    series = list(unique_frac_series or [])
    if series:
        values['unique_frac_min'] = float(min(series))
    return ScoreVector(values=values, quality_flag=quality_flag, unique_frac_series=series)


def _refine_once(
        input_stos: Path,
        output_stos: Path,
        *,
        trusted_mesh: bool,
        num_iterations: int = 3,
        cell_size: tuple[int, int] = (128, 128),
        grid_spacing: tuple[int, int] = (128, 128),
) -> tuple[float, bool]:
    """Run RefineStosFile once; return (wall_s, quality_flag)."""
    env_key = 'NORNIR_REFINE_TRUSTED_MESH'
    previous = os.environ.get(env_key)
    try:
        if trusted_mesh:
            os.environ[env_key] = '1'
        elif env_key in os.environ:
            del os.environ[env_key]
        # Refresh cached runtime config so the flag is visible.
        from nornir_imageregistration.refine_shared.runtime_config import get_runtime_config
        get_runtime_config(refresh=True)

        output_stos.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        nornir_imageregistration.local_distortion_correction.RefineStosFile(
            str(input_stos),
            str(output_stos),
            num_iterations=num_iterations,
            cell_size=cell_size,
            grid_spacing=grid_spacing,
            angles_to_search=[0.0],
        )
        wall = time.perf_counter() - t0
        quality_flag = False
        marker = output_stos.with_suffix('.quality_flag')
        if marker.is_file():
            quality_flag = True
        return wall, quality_flag
    finally:
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous
        from nornir_imageregistration.refine_shared.runtime_config import get_runtime_config
        get_runtime_config(refresh=True)


def run_ab_fixture(
        fixture_dir: Path,
        *,
        catalog: Catalog,
        runs: RunsDb,
        work_dir: Path,
        relative_dir: str | None = None,
        num_iterations: int = 3,
) -> AbResult:
    """A/B one fixture directory; write runs and return the verdict."""
    fixture_dir = Path(fixture_dir)
    input_stos = fixture_dir / 'input.stos'
    if not input_stos.is_file():
        raise FileNotFoundError(input_stos)

    rel = relative_dir or fixture_dir.name
    fixture_id = catalog.fixture_id_for_dir(rel)
    sha = _git_sha(Path(nornir_imageregistration.__file__).resolve().parents[2])

    base_out = work_dir / 'baseline' / f'{fixture_dir.name}.stos'
    flag_out = work_dir / 'trusted_mesh' / f'{fixture_dir.name}.stos'

    wall_b, q_b = _refine_once(input_stos, base_out, trusted_mesh=False, num_iterations=num_iterations)
    wall_f, q_f = _refine_once(input_stos, flag_out, trusted_mesh=True, num_iterations=num_iterations)

    scores_b = _score_output(base_out, quality_flag=q_b)
    scores_f = _score_output(flag_out, quality_flag=q_f)

    best = catalog.get_best_scores(fixture_id) if fixture_id is not None else {}
    verdict = compare_to_best(scores_f, best)

    run_b = runs.insert_run(
        str(fixture_dir), scores_b, fixture_id=fixture_id,
        imageregistration_sha=sha, flags='baseline', wall_s=wall_b,
        verdict=str(Verdict.UNCHANGED),
    )
    run_f = runs.insert_run(
        str(fixture_dir), scores_f, fixture_id=fixture_id,
        imageregistration_sha=sha, flags='NORNIR_REFINE_TRUSTED_MESH=1',
        wall_s=wall_f, verdict=str(verdict),
    )

    return AbResult(
        fixture_dir=str(fixture_dir),
        relative_dir=rel,
        baseline_run_id=run_b,
        flagged_run_id=run_f,
        verdict=str(verdict),
        baseline_scores=dict(scores_b.values),
        flagged_scores=dict(scores_f.values),
        wall_s_baseline=wall_b,
        wall_s_flagged=wall_f,
    )


def write_ab_report(
        results: Sequence[AbResult],
        out_dir: Path,
) -> tuple[Path, Path]:
    """Write CSV + HTML summary under *out_dir*; return (csv_path, html_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'ab_summary.csv'
    html_path = out_dir / 'ab_summary.html'

    fieldnames = [
        'relative_dir', 'verdict', 'pair_zncc_baseline', 'pair_zncc_flagged',
        'wall_s_baseline', 'wall_s_flagged', 'baseline_run_id', 'flagged_run_id',
    ]
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                'relative_dir': row.relative_dir,
                'verdict': row.verdict,
                'pair_zncc_baseline': row.baseline_scores.get('pair_zncc'),
                'pair_zncc_flagged': row.flagged_scores.get('pair_zncc'),
                'wall_s_baseline': f'{row.wall_s_baseline:.2f}',
                'wall_s_flagged': f'{row.wall_s_flagged:.2f}',
                'baseline_run_id': row.baseline_run_id,
                'flagged_run_id': row.flagged_run_id,
            })

    rows_html = []
    for row in results:
        rows_html.append(
            '<tr>'
            f'<td>{html.escape(row.relative_dir)}</td>'
            f'<td>{html.escape(row.verdict)}</td>'
            f'<td>{row.baseline_scores.get("pair_zncc", float("nan")):.4f}</td>'
            f'<td>{row.flagged_scores.get("pair_zncc", float("nan")):.4f}</td>'
            f'<td>{row.wall_s_baseline:.1f}</td>'
            f'<td>{row.wall_s_flagged:.1f}</td>'
            '</tr>'
        )
    html_path.write_text(
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<title>Refine A/B summary</title></head><body>'
        '<h1>Refine A/B (baseline vs trusted-mesh)</h1>'
        '<table border="1" cellpadding="4">'
        '<tr><th>Fixture</th><th>Verdict</th><th>ZNCC base</th>'
        '<th>ZNCC flagged</th><th>Wall base (s)</th><th>Wall flagged (s)</th></tr>'
        + '\n'.join(rows_html)
        + '</table></body></html>\n',
        encoding='utf-8',
    )
    (out_dir / 'ab_summary.json').write_text(
        json.dumps([row.__dict__ for row in results], indent=2) + '\n',
        encoding='utf-8',
    )
    return csv_path, html_path
