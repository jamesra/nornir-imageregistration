"""Adopt a run's scores into the catalog living-best ledger."""

from __future__ import annotations

from pathlib import Path

from nornir_imageregistration.refine_assessment.catalog import Catalog, RunsDb
from nornir_imageregistration.refine_assessment.verdict import (
    CANONICAL_METRICS,
    MetricDirection,
    ScoreVector,
)


def adopt_run(
        catalog: Catalog,
        runs: RunsDb,
        run_id: int,
        *,
        fixture_id: int | None = None,
        note: str | None = None,
        best_json_path: Path | None = None,
) -> int:
    """Copy *run_id* metrics into catalog bests; optionally write ``best.json``.

    Returns the new adopt id. Pytest must never call this.
    """
    run = runs.get_run(run_id)
    if run is None:
        raise KeyError(f'Unknown run_id: {run_id}')
    scores_map = runs.get_run_scores(run_id)
    fid = fixture_id if fixture_id is not None else run['fixture_id']
    if fid is None:
        raise ValueError('fixture_id is required when the run has none')

    directions = {spec.name: spec.direction for spec in CANONICAL_METRICS}
    bands = {
        spec.name: (spec.band_lo, spec.band_hi)
        for spec in CANONICAL_METRICS
        if spec.direction == MetricDirection.BAND
    }
    for name in scores_map:
        directions.setdefault(name, MetricDirection.MAXIMIZE)

    scores = ScoreVector(values=scores_map, quality_flag=bool(run['quality_flag']))
    adopt_id = catalog.adopt_scores(
        int(fid),
        scores,
        directions=directions,
        bands=bands,
        imageregistration_sha=run['imageregistration_sha'],
        code_path=run['flags'],
        note=note,
        run_id=run_id,
    )
    if best_json_path is not None:
        catalog.export_best_json(int(fid), best_json_path)
    return adopt_id
