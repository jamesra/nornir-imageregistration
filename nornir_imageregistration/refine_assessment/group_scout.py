"""Resumable StosGroup scout: screen, trial-refine, and rank local quality."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from nornir_imageregistration.files.stosfile import StosFile
from nornir_imageregistration.stos_quality import (
    CellZnccComparison,
    CellZnccRecord,
    compare_cell_zncc,
    compute_cell_zncc,
    compute_pair_zncc,
    entry_is_stale,
    load_quality_cache,
)

PAIR_NAME_RE = re.compile(r'^(\d+)-(\d+)')
DEFAULT_LIMIT = 50
PAIR_REGRESSION_TOLERANCE = 0.02
WEAK_QUANTILE = 0.25
CLUSTER_REPAIR_FRACTION = 0.5
FINE_SCORE_CELL_SIZE = 128
FINE_SCORE_GRID_SPACING = 128
COARSE_MAX_SIDE = 2048
DEFAULT_ITERATIONS = 10
SCOUT_DB_NAME = 'scout.sqlite'
SCHEMA_VERSION = 1

RefineCallable = Callable[[str, str, 'RefineSchedule'], None]
ScorePairCallable = Callable[[str], float]
ScoreCellsCallable = Callable[[str], list[CellZnccRecord]]


@dataclass(frozen=True)
class RefineSchedule:
    """One trial refinement window/spacing schedule."""

    name: str
    cell_size: tuple[int, int]
    grid_spacing: tuple[int, int]
    num_iterations: int = DEFAULT_ITERATIONS


DENSE_256 = RefineSchedule('dense256', (256, 256), (128, 128), DEFAULT_ITERATIONS)
FINE_128 = RefineSchedule('fine128', (128, 128), (64, 64), DEFAULT_ITERATIONS)
DEFAULT_SCHEDULES: tuple[RefineSchedule, ...] = (DENSE_256, FINE_128)


@dataclass(frozen=True)
class Candidate:
    """One group-root STOS eligible for screening."""

    pair: str
    stos_path: str
    relpath: str
    pair_zncc: float | None
    stos_checksum: str | None
    section_gap: int | None
    cache_valid: bool
    excluded_manual: bool = False


@dataclass(frozen=True)
class ScheduleMetrics:
    """Measured quality change for one pair and schedule."""

    pair: str
    schedule: str
    accepted: bool
    reject_reason: str | None
    quality_flag: bool
    pair_zncc_in: float | None
    pair_zncc_out: float | None
    pair_zncc_delta: float | None
    improved_count: int
    worse_count: int
    unchanged_count: int
    excluded_count: int
    net_improved: int
    delta_median: float | None
    delta_p05: float | None
    delta_min: float | None
    delta_max: float | None
    median_cell_in: float | None
    median_cell_out: float | None
    p05_cell_in: float | None
    baseline_weak_clusters: int
    repaired_clusters: int
    wall_s: float | None
    output_stos: str | None
    error: str | None = None


@dataclass
class ScoutConfig:
    """Immutable scout settings used for hashing and reports."""

    group_dir: str
    output_dir: str
    limit: int | None
    exclude_manual: bool
    schedules: tuple[RefineSchedule, ...]
    pair_regression_tolerance: float = PAIR_REGRESSION_TOLERANCE
    weak_quantile: float = WEAK_QUANTILE
    cluster_repair_fraction: float = CLUSTER_REPAIR_FRACTION
    score_cell_size: int = FINE_SCORE_CELL_SIZE
    score_grid_spacing: int = FINE_SCORE_GRID_SPACING
    coarse_max_side: int = COARSE_MAX_SIDE
    command: list[str] = field(default_factory=list)

    def settings_payload(self) -> dict[str, Any]:
        """Return the settings that must match for a checkpoint resume."""
        return {
            'exclude_manual': self.exclude_manual,
            'pair_regression_tolerance': self.pair_regression_tolerance,
            'weak_quantile': self.weak_quantile,
            'cluster_repair_fraction': self.cluster_repair_fraction,
            'score_cell_size': self.score_cell_size,
            'score_grid_spacing': self.score_grid_spacing,
            'coarse_max_side': self.coarse_max_side,
            'schedules': [
                {
                    'name': schedule.name,
                    'cell_size': list(schedule.cell_size),
                    'grid_spacing': list(schedule.grid_spacing),
                    'num_iterations': schedule.num_iterations,
                }
                for schedule in self.schedules
            ],
        }

    def settings_hash(self) -> str:
        """Return a stable SHA-256 of the resume-relevant settings."""
        encoded = json.dumps(self.settings_payload(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


_SCOUT_SCHEMA = '''
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidate (
    pair TEXT PRIMARY KEY,
    stos_relpath TEXT NOT NULL,
    stos_checksum TEXT,
    pair_zncc REAL,
    section_gap INTEGER,
    cache_valid INTEGER NOT NULL,
    excluded_manual INTEGER NOT NULL,
    screen_rank INTEGER
);

CREATE TABLE IF NOT EXISTS trial (
    pair TEXT NOT NULL,
    schedule TEXT NOT NULL,
    status TEXT NOT NULL,
    source_checksum TEXT,
    code_sha TEXT,
    settings_hash TEXT,
    quality_flag INTEGER,
    pair_zncc_in REAL,
    pair_zncc_out REAL,
    pair_zncc_delta REAL,
    median_cell_in REAL,
    median_cell_out REAL,
    delta_median REAL,
    delta_p05 REAL,
    improved_count INTEGER,
    worse_count INTEGER,
    unchanged_count INTEGER,
    excluded_count INTEGER,
    repaired_clusters INTEGER,
    baseline_weak_clusters INTEGER,
    wall_s REAL,
    error TEXT,
    output_stos TEXT,
    metrics_json TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (pair, schedule)
);
'''


def pair_id_from_name(name: str) -> str | None:
    """Return ``control-mapped`` section IDs from a STOS filename."""
    match = PAIR_NAME_RE.match(Path(name).name)
    if match is None:
        return None
    return f'{int(match.group(1))}-{int(match.group(2))}'


def section_gap_from_pair(pair: str) -> int | None:
    """Return the absolute section gap encoded in *pair*, or None."""
    match = PAIR_NAME_RE.match(pair)
    if match is None:
        return None
    return abs(int(match.group(1)) - int(match.group(2)))


def manual_pairs(group_dir: str | Path) -> set[str]:
    """Return pair IDs represented by files under ``Manual/``."""
    manual_dir = Path(group_dir) / 'Manual'
    found: set[str] = set()
    if not manual_dir.is_dir():
        return found
    for path in manual_dir.glob('*.stos'):
        pair = pair_id_from_name(path.name)
        if pair is not None:
            found.add(pair)
    return found


def list_group_root_stos(group_dir: str | Path) -> list[Path]:
    """Return ``.stos`` files in the group root, excluding subdirectories."""
    root = Path(group_dir)
    return sorted(path for path in root.glob('*.stos') if path.is_file())


def inventory_candidates(
        group_dir: str | Path,
        *,
        exclude_manual: bool = True,
) -> list[Candidate]:
    """List group-root transforms and attach non-stale cached pair ZNCC."""
    root = Path(group_dir)
    cache = load_quality_cache(str(root))
    excluded = manual_pairs(root) if exclude_manual else set()
    candidates: list[Candidate] = []
    for path in list_group_root_stos(root):
        pair = pair_id_from_name(path.name)
        if pair is None:
            continue
        is_manual = pair in excluded
        relpath = path.name.replace('\\', '/')
        entry = cache.entries.get(relpath)
        valid = not entry_is_stale(entry, str(path))
        pair_zncc: float | None = None
        checksum: str | None = None
        if valid and entry is not None:
            raw = entry.get('pair_zncc')
            try:
                pair_zncc = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                pair_zncc = None
                valid = False
            stored = entry.get('stos_checksum')
            checksum = str(stored) if stored else None
        if checksum is None:
            try:
                loaded = StosFile.LoadChecksum(str(path))
            except Exception:
                loaded = None
            checksum = str(loaded) if loaded else None
        candidates.append(Candidate(
            pair=pair,
            stos_path=str(path),
            relpath=relpath,
            pair_zncc=pair_zncc,
            stos_checksum=checksum,
            section_gap=section_gap_from_pair(pair),
            cache_valid=valid and pair_zncc is not None,
            excluded_manual=is_manual,
        ))
    return candidates


def screen_candidates(
        candidates: Sequence[Candidate],
        *,
        limit: int | None,
        exclude_manual: bool = True,
) -> list[Candidate]:
    """Return worst-first candidates with a valid cached pair ZNCC."""
    eligible = [
        candidate for candidate in candidates
        if candidate.cache_valid and candidate.pair_zncc is not None
        and not (exclude_manual and candidate.excluded_manual)
    ]
    eligible.sort(key=lambda item: (float(item.pair_zncc or 0.0), item.pair))
    if limit is None:
        return eligible
    return eligible[: max(0, int(limit))]


def _finite_scores(cells: Sequence[CellZnccRecord]) -> np.ndarray:
    return np.asarray(
        [float(cell.zncc) for cell in cells if cell.zncc is not None],
        dtype=np.float64,
    )


def weak_cell_threshold(
        cells: Sequence[CellZnccRecord],
        *,
        quantile: float = WEAK_QUANTILE,
) -> float | None:
    """Return the weak-cell ZNCC cutoff from finite *cells*."""
    scores = _finite_scores(cells)
    if scores.size == 0:
        return None
    return float(np.quantile(scores, float(quantile)))


def weak_cell_ids(
        cells: Sequence[CellZnccRecord],
        *,
        threshold: float,
) -> set[tuple[int, int]]:
    """Return lattice IDs whose ZNCC is at or below *threshold*."""
    return {
        (cell.grid_row, cell.grid_col)
        for cell in cells
        if cell.zncc is not None and float(cell.zncc) <= float(threshold)
    }


def clusters_from_ids(ids: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Return 4-connected clusters of lattice IDs."""
    remaining = set(ids)
    clusters: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        cluster = {start}
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    cluster.add(neighbor)
                    stack.append(neighbor)
        clusters.append(cluster)
    clusters.sort(key=lambda item: (min(item), len(item)))
    return clusters


def repaired_cluster_count(
        clusters: Sequence[set[tuple[int, int]]],
        comparison: CellZnccComparison,
        *,
        weak_threshold: float,
        repair_fraction: float = CLUSTER_REPAIR_FRACTION,
) -> int:
    """Count baseline weak clusters whose majority of cells improved or recovered."""
    by_id = {(cell.grid_row, cell.grid_col): cell for cell in comparison.cells}
    repaired = 0
    for cluster in clusters:
        scored = 0
        recovered = 0
        for key in cluster:
            record = by_id.get(key)
            if record is None:
                continue
            scored += 1
            if record.classification == 'improved' or record.candidate_zncc > weak_threshold:
                recovered += 1
        if scored > 0 and recovered / scored >= float(repair_fraction):
            repaired += 1
    return repaired


def ranking_key(metrics: ScheduleMetrics) -> tuple[Any, ...]:
    """Lexicographic key: accepted first, then local-quality improvement."""
    return (
        0 if metrics.accepted else 1,
        -int(metrics.repaired_clusters),
        -int(metrics.net_improved),
        -(metrics.delta_p05 if metrics.delta_p05 is not None else float('-inf')),
        -(metrics.delta_median if metrics.delta_median is not None else float('-inf')),
        -(metrics.pair_zncc_delta if metrics.pair_zncc_delta is not None else float('-inf')),
        metrics.pair,
        metrics.schedule,
    )


def choose_best_schedule(results: Sequence[ScheduleMetrics]) -> ScheduleMetrics:
    """Return the best schedule result for one pair."""
    if not results:
        raise ValueError('results must not be empty')
    return min(results, key=ranking_key)


def rank_pairs(results: Sequence[ScheduleMetrics]) -> list[ScheduleMetrics]:
    """Pick the best schedule per pair and order pairs by ranking key."""
    by_pair: dict[str, list[ScheduleMetrics]] = {}
    for result in results:
        if result.error and result.pair_zncc_out is None and result.improved_count == 0:
            by_pair.setdefault(result.pair, []).append(result)
            continue
        by_pair.setdefault(result.pair, []).append(result)
    ranked = [choose_best_schedule(group) for group in by_pair.values()]
    ranked.sort(key=ranking_key)
    return ranked


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_git_sha() -> str | None:
    """Return the imageregistration HEAD SHA, or None when git is unavailable."""
    import nornir_imageregistration
    package_root = Path(nornir_imageregistration.__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ['git', '-C', str(package_root), 'rev-parse', 'HEAD'],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


class ScoutStore:
    """SQLite checkpoint for candidates and per-schedule trials."""

    path: Path
    _conn: sqlite3.Connection

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCOUT_SCHEMA)
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),))
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> ScoutStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def set_meta(self, key: str, value: str) -> None:
        """Insert or replace a metadata value."""
        self._conn.execute(
            'INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value',
            (key, value),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        """Return a metadata value, or None."""
        row = self._conn.execute('SELECT value FROM meta WHERE key = ?', (key,)).fetchone()
        return str(row['value']) if row is not None else None

    def upsert_candidates(self, candidates: Sequence[Candidate]) -> None:
        """Replace the candidate inventory with *candidates*."""
        self._conn.execute('DELETE FROM candidate')
        for rank, candidate in enumerate(candidates, start=1):
            self._conn.execute(
                '''
                INSERT INTO candidate (
                    pair, stos_relpath, stos_checksum, pair_zncc, section_gap,
                    cache_valid, excluded_manual, screen_rank
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    candidate.pair,
                    candidate.relpath,
                    candidate.stos_checksum,
                    candidate.pair_zncc,
                    candidate.section_gap,
                    1 if candidate.cache_valid else 0,
                    1 if candidate.excluded_manual else 0,
                    rank,
                ),
            )
        self._conn.commit()

    def get_trial(self, pair: str, schedule: str) -> sqlite3.Row | None:
        """Return a stored trial row, or None."""
        return self._conn.execute(
            'SELECT * FROM trial WHERE pair = ? AND schedule = ?',
            (pair, schedule),
        ).fetchone()

    def trial_is_reusable(
            self,
            pair: str,
            schedule: str,
            *,
            source_checksum: str | None,
            code_sha: str | None,
            settings_hash: str,
    ) -> bool:
        """True when a completed trial matches the current inputs."""
        row = self.get_trial(pair, schedule)
        if row is None or row['status'] != 'completed':
            return False
        return (
            row['source_checksum'] == source_checksum
            and row['code_sha'] == code_sha
            and row['settings_hash'] == settings_hash
        )

    def save_trial(
            self,
            metrics: ScheduleMetrics,
            *,
            status: str,
            source_checksum: str | None,
            code_sha: str | None,
            settings_hash: str,
    ) -> None:
        """Insert or replace one pair/schedule trial."""
        self._conn.execute(
            '''
            INSERT INTO trial (
                pair, schedule, status, source_checksum, code_sha, settings_hash,
                quality_flag, pair_zncc_in, pair_zncc_out, pair_zncc_delta,
                median_cell_in, median_cell_out, delta_median, delta_p05,
                improved_count, worse_count, unchanged_count, excluded_count,
                repaired_clusters, baseline_weak_clusters, wall_s, error,
                output_stos, metrics_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pair, schedule) DO UPDATE SET
                status = excluded.status,
                source_checksum = excluded.source_checksum,
                code_sha = excluded.code_sha,
                settings_hash = excluded.settings_hash,
                quality_flag = excluded.quality_flag,
                pair_zncc_in = excluded.pair_zncc_in,
                pair_zncc_out = excluded.pair_zncc_out,
                pair_zncc_delta = excluded.pair_zncc_delta,
                median_cell_in = excluded.median_cell_in,
                median_cell_out = excluded.median_cell_out,
                delta_median = excluded.delta_median,
                delta_p05 = excluded.delta_p05,
                improved_count = excluded.improved_count,
                worse_count = excluded.worse_count,
                unchanged_count = excluded.unchanged_count,
                excluded_count = excluded.excluded_count,
                repaired_clusters = excluded.repaired_clusters,
                baseline_weak_clusters = excluded.baseline_weak_clusters,
                wall_s = excluded.wall_s,
                error = excluded.error,
                output_stos = excluded.output_stos,
                metrics_json = excluded.metrics_json,
                updated_at = excluded.updated_at
            ''',
            (
                metrics.pair,
                metrics.schedule,
                status,
                source_checksum,
                code_sha,
                settings_hash,
                1 if metrics.quality_flag else 0,
                metrics.pair_zncc_in,
                metrics.pair_zncc_out,
                metrics.pair_zncc_delta,
                metrics.median_cell_in,
                metrics.median_cell_out,
                metrics.delta_median,
                metrics.delta_p05,
                metrics.improved_count,
                metrics.worse_count,
                metrics.unchanged_count,
                metrics.excluded_count,
                metrics.repaired_clusters,
                metrics.baseline_weak_clusters,
                metrics.wall_s,
                metrics.error,
                metrics.output_stos,
                json.dumps(asdict(metrics), sort_keys=True),
                _utc_now(),
            ),
        )
        self._conn.commit()

    def load_completed_metrics(self, settings_hash: str) -> list[ScheduleMetrics]:
        """Return completed trial metrics matching *settings_hash*."""
        rows = self._conn.execute(
            "SELECT metrics_json FROM trial WHERE status = 'completed' AND settings_hash = ?",
            (settings_hash,),
        ).fetchall()
        loaded: list[ScheduleMetrics] = []
        for row in rows:
            payload = json.loads(row['metrics_json'])
            loaded.append(ScheduleMetrics(**payload))
        return loaded


def _cell_records_payload(cells: Sequence[CellZnccRecord]) -> list[dict[str, Any]]:
    return [asdict(cell) for cell in cells]


def _cells_from_payload(payload: Sequence[Mapping[str, Any]]) -> list[CellZnccRecord]:
    return [CellZnccRecord(**dict(item)) for item in payload]


def _median(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.median(values))


def _quantile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


def _score_pair(stos_path: str, *, max_side: int) -> float:
    return float(compute_pair_zncc(stos_path, max_side=max_side).pair_zncc)


def _score_cells(
        stos_path: str,
        *,
        cell_size: int,
        grid_spacing: int,
) -> list[CellZnccRecord]:
    result = compute_cell_zncc(
        stos_path,
        cell_size=cell_size,
        grid_spacing=grid_spacing,
        max_side=None,
    )
    return list(result.cells)


def default_refine(
        input_stos: str,
        output_stos: str,
        schedule: RefineSchedule,
) -> None:
    """Run ``RefineStosFile`` for *schedule* without modifying the input path."""
    os.environ.setdefault('NORNIR_HEADLESS', '1')
    os.environ['NORNIR_REFINE_PASS_DIAGNOSTICS'] = '1'
    from nornir_imageregistration.local_distortion_correction import RefineStosFile
    Path(output_stos).parent.mkdir(parents=True, exist_ok=True)
    RefineStosFile(
        input_stos,
        output_stos,
        num_iterations=schedule.num_iterations,
        cell_size=schedule.cell_size,
        grid_spacing=schedule.grid_spacing,
        angles_to_search=[0.0],
        SaveImages=False,
        SavePlots=False,
    )


def _quality_flag_present(output_stos: str) -> bool:
    return Path(output_stos).with_suffix('.quality_flag').is_file()


def build_schedule_metrics(
        *,
        pair: str,
        schedule: RefineSchedule,
        pair_zncc_in: float | None,
        pair_zncc_out: float | None,
        baseline_cells: Sequence[CellZnccRecord],
        output_cells: Sequence[CellZnccRecord],
        quality_flag: bool,
        wall_s: float | None,
        output_stos: str | None,
        error: str | None,
        pair_regression_tolerance: float = PAIR_REGRESSION_TOLERANCE,
        weak_quantile: float = WEAK_QUANTILE,
        cluster_repair_fraction: float = CLUSTER_REPAIR_FRACTION,
) -> ScheduleMetrics:
    """Build ranking metrics from baseline and output scores."""
    comparison = compare_cell_zncc(baseline_cells, output_cells)
    threshold = weak_cell_threshold(baseline_cells, quantile=weak_quantile)
    if threshold is None:
        clusters: list[set[tuple[int, int]]] = []
        repaired = 0
    else:
        clusters = clusters_from_ids(weak_cell_ids(baseline_cells, threshold=threshold))
        repaired = repaired_cluster_count(
            clusters,
            comparison,
            weak_threshold=threshold,
            repair_fraction=cluster_repair_fraction,
        )
    pair_delta: float | None = None
    if pair_zncc_in is not None and pair_zncc_out is not None:
        pair_delta = float(pair_zncc_out - pair_zncc_in)
    reject_reason: str | None = None
    if error:
        reject_reason = 'error'
    elif quality_flag:
        reject_reason = 'quality_flag'
    elif pair_delta is not None and pair_delta < -float(pair_regression_tolerance):
        reject_reason = 'pair_zncc_regression'
    baseline_scores = _finite_scores(baseline_cells)
    output_scores = _finite_scores(output_cells)
    return ScheduleMetrics(
        pair=pair,
        schedule=schedule.name,
        accepted=reject_reason is None,
        reject_reason=reject_reason,
        quality_flag=quality_flag,
        pair_zncc_in=pair_zncc_in,
        pair_zncc_out=pair_zncc_out,
        pair_zncc_delta=pair_delta,
        improved_count=comparison.improved_count,
        worse_count=comparison.worse_count,
        unchanged_count=comparison.unchanged_count,
        excluded_count=comparison.excluded_count,
        net_improved=comparison.improved_count - comparison.worse_count,
        delta_median=comparison.delta_median,
        delta_p05=comparison.delta_p05,
        delta_min=comparison.delta_min,
        delta_max=comparison.delta_max,
        median_cell_in=_median(baseline_scores),
        median_cell_out=_median(output_scores),
        p05_cell_in=_quantile(baseline_scores, 0.05),
        baseline_weak_clusters=len(clusters),
        repaired_clusters=repaired,
        wall_s=wall_s,
        output_stos=output_stos,
        error=error,
    )


def _load_or_score_baseline(
        pair_dir: Path,
        stos_path: str,
        checksum: str | None,
        score_pair: ScorePairCallable,
        score_cells: ScoreCellsCallable,
) -> tuple[float, list[CellZnccRecord]]:
    cache_path = pair_dir / 'baseline.json'
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding='utf-8'))
        if payload.get('stos_checksum') == checksum:
            return float(payload['pair_zncc']), _cells_from_payload(payload['cells'])
    pair_zncc = score_pair(stos_path)
    cells = score_cells(stos_path)
    pair_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({
            'stos_checksum': checksum,
            'pair_zncc': pair_zncc,
            'cells': _cell_records_payload(cells),
        }, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return pair_zncc, cells


def trial_pair_schedule(
        candidate: Candidate,
        schedule: RefineSchedule,
        *,
        config: ScoutConfig,
        store: ScoutStore,
        code_sha: str | None,
        refine: RefineCallable = default_refine,
        score_pair: ScorePairCallable | None = None,
        score_cells: ScoreCellsCallable | None = None,
) -> ScheduleMetrics:
    """Trial-refine one pair/schedule, resuming a matching completed checkpoint."""
    settings_hash = config.settings_hash()
    if store.trial_is_reusable(
            candidate.pair,
            schedule.name,
            source_checksum=candidate.stos_checksum,
            code_sha=code_sha,
            settings_hash=settings_hash,
    ):
        row = store.get_trial(candidate.pair, schedule.name)
        assert row is not None
        return ScheduleMetrics(**json.loads(row['metrics_json']))

    pair_fn = score_pair or (
        lambda path: _score_pair(path, max_side=config.coarse_max_side)
    )
    cells_fn = score_cells or (
        lambda path: _score_cells(
            path,
            cell_size=config.score_cell_size,
            grid_spacing=config.score_grid_spacing,
        )
    )
    pair_dir = Path(config.output_dir) / 'pairs' / candidate.pair
    output_stos = pair_dir / schedule.name / Path(candidate.relpath).name
    error: str | None = None
    wall_s: float | None = None
    quality_flag = False
    pair_zncc_in: float | None = candidate.pair_zncc
    pair_zncc_out: float | None = None
    baseline_cells: list[CellZnccRecord] = []
    output_cells: list[CellZnccRecord] = []
    try:
        pair_zncc_in, baseline_cells = _load_or_score_baseline(
            pair_dir,
            candidate.stos_path,
            candidate.stos_checksum,
            pair_fn,
            cells_fn,
        )
        started = time.perf_counter()
        refine(candidate.stos_path, str(output_stos), schedule)
        wall_s = time.perf_counter() - started
        if not output_stos.is_file():
            raise FileNotFoundError(output_stos)
        quality_flag = _quality_flag_present(str(output_stos))
        pair_zncc_out = pair_fn(str(output_stos))
        output_cells = cells_fn(str(output_stos))
        (pair_dir / schedule.name / 'output_cells.json').write_text(
            json.dumps(_cell_records_payload(output_cells), indent=2) + '\n',
            encoding='utf-8',
        )
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'

    metrics = build_schedule_metrics(
        pair=candidate.pair,
        schedule=schedule,
        pair_zncc_in=pair_zncc_in,
        pair_zncc_out=pair_zncc_out,
        baseline_cells=baseline_cells,
        output_cells=output_cells,
        quality_flag=quality_flag,
        wall_s=wall_s,
        output_stos=str(output_stos) if output_stos.is_file() else None,
        error=error,
        pair_regression_tolerance=config.pair_regression_tolerance,
        weak_quantile=config.weak_quantile,
        cluster_repair_fraction=config.cluster_repair_fraction,
    )
    store.save_trial(
        metrics,
        status='failed' if error else 'completed',
        source_checksum=candidate.stos_checksum,
        code_sha=code_sha,
        settings_hash=settings_hash,
    )
    return metrics


def write_reports(
        *,
        config: ScoutConfig,
        candidates: Sequence[Candidate],
        shortlist: Sequence[Candidate],
        results: Sequence[ScheduleMetrics],
        ranked: Sequence[ScheduleMetrics],
) -> dict[str, Path]:
    """Write JSON/CSV reports under ``config.output_dir``."""
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / 'config.json'
    config_path.write_text(
        json.dumps({
            'created': _utc_now(),
            'command': config.command,
            'group_dir': config.group_dir,
            'output_dir': config.output_dir,
            'limit': config.limit,
            'settings_hash': config.settings_hash(),
            'settings': config.settings_payload(),
            'screen_pairs': [item.pair for item in shortlist],
            'inventory_count': len(candidates),
            'shortlist_count': len(shortlist),
        }, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    trials_path = out / 'trials.csv'
    trial_fields = [
        'pair', 'schedule', 'accepted', 'reject_reason', 'quality_flag',
        'pair_zncc_in', 'pair_zncc_out', 'pair_zncc_delta',
        'improved_count', 'worse_count', 'net_improved',
        'delta_median', 'delta_p05', 'repaired_clusters',
        'baseline_weak_clusters', 'wall_s', 'error',
    ]
    with trials_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=trial_fields, lineterminator='\n')
        writer.writeheader()
        for row in results:
            writer.writerow({key: asdict(row).get(key) for key in trial_fields})

    priority_path = out / 'priority.csv'
    priority_fields = [
        'rank', 'pair', 'schedule', 'accepted', 'reject_reason',
        'repaired_clusters', 'net_improved', 'delta_p05', 'delta_median',
        'pair_zncc_in', 'pair_zncc_out', 'pair_zncc_delta', 'wall_s',
    ]
    with priority_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=priority_fields, lineterminator='\n')
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            payload = asdict(row)
            payload['rank'] = rank
            writer.writerow({key: payload.get(key) for key in priority_fields})

    json_path = out / 'priority.json'
    json_path.write_text(
        json.dumps({
            'ranked': [asdict(row) for row in ranked],
            'trials': [asdict(row) for row in results],
        }, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return {
        'config': config_path,
        'trials': trials_path,
        'priority_csv': priority_path,
        'priority_json': json_path,
    }


def default_output_dir(group_dir: str | Path) -> Path:
    """Return ``TESTOUTPUTPATH/refine_group_scout/<group>`` or a /tmp fallback."""
    root = os.environ.get('TESTOUTPUTPATH') or os.environ.get('TEST_OUTPUT_DIR')
    if not root:
        root = '/tmp/nornir-test-output'
    return Path(root) / 'refine_group_scout' / Path(group_dir).name


def run_scout(
        config: ScoutConfig,
        *,
        refine: RefineCallable = default_refine,
        score_pair: ScorePairCallable | None = None,
        score_cells: ScoreCellsCallable | None = None,
        store: ScoutStore | None = None,
) -> tuple[list[ScheduleMetrics], list[ScheduleMetrics], dict[str, Path]]:
    """Inventory, trial-refine the shortlist, persist reports, and return rankings."""
    owned_store = store is None
    if store is None:
        store = ScoutStore(Path(config.output_dir) / SCOUT_DB_NAME)
    try:
        code_sha = package_git_sha()
        store.set_meta('settings_hash', config.settings_hash())
        store.set_meta('code_sha', code_sha or '')
        store.set_meta('group_dir', config.group_dir)
        store.set_meta('command', json.dumps(config.command))
        inventory = inventory_candidates(config.group_dir, exclude_manual=config.exclude_manual)
        shortlist = screen_candidates(
            inventory,
            limit=config.limit,
            exclude_manual=config.exclude_manual,
        )
        store.upsert_candidates(shortlist)
        results: list[ScheduleMetrics] = []
        total = len(shortlist) * len(config.schedules)
        completed = 0
        for candidate in shortlist:
            for schedule in config.schedules:
                completed += 1
                print(
                    f'[{completed}/{total}] {candidate.pair} {schedule.name} '
                    f'baseline_zncc={candidate.pair_zncc}',
                    flush=True,
                )
                result = trial_pair_schedule(
                    candidate,
                    schedule,
                    config=config,
                    store=store,
                    code_sha=code_sha,
                    refine=refine,
                    score_pair=score_pair,
                    score_cells=score_cells,
                )
                print(
                    f'    accepted={result.accepted} repaired={result.repaired_clusters} '
                    f'net={result.net_improved} dPair={result.pair_zncc_delta} '
                    f'error={result.error}',
                    flush=True,
                )
                results.append(result)
        ranked = rank_pairs(results)
        reports = write_reports(
            config=config,
            candidates=inventory,
            shortlist=shortlist,
            results=results,
            ranked=ranked,
        )
        return results, ranked, reports
    finally:
        if owned_store:
            store.close()
