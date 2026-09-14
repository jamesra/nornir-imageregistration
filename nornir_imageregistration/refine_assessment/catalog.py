"""SQLite catalog (inventory / bests / adopts) and local runs database."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from nornir_imageregistration.refine_assessment.tags import (
    TagSource,
    TagStatus,
    ensure_seed_tags,
)
from nornir_imageregistration.refine_assessment.verdict import MetricDirection, ScoreVector

CATALOG_FILENAME: str = 'catalog.sqlite'
RUNS_FILENAME: str = 'refine_runs.sqlite'
SCHEMA_VERSION: int = 1

_CATALOG_SCHEMA = '''
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fixture (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    relative_dir TEXT NOT NULL UNIQUE,
    volume TEXT,
    group_name TEXT,
    downsample REAL,
    pair TEXT,
    channel TEXT,
    gold_kind TEXT,
    candidate_source TEXT,
    struggle_vs_damage TEXT,
    source_stos_checksum TEXT,
    bbox_json TEXT,
    halo_hops INTEGER DEFAULT 2,
    pass_index INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tag (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    suggestable INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS fixture_tag (
    fixture_id INTEGER NOT NULL REFERENCES fixture(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tag(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (fixture_id, tag_id)
);

CREATE TABLE IF NOT EXISTS certified_cell (
    fixture_id INTEGER NOT NULL REFERENCES fixture(id) ON DELETE CASCADE,
    grid_row INTEGER NOT NULL,
    grid_col INTEGER NOT NULL,
    gold_target_y REAL,
    gold_target_x REAL,
    PRIMARY KEY (fixture_id, grid_row, grid_col)
);

CREATE TABLE IF NOT EXISTS best_score (
    fixture_id INTEGER NOT NULL REFERENCES fixture(id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    direction TEXT NOT NULL,
    band_lo REAL,
    band_hi REAL,
    PRIMARY KEY (fixture_id, metric_name)
);

CREATE TABLE IF NOT EXISTS adopt (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER NOT NULL REFERENCES fixture(id) ON DELETE CASCADE,
    adopted_at TEXT NOT NULL,
    imageregistration_sha TEXT,
    umbrella_sha TEXT,
    code_path TEXT,
    note TEXT,
    run_id INTEGER
);

CREATE TABLE IF NOT EXISTS adopt_score (
    adopt_id INTEGER NOT NULL REFERENCES adopt(id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (adopt_id, metric_name)
);
'''

_RUNS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER,
    fixture_dir TEXT NOT NULL,
    started_at TEXT NOT NULL,
    imageregistration_sha TEXT,
    flags TEXT,
    wall_s REAL,
    verdict TEXT,
    quality_flag INTEGER DEFAULT 0,
    note TEXT
);

CREATE TABLE IF NOT EXISTS run_score (
    run_id INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (run_id, metric_name)
);

CREATE TABLE IF NOT EXISTS run_pass (
    run_id INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
    pass_index INTEGER NOT NULL,
    unique_frac REAL,
    lock_frac REAL,
    PRIMARY KEY (run_id, pass_index)
);
'''


def catalog_path(fixtures_root: str | Path) -> Path:
    """Return ``catalog.sqlite`` under *fixtures_root*."""
    return Path(fixtures_root) / CATALOG_FILENAME


def default_catalog_path(testinput: str | Path | None = None) -> Path:
    """Default catalog under ``TESTINPUTPATH/refine_fixtures``."""
    root = testinput or os.environ.get('TESTINPUTPATH', '')
    if not root:
        raise EnvironmentError('TESTINPUTPATH is not set')
    return catalog_path(Path(root) / 'refine_fixtures')


def runs_db_path(testoutput: str | Path | None = None) -> Path:
    """Default runs DB under ``TESTOUTPUTPATH``."""
    root = testoutput or os.environ.get('TESTOUTPUTPATH') or os.environ.get('TEST_OUTPUT_DIR')
    if not root:
        raise EnvironmentError('TESTOUTPUTPATH is not set')
    return Path(root) / RUNS_FILENAME


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Catalog:
    """Durable fixture inventory and adopted living bests."""

    path: Path
    _conn: sqlite3.Connection

    @classmethod
    def open(cls, path: str | Path, *, create: bool = True) -> Catalog:
        """Open or create the catalog at *path*."""
        path = Path(path)
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        catalog = cls(path=path, _conn=conn)
        catalog._migrate()
        return catalog

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _migrate(self) -> None:
        self._conn.executescript(_CATALOG_SCHEMA)
        ensure_seed_tags(self._conn)
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),))
        self._conn.commit()

    def upsert_fixture(
            self,
            relative_dir: str,
            *,
            volume: str | None = None,
            group_name: str | None = None,
            downsample: float | None = None,
            pair: str | None = None,
            channel: str | None = None,
            gold_kind: str | None = None,
            candidate_source: str | None = None,
            struggle_vs_damage: str | None = None,
            source_stos_checksum: str | None = None,
            bbox: Sequence[float] | None = None,
            halo_hops: int = 2,
            pass_index: int | None = None,
    ) -> int:
        """Insert or update a fixture row; returns fixture id."""
        bbox_json = json.dumps(list(bbox)) if bbox is not None else None
        existing = self._conn.execute(
            'SELECT id FROM fixture WHERE relative_dir = ?', (relative_dir,)).fetchone()
        if existing is None:
            cur = self._conn.execute(
                '''
                INSERT INTO fixture (
                    relative_dir, volume, group_name, downsample, pair, channel,
                    gold_kind, candidate_source, struggle_vs_damage,
                    source_stos_checksum, bbox_json, halo_hops, pass_index, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    relative_dir, volume, group_name, downsample, pair, channel,
                    gold_kind, candidate_source, struggle_vs_damage,
                    source_stos_checksum, bbox_json, int(halo_hops), pass_index, _utc_now(),
                ),
            )
            fixture_id = int(cur.lastrowid)
        else:
            fixture_id = int(existing['id'])
            self._conn.execute(
                '''
                UPDATE fixture SET
                    volume = COALESCE(?, volume),
                    group_name = COALESCE(?, group_name),
                    downsample = COALESCE(?, downsample),
                    pair = COALESCE(?, pair),
                    channel = COALESCE(?, channel),
                    gold_kind = COALESCE(?, gold_kind),
                    candidate_source = COALESCE(?, candidate_source),
                    struggle_vs_damage = COALESCE(?, struggle_vs_damage),
                    source_stos_checksum = COALESCE(?, source_stos_checksum),
                    bbox_json = COALESCE(?, bbox_json),
                    halo_hops = ?,
                    pass_index = COALESCE(?, pass_index)
                WHERE id = ?
                ''',
                (
                    volume, group_name, downsample, pair, channel, gold_kind,
                    candidate_source, struggle_vs_damage, source_stos_checksum,
                    bbox_json, int(halo_hops), pass_index, fixture_id,
                ),
            )
        self._conn.commit()
        return fixture_id

    def set_certified_cells(
            self,
            fixture_id: int,
            cells: Iterable[tuple[int, int, float | None, float | None]],
    ) -> None:
        """Replace certified cells for *fixture_id*."""
        self._conn.execute(
            'DELETE FROM certified_cell WHERE fixture_id = ?', (fixture_id,))
        self._conn.executemany(
            '''
            INSERT INTO certified_cell
                (fixture_id, grid_row, grid_col, gold_target_y, gold_target_x)
            VALUES (?, ?, ?, ?, ?)
            ''',
            [(fixture_id, int(r), int(c), gy, gx) for r, c, gy, gx in cells],
        )
        self._conn.commit()

    def tag_fixture(
            self,
            fixture_id: int,
            slug: str,
            *,
            status: TagStatus | str = TagStatus.SUGGESTED,
            source: TagSource | str = TagSource.DIAGNOSTICS,
    ) -> None:
        """Attach *slug* to *fixture_id* (upsert status/source)."""
        row = self._conn.execute(
            'SELECT id FROM tag WHERE slug = ?', (slug,)).fetchone()
        if row is None:
            raise KeyError(f'Unknown tag slug: {slug}')
        tag_id = int(row['id'])
        self._conn.execute(
            '''
            INSERT INTO fixture_tag (fixture_id, tag_id, status, source)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(fixture_id, tag_id) DO UPDATE SET
                status = excluded.status,
                source = excluded.source
            ''',
            (fixture_id, tag_id, str(status), str(source)),
        )
        self._conn.commit()

    def confirm_tags(self, fixture_id: int, slugs: Sequence[str]) -> None:
        """Mark *slugs* as confirmed human tags on *fixture_id*."""
        for slug in slugs:
            self.tag_fixture(
                fixture_id, slug, status=TagStatus.CONFIRMED, source=TagSource.HUMAN)

    def list_fixtures(
            self,
            *,
            tag: str | None = None,
            confirmed_only: bool = False,
    ) -> list[sqlite3.Row]:
        """Return fixture rows, optionally filtered by tag."""
        if tag is None:
            return list(self._conn.execute('SELECT * FROM fixture ORDER BY id'))
        status_clause = "AND ft.status = 'confirmed'" if confirmed_only else ''
        return list(self._conn.execute(
            f'''
            SELECT f.* FROM fixture f
            JOIN fixture_tag ft ON ft.fixture_id = f.id
            JOIN tag t ON t.id = ft.tag_id
            WHERE t.slug = ? {status_clause}
            ORDER BY f.id
            ''',
            (tag,),
        ))

    def fixture_tags(
            self,
            fixture_id: int,
            *,
            confirmed_only: bool = False,
    ) -> list[dict[str, str]]:
        """Return tag dicts for *fixture_id*."""
        clause = "AND ft.status = 'confirmed'" if confirmed_only else ''
        rows = self._conn.execute(
            f'''
            SELECT t.slug, ft.status, ft.source
            FROM fixture_tag ft
            JOIN tag t ON t.id = ft.tag_id
            WHERE ft.fixture_id = ? {clause}
            ORDER BY t.slug
            ''',
            (fixture_id,),
        ).fetchall()
        return [{'slug': r['slug'], 'status': r['status'], 'source': r['source']} for r in rows]

    def get_best_scores(self, fixture_id: int) -> dict[str, dict[str, Any]]:
        """Return adopted best metrics for *fixture_id*."""
        rows = self._conn.execute(
            'SELECT metric_name, value, direction, band_lo, band_hi FROM best_score WHERE fixture_id = ?',
            (fixture_id,),
        ).fetchall()
        return {
            r['metric_name']: {
                'value': float(r['value']),
                'direction': r['direction'],
                'band_lo': r['band_lo'],
                'band_hi': r['band_hi'],
            }
            for r in rows
        }

    def adopt_scores(
            self,
            fixture_id: int,
            scores: ScoreVector | Mapping[str, float],
            *,
            directions: Mapping[str, MetricDirection | str] | None = None,
            bands: Mapping[str, tuple[float | None, float | None]] | None = None,
            imageregistration_sha: str | None = None,
            umbrella_sha: str | None = None,
            code_path: str | None = None,
            note: str | None = None,
            run_id: int | None = None,
    ) -> int:
        """Write a new adopt record and update ``best_score``; return adopt id."""
        if isinstance(scores, ScoreVector):
            values = dict(scores.values)
        else:
            values = {k: float(v) for k, v in scores.items()}
        directions = directions or {}
        bands = bands or {}
        cur = self._conn.execute(
            '''
            INSERT INTO adopt (
                fixture_id, adopted_at, imageregistration_sha, umbrella_sha,
                code_path, note, run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                fixture_id, _utc_now(), imageregistration_sha, umbrella_sha,
                code_path, note, run_id,
            ),
        )
        adopt_id = int(cur.lastrowid)
        for name, value in values.items():
            self._conn.execute(
                'INSERT INTO adopt_score (adopt_id, metric_name, value) VALUES (?, ?, ?)',
                (adopt_id, name, float(value)),
            )
            direction = str(directions.get(name, MetricDirection.MAXIMIZE))
            band = bands.get(name, (None, None))
            self._conn.execute(
                '''
                INSERT INTO best_score
                    (fixture_id, metric_name, value, direction, band_lo, band_hi)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(fixture_id, metric_name) DO UPDATE SET
                    value = excluded.value,
                    direction = excluded.direction,
                    band_lo = excluded.band_lo,
                    band_hi = excluded.band_hi
                ''',
                (fixture_id, name, float(value), direction, band[0], band[1]),
            )
        self._conn.commit()
        return adopt_id

    def export_best_json(self, fixture_id: int, path: str | Path) -> Path:
        """Write git-diffable ``best.json`` for *fixture_id*."""
        path = Path(path)
        best = self.get_best_scores(fixture_id)
        adopt = self._conn.execute(
            '''
            SELECT adopted_at, imageregistration_sha, umbrella_sha, code_path, note
            FROM adopt WHERE fixture_id = ? ORDER BY id DESC LIMIT 1
            ''',
            (fixture_id,),
        ).fetchone()
        payload = {
            'fixture_id': fixture_id,
            'adopted_at': adopt['adopted_at'] if adopt else None,
            'imageregistration_sha': adopt['imageregistration_sha'] if adopt else None,
            'umbrella_sha': adopt['umbrella_sha'] if adopt else None,
            'code_path': adopt['code_path'] if adopt else None,
            'note': adopt['note'] if adopt else None,
            'metrics': best,
            'confirmed_tags': [t['slug'] for t in self.fixture_tags(fixture_id, confirmed_only=True)],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        return path

    def fixture_id_for_dir(self, relative_dir: str) -> int | None:
        """Return fixture id for *relative_dir*, or None."""
        row = self._conn.execute(
            'SELECT id FROM fixture WHERE relative_dir = ?', (relative_dir,)).fetchone()
        return int(row['id']) if row else None


@dataclass
class RunsDb:
    """Local experimental A/B run history (not git-tracked)."""

    path: Path
    _conn: sqlite3.Connection

    @classmethod
    def open(cls, path: str | Path, *, create: bool = True) -> RunsDb:
        """Open or create the runs database at *path*."""
        path = Path(path)
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        db = cls(path=path, _conn=conn)
        db._migrate()
        return db

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> RunsDb:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _migrate(self) -> None:
        self._conn.executescript(_RUNS_SCHEMA)
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),))
        self._conn.commit()

    def insert_run(
            self,
            fixture_dir: str,
            scores: ScoreVector,
            *,
            fixture_id: int | None = None,
            imageregistration_sha: str | None = None,
            flags: str | None = None,
            wall_s: float | None = None,
            verdict: str | None = None,
            note: str | None = None,
    ) -> int:
        """Insert one A/B run and its scores; return run id."""
        cur = self._conn.execute(
            '''
            INSERT INTO run (
                fixture_id, fixture_dir, started_at, imageregistration_sha,
                flags, wall_s, verdict, quality_flag, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                fixture_id, fixture_dir, _utc_now(), imageregistration_sha,
                flags, wall_s, verdict, 1 if scores.quality_flag else 0, note,
            ),
        )
        run_id = int(cur.lastrowid)
        for name, value in scores.values.items():
            self._conn.execute(
                'INSERT INTO run_score (run_id, metric_name, value) VALUES (?, ?, ?)',
                (run_id, name, float(value)),
            )
        for i, (uf, lf) in enumerate(
                zip(
                    scores.unique_frac_series,
                    [scores.values.get('lock_frac', float('nan'))] * len(scores.unique_frac_series),
                ),
                start=1,
        ):
            self._conn.execute(
                '''
                INSERT INTO run_pass (run_id, pass_index, unique_frac, lock_frac)
                VALUES (?, ?, ?, ?)
                ''',
                (run_id, i, float(uf), float(lf) if lf == lf else None),
            )
        self._conn.commit()
        return run_id

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        """Return a run row by id."""
        return self._conn.execute('SELECT * FROM run WHERE id = ?', (run_id,)).fetchone()

    def get_run_scores(self, run_id: int) -> dict[str, float]:
        """Return metric map for *run_id*."""
        rows = self._conn.execute(
            'SELECT metric_name, value FROM run_score WHERE run_id = ?', (run_id,)).fetchall()
        return {r['metric_name']: float(r['value']) for r in rows}
