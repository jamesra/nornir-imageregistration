"""Tests for StosGroup scout inventory, ranking, resume, and isolation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nornir_imageregistration.files.stosfile import StosFile
from nornir_imageregistration.refine_assessment.group_scout import (
    DENSE_256,
    FINE_128,
    Candidate,
    RefineSchedule,
    ScheduleMetrics,
    ScoutConfig,
    ScoutStore,
    build_schedule_metrics,
    clusters_from_ids,
    inventory_candidates,
    pair_id_from_name,
    rank_pairs,
    ranking_key,
    repaired_cluster_count,
    run_scout,
    screen_candidates,
    trial_pair_schedule,
    weak_cell_ids,
    weak_cell_threshold,
)
from nornir_imageregistration.stos_quality import (
    CellZnccComparison,
    CellZnccDeltaRecord,
    CellZnccRecord,
    QualityCache,
    save_quality_cache,
)
from nornir_imageregistration.transforms import factory
import nornir_imageregistration.core as core
import numpy as np


def _identity_transform(shape: tuple[int, int] = (16, 16)):
    dims = np.asarray(shape, dtype=np.float64)
    return factory.CreateRigidTransform((0, 0), 0.0, dims, dims)


def _write_stos(group: Path, name: str, *, seed: int = 1) -> Path:
    images = group.parent / 'images'
    images.mkdir(parents=True, exist_ok=True)
    ctrl = images / f'{name}_ctrl.png'
    mapped = images / f'{name}_map.png'
    rng = np.random.default_rng(seed)
    image = rng.integers(32, 224, size=(16, 16), dtype=np.uint8)
    core.SaveImage(str(ctrl), image)
    core.SaveImage(str(mapped), image)
    stos_path = group / name
    stos = StosFile()
    stos.ControlImageFullPath = str(ctrl)
    stos.MappedImageFullPath = str(mapped)
    stos.ControlImageDim = [1.0, 1.0, 16, 16]
    stos.MappedImageDim = [1.0, 1.0, 16, 16]
    stos.Downsample = 1
    stos.Transform = _identity_transform()
    stos.Save(str(stos_path))
    return stos_path


def _cache_entry(path: Path, pair_zncc: float) -> dict[str, object]:
    return {
        'pair_zncc': pair_zncc,
        'stos_checksum': StosFile.LoadChecksum(str(path)),
        'stos_mtime_ns': path.stat().st_mtime_ns,
    }


def _cell(row: int, col: int, score: float | None) -> CellZnccRecord:
    return CellZnccRecord(
        grid_row=row,
        grid_col=col,
        center_y=float(row),
        center_x=float(col),
        valid_pixel_count=4,
        total_pixel_count=4,
        exclusion_reason=None if score is not None else 'invalid',
        zncc=score,
    )


def _metrics(
        pair: str,
        schedule: str,
        *,
        accepted: bool = True,
        repaired: int = 0,
        net: int = 0,
        p05: float | None = 0.0,
        median: float | None = 0.0,
        pair_delta: float | None = 0.0,
        improved: int = 0,
        worse: int = 0,
        error: str | None = None,
) -> ScheduleMetrics:
    return ScheduleMetrics(
        pair=pair,
        schedule=schedule,
        accepted=accepted,
        reject_reason=None if accepted else 'quality_flag',
        quality_flag=not accepted,
        pair_zncc_in=0.2,
        pair_zncc_out=None if pair_delta is None else 0.2 + pair_delta,
        pair_zncc_delta=pair_delta,
        improved_count=improved,
        worse_count=worse,
        unchanged_count=0,
        excluded_count=0,
        net_improved=net,
        delta_median=median,
        delta_p05=p05,
        delta_min=p05,
        delta_max=median,
        median_cell_in=0.2,
        median_cell_out=0.2 if median is None else 0.2 + median,
        p05_cell_in=0.1,
        baseline_weak_clusters=repaired,
        repaired_clusters=repaired,
        wall_s=1.0,
        output_stos=None,
        error=error,
    )


def test_pair_id_from_manual_suffixes() -> None:
    """Manual -Rigid / -Mesh names still resolve to the section pair."""
    assert pair_id_from_name('826-822_ctrl-TEM_Leveled_map-TEM_Leveled.stos') == '826-822'
    assert pair_id_from_name('826-822-Rigid_ctrl-TEM_Leveled_map-TEM_Leveled.stos') == '826-822'
    assert pair_id_from_name('readme.txt') is None


def test_inventory_excludes_manual_and_uses_valid_cache(tmp_path: Path) -> None:
    """Group-root files are inventoried; Manual pairs are flagged and skipped in screening."""
    group = tmp_path / 'Grid16'
    manual = group / 'Manual'
    automatic = group / 'Automatic'
    group.mkdir()
    manual.mkdir()
    automatic.mkdir()
    low = _write_stos(group, '10-11_ctrl-TEM_Leveled_map-TEM_Leveled.stos', seed=1)
    high = _write_stos(group, '20-21_ctrl-TEM_Leveled_map-TEM_Leveled.stos', seed=2)
    manual_pair = _write_stos(group, '30-31_ctrl-TEM_Leveled_map-TEM_Leveled.stos', seed=3)
    _write_stos(manual, '30-31_ctrl-TEM_Leveled_map-TEM_Leveled.stos', seed=4)
    _write_stos(automatic, '40-41_ctrl-TEM_Leveled_map-TEM_Leveled.stos', seed=5)
    cache = QualityCache(entries={
        low.name: _cache_entry(low, 0.10),
        high.name: _cache_entry(high, 0.90),
        manual_pair.name: _cache_entry(manual_pair, 0.01),
    })
    save_quality_cache(str(group), cache)

    inventory = inventory_candidates(group, exclude_manual=True)
    pairs = {item.pair: item for item in inventory}
    assert set(pairs) == {'10-11', '20-21', '30-31'}
    assert pairs['30-31'].excluded_manual is True
    assert pairs['10-11'].cache_valid is True
    shortlist = screen_candidates(inventory, limit=10, exclude_manual=True)
    assert [item.pair for item in shortlist] == ['10-11', '20-21']


def test_stale_cache_is_not_used_for_screening(tmp_path: Path) -> None:
    """Checksum-mismatched cache entries are not ranked."""
    group = tmp_path / 'Grid16'
    group.mkdir()
    path = _write_stos(group, '1-2_ctrl-TEM_Leveled_map-TEM_Leveled.stos')
    cache = QualityCache(entries={
        path.name: {
            'pair_zncc': 0.01,
            'stos_checksum': 'not-the-real-checksum',
        },
    })
    save_quality_cache(str(group), cache)
    inventory = inventory_candidates(group)
    assert inventory[0].cache_valid is False
    assert screen_candidates(inventory, limit=10) == []


def test_screen_order_is_worst_pair_zncc_then_pair_id(tmp_path: Path) -> None:
    """Screening is deterministic: lowest pair ZNCC first, then pair id."""
    group = tmp_path / 'Grid16'
    group.mkdir()
    a = _write_stos(group, '5-6_ctrl-TEM_Leveled_map-TEM_Leveled.stos', seed=1)
    b = _write_stos(group, '3-4_ctrl-TEM_Leveled_map-TEM_Leveled.stos', seed=2)
    c = _write_stos(group, '7-8_ctrl-TEM_Leveled_map-TEM_Leveled.stos', seed=3)
    cache = QualityCache(entries={
        a.name: _cache_entry(a, 0.20),
        b.name: _cache_entry(b, 0.20),
        c.name: _cache_entry(c, 0.05),
    })
    save_quality_cache(str(group), cache)
    shortlist = screen_candidates(inventory_candidates(group), limit=None)
    assert [item.pair for item in shortlist] == ['7-8', '3-4', '5-6']


def test_weak_cell_clusters_are_4_connected() -> None:
    """Adjacent weak cells form one cluster; a diagonal neighbor does not."""
    cells = [
        _cell(0, 0, 0.05),
        _cell(0, 1, 0.06),
        _cell(1, 0, 0.90),
        _cell(2, 2, 0.04),
    ]
    assert weak_cell_threshold(cells, quantile=0.75) is not None
    ids = weak_cell_ids(cells, threshold=0.07)
    clusters = clusters_from_ids(ids)
    assert {frozenset(cluster) for cluster in clusters} == {
        frozenset({(0, 0), (0, 1)}),
        frozenset({(2, 2)}),
    }


def test_repaired_cluster_requires_majority() -> None:
    """A cluster counts as repaired only when a majority of cells recover."""
    comparison = CellZnccComparison(
        cells=[
            CellZnccDeltaRecord(0, 0, 0.0, 0.0, 0.05, 0.40, 0.35, 'improved'),
            CellZnccDeltaRecord(0, 1, 0.0, 1.0, 0.06, 0.07, 0.01, 'improved'),
            CellZnccDeltaRecord(2, 2, 2.0, 2.0, 0.04, 0.03, -0.01, 'worse'),
        ],
        improved_count=2,
        worse_count=1,
        unchanged_count=0,
        excluded_count=0,
        delta_median=0.01,
        delta_min=-0.01,
        delta_max=0.35,
        delta_p05=-0.01,
        delta_p95=0.35,
        tolerance=1e-4,
    )
    clusters = [{(0, 0), (0, 1)}, {(2, 2)}]
    assert repaired_cluster_count(clusters, comparison, weak_threshold=0.10) == 1


def test_ranking_rejects_quality_flag_and_prefers_local_repair() -> None:
    """Rejected results sort after accepted ones; local repair beats pair ZNCC."""
    rejected = _metrics('1-2', 'dense256', accepted=False, repaired=9, net=20)
    weaker = _metrics('3-4', 'dense256', repaired=1, net=1, p05=0.01, pair_delta=0.20)
    stronger = _metrics('5-6', 'fine128', repaired=3, net=4, p05=0.02, pair_delta=0.01)
    ranked = rank_pairs([rejected, weaker, stronger])
    assert [row.pair for row in ranked] == ['5-6', '3-4', '1-2']


def test_choose_best_schedule_is_isolated() -> None:
    """The better schedule for a pair is selected without mixing metrics."""
    dense = _metrics('9-10', 'dense256', repaired=1, net=2, p05=0.01)
    fine = _metrics('9-10', 'fine128', repaired=4, net=1, p05=0.00)
    ranked = rank_pairs([dense, fine])
    assert len(ranked) == 1
    assert ranked[0].schedule == 'fine128'
    assert ranked[0].repaired_clusters == 4


def _fake_cells(score: float) -> list[CellZnccRecord]:
    return [_cell(0, 0, score), _cell(0, 1, score), _cell(1, 0, score)]


def test_resume_skips_matching_completed_trial(tmp_path: Path) -> None:
    """A completed trial with matching checksum/SHA/hash is not refined again."""
    group = tmp_path / 'Grid16'
    out = tmp_path / 'scout'
    group.mkdir()
    path = _write_stos(group, '1-2_ctrl-TEM_Leveled_map-TEM_Leveled.stos')
    cache = QualityCache(entries={path.name: _cache_entry(path, 0.11)})
    save_quality_cache(str(group), cache)
    candidate = screen_candidates(inventory_candidates(group), limit=1)[0]
    config = ScoutConfig(
        group_dir=str(group),
        output_dir=str(out),
        limit=1,
        exclude_manual=True,
        schedules=(DENSE_256,),
    )
    calls = {'refine': 0}

    def refine(src: str, dest: str, schedule: RefineSchedule) -> None:
        calls['refine'] += 1
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    with ScoutStore(out / 'scout.sqlite') as store:
        first = trial_pair_schedule(
            candidate,
            DENSE_256,
            config=config,
            store=store,
            code_sha='abc',
            refine=refine,
            score_pair=lambda _path: 0.11 if calls['refine'] == 0 else 0.20,
            score_cells=lambda _path: _fake_cells(0.20 if 'dense256' in _path else 0.10),
        )
        second = trial_pair_schedule(
            candidate,
            DENSE_256,
            config=config,
            store=store,
            code_sha='abc',
            refine=refine,
            score_pair=lambda _path: 0.99,
            score_cells=lambda _path: _fake_cells(0.99),
        )
    assert calls['refine'] == 1
    assert second.pair_zncc_out == first.pair_zncc_out


def test_failed_trial_is_checkpointed_and_does_not_abort(tmp_path: Path) -> None:
    """A refine exception is recorded and the batch continues."""
    group = tmp_path / 'Grid16'
    out = tmp_path / 'scout'
    group.mkdir()
    low = _write_stos(group, '1-2_ctrl-TEM_Leveled_map-TEM_Leveled.stos', seed=1)
    high = _write_stos(group, '3-4_ctrl-TEM_Leveled_map-TEM_Leveled.stos', seed=2)
    cache = QualityCache(entries={
        low.name: _cache_entry(low, 0.05),
        high.name: _cache_entry(high, 0.06),
    })
    save_quality_cache(str(group), cache)
    config = ScoutConfig(
        group_dir=str(group),
        output_dir=str(out),
        limit=2,
        exclude_manual=True,
        schedules=(FINE_128,),
        command=['nornir-stos-group-scout', '--limit', '2'],
    )

    def refine(src: str, dest: str, schedule: RefineSchedule) -> None:
        if '1-2' in src:
            raise RuntimeError('boom')
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    results, ranked, reports = run_scout(
        config,
        refine=refine,
        score_pair=lambda path: 0.30 if Path(path).parent.name == 'fine128' else 0.10,
        score_cells=lambda path: _fake_cells(0.40 if Path(path).parent.name == 'fine128' else 0.10),
    )
    assert len(results) == 2
    failed = next(row for row in results if row.pair == '1-2')
    ok = next(row for row in results if row.pair == '3-4')
    assert failed.error is not None
    assert failed.accepted is False
    assert ok.error is None
    assert reports['priority_csv'].is_file()
    assert len(ranked) == 2


def test_build_schedule_metrics_rejects_pair_regression() -> None:
    """A material coarse pair-ZNCC drop is not accepted."""
    baseline = _fake_cells(0.20)
    output = _fake_cells(0.25)
    metrics = build_schedule_metrics(
        pair='1-2',
        schedule=DENSE_256,
        pair_zncc_in=0.40,
        pair_zncc_out=0.30,
        baseline_cells=baseline,
        output_cells=output,
        quality_flag=False,
        wall_s=1.0,
        output_stos=None,
        error=None,
        pair_regression_tolerance=0.02,
    )
    assert metrics.accepted is False
    assert metrics.reject_reason == 'pair_zncc_regression'


@given(
    st.lists(
        st.tuples(
            st.text(alphabet='abc', min_size=1, max_size=3),
            st.sampled_from(['dense256', 'fine128']),
            st.booleans(),
            st.integers(min_value=0, max_value=8),
            st.integers(min_value=-4, max_value=8),
            st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=8,
    )
)
@settings(max_examples=40, deadline=None)
def test_ranking_is_deterministic(rows: list[tuple]) -> None:
    """The lexicographic ranking key is a total order independent of input order."""
    metrics = [
        _metrics(
            f'{index}-{pair}',
            schedule,
            accepted=accepted,
            repaired=repaired,
            net=net,
            p05=p05,
            median=median,
            pair_delta=pair_delta,
        )
        for index, (pair, schedule, accepted, repaired, net, p05, median, pair_delta)
        in enumerate(rows)
    ]
    ranked = rank_pairs(metrics)
    keys = [ranking_key(item) for item in ranked]
    assert keys == sorted(keys)
    by_pair = {item.pair for item in ranked}
    assert len(by_pair) == len(ranked)
