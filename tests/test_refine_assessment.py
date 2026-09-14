"""Unit tests for refine assessment catalog, verdicts, and strain helpers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from nornir_imageregistration.refine_assessment.catalog import Catalog, RunsDb
from nornir_imageregistration.refine_assessment.importer import import_refine_fixture
from nornir_imageregistration.refine_assessment.strain import (
    classify_struggle_vs_damage,
    is_localized_high_strain,
    suggest_tags_from_strain,
)
from nornir_imageregistration.refine_assessment.tags import TagStatus
from nornir_imageregistration.refine_assessment.verdict import (
    MetricDirection,
    ScoreVector,
    Verdict,
    compare_to_best,
)
from nornir_imageregistration.transforms.factory import CreateRigidTransform
from nornir_imageregistration.files.stosfile import StosFile
import nornir_imageregistration


def test_catalog_round_trip_and_tags(tmp_path: Path) -> None:
    """Catalog stores fixtures, tags, certified cells, and adopted bests."""
    catalog = Catalog.open(tmp_path / 'catalog.sqlite')
    try:
        fid = catalog.upsert_fixture(
            'RC2/Grid16/240-241',
            volume='RC2',
            group_name='Grid16',
            pair='240-241',
            gold_kind='manual',
            candidate_source='manual',
        )
        catalog.set_certified_cells(fid, [(1, 2, 10.0, 20.0), (1, 3, 11.0, 21.0)])
        catalog.tag_fixture(fid, 'tear', status=TagStatus.CONFIRMED)
        catalog.tag_fixture(fid, 'high-relative-distortion', status=TagStatus.SUGGESTED)
        catalog.adopt_scores(
            fid,
            {'pair_zncc': 0.71, 'lock_frac': 0.33},
            directions={
                'pair_zncc': MetricDirection.MAXIMIZE,
                'lock_frac': MetricDirection.BAND,
            },
            bands={'lock_frac': (0.29, 0.40)},
            note='unit test adopt',
        )
        best = catalog.get_best_scores(fid)
        assert best['pair_zncc']['value'] == pytest.approx(0.71)
        tags = catalog.fixture_tags(fid)
        assert {t['slug'] for t in tags} >= {'tear', 'high-relative-distortion'}
        confirmed = catalog.fixture_tags(fid, confirmed_only=True)
        assert [t['slug'] for t in confirmed] == ['tear']
        out = catalog.export_best_json(fid, tmp_path / 'best.json')
        assert out.is_file()
        listed = catalog.list_fixtures(tag='tear', confirmed_only=True)
        assert len(listed) == 1
    finally:
        catalog.close()


def test_compare_to_best_broken_and_helped() -> None:
    """Maximize regression is broken; improvement is helped."""
    best = {
        'pair_zncc': {'value': 0.70, 'direction': 'maximize', 'band_lo': None, 'band_hi': None},
    }
    broken = compare_to_best(ScoreVector(values={'pair_zncc': 0.60}), best)
    assert broken == Verdict.BROKEN
    helped = compare_to_best(ScoreVector(values={'pair_zncc': 0.80}), best)
    assert helped == Verdict.HELPED


def test_localized_strain_and_struggle_classifier() -> None:
    """Island residuals are localized; Manual≫Auto with unique tissue is struggle."""
    residual = np.ones(100, dtype=np.float64)
    residual[30:55] = 40.0
    assert is_localized_high_strain(residual)
    soup = np.linspace(10, 50, 100)
    assert not is_localized_high_strain(soup)
    assert classify_struggle_vs_damage(
        manual_zncc=0.8, automatic_zncc=0.4, unique_frac=0.1, localized=True) == 'struggle'
    assert classify_struggle_vs_damage(
        manual_zncc=0.1, automatic_zncc=0.1, unique_frac=0.0, localized=False) == 'damage'


def test_runs_db_insert(tmp_path: Path) -> None:
    """Runs DB records scores without touching the catalog."""
    runs = RunsDb.open(tmp_path / 'runs.sqlite')
    try:
        run_id = runs.insert_run(
            'RC2/Grid16/x',
            ScoreVector(values={'pair_zncc': 0.5}, quality_flag=True, unique_frac_series=[0.2, 0.1]),
            flags='baseline',
            wall_s=1.5,
            verdict='unchanged',
        )
        assert runs.get_run_scores(run_id)['pair_zncc'] == pytest.approx(0.5)
        assert runs.get_run(run_id)['quality_flag'] == 1
    finally:
        runs.close()


def _write_tiny_stos(tmp_path: Path) -> Path:
    """Write a tiny identity .stos with patterned images."""
    images = tmp_path / 'images'
    images.mkdir(parents=True, exist_ok=True)
    ctrl = images / 'c.png'
    mapped = images / 'm.png'
    rng = np.random.default_rng(0)
    img = rng.integers(32, 200, size=(64, 64), dtype=np.uint8)
    nornir_imageregistration.SaveImage(str(ctrl), img)
    nornir_imageregistration.SaveImage(str(mapped), img)
    transform = CreateRigidTransform(
        (0, 0), 0.0,
        np.asarray((64, 64), dtype=np.int64),
        np.asarray((64, 64), dtype=np.int64),
    )
    stos_path = tmp_path / 'pair.stos'
    stos = StosFile()
    stos.ControlImageFullPath = str(ctrl)
    stos.MappedImageFullPath = str(mapped)
    stos.ControlImageDim = [1.0, 1.0, 64, 64]
    stos.MappedImageDim = [1.0, 1.0, 64, 64]
    stos.Downsample = 16
    stos.Transform = transform
    stos.Save(str(stos_path))
    return stos_path


def test_import_refine_fixture_writes_manifest(tmp_path: Path) -> None:
    """Importer copies images, writes input.stos + manifest, registers catalog."""
    stos_path = _write_tiny_stos(tmp_path / 'src')
    out_root = tmp_path / 'refine_fixtures'
    result = import_refine_fixture(
        stos_path,
        out_root,
        volume='Synthetic',
        group_name='Grid16',
        pair='1-2',
        bbox_yxhw=(8, 8, 32, 32),
        confirmed_tags=['healthy'],
    )
    assert result.fixture_dir.is_dir()
    assert (result.fixture_dir / 'input.stos').is_file()
    assert (result.fixture_dir / 'control.png').is_file()
    assert (result.fixture_dir / 'manifest.json').is_file()
    catalog = Catalog.open(out_root / 'catalog.sqlite')
    try:
        assert catalog.fixture_id_for_dir(result.relative_dir) == result.fixture_id
        tags = catalog.fixture_tags(result.fixture_id, confirmed_only=True)
        assert [t['slug'] for t in tags] == ['healthy']
    finally:
        catalog.close()


@pytest.mark.skipif(
    not os.environ.get('TESTINPUTPATH'),
    reason='mini-stos refine needs TESTINPUTPATH when corpus fixtures are present',
)
def test_refine_fixture_corpus_optional() -> None:
    """Skip-gated: when refine_fixtures exist under TESTINPUTPATH, list at least one."""
    root = Path(os.environ['TESTINPUTPATH']) / 'refine_fixtures'
    if not root.is_dir():
        pytest.skip('no refine_fixtures directory yet')
    stos_files = list(root.rglob('input.stos'))
    if not stos_files:
        pytest.skip('no imported refine fixtures yet')
    assert stos_files[0].is_file()
