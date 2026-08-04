"""Tests for STOS pair quality scoring and folder-local cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nornir_imageregistration.core as core
from nornir_imageregistration.files.stosfile import StosFile
from nornir_imageregistration.refine_shared.cell_roles import Role
from nornir_imageregistration.stos_quality import (
    QualityCache,
    attach_scores_to_paths,
    build_quality_histogram,
    cache_key_for_stos,
    compute_pair_zncc,
    entry_is_stale,
    load_quality_cache,
    merge_entry,
    refine_summary_from_diagnostics,
    save_quality_cache,
    score_stos_into_cache,
)
from nornir_imageregistration.transforms import factory


def _identity_transform(shape: tuple[int, int] = (32, 32)):
    """Return a rigid identity transform for square test images."""
    dims = np.asarray(shape, dtype=np.float64)
    return factory.CreateRigidTransform((0, 0), 0.0, dims, dims)


def _write_pattern_png(path: Path, *, seed: int = 0, shift: int = 0) -> None:
    """Write a small patterned image with non-zero variance for ZNCC."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    image = rng.integers(32, 224, size=(32, 32), dtype=np.uint8)
    if shift:
        image = np.roll(image, shift=shift, axis=1)
    core.SaveImage(str(path), image)


def _write_identity_stos(tmp_path: Path, *, name: str = 'pair.stos') -> Path:
    images = tmp_path / 'images'
    ctrl = images / 'ctrl.png'
    mapped = images / 'map.png'
    _write_pattern_png(ctrl, seed=1)
    # Identical content → near-perfect ZNCC under identity transform.
    _write_pattern_png(mapped, seed=1)
    group = tmp_path / 'StosGroup'
    group.mkdir(parents=True, exist_ok=True)
    stos_path = group / name
    stos = StosFile()
    stos.ControlImageFullPath = str(ctrl)
    stos.MappedImageFullPath = str(mapped)
    stos.ControlImageDim = [1.0, 1.0, 32, 32]
    stos.MappedImageDim = [1.0, 1.0, 32, 32]
    stos.Downsample = 1
    stos.Transform = _identity_transform()
    stos.Save(str(stos_path))
    return stos_path


def test_compute_pair_zncc_identity_near_one(tmp_path: Path) -> None:
    """Identity warp of matching images yields a high pair ZNCC."""
    stos_path = _write_identity_stos(tmp_path)
    result = compute_pair_zncc(str(stos_path), max_side=64)
    assert result.pair_zncc > 0.95
    assert result.stos_checksum
    assert result.stos_mtime_ns > 0


def test_quality_cache_round_trip_and_stale(tmp_path: Path) -> None:
    """Cache save/load preserves entries; checksum mismatch marks stale."""
    stos_path = _write_identity_stos(tmp_path)
    group = stos_path.parent
    cache, entry, computed = score_stos_into_cache(str(group), str(stos_path), max_side=64)
    assert computed is True
    assert entry['pair_zncc'] > 0.95
    save_quality_cache(str(group), cache)

    reloaded = load_quality_cache(str(group))
    key = cache_key_for_stos(str(group), str(stos_path))
    assert key in reloaded.entries
    assert not entry_is_stale(reloaded.entries[key], str(stos_path))

    # Corrupt checksum → stale.
    reloaded.entries[key]['stos_checksum'] = 'deadbeef'
    assert entry_is_stale(reloaded.entries[key], str(stos_path))


def test_entry_is_stale_missing_pair_zncc(tmp_path: Path) -> None:
    """Missing pair_zncc forces a recompute."""
    stos_path = _write_identity_stos(tmp_path)
    assert entry_is_stale({'stos_checksum': 'x', 'stos_mtime_ns': 1}, str(stos_path))


def test_build_quality_histogram_bins_and_median() -> None:
    """Histogram.Init/Add bins scores and reports a sensible median."""
    scores = [-0.5, 0.0, 0.25, 0.5, 0.75]
    hist = build_quality_histogram(scores, num_bins=20)
    assert hist.NumSamples == len(scores)
    assert sum(hist.Bins) == len(scores)
    median = hist.Median()
    assert 0.0 <= float(median) <= 0.5


def test_attach_scores_and_manual_key(tmp_path: Path) -> None:
    """Manual/ relative keys attach independently of Automatic."""
    auto = _write_identity_stos(tmp_path, name='pair.stos')
    group = auto.parent
    manual_dir = group / 'Manual'
    manual_dir.mkdir()
    manual = manual_dir / 'pair.stos'
    # Shifted mapped content → lower score than identity auto.
    images = tmp_path / 'images_manual'
    ctrl = images / 'ctrl.png'
    mapped = images / 'map.png'
    _write_pattern_png(ctrl, seed=2)
    _write_pattern_png(mapped, seed=2, shift=4)
    stos = StosFile()
    stos.ControlImageFullPath = str(ctrl)
    stos.MappedImageFullPath = str(mapped)
    stos.ControlImageDim = [1.0, 1.0, 32, 32]
    stos.MappedImageDim = [1.0, 1.0, 32, 32]
    stos.Downsample = 1
    stos.Transform = _identity_transform()
    stos.Save(str(manual))

    cache = QualityCache()
    cache, _, _ = score_stos_into_cache(str(group), str(auto), cache=cache, max_side=64)
    cache, _, _ = score_stos_into_cache(str(group), str(manual), cache=cache, max_side=64)
    save_quality_cache(str(group), cache)

    attached = attach_scores_to_paths(
        str(group),
        {'auto': str(auto), 'manual': str(manual)},
        cache=load_quality_cache(str(group)),
    )
    assert attached['auto'] is not None and attached['auto'] > 0.95
    assert attached['manual'] is not None
    assert 'Manual/pair.stos' in cache.entries


def test_refine_summary_from_diagnostics(tmp_path: Path) -> None:
    """NPZ refine diagnostics yield median lock ZNCC and lock fraction."""
    npz = tmp_path / 'refine_pass04_diagnostics.npz'
    zncc = np.asarray([0.1, 0.8, 0.9, np.nan], dtype=np.float64)
    locked = np.asarray([False, True, True, False], dtype=bool)
    role = np.asarray(
        [int(Role.REJECT), int(Role.LOCKABLE), int(Role.LOCKABLE), int(Role.FREE)],
        dtype=np.int64,
    )
    np.savez_compressed(npz, zncc=zncc, locked=locked, role=role)
    summary = refine_summary_from_diagnostics(str(npz), pass_index=4)
    assert summary is not None
    assert summary['pass'] == 4
    assert summary['lock_frac'] == pytest.approx(0.5)
    assert summary['median_lock_zncc'] == pytest.approx(0.85)


def test_merge_refine_without_forcing_pair(tmp_path: Path) -> None:
    """Refine merge updates refine block without clearing pair_zncc."""
    cache = QualityCache()
    from nornir_imageregistration.stos_quality import PairZnccResult

    pair = PairZnccResult(
        pair_zncc=0.7,
        downsample=16.0,
        stos_checksum='abc',
        stos_mtime_ns=1,
        max_side=2048,
    )
    merge_entry(cache, 'a.stos', pair=pair)
    merge_entry(cache, 'a.stos', refine={'median_lock_zncc': 0.6, 'lock_frac': 0.5, 'pass': 3})
    entry = cache.entries['a.stos']
    assert entry['pair_zncc'] == 0.7
    assert entry['refine']['pass'] == 3
