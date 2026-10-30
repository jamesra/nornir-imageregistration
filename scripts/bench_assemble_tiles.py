#!/usr/bin/env python3
"""Benchmark optimized fixed-size tile assembly (GenerateOptimizedTiles).

Compares NumPy (TilesToImageParallel) vs CuPy (TilesToImage) backends for the
active AssembleTilesetNumpy / GenerateOptimizedTiles production path.

Usage
-----
  python bench_assemble_tiles.py \\
      --volume-root \"$TESTINPUTPATH/Volumes/RPC3\" --section 601 \\
      --tile-size 512 512 --backends both --iterations 3 \\
      --output \"$TESTOUTPUTPATH/assemble_bench/rpc3_601/profiles/baseline.json\"

  python bench_assemble_tiles.py ... --profile --include-save
  python bench_assemble_tiles.py ... --save-golden
  python bench_assemble_tiles.py ... --verify

Production I/O (two-stage save: encode pool + copy_workers=2 default, mirrors AssembleTilesetNumpy)::

  export NORNIR_HEADLESS=1
  pip install -e /workspace/nornir-shared

  # Defaults: NORNIR_TILE_IO_WORKERS=16, NORNIR_TILE_COPY_WORKERS=2 (two-stage always on).
  # Monolithic encode+copy A/B uses _SaveImageAndCopy via io-sweep helpers, not production assemble.

  # One production-width strip: assemble once, sweep save workers (section 815, CIFS)
  python bench_assemble_tiles.py \\
      --section 815 --volume-root /storage4/RPC3 \\
      --production-save --one-strip --io-sweep-only \\
      --cifs-dest /storage4/RPC3/_assemble_io_bench/section_0815/tiles \\
      --workers 1,2,4,8,16,32 --backends gpu

  # Local devcontainer smoke (section 601, dest under TESTOUTPUTPATH)
  python bench_assemble_tiles.py \\
      --section 601 --volume-root \"$TESTINPUTPATH/Volumes/RPC3\" \\
      --production-save --one-strip --io-sweep-only \\
      --cifs-dest \"$TESTOUTPUTPATH/assemble_bench/rpc3_601/production_save/tiles\" \\
      --workers 1,2,4,8,16 --backends gpu

  # Section 815 on CIFS (/storage4/RPC3) — required for production I/O benchmarks
  python bench_assemble_tiles.py \\
      --section 815 --volume-root /storage4/RPC3 \\
      --production-save --one-strip --io-sweep-only \\
      --cifs-dest /storage4/RPC3/_assemble_io_bench/section_0815/tiles \\
      --workers 1,2,4,8,16,32 --backends gpu

  # Full section e2e with worker sweep
  python bench_assemble_tiles.py \\
      --section 815 --volume-root /storage4/RPC3 \\
      --production-save --workers 8,16,32 --backends gpu

Fixture layout (RPC3 section 601)::
  $TESTINPUTPATH/Volumes/RPC3/VolumeData.xml
  $TESTINPUTPATH/Volumes/RPC3/TEM/0601/TEM/Leveled/TilePyramid/001/
  $TESTINPUTPATH/Volumes/RPC3/TEM/0601/TEM/Grid_*.mosaic  (Name=\"Grid\")

Outputs under ``--output-root`` (default ``$TESTOUTPUTPATH/assemble_bench/rpc3_601``).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import cProfile
import hashlib
import json
import math
import multiprocessing
import os
import pstats
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import partial
from io import StringIO
from pathlib import Path
from typing import Any

# Force 'fork' so multiprocess NumPy workers inherit the parent's computation lib.
if multiprocessing.get_start_method(allow_none=True) != 'fork':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILDMANAGER_ROOT = _REPO_ROOT.parent / 'nornir-buildmanager'
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if _BUILDMANAGER_ROOT.is_dir() and str(_BUILDMANAGER_ROOT) not in sys.path:
    sys.path.insert(0, str(_BUILDMANAGER_ROOT))

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.computational_lib import ComputationLib

PROFILE_NEEDLES: tuple[str, ...] = (
    'TransformTile',
    'SourceImageToTargetSpace',
    '_TransformImageUsingCoords',
    'map_coordinates',
    'CompositeImageWithZBuffer',
    'CreateDistanceImage',
    'ImageToTilesGenerator',
    'TilesToImage',
    'TilesToImageParallel',
    'SaveImage',
)


@dataclass
class FixturePaths:
    volume_root: Path
    section_number: int
    section_dir: Path
    mosaic_path: Path
    tile_dir: Path
    channel: str = 'TEM'
    filter_name: str = 'Leveled'
    transform_name: str = 'Grid'


@dataclass
class StageTimes:
    load_mosaic: float = 0.0
    assemble_warp: float = 0.0
    save_png: float = 0.0
    total: float = 0.0
    tiles_generated: int = 0
    grid_dims: tuple[int, int] = (0, 0)
    source_tile_count: int = 0
    bbox_shape: tuple[int, int] = (0, 0)


@dataclass
class BackendResult:
    backend: str
    times: list[StageTimes] = field(default_factory=list)
    profile_path: str | None = None


@dataclass
class StripLayout:
    """Production strip geometry derived from mosaic_tileset column math."""

    full_grid_cols: int
    full_grid_rows: int
    full_grid_cells: int
    strip_cols: int
    num_strips: int
    strip_width_px: int
    strip_height_px: int
    strip_area_px: int
    max_temp_image_area_px: int

    @property
    def one_strip_grid_cells(self) -> int:
        return self.strip_cols * self.full_grid_rows


@dataclass
class WorkerSweepRow:
    workers: int
    max_active: int
    tiles: int
    wall_s: float | None
    error: str | None = None
    output_io_wait_s: float | None = None
    output_io_wait_backpressure_s: float | None = None
    output_io_wait_drain_s: float | None = None
    sum_encode_thread_s: float | None = None
    sum_copy_thread_s: float | None = None
    level001_wall_s: float | None = None
    last_strip_yield_to_drain_s: float | None = None
    encode_workers: int | None = None
    copy_workers: int | None = None
    max_active_encode: int | None = None
    tasks_at_drain: int | None = None
    two_stage: bool | None = None


@dataclass
class FullSectionProductionResult:
    """Pipelined full-section assemble + save metrics (mirrors AssembleTilesetNumpy)."""

    tiles_saved: int
    wall_s: float
    output_io_wait_backpressure_s: float
    output_io_wait_drain_s: float
    save_encode_thread_s: float
    save_copy_thread_s: float
    last_strip_yield_to_drain_s: float | None
    encode_workers: int
    copy_workers: int
    max_active_encode: int
    tasks_at_drain: int
    two_stage: bool

    @property
    def output_io_wait_s(self) -> float:
        return self.output_io_wait_backpressure_s + self.output_io_wait_drain_s

    def as_timeline_dict(self) -> dict[str, float | int | bool | None]:
        return {
            'level001_wall_s': self.wall_s,
            'output_io_wait_s': self.output_io_wait_s,
            'output_io_wait_backpressure_s': self.output_io_wait_backpressure_s,
            'output_io_wait_drain_s': self.output_io_wait_drain_s,
            'save_encode_thread_s': self.save_encode_thread_s,
            'save_copy_thread_s': self.save_copy_thread_s,
            'tiles_saved': self.tiles_saved,
            'encode_workers': self.encode_workers,
            'copy_workers': self.copy_workers,
            'max_active_encode': self.max_active_encode,
            'last_strip_yield_to_drain_s': self.last_strip_yield_to_drain_s,
            'tasks_at_drain': self.tasks_at_drain,
            'two_stage': self.two_stage,
        }


@dataclass
class ProductionSaveTimings:
    """Wall and per-thread encode/copy totals from a production-save sweep."""

    wall_s: float
    sum_encode_thread_s: float = 0.0
    sum_copy_thread_s: float = 0.0


TileRecord = tuple[int, int, Any]


def _default_volume_root() -> Path:
    test_input = os.environ.get('TESTINPUTPATH', '/nornir-testdata')
    return Path(test_input) / 'Volumes' / 'RPC3'


def _default_output_root() -> Path:
    test_output = os.environ.get('TESTOUTPUTPATH', '/tmp/nornir-test-output')
    return Path(test_output) / 'assemble_bench' / 'rpc3_601'


def _estimate_max_temp_image_area() -> int:
    """Mirror buildmanager EstimateMaxTempImageArea when available."""
    try:
        from nornir_buildmanager.operations.tile import EstimateMaxTempImageArea
        return int(EstimateMaxTempImageArea())
    except Exception:
        try:
            import psutil
            memory_data = psutil.virtual_memory()
            bytes_per_pixel = 2
            num_images_per_tile = 2
            num_duplicate_copies_in_memory = 3
            safety_factor = 2
            return int(memory_data.available / (
                bytes_per_pixel * num_images_per_tile * num_duplicate_copies_in_memory * safety_factor))
        except Exception:
            return 1 << 30


def _find_section_channel_dir(volume_root: Path, section: int, channel: str) -> Path:
    """Locate TEM/.../TEM channel directory for the section."""
    padded = f'{section:04d}'
    candidates = [
        volume_root / 'TEM' / padded / channel,
        volume_root / 'TEM' / str(section) / channel,
        volume_root / 'TEM' / padded,
        volume_root / 'TEM' / str(section),
    ]
    for path in candidates:
        if path.is_dir() and (path / 'VolumeData.xml').is_file():
            return path
        if path.is_dir() and any(path.glob('*.mosaic')):
            return path
    raise FileNotFoundError(
        f'Could not find section {section} channel={channel} under {volume_root}. '
        f'Tried: {", ".join(str(p) for p in candidates)}')


def _parse_grid_mosaic_from_volume_xml(channel_dir: Path, transform_name: str) -> Path:
    """Pick the newest Transform named ``transform_name`` from channel VolumeData.xml."""
    xml_path = channel_dir / 'VolumeData.xml'
    if not xml_path.is_file():
        # Fall back to glob for Grid_*.mosaic
        mosaics = sorted(channel_dir.glob('Grid_*.mosaic'), key=lambda p: p.stat().st_mtime)
        if not mosaics:
            raise FileNotFoundError(f'No VolumeData.xml or Grid_*.mosaic in {channel_dir}')
        return mosaics[-1]

    tree = ET.parse(xml_path)
    root = tree.getroot()
    matches: list[tuple[str, str]] = []  # (creation_date, path)
    for elem in root.iter('Transform'):
        if elem.attrib.get('Name') != transform_name:
            continue
        path = elem.attrib.get('Path')
        if not path:
            continue
        matches.append((elem.attrib.get('CreationDate', ''), path))

    if not matches:
        mosaics = sorted(channel_dir.glob(f'{transform_name}*.mosaic'),
                         key=lambda p: p.stat().st_mtime)
        if mosaics:
            return mosaics[-1]
        raise FileNotFoundError(
            f'No Transform Name="{transform_name}" in {xml_path}')

    # Prefer newest CreationDate string (ISO-like sorts lexicographically for our dates)
    matches.sort(key=lambda t: t[0])
    mosaic_rel = matches[-1][1]
    mosaic_path = channel_dir / mosaic_rel
    if not mosaic_path.is_file():
        raise FileNotFoundError(f'Transform path missing: {mosaic_path}')
    return mosaic_path


def discover_fixture(volume_root: Path, section: int, channel: str = 'TEM',
                     filter_name: str = 'Leveled', transform_name: str = 'Grid',
                     mosaic: Path | None = None, tile_dir: Path | None = None) -> FixturePaths:
    """Resolve mosaic and TilePyramid/001 paths from the volume tree."""
    if not volume_root.is_dir():
        raise FileNotFoundError(f'Volume root not found: {volume_root}')
    volume_xml = volume_root / 'VolumeData.xml'
    if not volume_xml.is_file():
        raise FileNotFoundError(
            f'Expected VolumeData.xml at {volume_xml} (check TESTINPUTPATH mount)')

    channel_dir = _find_section_channel_dir(volume_root, section, channel)
    mosaic_path = Path(mosaic) if mosaic else _parse_grid_mosaic_from_volume_xml(
        channel_dir, transform_name)
    if tile_dir is None:
        tile_dir_path = channel_dir / filter_name / 'TilePyramid' / '001'
    else:
        tile_dir_path = Path(tile_dir)
    if not tile_dir_path.is_dir():
        raise FileNotFoundError(f'Tile directory not found: {tile_dir_path}')

    section_dir = channel_dir.parent if channel_dir.name == channel else channel_dir
    return FixturePaths(
        volume_root=volume_root,
        section_number=section,
        section_dir=section_dir,
        mosaic_path=mosaic_path,
        tile_dir=tile_dir_path,
        channel=channel,
        filter_name=filter_name,
        transform_name=transform_name,
    )


def _to_numpy_image(tile_image) -> np.ndarray:
    """Host ndarray for checksum/save (handles CuPy without implicit conversion)."""
    if hasattr(tile_image, 'get'):
        tile_image = tile_image.get()
    return np.ascontiguousarray(np.asarray(tile_image))


def _tile_checksum(tile_image) -> str:
    arr = _to_numpy_image(tile_image)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _save_tile_png(path: Path, tile_image, bpp: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nornir_imageregistration.SaveImage(
        str(path), _to_numpy_image(tile_image), bpp=bpp, optimize=True)


def _process_tile_stream(
    tile_iter,
    stages: StageTimes,
    include_save: bool,
    tileset_out: Path | None,
    collect_checksums: bool,
    checksums: dict[str, str],
) -> None:
    """Consume (row, col, image) yields: checksum / optional PNG save."""
    save_pool: concurrent.futures.ThreadPoolExecutor | None = None
    save_futures: list[concurrent.futures.Future] = []
    if include_save and tileset_out is not None:
        tileset_out.mkdir(parents=True, exist_ok=True)
        save_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(2, (os.cpu_count() or 4) * 2))

    t_save = 0.0
    tiles_generated = 0
    t0 = time.perf_counter()
    try:
        for i_row, i_col, tile_image in tile_iter:
            tiles_generated += 1
            key = f'Y{i_row:03d}_X{i_col:03d}'
            if collect_checksums:
                checksums[key] = _tile_checksum(tile_image)
            if include_save and tileset_out is not None and save_pool is not None:
                out_path = tileset_out / f'Leveled_X{i_col:03d}_Y{i_row:03d}.png'
                tile_host = _to_numpy_image(tile_image)
                t_save0 = time.perf_counter()
                save_futures.append(
                    save_pool.submit(_save_tile_png, out_path, tile_host))
                t_save += time.perf_counter() - t_save0
    finally:
        stages.assemble_warp = time.perf_counter() - t0
        if save_pool is not None:
            t_join0 = time.perf_counter()
            for fut in concurrent.futures.as_completed(save_futures):
                fut.result()
            t_save += time.perf_counter() - t_join0
            save_pool.shutdown(wait=True)
        stages.save_png = t_save
        stages.tiles_generated = tiles_generated


def run_once(
    fixture: FixturePaths,
    tile_size: tuple[int, int],
    max_temp_image_area: int | None,
    include_save: bool,
    tileset_out: Path | None,
    max_columns: int | None,
    collect_checksums: bool,
) -> tuple[StageTimes, dict[str, str]]:
    """Run one optimized-tile assembly pass; return timings and optional checksums.

    Full-grid path uses ``GenerateOptimizedTiles`` (production). When
    ``max_columns`` is set, only the left N columns are assembled via
    ``AssembleImage`` + ``ImageToTilesGenerator`` so truncated benches do less
    work (GenerateOptimizedTiles otherwise submits every column up front).
    """
    stages = StageTimes()
    checksums: dict[str, str] = {}
    t_total0 = time.perf_counter()

    t0 = time.perf_counter()
    mosaic = nornir_imageregistration.Mosaic.LoadFromMosaicFile(str(fixture.mosaic_path))
    image_to_source_space_scale = 1.0
    mosaic_tileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(
        mosaic,
        image_folder=str(fixture.tile_dir),
        image_to_source_space_scale=image_to_source_space_scale,
    )
    stages.source_tile_count = len(mosaic_tileset)
    bbox = mosaic_tileset.TargetBoundingBox
    if not np.array_equal(bbox.BottomLeft, np.asarray((0, 0))):
        mosaic_tileset.TranslateToZeroOrigin()
        bbox = mosaic_tileset.TargetBoundingBox
    stages.bbox_shape = (int(bbox.shape[0]), int(bbox.shape[1]))
    stages.load_mosaic = time.perf_counter() - t0

    tile_dims = np.asarray(tile_size, dtype=np.int64)
    target_space_scale = 1.0
    scaled_bbox_shape = np.ceil(bbox.shape * target_space_scale).astype(np.int64)
    expected_grid = nornir_imageregistration.TileGridShape(
        scaled_bbox_shape, tile_size=tile_dims)
    stages.grid_dims = (int(expected_grid[0]), int(expected_grid[1]))

    area = max_temp_image_area
    if area is None:
        area = _estimate_max_temp_image_area()

    if max_columns is None:
        tile_iter = mosaic_tileset.GenerateOptimizedTiles(
            target_space_scale=target_space_scale,
            tile_dims=tile_dims,
            max_temp_image_area=area,
        )
        _process_tile_stream(
            tile_iter, stages, include_save, tileset_out, collect_checksums, checksums)
    else:
        # Left strip of N columns — mirrors one GenerateOptimizedTiles column strip.
        n_cols = min(int(max_columns), int(expected_grid[1]))
        n_rows = int(expected_grid[0])
        scaled_tile_dims = tile_dims / target_space_scale
        working_shape = np.asarray((n_rows, n_cols), dtype=np.float64) * scaled_tile_dims
        fixed_region = nornir_imageregistration.Rectangle.CreateFromPointAndArea(
            (0.0, 0.0), working_shape)

        def _strip_tiles():
            working_image, _mask = mosaic_tileset.AssembleImage(
                FixedRegion=fixed_region, target_space_scale=target_space_scale)
            yield from nornir_imageregistration.ImageToTilesGenerator(
                source_image=working_image,
                tile_size=tile_dims,
                grid_shape=np.asarray((n_rows, n_cols)),
                coord_offset=(0, 0),
            )

        _process_tile_stream(
            _strip_tiles(), stages, include_save, tileset_out, collect_checksums, checksums)
        # Report truncated grid for clarity
        stages.grid_dims = (n_rows, n_cols)

    stages.total = time.perf_counter() - t_total0
    return stages, checksums


def _profile_needles_report(profile_path: Path) -> dict[str, dict[str, float | int]]:
    stream = StringIO()
    stats = pstats.Stats(str(profile_path), stream=stream)
    report: dict[str, dict[str, float | int]] = {}
    for needle in PROFILE_NEEDLES:
        stream.truncate(0)
        stream.seek(0)
        stats.print_stats(needle)
        text = stream.getvalue()
        best: tuple[int, float] | None = None
        for line in text.splitlines():
            if needle not in line or 'function calls' in line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                ncalls = int(parts[0].split('/')[0])
                cumtime = float(parts[3])
            except ValueError:
                continue
            if best is None or cumtime > best[1]:
                best = (ncalls, cumtime)
        if best is not None:
            report[needle] = {'ncalls': best[0], 'cumtime': best[1]}
    return report


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _format_table(results: list[BackendResult]) -> str:
    hdr = (f"{'Backend':<8} {'N':>3} {'tiles':>7} {'load':>8} {'assemble':>10} "
           f"{'save':>8} {'total':>10} {'tiles/s':>8}")
    lines = [hdr, '-' * len(hdr)]
    for result in results:
        if not result.times:
            continue
        measured = result.times[1:] if len(result.times) > 1 else result.times
        totals = [t.total for t in measured]
        assembles = [t.assemble_warp for t in measured]
        loads = [t.load_mosaic for t in measured]
        saves = [t.save_png for t in measured]
        tiles = measured[-1].tiles_generated
        t_mean = _mean(totals)
        lines.append(
            f"{result.backend:<8} {len(measured):>3} {tiles:>7} "
            f"{_mean(loads):>8.2f} {_mean(assembles):>10.2f} {_mean(saves):>8.2f} "
            f"{t_mean:>10.2f} "
            f"{(tiles / t_mean) if t_mean > 0 else 0:>8.1f}"
        )
        if len(totals) > 1:
            lines.append(
                f"{'':8} {'std':>3} {'':>7} {_stdev(loads):>8.2f} {_stdev(assembles):>10.2f} "
                f"{_stdev(saves):>8.2f} {_stdev(totals):>10.2f}")
    return '\n'.join(lines)


def _results_to_json(results: list[BackendResult], fixture: FixturePaths,
                     tile_size: tuple[int, int]) -> dict:
    backends = []
    for result in results:
        measured = result.times[1:] if len(result.times) > 1 else result.times
        sample = measured[-1] if measured else StageTimes()
        backends.append({
            'backend': result.backend,
            'profile_path': result.profile_path,
            'iterations': len(measured),
            'warmup_discarded': len(result.times) > 1,
            'mean_total_s': _mean([t.total for t in measured]),
            'mean_assemble_warp_s': _mean([t.assemble_warp for t in measured]),
            'mean_load_mosaic_s': _mean([t.load_mosaic for t in measured]),
            'mean_save_png_s': _mean([t.save_png for t in measured]),
            'stdev_total_s': _stdev([t.total for t in measured]),
            'tiles_generated': sample.tiles_generated,
            'grid_dims_yx': list(sample.grid_dims),
            'source_tile_count': sample.source_tile_count,
            'bbox_shape_yx': list(sample.bbox_shape),
            'per_iteration': [
                {
                    'load_mosaic': t.load_mosaic,
                    'assemble_warp': t.assemble_warp,
                    'save_png': t.save_png,
                    'total': t.total,
                    'tiles_generated': t.tiles_generated,
                }
                for t in result.times
            ],
        })
    return {
        'pipeline': 'AssembleTiles',
        'fixture': {
            'volume_root': str(fixture.volume_root),
            'section': fixture.section_number,
            'mosaic': str(fixture.mosaic_path),
            'tile_dir': str(fixture.tile_dir),
            'channel': fixture.channel,
            'filter': fixture.filter_name,
            'transform': fixture.transform_name,
        },
        'tile_size': list(tile_size),
        'backends': backends,
    }


def _save_golden(golden_dir: Path, checksums: dict[str, str], meta: dict) -> None:
    golden_dir.mkdir(parents=True, exist_ok=True)
    (golden_dir / 'checksums.json').write_text(
        json.dumps(checksums, indent=2, sort_keys=True), encoding='utf-8')
    (golden_dir / 'meta.json').write_text(
        json.dumps(meta, indent=2), encoding='utf-8')


def _verify_golden(golden_dir: Path, checksums: dict[str, str]) -> tuple[bool, str]:
    path = golden_dir / 'checksums.json'
    if not path.is_file():
        return False, f'No golden checksums at {path} (run --save-golden first)'
    expected = json.loads(path.read_text(encoding='utf-8'))
    if set(expected) != set(checksums):
        missing = sorted(set(expected) - set(checksums))
        extra = sorted(set(checksums) - set(expected))
        return False, f'Tile key mismatch: missing={missing[:5]} extra={extra[:5]}'
    mismatches = [k for k in expected if expected[k] != checksums[k]]
    if mismatches:
        return False, f'{len(mismatches)} tile checksum mismatches (e.g. {mismatches[:5]})'
    return True, f'OK — {len(checksums)} tiles match golden'


def _import_production_save() -> tuple[Any, Any, Any, Any, Any]:
    """Import production save helpers from buildmanager / shared."""
    try:
        from nornir_buildmanager.operations.tile import (
            _SaveImageAndCopy,
            _TwoStageTileSavePipeline,
            _tile_copy_worker_count,
            _use_two_stage_tile_save,
        )
        from nornir_shared.files import ensure_directory
        return (
            _SaveImageAndCopy,
            ensure_directory,
            _TwoStageTileSavePipeline,
            _tile_copy_worker_count,
            _use_two_stage_tile_save,
        )
    except ImportError as exc:
        raise SystemExit(
            'Production save requires nornir-buildmanager and nornir-shared. '
            'Install editable nornir-shared and ensure nornir-buildmanager is on PYTHONPATH.'
        ) from exc


def _save_image_and_copy_timed(
    ImageFullPath: str,
    temp_output_tile_fullpath: str,
    tile_image: Any,
    bpp: int | None,
    optimize: bool = True,
) -> tuple[float, float]:
    """Bench helper: time SaveImage (encode+local write) vs copyfile separately."""
    t_encode = time.perf_counter()
    nornir_imageregistration.SaveImage(
        ImageFullPath=temp_output_tile_fullpath,
        image=tile_image,
        bpp=bpp,
        optimize=optimize,
    )
    encode_s = time.perf_counter() - t_encode
    t_copy = time.perf_counter()
    shutil.copyfile(temp_output_tile_fullpath, ImageFullPath)
    copy_s = time.perf_counter() - t_copy
    return encode_s, copy_s


def _accumulate_save_future(
    finished: concurrent.futures.Future,
    *,
    sum_encode: list[float],
    sum_copy: list[float],
) -> None:
    """Add encode/copy thread-seconds from a timed save future."""
    result = finished.result()
    if isinstance(result, tuple) and len(result) == 2:
        sum_encode[0] += float(result[0])
        sum_copy[0] += float(result[1])


def _parse_workers_list(workers_arg: str | None, *, one_strip: bool) -> list[int]:
    """Parse comma-separated worker counts; apply mode-specific defaults."""
    if workers_arg is not None and workers_arg.strip():
        return [int(x.strip()) for x in workers_arg.split(',') if x.strip()]
    if one_strip:
        return [1, 2, 4, 8, 16, 32]
    return [8, 16, 32]


def _compute_strip_layout(
    mosaic_tileset: Any,
    tile_dims: np.ndarray,
    max_temp_image_area: int,
) -> StripLayout:
    """Mirror strip column math in mosaic_tileset.GenerateOptimizedTiles."""
    bbox = mosaic_tileset.TargetBoundingBox
    if not np.array_equal(bbox.BottomLeft, np.asarray((0, 0))):
        mosaic_tileset.TranslateToZeroOrigin()
    grid_dims = nornir_imageregistration.TileGridShape(
        np.ceil(mosaic_tileset.TargetBoundingBox.shape).astype(np.int64),
        tile_size=tile_dims,
    )
    scaled_shape = grid_dims * tile_dims
    if max_temp_image_area >= int(np.prod(scaled_shape)):
        strip_cols = int(grid_dims[1])
    else:
        strip_cols = max(1, int(np.floor((max_temp_image_area / scaled_shape[0]) / tile_dims[1])))
    num_strips = math.ceil(int(grid_dims[1]) / strip_cols)
    strip_width_px = int(strip_cols * tile_dims[1])
    strip_height_px = int(grid_dims[0] * tile_dims[0])
    return StripLayout(
        full_grid_cols=int(grid_dims[1]),
        full_grid_rows=int(grid_dims[0]),
        full_grid_cells=int(np.prod(grid_dims)),
        strip_cols=strip_cols,
        num_strips=num_strips,
        strip_width_px=strip_width_px,
        strip_height_px=strip_height_px,
        strip_area_px=strip_width_px * strip_height_px,
        max_temp_image_area_px=max_temp_image_area,
    )


def _load_mosaic_tileset(fixture: FixturePaths) -> Any:
    """Load mosaic and build a MosaicTileset for assembly benchmarks."""
    mosaic = nornir_imageregistration.Mosaic.LoadFromMosaicFile(str(fixture.mosaic_path))
    mosaic_tileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(
        mosaic,
        image_folder=str(fixture.tile_dir),
        image_to_source_space_scale=1.0,
    )
    if not np.array_equal(mosaic_tileset.TargetBoundingBox.BottomLeft, np.asarray((0, 0))):
        mosaic_tileset.TranslateToZeroOrigin()
    return mosaic_tileset


def _collect_one_strip_tiles(
    mosaic_tileset: Any,
    tile_dims: np.ndarray,
    strip_cols: int,
    target_space_scale: float = 1.0,
) -> tuple[list[TileRecord], float]:
    """Assemble one production-width strip and slice to tiles (with coverage mask)."""
    grid_dims = nornir_imageregistration.TileGridShape(
        np.ceil(mosaic_tileset.TargetBoundingBox.shape).astype(np.int64),
        tile_size=tile_dims,
    )
    n_cols = min(int(strip_cols), int(grid_dims[1]))
    n_rows = int(grid_dims[0])
    scaled_tile_dims = tile_dims / target_space_scale
    working_shape = np.asarray((n_rows, n_cols), dtype=np.float64) * scaled_tile_dims.astype(np.float64)
    fixed_region = nornir_imageregistration.Rectangle.CreateFromPointAndArea(
        (0.0, 0.0), working_shape)
    t0 = time.perf_counter()
    working_image, mask = mosaic_tileset.AssembleImage(
        FixedRegion=fixed_region, target_space_scale=target_space_scale)
    tiles = list(
        nornir_imageregistration.ImageToTilesGenerator(
            source_image=working_image,
            tile_size=tile_dims,
            grid_shape=np.asarray((n_rows, n_cols)),
            coord_offset=(0, 0),
            coverage_mask=mask,
        )
    )
    return tiles, time.perf_counter() - t0


def _tile_output_name(i_row: int, i_col: int, file_prefix: str, file_postfix: str) -> str:
    return f'{file_prefix}X{i_col:03d}_Y{i_row:03d}{file_postfix}'


def _prepare_output_dirs(
    dest_dir: Path,
    temp_dir: Path,
    ensure_directory: Any,
    *,
    keep_output: bool,
) -> None:
    """Reset or ensure production save directories."""
    if not keep_output:
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    ensure_directory(str(dest_dir))
    ensure_directory(str(temp_dir))


def _production_save_tiles(
    tiles: Sequence[TileRecord],
    workers: int,
    local_temp_dir: Path,
    dest_dir: Path,
    *,
    bpp: int,
    file_prefix: str,
    file_postfix: str,
    keep_output: bool,
    split_encode_copy: bool = True,
) -> ProductionSaveTimings:
    """Save tiles with production two-stage backpressure (encode pool + bounded copy queue)."""
    (
        _save_image_and_copy,
        ensure_directory,
        two_stage_pipeline_cls,
        tile_copy_worker_count,
        _use_two_stage_tile_save,
    ) = _import_production_save()
    assert _use_two_stage_tile_save()
    _prepare_output_dirs(dest_dir, local_temp_dir, ensure_directory, keep_output=keep_output)
    max_active = workers * 2
    t0 = time.perf_counter()

    copy_workers = tile_copy_worker_count()
    pipeline = two_stage_pipeline_cls(
        encode_workers=workers,
        copy_workers=copy_workers,
        bpp=bpp,
        optimize=True,
        collect_thread_timings=split_encode_copy,
    )
    for i_row, i_col, tile_image in tiles:
        tilename = _tile_output_name(i_row, i_col, file_prefix, file_postfix)
        pipeline.submit(
            str(dest_dir / tilename),
            str(local_temp_dir / tilename),
            tile_image,
        )
        while pipeline.active_encode_count >= max_active:
            pipeline.wait_one_encode()
    pipeline.finish()
    return ProductionSaveTimings(
        wall_s=time.perf_counter() - t0,
        sum_encode_thread_s=pipeline.sum_encode_thread_s,
        sum_copy_thread_s=pipeline.sum_copy_thread_s,
    )


def _production_save_tiles_monolithic(
    tiles: Sequence[TileRecord],
    workers: int,
    local_temp_dir: Path,
    dest_dir: Path,
    *,
    bpp: int,
    file_prefix: str,
    file_postfix: str,
    keep_output: bool,
    split_encode_copy: bool = True,
) -> ProductionSaveTimings:
    """Bench-only monolithic save via ``_SaveImageAndCopy`` (not used in production assemble)."""
    (
        save_image_and_copy,
        ensure_directory,
        _two_stage_pipeline_cls,
        _tile_copy_worker_count,
        _use_two_stage_tile_save,
    ) = _import_production_save()
    _prepare_output_dirs(dest_dir, local_temp_dir, ensure_directory, keep_output=keep_output)
    max_active = workers * 2
    t0 = time.perf_counter()

    if split_encode_copy:
        save_fn: Any = partial(_save_image_and_copy_timed, bpp=bpp, optimize=True)
    else:
        save_fn = partial(save_image_and_copy, bpp=bpp, optimize=True)
    active: list[concurrent.futures.Future] = []
    sum_encode = [0.0]
    sum_copy = [0.0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i_row, i_col, tile_image in tiles:
            tilename = _tile_output_name(i_row, i_col, file_prefix, file_postfix)
            task = pool.submit(
                save_fn,
                ImageFullPath=str(dest_dir / tilename),
                temp_output_tile_fullpath=str(local_temp_dir / tilename),
                tile_image=tile_image,
            )
            active.append(task)
            while len(active) >= max_active:
                done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                for finished in done:
                    if split_encode_copy:
                        _accumulate_save_future(
                            finished, sum_encode=sum_encode, sum_copy=sum_copy)
                    else:
                        finished.result()
                    active.remove(finished)
        for finished in concurrent.futures.as_completed(active):
            if split_encode_copy:
                _accumulate_save_future(
                    finished, sum_encode=sum_encode, sum_copy=sum_copy)
            else:
                finished.result()
    return ProductionSaveTimings(
        wall_s=time.perf_counter() - t0,
        sum_encode_thread_s=sum_encode[0],
        sum_copy_thread_s=sum_copy[0],
    )


def _worker_sweep_row_from_full_section(
    workers: int,
    result: FullSectionProductionResult,
) -> WorkerSweepRow:
    """Build a worker sweep row from full-section production metrics."""
    return WorkerSweepRow(
        workers=workers,
        max_active=result.max_active_encode,
        tiles=result.tiles_saved,
        wall_s=result.wall_s,
        output_io_wait_s=result.output_io_wait_s,
        output_io_wait_backpressure_s=result.output_io_wait_backpressure_s,
        output_io_wait_drain_s=result.output_io_wait_drain_s,
        sum_encode_thread_s=result.save_encode_thread_s,
        sum_copy_thread_s=result.save_copy_thread_s,
        level001_wall_s=result.wall_s,
        last_strip_yield_to_drain_s=result.last_strip_yield_to_drain_s,
        encode_workers=result.encode_workers,
        copy_workers=result.copy_workers,
        max_active_encode=result.max_active_encode,
        tasks_at_drain=result.tasks_at_drain,
        two_stage=result.two_stage,
    )


def _run_full_section_production(
    mosaic_tileset: Any,
    tile_dims: np.ndarray,
    max_temp_image_area: int,
    workers: int,
    local_temp_dir: Path,
    dest_dir: Path,
    *,
    bpp: int,
    file_prefix: str,
    file_postfix: str,
    keep_output: bool,
) -> FullSectionProductionResult:
    """Pipelined GenerateOptimizedTiles + production save (mirrors AssembleTilesetNumpy)."""
    (
        _save_image_and_copy,
        ensure_directory,
        two_stage_pipeline_cls,
        tile_copy_worker_count,
        _use_two_stage_tile_save,
    ) = _import_production_save()
    assert _use_two_stage_tile_save()
    _prepare_output_dirs(dest_dir, local_temp_dir, ensure_directory, keep_output=keep_output)
    tiles_saved = 0
    output_io_wait_backpressure_s = 0.0
    output_io_wait_drain_s = 0.0
    last_tile_submit_time: float | None = None
    t_drain: float | None = None
    encode_workers = workers
    max_active = workers * 2
    t0 = time.perf_counter()

    copy_workers = tile_copy_worker_count()
    pipeline = two_stage_pipeline_cls(
        encode_workers=encode_workers,
        copy_workers=copy_workers,
        bpp=bpp,
        optimize=True,
        collect_thread_timings=True,
    )
    for i_row, i_col, tile_image in mosaic_tileset.GenerateOptimizedTiles(
        target_space_scale=1.0,
        tile_dims=tile_dims,
        max_temp_image_area=max_temp_image_area,
    ):
        tilename = _tile_output_name(i_row, i_col, file_prefix, file_postfix)
        pipeline.submit(
            str(dest_dir / tilename),
            str(local_temp_dir / tilename),
            tile_image,
        )
        tiles_saved += 1
        last_tile_submit_time = time.perf_counter()
        while pipeline.active_encode_count >= max_active:
            t_wait = time.perf_counter()
            pipeline.wait_one_encode()
            output_io_wait_backpressure_s += time.perf_counter() - t_wait
    t_drain = time.perf_counter()
    tasks_at_drain = pipeline.finish()
    output_io_wait_drain_s = time.perf_counter() - t_drain

    last_strip_yield_to_drain_s: float | None = None
    if last_tile_submit_time is not None and t_drain is not None:
        last_strip_yield_to_drain_s = max(0.0, t_drain - last_tile_submit_time)

    return FullSectionProductionResult(
        tiles_saved=tiles_saved,
        wall_s=time.perf_counter() - t0,
        output_io_wait_backpressure_s=output_io_wait_backpressure_s,
        output_io_wait_drain_s=output_io_wait_drain_s,
        save_encode_thread_s=pipeline.sum_encode_thread_s,
        save_copy_thread_s=pipeline.sum_copy_thread_s,
        last_strip_yield_to_drain_s=last_strip_yield_to_drain_s,
        encode_workers=encode_workers,
        copy_workers=copy_workers,
        max_active_encode=max_active,
        tasks_at_drain=tasks_at_drain,
        two_stage=True,
    )


def _format_timing_row(label: str, tiles: int, wall_s: float) -> str:
    ms = (wall_s * 1000.0 / tiles) if tiles else 0.0
    tps = (tiles / wall_s) if wall_s > 0 else 0.0
    return f'{label:<28} {tiles:>7} {wall_s:>10.2f} {ms:>9.1f} {tps:>9.1f}'


def _log_save_thread_split(log: Any, timings: ProductionSaveTimings, tiles: int) -> None:
    """Log aggregated encode vs copy thread-seconds for a save sweep."""
    if tiles <= 0:
        return
    enc = timings.sum_encode_thread_s
    cp = timings.sum_copy_thread_s
    log(
        f'  encode thread_s={enc:.2f} ({enc * 1000 / tiles:.1f} ms/tile)  '
        f'copy thread_s={cp:.2f} ({cp * 1000 / tiles:.1f} ms/tile)  '
        f'ratio copy/(enc+copy)={(cp / (enc + cp) if enc + cp > 0 else 0):.0%}'
    )


def _print_strip_context(
    log: Any,
    *,
    title: str,
    fixture: FixturePaths,
    tile_size: tuple[int, int],
    layout: StripLayout,
    local_temp: Path,
    cifs_dest: Path,
    workers: list[int],
    one_strip: bool,
) -> None:
    log(f'=== {title} ===')
    log(f'Section: {fixture.section_number}')
    log(f'Mosaic: {fixture.mosaic_path}')
    log(f'Tile source: {fixture.tile_dir}')
    log(f'Tile size: {tile_size[0]} x {tile_size[1]} px')
    if one_strip:
        log(f'Strip grid: {layout.strip_cols} cols x {layout.full_grid_rows} rows '
            f'({layout.one_strip_grid_cells} grid cells)')
    else:
        log(f'Full grid: {layout.full_grid_cols} cols x {layout.full_grid_rows} rows '
            f'({layout.full_grid_cells} grid cells)')
        log(f'Strip width: {layout.strip_cols} cols -> {layout.num_strips} strip(s)')
    log(f'max_temp_image_area: {layout.max_temp_image_area_px} px '
        f'({layout.max_temp_image_area_px / (1 << 20):.0f} Mpx cap)')
    log(f'Local temp: {local_temp}')
    log(f'Dest: {cifs_dest}')
    log(f'Workers: {workers}')
    log('')


def _print_worker_sweep_table(log: Any, rows: list[WorkerSweepRow], title: str) -> None:
    log(title)
    log(f"{'workers':>8} {'max_active':>11} {'tiles':>7} {'wall_s':>10} {'ms/tile':>9} {'tiles/s':>9}")
    log('-' * 68)
    for row in rows:
        if row.error:
            log(f"{row.workers:>8} {row.max_active:>11} {row.tiles:>7} {'FAILED':>10} {row.error[:35]}")
        elif row.wall_s is not None:
            wall_s = row.wall_s
            log(
                f"{row.workers:>8} {row.max_active:>11} {row.tiles:>7} "
                f"{wall_s:>10.2f} {wall_s * 1000 / row.tiles:>9.1f} {row.tiles / wall_s:>9.1f}"
            )


def _production_save_results_to_json(
    fixture: FixturePaths,
    tile_size: tuple[int, int],
    layout: StripLayout,
    *,
    mode: str,
    local_temp: Path,
    dest_root: Path,
    assemble_once: dict[str, float] | None,
    worker_sweep: list[WorkerSweepRow],
    tiles_saved: int,
) -> dict:
    return {
        'pipeline': 'AssembleTiles',
        'mode': mode,
        'production_save': True,
        'fixture': {
            'volume_root': str(fixture.volume_root),
            'section': fixture.section_number,
            'mosaic': str(fixture.mosaic_path),
            'tile_dir': str(fixture.tile_dir),
        },
        'tile_size': list(tile_size),
        'strip': {
            'grid_cols': layout.strip_cols if mode.startswith('one_strip') else layout.full_grid_cols,
            'grid_rows': layout.full_grid_rows,
            'grid_cells': layout.one_strip_grid_cells if mode.startswith('one_strip') else layout.full_grid_cells,
            'tiles_saved': tiles_saved,
            'strip_width_px': layout.strip_width_px,
            'strip_height_px': layout.strip_height_px,
            'strip_area_px': layout.strip_area_px,
            'max_temp_image_area_px': layout.max_temp_image_area_px,
            'num_strips': layout.num_strips,
            'local_temp_dir': str(local_temp),
            'dest_dir': str(dest_root),
        },
        'assemble_once': assemble_once,
        'worker_sweep': [
            {
                'workers': row.workers,
                'max_active': row.max_active,
                'tiles_saved': row.tiles,
                'wall_s': row.wall_s,
                'ms_per_tile': (row.wall_s * 1000 / row.tiles) if row.wall_s and row.tiles else None,
                'tiles_per_s': (row.tiles / row.wall_s) if row.wall_s and row.wall_s > 0 else None,
                'output_io_wait_s': row.output_io_wait_s,
                'output_io_wait_backpressure_s': row.output_io_wait_backpressure_s,
                'output_io_wait_drain_s': row.output_io_wait_drain_s,
                'level001_wall_s': row.level001_wall_s,
                'last_strip_yield_to_drain_s': row.last_strip_yield_to_drain_s,
                'encode_workers': row.encode_workers,
                'copy_workers': row.copy_workers,
                'max_active_encode': row.max_active_encode,
                'tasks_at_drain': row.tasks_at_drain,
                'two_stage': row.two_stage,
                'sum_encode_thread_s': row.sum_encode_thread_s,
                'sum_copy_thread_s': row.sum_copy_thread_s,
                'error': row.error,
            }
            for row in worker_sweep
        ],
    }


def run_production_save(
    args: argparse.Namespace,
    fixture: FixturePaths,
    tile_size: tuple[int, int],
    max_area: int | None,
    output_root: Path,
    log: Any,
) -> int:
    """Run production-faithful save benchmarks (_SaveImageAndCopy)."""
    (
        _save_image_and_copy,
        _ensure_directory,
        _two_stage_pipeline_cls,
        tile_copy_worker_count,
        _use_two_stage_tile_save,
    ) = _import_production_save()
    assert _use_two_stage_tile_save()
    log(
        f'Tile save mode: two-stage '
        f'(encode from --workers sweep, copy_workers={tile_copy_worker_count()}, '
        f'copy_queue_max={tile_copy_worker_count() * 2})'
    )

    if args.backends not in ('gpu', 'both'):
        log('Production save bench uses GPU backend (set --backends gpu).')
    if nornir_imageregistration.HasCupy():
        nornir_imageregistration.SetActiveComputationLib(ComputationLib.cupy)
    else:
        log('WARNING: CuPy unavailable; using NumPy backend.')

    tile_dims = np.asarray(tile_size, dtype=np.int64)
    max_temp_image_area = max_area if max_area is not None else _estimate_max_temp_image_area()
    workers_list = _parse_workers_list(args.workers, one_strip=bool(args.one_strip))
    bpp = int(args.bpp)
    file_prefix = args.file_prefix
    file_postfix = args.file_postfix

    bench_tag = f"section_{fixture.section_number:04d}"
    if args.one_strip:
        bench_tag = f'one_strip_{bench_tag}'
    elif args.io_sweep_only:
        bench_tag = f'io_sweep_{bench_tag}'
    else:
        bench_tag = f'full_section_{bench_tag}'

    local_temp_root = (
        Path(args.local_temp_dir) if args.local_temp_dir else
        Path(nornir_imageregistration.gettempdir()) / 'nornir' / 'assemble_io_bench' / bench_tag
    )
    dest_root = Path(args.cifs_dest) if args.cifs_dest else output_root / 'production_save' / bench_tag

    mosaic_tileset = _load_mosaic_tileset(fixture)
    layout = _compute_strip_layout(mosaic_tileset, tile_dims, max_temp_image_area)

    mode = 'one_strip_io_sweep' if args.one_strip and args.io_sweep_only else (
        'one_strip_e2e' if args.one_strip else 'full_section_e2e'
    )
    _print_strip_context(
        log,
        title='Production-save bench',
        fixture=fixture,
        tile_size=tile_size,
        layout=layout,
        local_temp=local_temp_root,
        cifs_dest=dest_root,
        workers=workers_list,
        one_strip=bool(args.one_strip),
    )

    assemble_once: dict[str, float] | None = None
    buffered_tiles: list[TileRecord] | None = None
    tiles_saved = 0
    sweep_rows: list[WorkerSweepRow] = []

    if args.one_strip and args.io_sweep_only:
        buffered_tiles, assemble_s = _collect_one_strip_tiles(
            mosaic_tileset, tile_dims, layout.strip_cols)
        tiles_saved = len(buffered_tiles)
        assemble_once = {
            'wall_s': assemble_s,
            'ms_per_tile': assemble_s * 1000 / tiles_saved if tiles_saved else 0.0,
            'tiles_per_s': tiles_saved / assemble_s if assemble_s > 0 else 0.0,
        }
        log(f'Tiles saved: {tiles_saved} (coverage mask; fewer than grid cells)')
        log(f'Strip area: {layout.strip_width_px} x {layout.strip_height_px} px = {layout.strip_area_px} px')
        log('')
        log(f"{'Phase':<28} {'tiles':>7} {'wall_s':>10} {'ms/tile':>9} {'tiles/s':>9}")
        log('-' * 68)
        log(_format_timing_row('assemble+slice (GPU)', tiles_saved, assemble_s))
        log('')

        for workers in workers_list:
            log(f'--- workers={workers} max_active={workers * 2} ---')
            temp_dir = local_temp_root / f'w{workers}'
            dest_dir = dest_root / f'w{workers}'
            try:
                save_timings = _production_save_tiles(
                    buffered_tiles,
                    workers,
                    temp_dir,
                    dest_dir,
                    bpp=bpp,
                    file_prefix=file_prefix,
                    file_postfix=file_postfix,
                    keep_output=args.keep_output,
                )
                sweep_rows.append(WorkerSweepRow(
                    workers, workers * 2, tiles_saved, save_timings.wall_s,
                    sum_encode_thread_s=save_timings.sum_encode_thread_s,
                    sum_copy_thread_s=save_timings.sum_copy_thread_s,
                ))
                log(_format_timing_row('save (local tmp->dest)', tiles_saved, save_timings.wall_s))
                _log_save_thread_split(log, save_timings, tiles_saved)
            except OSError as exc:
                sweep_rows.append(WorkerSweepRow(workers, workers * 2, tiles_saved, None, str(exc)))
                log(f'  FAILED: {exc}')

        log('')
        _print_worker_sweep_table(
            log, sweep_rows, 'Worker sweep summary (encode to local temp + copyfile to dest):')

    elif args.one_strip:
        for workers in workers_list:
            log(f'--- workers={workers} max_active={workers * 2} (one strip e2e) ---')
            mosaic_tileset = _load_mosaic_tileset(fixture)
            temp_dir = local_temp_root / f'w{workers}'
            dest_dir = dest_root / f'w{workers}'
            try:
                t0 = time.perf_counter()
                tiles, _assemble_s = _collect_one_strip_tiles(
                    mosaic_tileset, tile_dims, layout.strip_cols)
                save_timings = _production_save_tiles(
                    tiles,
                    workers,
                    temp_dir,
                    dest_dir,
                    bpp=bpp,
                    file_prefix=file_prefix,
                    file_postfix=file_postfix,
                    keep_output=args.keep_output,
                )
                wall_s = time.perf_counter() - t0
                tiles_saved = len(tiles)
                sweep_rows.append(WorkerSweepRow(
                    workers, workers * 2, tiles_saved, wall_s,
                    sum_encode_thread_s=save_timings.sum_encode_thread_s,
                    sum_copy_thread_s=save_timings.sum_copy_thread_s,
                ))
                log(_format_timing_row('assemble+save', tiles_saved, wall_s))
                log(f'  (assemble ~{_assemble_s:.2f}s, save ~{save_timings.wall_s:.2f}s)')
                _log_save_thread_split(log, save_timings, tiles_saved)
            except OSError as exc:
                sweep_rows.append(WorkerSweepRow(workers, workers * 2, 0, None, str(exc)))
                log(f'  FAILED: {exc}')
        log('')
        _print_worker_sweep_table(log, sweep_rows, 'One-strip e2e summary:')

    else:
        log(f'Tiles saved: (per worker run, full grid)')
        log('')
        for workers in workers_list:
            log(f'--- workers={workers} max_active={workers * 2} (full section e2e) ---')
            mosaic_tileset = _load_mosaic_tileset(fixture)
            temp_dir = local_temp_root / f'w{workers}'
            dest_dir = dest_root / f'w{workers}'
            try:
                result = _run_full_section_production(
                    mosaic_tileset,
                    tile_dims,
                    max_temp_image_area,
                    workers,
                    temp_dir,
                    dest_dir,
                    bpp=bpp,
                    file_prefix=file_prefix,
                    file_postfix=file_postfix,
                    keep_output=args.keep_output,
                )
                tiles_saved = result.tiles_saved
                sweep_rows.append(_worker_sweep_row_from_full_section(workers, result))
                log(_format_timing_row('full section (asm+save)', result.tiles_saved, result.wall_s))
                log(
                    f'  save tail: drain={result.output_io_wait_drain_s:.2f}s  '
                    f'backpressure={result.output_io_wait_backpressure_s:.2f}s  '
                    f'encode_thread_s={result.save_encode_thread_s:.2f}  '
                    f'copy_thread_s={result.save_copy_thread_s:.2f}'
                )
                if result.last_strip_yield_to_drain_s is not None:
                    log(f'  last_strip_yield_to_drain_s={result.last_strip_yield_to_drain_s:.2f}s')
            except OSError as exc:
                sweep_rows.append(WorkerSweepRow(workers, workers * 2, 0, None, str(exc)))
                log(f'  FAILED: {exc}')
        log('')
        _print_worker_sweep_table(
            log, sweep_rows, 'Full-section summary (GenerateOptimizedTiles + production save):')

    json_data = _production_save_results_to_json(
        fixture,
        tile_size,
        layout,
        mode=mode,
        local_temp=local_temp_root,
        dest_root=dest_root,
        assemble_once=assemble_once,
        worker_sweep=sweep_rows,
        tiles_saved=tiles_saved,
    )
    out_path = args.output or (output_root / 'profiles' / f'production_save_{bench_tag}.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(json_data, indent=2), encoding='utf-8')
    log(f'\nJSON written to: {out_path}')
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark GenerateOptimizedTiles (AssembleTilesetNumpy path)')
    parser.add_argument('--volume-root', type=Path, default=None,
                        help='Volume root containing VolumeData.xml '
                             '(default: $TESTINPUTPATH/Volumes/RPC3)')
    parser.add_argument('--section', type=int, default=601)
    parser.add_argument('--channel', type=str, default='TEM')
    parser.add_argument('--filter', dest='filter_name', type=str, default='Leveled')
    parser.add_argument('--transform', dest='transform_name', type=str, default='Grid')
    parser.add_argument('--mosaic', type=Path, default=None,
                        help='Override mosaic path (skip VolumeData discovery)')
    parser.add_argument('--tile-dir', type=Path, default=None,
                        help='Override source TilePyramid/001 directory')
    parser.add_argument('--output-root', type=Path, default=None,
                        help='Writable output root '
                             '(default: $TESTOUTPUTPATH/assemble_bench/rpc3_601)')
    parser.add_argument('--tile-size', type=int, nargs=2, default=[512, 512],
                        metavar=('H', 'W'))
    parser.add_argument('--max-temp-image-area', type=float, default=None,
                        help='Working buffer area in pixels (default: EstimateMaxTempImageArea)')
    parser.add_argument('--max-columns', type=int, default=None,
                        help='Optional: only assemble the first N output columns '
                             '(faster iteration; omit for full section)')
    parser.add_argument('--backends', choices=('cpu', 'gpu', 'both'), default='both')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Timed iterations after one warmup when >1 (default: 3)')
    parser.add_argument('--profile', action='store_true',
                        help='Write cProfile stats for the last timed iteration per backend')
    parser.add_argument('--include-save', action='store_true',
                        help='Also time PNG encode + write (direct SaveImage; not production path)')
    parser.add_argument('--production-save', action='store_true',
                        help='Benchmark _SaveImageAndCopy (local temp encode + copy to dest)')
    parser.add_argument('--one-strip', action='store_true',
                        help='Use production strip width (EstimateMaxTempImageArea column math)')
    parser.add_argument('--io-sweep-only', action='store_true',
                        help='Assemble once (one-strip only), then sweep save worker counts')
    parser.add_argument('--cifs-dest', type=Path, default=None,
                        help='Final tile output directory (default: under --output-root)')
    parser.add_argument('--local-temp-dir', type=Path, default=None,
                        help='Local temp root for encoded tiles (default: gettempdir()/assemble_io_bench/...)')
    parser.add_argument('--workers', type=str, default=None,
                        help='Comma-separated tile I/O worker counts (default: 1,2,4,8,16,32 one-strip; 8,16,32 full)')
    parser.add_argument('--keep-output', action='store_true',
                        help='Do not delete dest/temp dirs between worker sweep iterations')
    parser.add_argument('--bpp', type=int, default=8,
                        help='Bits per pixel for production save (default: 8)')
    parser.add_argument('--file-prefix', type=str, default='Leveled_',
                        help='Output tile filename prefix (default: Leveled_)')
    parser.add_argument('--file-postfix', type=str, default='.png',
                        help='Output tile filename postfix (default: .png)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Write StageTimings-compatible JSON to this file')
    parser.add_argument('--save-golden', action='store_true',
                        help='Write checksum golden from the first CPU (or only) backend run')
    parser.add_argument('--verify', action='store_true',
                        help='Compare tile checksums against golden/')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    volume_root = args.volume_root or _default_volume_root()
    output_root = args.output_root or _default_output_root()
    profiles_dir = output_root / 'profiles'
    tileset_dir = output_root / 'tileset'
    golden_dir = output_root / 'golden'
    profiles_dir.mkdir(parents=True, exist_ok=True)

    fixture = discover_fixture(
        volume_root=volume_root,
        section=args.section,
        channel=args.channel,
        filter_name=args.filter_name,
        transform_name=args.transform_name,
        mosaic=args.mosaic,
        tile_dir=args.tile_dir,
    )
    tile_size = (int(args.tile_size[0]), int(args.tile_size[1]))
    max_area = int(args.max_temp_image_area) if args.max_temp_image_area else None

    def _log(msg: str) -> None:
        print(msg, flush=True)

    if args.production_save:
        if args.include_save:
            _log('ERROR: --production-save and --include-save are mutually exclusive.')
            return 1
        if args.io_sweep_only and not args.one_strip:
            _log('ERROR: --io-sweep-only requires --one-strip (full-grid buffer would exceed RAM).')
            return 1
        return run_production_save(args, fixture, tile_size, max_area, output_root, _log)

    if args.io_sweep_only or args.cifs_dest is not None or args.workers is not None:
        _log('ERROR: --io-sweep-only, --cifs-dest, and --workers require --production-save.')
        return 1

    _log(f'Volume root     : {fixture.volume_root}')
    _log(f'Section         : {fixture.section_number}')
    _log(f'Mosaic          : {fixture.mosaic_path}')
    _log(f'Tile dir        : {fixture.tile_dir}')
    _log(f'Tile size       : {tile_size[0]}x{tile_size[1]}')
    _log(f'Output root     : {output_root}')
    _log(f'Max temp area   : {max_area or "(EstimateMaxTempImageArea)"}')
    _log(f'Max columns     : {args.max_columns or "(full grid)"}')
    _log(f'Backends        : {args.backends}')
    _log(f'Iterations      : {args.iterations} (+ warmup if iterations>1)')

    backends_to_run: list[tuple[str, ComputationLib]] = []
    if args.backends in ('cpu', 'both'):
        backends_to_run.append(('cpu', ComputationLib.numpy))
    if args.backends in ('gpu', 'both'):
        if nornir_imageregistration.HasCupy():
            backends_to_run.append(('gpu', ComputationLib.cupy))
        else:
            print('\n[GPU] CuPy not available — skipping.')

    results: list[BackendResult] = []
    verify_checksums: dict[str, str] | None = None

    for backend_name, lib in backends_to_run:
        nornir_imageregistration.SetActiveComputationLib(lib)
        result = BackendResult(backend=backend_name)
        # iterations: first is warmup when iterations>1
        n_runs = args.iterations + (1 if args.iterations > 1 else 0)
        _log(f'\n[{backend_name.upper()}] GenerateOptimizedTiles  x{n_runs} '
             f'({"1 warmup + " + str(args.iterations) if args.iterations > 1 else "no warmup"}) ...')

        for i in range(n_runs):
            is_last = (i == n_runs - 1)
            is_warmup = (args.iterations > 1 and i == 0)
            # Checksums only when needed — never during timed profile samples.
            want_golden = args.save_golden and backend_name == backends_to_run[0][0]
            collect = bool((want_golden or args.verify) and is_last)
            # Profile a timed iteration without checksum overhead when verifying/golden.
            profile_this = bool(args.profile and not is_warmup and (
                (is_last and not collect) or (i == n_runs - 2 and collect and n_runs >= 2)))
            run_tileset = tileset_dir / backend_name if args.include_save else None
            if run_tileset is not None and run_tileset.exists():
                shutil.rmtree(run_tileset)

            profiler: cProfile.Profile | None = None
            if profile_this:
                profiler = cProfile.Profile()
                profiler.enable()

            stages, checksums = run_once(
                fixture,
                tile_size,
                max_area,
                args.include_save,
                run_tileset,
                args.max_columns,
                collect,
            )

            if profiler is not None:
                profiler.disable()
                profile_path = profiles_dir / f'{backend_name}_assemble.profile'
                profiler.dump_stats(str(profile_path))
                result.profile_path = str(profile_path)
                stream = StringIO()
                stats = pstats.Stats(str(profile_path), stream=stream)
                stats.sort_stats('cumulative')
                stats.print_stats(25)
                _log(stream.getvalue())
                needles = _profile_needles_report(profile_path)
                needles_path = profiles_dir / f'{backend_name}_needles.json'
                needles_path.write_text(json.dumps(needles, indent=2), encoding='utf-8')
                _log(f'Profile needles → {needles_path}')

            result.times.append(stages)
            label = 'warmup' if is_warmup else f'iter {i if args.iterations == 1 else i}'
            _log(f'  {label}: total={stages.total:.2f}s  assemble={stages.assemble_warp:.2f}s  '
                 f'load={stages.load_mosaic:.2f}s  save={stages.save_png:.2f}s  '
                 f'tiles={stages.tiles_generated}  grid={stages.grid_dims}')

            if collect and checksums:
                if want_golden:
                    _save_golden(golden_dir, checksums, {
                        'backend': backend_name,
                        'mosaic': str(fixture.mosaic_path),
                        'tile_size': list(tile_size),
                        'tiles_generated': stages.tiles_generated,
                        'grid_dims_yx': list(stages.grid_dims),
                        'max_temp_image_area': max_area,
                        'max_columns': args.max_columns,
                    })
                    _log(f'  Golden saved → {golden_dir / "checksums.json"} '
                         f'({len(checksums)} tiles)')
                if args.verify:
                    verify_checksums = checksums

        results.append(result)

    _log('\n' + _format_table(results))

    if args.verify and verify_checksums is not None:
        ok, msg = _verify_golden(golden_dir, verify_checksums)
        _log(f'\nVerify: {msg}')
        if not ok:
            return 1
    elif args.verify:
        _log('\nVerify: no checksums collected')
        return 1

    json_data = _results_to_json(results, fixture, tile_size)
    json_data['max_temp_image_area'] = max_area
    json_data['max_columns'] = args.max_columns
    json_str = json.dumps(json_data, indent=2)
    out_path = args.output
    if out_path is None:
        out_path = profiles_dir / 'baseline.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json_str, encoding='utf-8')
    _log(f'\nJSON written to: {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
