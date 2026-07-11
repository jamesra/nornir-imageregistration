#!/usr/bin/env python3
"""Cross-scope TEM1/TEM2 scale assessment (methods 1–5).

Inventory the first N numeric StosBrute64 pairs by Microscope notes, then:
  1) VolumeData Scale X/Y metadata
  2) Same- vs cross-scope anisotropic image estimates
  3) 90° axis-swap control
  4) Target- vs source-axis scale framing
  5) 1D strip correlators

Prefers Manual transforms when present under StosBrute64/Manual/.

Example::

    python scripts/assess_cross_scope_scale.py \\
        --volume-root /storage4/RPC3 --limit 100 --max-dim 1024 \\
        --write-csv /tmp/cross_scope_scale/results.csv \\
        --write-json /tmp/cross_scope_scale/results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import nornir_imageregistration  # noqa: E402
import nornir_imageregistration.stos_brute as stos_brute  # noqa: E402
from nornir_imageregistration.files.stosfile import StosFile  # noqa: E402

import assess_stos_anisotropic_scale as aniso  # noqa: E402

_PAIR_RE = re.compile(r'^(\d+)-(\d+)_')
_MIN_OVERLAP = aniso._MIN_OVERLAP


@dataclass
class InventoryRow:
    pair: str
    ctrl_section: int
    map_section: int
    ctrl_mic: str | None
    map_mic: str | None
    cross_scope: bool | None
    used_manual: bool
    stos_path: str
    transform_path: str
    angle_deg: float | None
    scalar: float | None
    ctrl_nm_x: float | None = None
    ctrl_nm_y: float | None = None
    map_nm_x: float | None = None
    map_nm_y: float | None = None
    sx_meta: float | None = None
    sy_meta: float | None = None
    s_meta: float | None = None
    abs_sy_sx_meta: float | None = None
    scalar_vs_s_meta: float | None = None
    notes_missing: bool = False


@dataclass
class ImageScaleRow:
    pair: str
    cross_scope: bool
    used_manual: bool
    ctrl_mic: str | None
    map_mic: str | None
    angle_deg: float
    transform_scalar: float
    s_meta: float | None
    scale_iso_refine: float
    sy: float
    sx: float
    abs_sy_sx: float
    geo_mean: float
    geo_mean_vs_s_meta: float | None
    geo_mean_vs_transform: float
    weight_aniso: float
    weight_iso_refine: float
    weight_lift_vs_iso: float
    elapsed_s: float
    method: str = 'source_axis'


@dataclass
class SubsetExtraRow:
    pair: str
    cross_scope: bool
    sy_src: float
    sx_src: float
    abs_src: float
    sy_swap: float
    sx_swap: float
    abs_swap: float
    swap_tracks_axes: bool
    sy_tgt: float
    sx_tgt: float
    abs_tgt: float
    strip_sy: float
    strip_sx: float
    abs_strip: float
    elapsed_s: float


def _section_dir(volume_root: Path, section: int) -> Path:
    return volume_root / 'TEM' / f'{section:04d}' / 'TEM'


def _read_microscope(volume_root: Path, section: int) -> str | None:
    notes = _section_dir(volume_root, section) / f'{section:04d}.txt'
    if not notes.is_file():
        return None
    for line in notes.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.startswith('Microscope:'):
            return line.split(':', 1)[1].strip()
    return None


def _read_scale_nm(volume_root: Path, section: int) -> tuple[float | None, float | None]:
    """Return (UnitsPerPixel X, UnitsPerPixel Y) from VolumeData.xml Channel Scale."""
    xml_path = _section_dir(volume_root, section) / 'VolumeData.xml'
    if not xml_path.is_file():
        return None, None
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None, None
    for scale in root.findall('.//Scale'):
        x_el = scale.find('X')
        y_el = scale.find('Y')
        if x_el is None or y_el is None:
            continue
        try:
            return float(x_el.get('UnitsPerPixel')), float(y_el.get('UnitsPerPixel'))
        except (TypeError, ValueError):
            continue
    return None, None


def _list_stos_numeric(stos_dir: Path, limit: int) -> list[Path]:
    files = [p for p in stos_dir.glob('*.stos') if p.is_file()]

    def _key(p: Path) -> tuple[int, int]:
        m = _PAIR_RE.match(p.name)
        if not m:
            return (10**9, 10**9)
        return int(m.group(1)), int(m.group(2))

    return sorted(files, key=_key)[:limit]


def _parse_pair(name: str) -> tuple[str, int, int]:
    m = _PAIR_RE.match(name)
    if not m:
        raise ValueError(f'cannot parse pair from {name}')
    a, b = int(m.group(1)), int(m.group(2))
    return f'{a}-{b}', a, b


def _transform_angle_scalar(stos_path: Path) -> tuple[float, float]:
    stos = StosFile.Load(str(stos_path), resolve_paths=True)
    return aniso._manual_angle_and_scalar(stos)


def _resolve_transform_path(stos_path: Path, manual_dir: Path) -> tuple[Path, bool]:
    manual = manual_dir / stos_path.name
    if manual.is_file():
        return manual, True
    return stos_path, False


def build_inventory(
        volume_root: Path,
        stos_dir: Path,
        manual_dir: Path,
        limit: int,
) -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    for stos_path in _list_stos_numeric(stos_dir, limit):
        pair, ctrl, mapped = _parse_pair(stos_path.name)
        ctrl_mic = _read_microscope(volume_root, ctrl)
        map_mic = _read_microscope(volume_root, mapped)
        notes_missing = ctrl_mic is None or map_mic is None
        cross = None if notes_missing else (ctrl_mic != map_mic)
        transform_path, used_manual = _resolve_transform_path(stos_path, manual_dir)
        angle_deg = scalar = None
        try:
            angle_deg, scalar = _transform_angle_scalar(transform_path)
        except Exception as exc:
            print(f'WARN {pair}: transform parse failed: {exc}', file=sys.stderr)

        ctrl_nm_x, ctrl_nm_y = _read_scale_nm(volume_root, ctrl)
        map_nm_x, map_nm_y = _read_scale_nm(volume_root, mapped)
        sx_meta = sy_meta = s_meta = abs_meta = scalar_vs = None
        if None not in (ctrl_nm_x, ctrl_nm_y, map_nm_x, map_nm_y):
            assert ctrl_nm_x and ctrl_nm_y and map_nm_x and map_nm_y
            sx_meta = float(ctrl_nm_x / map_nm_x)
            sy_meta = float(ctrl_nm_y / map_nm_y)
            s_meta = float(np.sqrt(sx_meta * sy_meta))
            abs_meta = abs(sy_meta - sx_meta)
            if scalar is not None:
                scalar_vs = abs(float(scalar) - s_meta)

        rows.append(InventoryRow(
            pair=pair,
            ctrl_section=ctrl,
            map_section=mapped,
            ctrl_mic=ctrl_mic,
            map_mic=map_mic,
            cross_scope=cross,
            used_manual=used_manual,
            stos_path=str(stos_path),
            transform_path=str(transform_path),
            angle_deg=angle_deg,
            scalar=scalar,
            ctrl_nm_x=ctrl_nm_x,
            ctrl_nm_y=ctrl_nm_y,
            map_nm_x=map_nm_x,
            map_nm_y=map_nm_y,
            sx_meta=sx_meta,
            sy_meta=sy_meta,
            s_meta=s_meta,
            abs_sy_sx_meta=abs_meta,
            scalar_vs_s_meta=scalar_vs,
            notes_missing=notes_missing,
        ))
    return rows


def _load_pair_images(
        row: InventoryRow,
        volume_root: Path,
        max_dim: int,
) -> tuple[np.ndarray, np.ndarray,
           nornir_imageregistration.ImageStats,
           nornir_imageregistration.ImageStats]:
    """Load images; prefer Manual stos for relative-path resolution (../../section)."""
    # Auto StosBrute64/*.stos resolves ../../NNNN one level too high (missing TEM/).
    # Manual/*.stos resolves correctly; fall back to volume_root/TEM rewrite.
    load_path = Path(row.transform_path if row.used_manual else row.stos_path)
    stos = StosFile.Load(str(load_path), resolve_paths=True)

    def _resolve(stored: str | None) -> str | None:
        if not stored:
            return None
        if os.path.isfile(stored):
            return stored
        # Rewrite .../RPC3/NNNN/TEM/... -> .../RPC3/TEM/NNNN/TEM/...
        norm = stored.replace('\\', '/')
        marker = str(volume_root).replace('\\', '/').rstrip('/')
        # Pattern: <volume>/0xxx/TEM/ -> <volume>/TEM/0xxx/TEM/
        m = re.search(r'/(\d{4})/TEM/', norm)
        if m and marker in norm:
            sec = m.group(1)
            # If path lacks /TEM/<sec> and has /<sec>/TEM
            bad = f'{marker}/{sec}/TEM/'
            good = f'{marker}/TEM/{sec}/TEM/'
            if bad in norm:
                candidate = norm.replace(bad, good, 1)
                if os.path.isfile(candidate):
                    return candidate
        # Try joining from volume TEM root using trailing section-relative suffix
        idx = norm.find('/TEM/')
        if idx >= 0:
            # already has TEM somewhere
            pass
        return stored

    source_path = _resolve(stos.MappedImageFullPath)
    target_path = _resolve(stos.ControlImageFullPath)
    source_mask = _resolve(stos.MappedMaskFullPath)
    target_mask = _resolve(stos.ControlMaskFullPath)
    if not source_path or not target_path:
        raise FileNotFoundError(f'missing image paths in {load_path}')
    if not os.path.isfile(source_path) or not os.path.isfile(target_path):
        raise FileNotFoundError(f'images not found: {source_path!r} / {target_path!r}')

    source_h = nornir_imageregistration.ImagePermutationHelper(
        source_path,
        source_mask if source_mask and os.path.isfile(source_mask) else None)
    target_h = nornir_imageregistration.ImagePermutationHelper(
        target_path,
        target_mask if target_mask and os.path.isfile(target_mask) else None)
    source_raw = source_h.ImageWithMaskAsNoise
    target_raw = target_h.ImageWithMaskAsNoise
    if hasattr(source_raw, 'get'):
        source_raw = source_raw.get()
    if hasattr(target_raw, 'get'):
        target_raw = target_raw.get()
    source = np.asarray(source_raw, dtype=np.float32)
    target = np.asarray(target_raw, dtype=np.float32)
    source, target, _, _ = aniso._maybe_downscale(source, target, None, None, max_dim)
    return (
        source,
        target,
        nornir_imageregistration.ImageStats.CalcStats(source),
        nornir_imageregistration.ImageStats.CalcStats(target),
    )


def _estimate_source_axis(
        source: np.ndarray,
        target: np.ndarray,
        source_stats: nornir_imageregistration.ImageStats,
        target_stats: nornir_imageregistration.ImageStats,
        angle_deg: float,
        seed_scalar: float,
) -> tuple[float, float, float, float, float]:
    """Return iso_refine, sy, sx, weight_aniso, weight_iso."""
    seed = float(np.clip(seed_scalar, stos_brute._SCALE_REFINE_MIN, stos_brute._SCALE_REFINE_MAX))
    iso = stos_brute._refine_scale_local(
        source, target, source_stats, target_stats,
        angle=angle_deg, initial_scale=seed, min_overlap=_MIN_OVERLAP, wide_search=False)
    score = aniso._make_aniso_scorer(source, target, source_stats, target_stats, angle_deg)
    w_iso = score(iso, iso)
    sy, sx, w_aniso = aniso._refine_anisotropic(score, float(iso))
    return float(iso), float(sy), float(sx), float(w_aniso), float(w_iso)


def _make_target_axis_scorer(
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_stats: nornir_imageregistration.ImageStats,
        target_stats: nornir_imageregistration.ImageStats,
        angle_deg: float,
):
    """Scale in target axes: rotate source first, then zoom target by (1/sy, 1/sx) equivalently
    by zooming source after rotation is awkward; instead zoom target by (sy,sx) inverse framing:
    apply (sy,sx) to target (control) then correlate against rotated source at scale 1.
    Convention: reported (sy,sx) still means mapped→control scale factors.
    """
    xp = np
    rotated_source = stos_brute.pad_and_rotate_image(
        image=source_image,
        angle=angle_deg,
        image_stats=source_stats,
        min_overlap=_MIN_OVERLAP)
    rot_stats = nornir_imageregistration.ImageStats.CalcStats(rotated_source)
    source_mean = float(rot_stats.mean)
    fft_source_cache: dict[tuple[int, int], np.ndarray] = {}

    def score(sy: float, sx: float) -> float:
        # Zoom target so that matching source needs scale (sy,sx) in target frame.
        sy = float(np.clip(sy, stos_brute._SCALE_REFINE_MIN, stos_brute._SCALE_REFINE_MAX))
        sx = float(np.clip(sx, stos_brute._SCALE_REFINE_MIN, stos_brute._SCALE_REFINE_MAX))
        inv_y = 1.0 / sy
        inv_x = 1.0 / sx
        working_target = target_image
        working_stats = target_stats
        working_shape = target_image.shape
        if not (np.isclose(inv_y, 1.0) and np.isclose(inv_x, 1.0)):
            working_target = stos_brute._scale_registration_image(target_image, inv_y, inv_x)
            working_stats = nornir_imageregistration.ImageStats.CalcStats(working_target)
            working_shape = working_target.shape

        padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
            working_target,
            image_median=working_stats.median,
            image_stddev=working_stats.std,
            min_overlap=_MIN_OVERLAP,
            original_shape=working_shape)

        corr_h = max(padded_target.shape[0], rotated_source.shape[0])
        corr_w = max(padded_target.shape[1], rotated_source.shape[1])
        corr_shape = (int(corr_h), int(corr_w))

        if padded_target.shape != corr_shape:
            padded_target = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                padded_target,
                new_width=corr_shape[1],
                new_height=corr_shape[0],
                image_median=working_stats.median,
                image_stddev=working_stats.std,
                min_overlap=1.0,
                power_of_two=False)
        if rotated_source.shape != corr_shape:
            source_for_corr = nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                rotated_source,
                new_width=corr_shape[1],
                new_height=corr_shape[0],
                image_median=rot_stats.median,
                image_stddev=rot_stats.std,
                min_overlap=1.0,
                power_of_two=False)
        else:
            source_for_corr = rotated_source

        if corr_shape not in fft_source_cache:
            fft_source_cache[corr_shape] = xp.fft.fft2(source_for_corr - source_mean)
        target_fft = xp.fft.fft2(padded_target - float(working_stats.mean))
        # Correlate target vs source: swap roles so peak weight still meaningful.
        correlation_image = nornir_imageregistration.phasecorrelation.fft_phase_correlation(
            target_fft, fft_source_cache[corr_shape], delete_input=False, correlation_coefficient=.66)
        del target_fft
        record = stos_brute._peak_from_correlation_image(
            correlation_image, working_shape, source_image.shape, angle_deg, _MIN_OVERLAP, xp)
        return float(record.weight)

    return score


def _estimate_target_axis(
        source: np.ndarray,
        target: np.ndarray,
        source_stats: nornir_imageregistration.ImageStats,
        target_stats: nornir_imageregistration.ImageStats,
        angle_deg: float,
        seed_scalar: float,
) -> tuple[float, float, float]:
    seed = float(np.clip(seed_scalar, stos_brute._SCALE_REFINE_MIN, stos_brute._SCALE_REFINE_MAX))
    score = _make_target_axis_scorer(source, target, source_stats, target_stats, angle_deg)
    sy, sx, w = aniso._refine_anisotropic(score, seed)
    return sy, sx, w


def _rotate_both_90(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.rot90(source, k=1), np.rot90(target, k=1)


def _strip_1d_scales(
        source: np.ndarray,
        target: np.ndarray,
        source_stats: nornir_imageregistration.ImageStats,
        angle_deg: float,
        seed: float,
) -> tuple[float, float]:
    """Estimate sy/sx from 1D phase correlation on mid strips after Manual rotate."""
    rotated = stos_brute.pad_and_rotate_image(
        image=source, angle=angle_deg, image_stats=source_stats, min_overlap=_MIN_OVERLAP)
    # Crop to overlapping central region.
    h = min(rotated.shape[0], target.shape[0])
    w = min(rotated.shape[1], target.shape[1])
    rs = rotated[:h, :w]
    tg = target[:h, :w]
    band = max(8, min(h, w) // 16)

    def _axis_scale(horizontal: bool) -> float:
        if horizontal:
            mid = h // 2
            a = np.mean(tg[max(0, mid - band):min(h, mid + band), :], axis=0)
            b = np.mean(rs[max(0, mid - band):min(h, mid + band), :], axis=0)
        else:
            mid = w // 2
            a = np.mean(tg[:, max(0, mid - band):min(w, mid + band)], axis=1)
            b = np.mean(rs[:, max(0, mid - band):min(w, mid + band)], axis=1)
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float(seed)
        # Search isotropic 1D zoom of b vs a via coarse grid on correlation peak.
        best_s = float(seed)
        best_c = -1.0
        for delta in stos_brute._SCALE_REFINE_NARROW_DELTAS:
            s = float(np.clip(seed + delta, stos_brute._SCALE_REFINE_MIN, stos_brute._SCALE_REFINE_MAX))
            zoomed = ndimage.zoom(b, s, order=1)
            n = min(len(a), len(zoomed))
            aa = a[:n] - a[:n].mean()
            zz = zoomed[:n] - zoomed[:n].mean()
            denom = (np.linalg.norm(aa) * np.linalg.norm(zz)) + 1e-12
            corr = float(np.dot(aa, zz) / denom)
            if corr > best_c:
                best_c = corr
                best_s = s
        return best_s

    sx = _axis_scale(horizontal=True)   # along X
    sy = _axis_scale(horizontal=False)  # along Y
    return sy, sx


def _select_method2_pairs(
        inventory: list[InventoryRow],
        max_pairs: int | None,
) -> tuple[list[InventoryRow], list[InventoryRow]]:
    cross = [r for r in inventory if r.cross_scope is True and r.angle_deg is not None]
    same = [r for r in inventory if r.cross_scope is False and r.angle_deg is not None]
    cross_manual = [r for r in cross if r.used_manual]
    cross_auto = [r for r in cross if not r.used_manual]
    cross_ordered = cross_manual + cross_auto
    if max_pairs is not None:
        cross_ordered = cross_ordered[:max_pairs]
    n_same = len(cross_ordered) if max_pairs is None else min(len(same), len(cross_ordered))
    # Balance TEM1-TEM1 and TEM2-TEM2 when possible.
    same_t1 = [r for r in same if r.ctrl_mic == 'TEM1']
    same_t2 = [r for r in same if r.ctrl_mic == 'TEM2']
    same_sel: list[InventoryRow] = []
    i1 = i2 = 0
    while len(same_sel) < n_same and (i1 < len(same_t1) or i2 < len(same_t2)):
        if i1 < len(same_t1):
            same_sel.append(same_t1[i1])
            i1 += 1
        if len(same_sel) >= n_same:
            break
        if i2 < len(same_t2):
            same_sel.append(same_t2[i2])
            i2 += 1
    return cross_ordered, same_sel


def _select_subset8(inventory: list[InventoryRow]) -> list[InventoryRow]:
    cross_m = [r for r in inventory if r.cross_scope and r.used_manual and r.angle_deg is not None]
    same = [r for r in inventory if r.cross_scope is False and r.angle_deg is not None]
    return cross_m[:4] + same[:4]


def run_method2(
        rows: list[InventoryRow],
        volume_root: Path,
        max_dim: int,
        label: str,
) -> list[ImageScaleRow]:
    out: list[ImageScaleRow] = []
    for row in rows:
        t0 = time.perf_counter()
        try:
            print(f'  M2 {label} {row.pair} (manual={row.used_manual}) ...', flush=True)
            source, target, ss, ts = _load_pair_images(row, volume_root, max_dim)
            assert row.angle_deg is not None and row.scalar is not None
            iso, sy, sx, w_aniso, w_iso = _estimate_source_axis(
                source, target, ss, ts, row.angle_deg, row.scalar)
            geo = float(np.sqrt(sy * sx))
            out.append(ImageScaleRow(
                pair=row.pair,
                cross_scope=bool(row.cross_scope),
                used_manual=row.used_manual,
                ctrl_mic=row.ctrl_mic,
                map_mic=row.map_mic,
                angle_deg=row.angle_deg,
                transform_scalar=row.scalar,
                s_meta=row.s_meta,
                scale_iso_refine=iso,
                sy=sy,
                sx=sx,
                abs_sy_sx=abs(sy - sx),
                geo_mean=geo,
                geo_mean_vs_s_meta=(abs(geo - row.s_meta) if row.s_meta is not None else None),
                geo_mean_vs_transform=abs(geo - row.scalar),
                weight_aniso=w_aniso,
                weight_iso_refine=w_iso,
                weight_lift_vs_iso=w_aniso / w_iso if w_iso > 1e-12 else float('nan'),
                elapsed_s=time.perf_counter() - t0,
                method='source_axis',
            ))
            print(
                f'    sy={sy:.4f} sx={sx:.4f} |d|={abs(sy - sx):.4f} '
                f's_meta={row.s_meta} ({time.perf_counter() - t0:.1f}s)',
                flush=True,
            )
        except Exception as exc:
            print(f'  SKIP M2 {row.pair}: {exc}', file=sys.stderr)
    return out


def run_methods_3_5(
        rows: list[InventoryRow],
        volume_root: Path,
        max_dim: int,
) -> list[SubsetExtraRow]:
    out: list[SubsetExtraRow] = []
    for row in rows:
        t0 = time.perf_counter()
        try:
            print(f'  M3-5 {row.pair} ...', flush=True)
            source, target, ss, ts = _load_pair_images(row, volume_root, max_dim)
            assert row.angle_deg is not None and row.scalar is not None
            seed = row.scalar
            _, sy, sx, _, _ = _estimate_source_axis(source, target, ss, ts, row.angle_deg, seed)

            # Method 3: 90° swap both images; angle unchanged in content frame after dual rot90
            # is equivalent to angle-90 for relative pose; use angle_deg - 90.
            s90, t90 = _rotate_both_90(source, target)
            ss90 = nornir_imageregistration.ImageStats.CalcStats(s90)
            ts90 = nornir_imageregistration.ImageStats.CalcStats(t90)
            angle_swap = stos_brute._normalize_angle_degrees(row.angle_deg - 90.0)
            _, sy_s, sx_s, _, _ = _estimate_source_axis(s90, t90, ss90, ts90, angle_swap, seed)
            # Camera-axis anisotropy should swap: (sy,sx) -> ~(sx,sy)
            swap_tracks = abs(sy_s - sx) < abs(sy_s - sy) and abs(sx_s - sy) < abs(sx_s - sx)

            # Method 4: target-axis framing
            sy_t, sx_t, _ = _estimate_target_axis(source, target, ss, ts, row.angle_deg, seed)

            # Method 5: 1D strips
            strip_sy, strip_sx = _strip_1d_scales(source, target, ss, row.angle_deg, seed)

            out.append(SubsetExtraRow(
                pair=row.pair,
                cross_scope=bool(row.cross_scope),
                sy_src=sy,
                sx_src=sx,
                abs_src=abs(sy - sx),
                sy_swap=sy_s,
                sx_swap=sx_s,
                abs_swap=abs(sy_s - sx_s),
                swap_tracks_axes=bool(swap_tracks),
                sy_tgt=sy_t,
                sx_tgt=sx_t,
                abs_tgt=abs(sy_t - sx_t),
                strip_sy=strip_sy,
                strip_sx=strip_sx,
                abs_strip=abs(strip_sy - strip_sx),
                elapsed_s=time.perf_counter() - t0,
            ))
            print(
                f'    src|d|={abs(sy - sx):.4f} swap|d|={abs(sy_s - sx_s):.4f} '
                f'track={swap_tracks} tgt|d|={abs(sy_t - sx_t):.4f} '
                f'strip|d|={abs(strip_sy - strip_sx):.4f}',
                flush=True,
            )
        except Exception as exc:
            print(f'  SKIP M3-5 {row.pair}: {exc}', file=sys.stderr)
    return out


def _summarize_method1(inventory: list[InventoryRow]) -> dict:
    tem1_x: list[float] = []
    tem2_x: list[float] = []
    for r in inventory:
        if r.ctrl_mic == 'TEM1' and r.ctrl_nm_x is not None:
            tem1_x.append(r.ctrl_nm_x)
        if r.ctrl_mic == 'TEM2' and r.ctrl_nm_x is not None:
            tem2_x.append(r.ctrl_nm_x)
        if r.map_mic == 'TEM1' and r.map_nm_x is not None:
            tem1_x.append(r.map_nm_x)
        if r.map_mic == 'TEM2' and r.map_nm_x is not None:
            tem2_x.append(r.map_nm_x)

    cross = [r for r in inventory if r.cross_scope and r.s_meta is not None and r.scalar is not None]
    abs_meta = [r.abs_sy_sx_meta for r in cross if r.abs_sy_sx_meta is not None]
    scalar_err = [r.scalar_vs_s_meta for r in cross if r.scalar_vs_s_meta is not None]
    corr = float('nan')
    if len(cross) >= 2:
        s_meta = np.array([r.s_meta for r in cross], dtype=float)
        scalars = np.array([r.scalar for r in cross], dtype=float)
        if s_meta.std() > 1e-12 and scalars.std() > 1e-12:
            corr = float(np.corrcoef(s_meta, scalars)[0, 1])

    return {
        'tem1_nm_per_px_mean': float(np.mean(tem1_x)) if tem1_x else None,
        'tem2_nm_per_px_mean': float(np.mean(tem2_x)) if tem2_x else None,
        'tem1_nm_per_px_std': float(np.std(tem1_x)) if tem1_x else None,
        'tem2_nm_per_px_std': float(np.std(tem2_x)) if tem2_x else None,
        'cross_n': len(cross),
        'cross_mean_abs_sy_sx_meta': float(np.mean(abs_meta)) if abs_meta else None,
        'cross_mean_s_meta': float(np.mean([r.s_meta for r in cross])) if cross else None,
        'cross_mean_scalar_vs_s_meta': float(np.mean(scalar_err)) if scalar_err else None,
        'cross_corr_scalar_vs_s_meta': corr,
        'meta_isotropic': (float(np.mean(abs_meta)) < 1e-6) if abs_meta else None,
    }


def _group_stats(rows: list[ImageScaleRow]) -> dict:
    if not rows:
        return {'n': 0}
    abs_d = np.array([r.abs_sy_sx for r in rows], dtype=float)
    vs_meta = [r.geo_mean_vs_s_meta for r in rows if r.geo_mean_vs_s_meta is not None]
    return {
        'n': len(rows),
        'mean_abs_sy_sx': float(np.mean(abs_d)),
        'median_abs_sy_sx': float(np.median(abs_d)),
        'mean_geo_vs_s_meta': float(np.mean(vs_meta)) if vs_meta else None,
        'mean_geo_vs_transform': float(np.mean([r.geo_mean_vs_transform for r in rows])),
        'mean_weight_lift': float(np.nanmean([r.weight_lift_vs_iso for r in rows])),
    }


def _decision(
        m1: dict,
        m2_cross: dict,
        m2_same: dict,
        subset: list[SubsetExtraRow],
) -> dict:
    meta_iso = bool(m1.get('meta_isotropic'))
    tem1 = m1.get('tem1_nm_per_px_mean')
    tem2 = m1.get('tem2_nm_per_px_mean')
    scopes_differ = (
        tem1 is not None and tem2 is not None and abs(float(tem1) - float(tem2)) > 0.01
    )
    corr = m1.get('cross_corr_scalar_vs_s_meta')
    scalar_tracks_meta = corr is not None and np.isfinite(corr) and corr > 0.5

    cross_abs = m2_cross.get('mean_abs_sy_sx')
    same_abs = m2_same.get('mean_abs_sy_sx')
    aniso_cross_larger = (
        cross_abs is not None and same_abs is not None and float(cross_abs) > float(same_abs) + 0.01
    )
    n_swap_track = sum(1 for r in subset if r.swap_tracks_axes)
    meta_aniso = m1.get('cross_mean_abs_sy_sx_meta')
    meta_aniso_nontrivial = meta_aniso is not None and float(meta_aniso) >= 0.01

    recommend_affine = bool(aniso_cross_larger and meta_aniso_nontrivial)
    recommend_scope_scale = bool(meta_iso and scopes_differ and scalar_tracks_meta)

    if recommend_affine:
        verdict = (
            'GO Affine/per-axis: cross-scope |sy-sx| exceeds same-scope and metadata '
            'shows per-axis scale difference.'
        )
    elif recommend_scope_scale:
        verdict = (
            'NO-GO Affine; GO scope isotropic WarpedImageScaleFactors: TEM1/TEM2 nm/px '
            'differ isotropically and Manual scalars track metadata scale.'
        )
    else:
        verdict = (
            'NO-GO Affine; inconclusive scope wiring — inspect Method 1/2 tables.'
        )

    return {
        'recommend_affine_per_axis': recommend_affine,
        'recommend_scope_isotropic_scale_factors': recommend_scope_scale,
        'verdict': verdict,
        'n_subset_swap_tracks_axes': n_swap_track,
        'n_subset': len(subset),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--volume-root', type=Path, default=Path('/storage4/RPC3'))
    parser.add_argument('--stos-dir', type=Path, default=None,
                        help='Default: <volume-root>/TEM/StosBrute64')
    parser.add_argument('--manual-dir', type=Path, default=None,
                        help='Default: <stos-dir>/Manual')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--max-dim', type=int, default=1024)
    parser.add_argument(
        '--max-pairs', type=int, default=None,
        help='Cap Method 2 cross-scope pairs (default: all Manual cross, then auto fill)')
    parser.add_argument('--skip-method2', action='store_true')
    parser.add_argument('--skip-method345', action='store_true')
    parser.add_argument('--backend', choices=('numpy', 'cupy'), default='numpy')
    parser.add_argument('--write-csv', type=Path, default=None)
    parser.add_argument('--write-json', type=Path, default=None)
    args = parser.parse_args()

    aniso._set_backend(args.backend)
    volume_root = args.volume_root
    stos_dir = args.stos_dir or (volume_root / 'TEM' / 'StosBrute64')
    manual_dir = args.manual_dir or (stos_dir / 'Manual')
    if not stos_dir.is_dir():
        print(f'stos-dir not found: {stos_dir}', file=sys.stderr)
        return 1

    print('=== Phase A: inventory ===', flush=True)
    inventory = build_inventory(volume_root, stos_dir, manual_dir, args.limit)
    n_cross = sum(1 for r in inventory if r.cross_scope is True)
    n_same = sum(1 for r in inventory if r.cross_scope is False)
    n_miss = sum(1 for r in inventory if r.notes_missing)
    n_manual_cross = sum(1 for r in inventory if r.cross_scope and r.used_manual)
    print(f'first {len(inventory)}: cross={n_cross} same={n_same} notes_missing={n_miss} '
          f'manual_cross={n_manual_cross}')

    print('=== Phase B: Method 1 metadata ===', flush=True)
    m1 = _summarize_method1(inventory)
    print(json.dumps(m1, indent=2))

    m2_cross_rows: list[ImageScaleRow] = []
    m2_same_rows: list[ImageScaleRow] = []
    if not args.skip_method2:
        print('=== Phase C: Method 2 same vs cross ===', flush=True)
        cross_sel, same_sel = _select_method2_pairs(inventory, args.max_pairs)
        print(f'Method2 assessing cross={len(cross_sel)} same={len(same_sel)}')
        m2_cross_rows = run_method2(cross_sel, volume_root, args.max_dim, 'cross')
        m2_same_rows = run_method2(same_sel, volume_root, args.max_dim, 'same')
    m2_cross_stats = _group_stats(m2_cross_rows)
    m2_same_stats = _group_stats(m2_same_rows)
    print('M2 cross', m2_cross_stats)
    print('M2 same', m2_same_stats)

    subset_rows: list[SubsetExtraRow] = []
    if not args.skip_method345:
        print('=== Phase D: Methods 3–5 subset ===', flush=True)
        subset = _select_subset8(inventory)
        print(f'subset pairs: {[r.pair for r in subset]}')
        subset_rows = run_methods_3_5(subset, volume_root, args.max_dim)

    print('=== Phase E: decision ===', flush=True)
    decision = _decision(m1, m2_cross_stats, m2_same_stats, subset_rows)
    print(decision['verdict'])

    summary = {
        'volume_root': str(volume_root),
        'stos_dir': str(stos_dir),
        'limit': args.limit,
        'inventory_counts': {
            'n': len(inventory),
            'cross': n_cross,
            'same': n_same,
            'notes_missing': n_miss,
            'manual_cross': n_manual_cross,
        },
        'method1': m1,
        'method2_cross': m2_cross_stats,
        'method2_same': m2_same_stats,
        'method345_n': len(subset_rows),
        'decision': decision,
        'cross_scope_pairs': [r.pair for r in inventory if r.cross_scope],
    }

    inv_dicts = [asdict(r) for r in inventory]
    if args.write_csv:
        args.write_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.write_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(inv_dicts[0].keys()))
            writer.writeheader()
            writer.writerows(inv_dicts)
        # Also write method2 / subset beside it
        stem = args.write_csv.with_suffix('')
        if m2_cross_rows or m2_same_rows:
            m2_path = Path(str(stem) + '_method2.csv')
            m2_dicts = [asdict(r) for r in (m2_cross_rows + m2_same_rows)]
            with m2_path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(m2_dicts[0].keys()))
                writer.writeheader()
                writer.writerows(m2_dicts)
            print(f'Wrote {m2_path}')
        if subset_rows:
            s_path = Path(str(stem) + '_method345.csv')
            s_dicts = [asdict(r) for r in subset_rows]
            with s_path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(s_dicts[0].keys()))
                writer.writeheader()
                writer.writerows(s_dicts)
            print(f'Wrote {s_path}')
        print(f'Wrote CSV {args.write_csv}')

    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'summary': summary,
            'inventory': inv_dicts,
            'method2': [asdict(r) for r in (m2_cross_rows + m2_same_rows)],
            'method345': [asdict(r) for r in subset_rows],
        }
        args.write_json.write_text(json.dumps(payload, indent=2))
        print(f'Wrote JSON {args.write_json}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
