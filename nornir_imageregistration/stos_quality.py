"""STOS pair quality scores (full-image ZNCC) and folder-local JSON cache."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.assemble import ParameterToStosTransform, SourceImageToTargetSpace
from nornir_imageregistration.files.stosfile import StosFile
from nornir_imageregistration.refine_shared.cell_roles import Role, masked_zncc
from nornir_shared.files import file_mtime_ns
from nornir_shared.histogram import Histogram

QUALITY_CACHE_FILENAME: str = 'stos_quality.json'
QUALITY_CACHE_VERSION: int = 1
DEFAULT_MAX_SIDE: int = 2048
DEFAULT_HIST_BINS: int = 50
DEFAULT_HIST_MIN: float = -1.0
DEFAULT_HIST_MAX: float = 1.0


@dataclass
class PairZnccResult:
    """Result of a full-pair ZNCC score for one ``.stos`` file."""

    pair_zncc: float
    downsample: float
    stos_checksum: str
    stos_mtime_ns: int
    max_side: int
    scored_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class QualityCache:
    """Folder-local map of relative ``.stos`` paths to quality entries."""

    version: int = QUALITY_CACHE_VERSION
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)


def quality_cache_path(group_folder: str) -> str:
    """Return the path of ``stos_quality.json`` under *group_folder*."""
    return os.path.join(group_folder, QUALITY_CACHE_FILENAME)


def cache_key_for_stos(group_folder: str, stos_path: str) -> str:
    """Return a forward-slash relative key for *stos_path* under *group_folder*."""
    rel = os.path.relpath(os.path.abspath(stos_path), os.path.abspath(group_folder))
    return rel.replace('\\', '/')


def load_quality_cache(group_folder: str) -> QualityCache:
    """Load ``stos_quality.json`` from *group_folder*, or an empty cache."""
    path = quality_cache_path(group_folder)
    if not os.path.isfile(path):
        return QualityCache()
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return QualityCache()
    if not isinstance(data, dict):
        return QualityCache()
    entries = data.get('entries', {})
    if not isinstance(entries, dict):
        entries = {}
    version = int(data.get('version', QUALITY_CACHE_VERSION))
    return QualityCache(version=version, entries=dict(entries))


def save_quality_cache(group_folder: str, cache: QualityCache) -> str:
    """Write *cache* to ``stos_quality.json`` under *group_folder*."""
    os.makedirs(group_folder, exist_ok=True)
    path = quality_cache_path(group_folder)
    payload = {
        'version': int(cache.version),
        'entries': cache.entries,
    }
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)
    return path


def entry_is_stale(entry: Mapping[str, Any] | None, stos_path: str) -> bool:
    """Return True when *entry* is missing, incomplete, or out of sync with *stos_path*."""
    if entry is None:
        return True
    if not os.path.isfile(stos_path):
        return True
    if entry.get('pair_zncc') is None:
        return True

    stored_checksum = entry.get('stos_checksum')
    if stored_checksum:
        try:
            current = StosFile.LoadChecksum(stos_path)
        except Exception:
            current = None
        if current is None or str(current) != str(stored_checksum):
            return True
        return False

    stored_mtime = entry.get('stos_mtime_ns')
    try:
        current_mtime = file_mtime_ns(stos_path)
    except OSError:
        return True
    if stored_mtime is None:
        return True
    return int(stored_mtime) != int(current_mtime)


def _to_numpy(image: NDArray) -> NDArray[np.floating]:
    """Return a host float array for *image* (CuPy-safe)."""
    host = np.asarray(getattr(image, 'get', lambda: image)(), dtype=np.float64)
    return host


def _load_optional_mask(path: str | None, shape: tuple[int, ...]) -> NDArray[np.bool_] | None:
    """Load a mask image if *path* exists; otherwise return None."""
    if path is None or not os.path.isfile(path):
        return None
    mask_img = nornir_imageregistration.ImageParamToImageArray(path)
    mask = _to_numpy(mask_img)
    target_shape = (int(shape[0]), int(shape[1]))
    if mask.shape[:2] != target_shape:
        sy = float(target_shape[0]) / float(max(mask.shape[0], 1))
        sx = float(target_shape[1]) / float(max(mask.shape[1], 1))
        mask = _to_numpy(nornir_imageregistration.ResizeImage(mask.astype(np.float64, copy=False), (sy, sx)))
    return np.asarray(mask > 0.5, dtype=bool)


def compute_pair_zncc(
        stos_path: str,
        *,
        downsample: float | None = None,
        max_side: int | None = DEFAULT_MAX_SIDE,
) -> PairZnccResult:
    """Warp mapped→control for *stos_path* and return masked full-pair ZNCC.

    Images are optionally downsampled so the longest side is at most *max_side*
    (default 2048). The transform is temporarily scaled for that downsample and
    restored afterward. Scoring uses NumPy on the host.
    """
    if not os.path.isfile(stos_path):
        raise FileNotFoundError(stos_path)

    stos = StosFile.Load(stos_path)
    if stos is None or stos.Transform is None:
        raise ValueError(f'Could not load STOS transform: {stos_path}')

    control = _to_numpy(
        nornir_imageregistration.ImageParamToImageArray(
            stos.ControlImageFullPath,
            dtype=nornir_imageregistration.default_image_dtype(),
        )
    )
    mapped = _to_numpy(
        nornir_imageregistration.ImageParamToImageArray(
            stos.MappedImageFullPath,
            dtype=nornir_imageregistration.default_image_dtype(),
        )
    )

    transform = ParameterToStosTransform(stos_path)
    if transform is None:
        raise ValueError(f'Could not parse transform from {stos_path}')

    stos_downsample = float(stos.Downsample) if stos.Downsample is not None else 1.0
    if downsample is not None and float(downsample) > 0.0:
        # Additional relative scale vs the STOS image resolution.
        requested = float(stos_downsample) / float(downsample)
        scalar = min(1.0, requested) if requested > 0.0 else 1.0
    else:
        scalar = 1.0

    max_dim = int(max_side) if max_side is not None and max_side > 0 else 0
    if max_dim > 0:
        longest = float(max(int(control.shape[0]), int(control.shape[1]), 1))
        clamp = float(max_dim) / longest
        if 0.0 < clamp < 1.0:
            scalar = min(scalar, clamp)

    if 0.0 < scalar < 1.0:
        control = _to_numpy(nornir_imageregistration.ResizeImage(control, scalar))
        mapped = _to_numpy(nornir_imageregistration.ResizeImage(mapped, scalar))
    else:
        scalar = 1.0

    control_mask = _load_optional_mask(stos.ControlMaskFullPath, control.shape)
    mapped_mask = _load_optional_mask(stos.MappedMaskFullPath, mapped.shape)

    scale_fn = getattr(transform, 'Scale', None)
    scaled = False
    if callable(scale_fn) and scalar != 1.0:
        scale_fn(scalar)
        scaled = True

    try:
        area = np.asarray(control.shape[:2], dtype=np.float64)
        warped_result = SourceImageToTargetSpace(
            transform,
            DataToTransform=mapped,
            output_botleft=np.asarray((0.0, 0.0), dtype=np.float64),
            output_area=area,
            extrapolate=True,
            cval=np.nan,
            return_valid_mask=True,
            interpolation_order=1,
        )
        warped, valid_mask = warped_result  # type: ignore[misc]
        warped = _to_numpy(warped)
        valid = np.asarray(getattr(valid_mask, 'get', lambda: valid_mask)(), dtype=bool)

        if mapped_mask is not None:
            warped_mask_result = SourceImageToTargetSpace(
                transform,
                DataToTransform=mapped_mask.astype(np.float64, copy=False),
                output_botleft=np.asarray((0.0, 0.0), dtype=np.float64),
                output_area=area,
                extrapolate=True,
                cval=0.0,
                return_valid_mask=False,
                interpolation_order=0,
            )
            warped_mask = _to_numpy(warped_mask_result) > 0.5
            valid = valid & warped_mask

        if control_mask is not None:
            valid = valid & control_mask

        score = float(masked_zncc(control, warped, mask=valid))
    finally:
        if scaled and callable(scale_fn):
            undo = 1.0 / scalar if scalar != 0.0 else 1.0
            scale_fn(undo)

    try:
        mtime_ns = file_mtime_ns(stos_path)
    except OSError:
        mtime_ns = 0

    # Effective downsample relative to full-resolution section imagery.
    applied_downsample = float(stos_downsample / scalar) if scalar != 0.0 else float(stos_downsample)
    return PairZnccResult(
        pair_zncc=score,
        downsample=applied_downsample,
        stos_checksum=str(stos.Checksum or ''),
        stos_mtime_ns=int(mtime_ns),
        max_side=int(max_dim) if max_dim > 0 else 0,
    )


def pair_result_to_entry(
        result: PairZnccResult,
        *,
        refine: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a :class:`PairZnccResult` into a cache entry dict."""
    entry = asdict(result)
    if refine is not None:
        entry['refine'] = dict(refine)
    return entry


def merge_entry(
        cache: QualityCache,
        key: str,
        *,
        pair: PairZnccResult | None = None,
        refine: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert or update cache entry *key*; returns the updated entry."""
    entry = dict(cache.entries.get(key, {}))
    if pair is not None:
        entry.update(asdict(pair))
    if refine is not None:
        existing_refine = entry.get('refine')
        if isinstance(existing_refine, dict):
            merged = dict(existing_refine)
            merged.update(refine)
            entry['refine'] = merged
        else:
            entry['refine'] = dict(refine)
    cache.entries[key] = entry
    return entry


def refine_summary_from_diagnostics(
        npz_path: str,
        *,
        pass_index: int | None = None,
) -> dict[str, Any] | None:
    """Build a refine summary dict from a ``refine_passNN_diagnostics.npz`` file.

    Uses LOCKABLE roles when present; otherwise locked cells. Returns None when
    the file is missing or unreadable.
    """
    if not os.path.isfile(npz_path):
        return None
    try:
        data = np.load(npz_path)
    except (OSError, ValueError):
        return None

    zncc = np.asarray(data['zncc'], dtype=np.float64) if 'zncc' in data.files else None
    locked = np.asarray(data['locked'], dtype=bool) if 'locked' in data.files else None
    role = np.asarray(data['role'], dtype=np.int64) if 'role' in data.files else None

    summary: dict[str, Any] = {}
    if pass_index is not None:
        summary['pass'] = int(pass_index)
    else:
        # Infer pass from filename refine_passNN_diagnostics.npz when possible.
        base = os.path.basename(npz_path)
        if base.startswith('refine_pass') and '_diagnostics' in base:
            try:
                summary['pass'] = int(base[len('refine_pass'):].split('_', 1)[0])
            except ValueError:
                pass

    n = 0
    if locked is not None:
        n = int(locked.size)
        summary['lock_frac'] = float(np.mean(locked)) if n > 0 else 0.0

    scores: NDArray[np.floating] | None = None
    if zncc is not None:
        if role is not None and role.shape == zncc.shape:
            mask = role == int(Role.LOCKABLE)
            scores = zncc[mask & np.isfinite(zncc)]
        if (scores is None or scores.size == 0) and locked is not None and locked.shape == zncc.shape:
            scores = zncc[locked & np.isfinite(zncc)]
        if scores is None or scores.size == 0:
            scores = zncc[np.isfinite(zncc)]
        if scores is not None and scores.size > 0:
            summary['median_lock_zncc'] = float(np.median(scores))

    return summary if summary else None


def find_latest_refine_diagnostics(directory: str) -> tuple[str, int] | None:
    """Return ``(npz_path, pass_index)`` for the highest refine pass under *directory*."""
    if not os.path.isdir(directory):
        return None
    best: tuple[str, int] | None = None
    for name in os.listdir(directory):
        if not (name.startswith('refine_pass') and name.endswith('_diagnostics.npz')):
            continue
        mid = name[len('refine_pass'):].split('_', 1)[0]
        try:
            pass_index = int(mid)
        except ValueError:
            continue
        path = os.path.join(directory, name)
        if best is None or pass_index > best[1]:
            best = (path, pass_index)
    return best


def score_stos_into_cache(
        group_folder: str,
        stos_path: str,
        *,
        cache: QualityCache | None = None,
        force: bool = False,
        max_side: int | None = DEFAULT_MAX_SIDE,
        downsample: float | None = None,
        refine_diagnostics_dir: str | None = None,
) -> tuple[QualityCache, dict[str, Any], bool]:
    """Ensure *stos_path* has a fresh pair_zncc entry in the group cache.

    Returns ``(cache, entry, computed)`` where *computed* is True when ZNCC ran.
    """
    if cache is None:
        cache = load_quality_cache(group_folder)
    key = cache_key_for_stos(group_folder, stos_path)
    entry = cache.entries.get(key)
    computed = False
    if force or entry_is_stale(entry, stos_path):
        result = compute_pair_zncc(stos_path, downsample=downsample, max_side=max_side)
        entry = merge_entry(cache, key, pair=result)
        computed = True
    else:
        entry = dict(entry or {})

    if refine_diagnostics_dir is not None:
        found = find_latest_refine_diagnostics(refine_diagnostics_dir)
        if found is not None:
            npz_path, pass_index = found
            refine = refine_summary_from_diagnostics(npz_path, pass_index=pass_index)
            if refine is not None:
                entry = merge_entry(cache, key, refine=refine)

    return cache, entry, computed


def non_stale_pair_scores(
        group_folder: str,
        stos_paths: Sequence[str],
        *,
        cache: QualityCache | None = None,
) -> list[float]:
    """Return non-stale ``pair_zncc`` values for *stos_paths* under *group_folder*."""
    if cache is None:
        cache = load_quality_cache(group_folder)
    scores: list[float] = []
    for path in stos_paths:
        key = cache_key_for_stos(group_folder, path)
        entry = cache.entries.get(key)
        if entry_is_stale(entry, path):
            continue
        value = entry.get('pair_zncc') if entry is not None else None
        if value is None:
            continue
        try:
            scores.append(float(value))
        except (TypeError, ValueError):
            continue
    return scores


def build_quality_histogram(
        scores: Sequence[float],
        *,
        min_val: float = DEFAULT_HIST_MIN,
        max_val: float = DEFAULT_HIST_MAX,
        num_bins: int = DEFAULT_HIST_BINS,
) -> Histogram:
    """Build an in-memory :class:`Histogram` from *scores* via ``Init`` + ``Add``."""
    hist = Histogram.Init(minVal=float(min_val), maxVal=float(max_val), numBins=int(num_bins))
    if scores:
        hist.Add([float(v) for v in scores])
    return hist


def attach_scores_to_paths(
        group_folder: str,
        path_by_key: Mapping[str, str | None],
        *,
        cache: QualityCache | None = None,
) -> dict[str, float | None]:
    """Map logical keys to non-stale ``pair_zncc`` (or None) for each path."""
    if cache is None:
        cache = load_quality_cache(group_folder)
    out: dict[str, float | None] = {}
    for name, path in path_by_key.items():
        if path is None or not os.path.isfile(path):
            out[name] = None
            continue
        key = cache_key_for_stos(group_folder, path)
        entry = cache.entries.get(key)
        if entry_is_stale(entry, path):
            out[name] = None
            continue
        value = entry.get('pair_zncc') if entry is not None else None
        try:
            out[name] = float(value) if value is not None else None
        except (TypeError, ValueError):
            out[name] = None
    return out
