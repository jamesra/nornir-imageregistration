"""Per-cell Role and pass-level FieldMode for STOS grid refine.

Theory: docs/grid16_stos_cell_role_theory.md

Role decision (one pass)::

    REJECT  <- low content OR known peak_ratio < PEAK_RATIO_MIN
    FREE    <- PC-pass, not lock-candidate (ZNCC deferred)
    LOCKABLE / IDENTITY_SUSPECT <- lock-candidate + (field suspect OR ZNCC)

Field consistency (cold-half under ASYMMETRIC, or active unique neighbor) brands
``IDENTITY_SUSPECT`` even when ZNCC passes. Absolute ZNCC remains a secondary gate.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.refine_shared.coherent_residual import (
    COHERENCE_MIN,
    INLIER_COS_MIN,
    LOCK_FRAC_TRIGGER,
    MIN_UNIQUE_PEAKS,
    estimate_coherent_residual_translation,
)
from nornir_imageregistration.refine_shared.failure_mode_stats import peak_direction_coherence
from nornir_imageregistration.refine_shared.peak_ratio_gates import (
    PEAK_RATIO_EARLY,
    PEAK_RATIO_MIN,
    finite_peak_ratio,
    is_ambiguous_peak,
)
from nornir_imageregistration.refine_shared.runtime_config import get_runtime_config

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp

# Default min masked ZNCC for lock-candidate PC-pass cells.
# Override: NORNIR_REFINE_IDENTITY_ZNCC_MIN (unset = this default).
# Below → IDENTITY_SUSPECT (never lock); at/above → may be LOCKABLE.
DEFAULT_IDENTITY_ZNCC_MIN: float = 0.25


class Role(enum.IntEnum):
    """Per-cell mesh / lock role for one refine pass."""

    REJECT = 0
    FREE = 1
    LOCKABLE = 2
    IDENTITY_SUSPECT = 3


class RejectReason(enum.IntEnum):
    """Why a cell is REJECT (or NONE when not rejected)."""

    NONE = 0
    PEAK_AMBIGUOUS = 1
    LOW_CONTENT = 2


class FieldMode(enum.IntEnum):
    """Pass-level aggregate mode."""

    LOCAL = 0
    RIGID_RESIDUAL = 1
    ASYMMETRIC = 2


class _AlignmentRecordLike(Protocol):
    ID: tuple[int, int]
    peak: NDArray[np.floating]
    weight: float
    peak_ratio: float | None


@dataclass(frozen=True)
class RoleClassificationResult:
    """Aligned Role / reject / ZNCC metadata for one pass's free records."""

    roles: list[Role]
    reject_reasons: list[RejectReason]
    lock_candidate: NDArray[np.bool_]
    zncc: NDArray[np.float64]
    field_mode: FieldMode
    n_reject: int = 0
    n_free: int = 0
    n_lockable: int = 0
    n_identity_suspect: int = 0
    n_peak_ambiguous: int = 0
    n_low_content: int = 0
    n_lock_cand: int = 0
    n_zncc_eval: int = 0
    n_zncc_pass: int = 0
    n_zncc_fail: int = 0
    identity_zncc_min: float = DEFAULT_IDENTITY_ZNCC_MIN
    role_by_id: dict[tuple[int, int], Role] = field(default_factory=dict)


def identity_zncc_min_threshold() -> float:
    """Return the ZNCC lock bar (env ``NORNIR_REFINE_IDENTITY_ZNCC_MIN`` or default).

    Meaning
        Minimum masked zero-mean normalized cross-correlation between fixed and
        moving ROIs at the claimed peak offset for a lock-candidate cell that
        already passed PhaseCorrelation (``peak_ratio >= PEAK_RATIO_MIN``).

    Default
        ``DEFAULT_IDENTITY_ZNCC_MIN`` (0.25) when the env is unset or invalid.

    Valid values
        Finite float. Typical useful range is roughly ``0.0``–``1.0``.

    Effect
        Score ``<`` threshold → ``Role.IDENTITY_SUSPECT`` (mesh OK, never lock).
        Score ``>=`` threshold → eligible for ``Role.LOCKABLE`` (with stability).

    Reads the cached config; refine entry points refresh once per pass. See
    ``low_content_std_min_threshold`` for why ``refresh=True`` is not used here.
    """
    return float(get_runtime_config().identity_zncc_min)


def masked_zncc(
        fixed: NDArray,
        moving: NDArray,
        *,
        mask: NDArray[np.bool_] | None = None,
) -> float:
    """Return masked zero-mean normalized cross-correlation of two ROIs.

    Arrays stay on their input backend (``xp``). Returns ``0.0`` when variance
    is degenerate or fewer than two valid pixels remain.
    """
    if fixed is None or moving is None:
        return 0.0
    xp = cp.get_array_module(fixed)
    a = xp.asarray(fixed, dtype=xp.float64)
    b = xp.asarray(moving, dtype=xp.float64)
    if a.shape != b.shape or a.size == 0:
        return 0.0

    if mask is None:
        valid = xp.ones(a.shape, dtype=bool)
        # Ignore NaNs if present.
        valid = valid & xp.isfinite(a) & xp.isfinite(b)
    else:
        valid = xp.asarray(mask, dtype=bool) & xp.isfinite(a) & xp.isfinite(b)

    n = int(xp.count_nonzero(valid))
    if n < 2:
        return 0.0

    av = a[valid]
    bv = b[valid]
    av = av - xp.mean(av)
    bv = bv - xp.mean(bv)
    denom = float(xp.sqrt(xp.sum(av * av) * xp.sum(bv * bv)))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0
    score = float(xp.sum(av * bv) / denom)
    if not np.isfinite(score):
        return 0.0
    return score


def free_peak_half_stats(
        records: Sequence[_AlignmentRecordLike],
        *,
        travel_eps: float = 0.5,
) -> dict[str, float | dict[str, float]]:
    """Half stats from free measured peaks only (exclude baked travel≈0 locks).

    Returns keys ``low_x`` / ``high_x`` (each with ``n``, ``travel_med``,
    ``identity_frac``) and scalar ``mid_source_x``.
    """
    empty_half = {'n': 0.0, 'travel_med': float('nan'), 'identity_frac': 0.0}
    if len(records) == 0:
        return {
            'low_x': dict(empty_half),
            'high_x': dict(empty_half),
            'mid_source_x': float('nan'),
        }

    source_x = np.asarray(
        [float(np.asarray(getattr(r, 'SourcePoint'), dtype=np.float64).reshape(2)[1])
         for r in records],
        dtype=np.float64)
    travel = np.asarray(
        [float(np.linalg.norm(np.asarray(r.peak, dtype=np.float64).reshape(2)))
         for r in records],
        dtype=np.float64)
    mid = float(np.median(source_x))
    out: dict[str, float | dict[str, float]] = {'mid_source_x': mid}
    for name, mask in (('low_x', source_x <= mid), ('high_x', source_x > mid)):
        if not np.any(mask):
            out[name] = dict(empty_half)
            continue
        t = travel[mask]
        out[name] = {
            'n': float(mask.sum()),
            'travel_med': float(np.median(t)),
            'identity_frac': float(np.mean(t < float(travel_eps))),
        }
    return out


def classify_field(
        records: Sequence[_AlignmentRecordLike],
        lock_fraction: float,
        *,
        max_travel: float,
        travel_eps: float = 0.5,
        locked_records: Sequence[_AlignmentRecordLike] | None = None,
) -> FieldMode:
    """Detect pass-level FieldMode from lock fraction and free peaks.

    ``RIGID_RESIDUAL`` when coherent residual translation would trigger.
    ``ASYMMETRIC`` when free-peak travel medians disagree across low-x / high-x
    halves (one half still sliding, the other near identity). Otherwise ``LOCAL``.
    """
    del locked_records  # Half detection uses free measured peaks only.
    residual = estimate_coherent_residual_translation(records, lock_fraction=lock_fraction)
    if residual is not None:
        return FieldMode.RIGID_RESIDUAL

    if len(records) < 4:
        return FieldMode.LOCAL

    halves = free_peak_half_stats(records, travel_eps=travel_eps)
    low = halves.get('low_x', {})
    high = halves.get('high_x', {})
    if not isinstance(low, dict) or not isinstance(high, dict):
        return FieldMode.LOCAL
    if float(low.get('n', 0)) < 1 or float(high.get('n', 0)) < 1:
        return FieldMode.LOCAL

    active_bar = float(max_travel) * 0.5
    low_med = float(low.get('travel_med', float('nan')))
    high_med = float(high.get('travel_med', float('nan')))
    if not (np.isfinite(low_med) and np.isfinite(high_med)):
        return FieldMode.LOCAL

    low_active = low_med > active_bar
    high_active = high_med > active_bar
    low_cold = low_med <= float(travel_eps) or float(low.get('identity_frac', 0.0)) >= 0.5
    high_cold = high_med <= float(travel_eps) or float(high.get('identity_frac', 0.0)) >= 0.5
    if (low_active and high_cold) or (high_active and low_cold):
        return FieldMode.ASYMMETRIC
    return FieldMode.LOCAL


def cold_half_source_x_threshold(
        records: Sequence[_AlignmentRecordLike],
        *,
        max_travel: float,
        travel_eps: float = 0.5,
) -> tuple[float, str] | None:
    """Return ``(mid_source_x, cold_side)`` when free halves are ASYMMETRIC.

    ``cold_side`` is ``'low_x'`` or ``'high_x'``. ``None`` when not asymmetric.
    """
    if len(records) < 4:
        return None
    halves = free_peak_half_stats(records, travel_eps=travel_eps)
    low = halves.get('low_x', {})
    high = halves.get('high_x', {})
    mid_raw = halves.get('mid_source_x', float('nan'))
    mid = float(mid_raw) if not isinstance(mid_raw, dict) else float('nan')
    if not isinstance(low, dict) or not isinstance(high, dict):
        return None
    if not np.isfinite(mid):
        return None
    active_bar = float(max_travel) * 0.5
    low_med = float(low.get('travel_med', float('nan')))
    high_med = float(high.get('travel_med', float('nan')))
    if not (np.isfinite(low_med) and np.isfinite(high_med)):
        return None
    low_active = low_med > active_bar
    high_active = high_med > active_bar
    low_cold = low_med <= float(travel_eps) or float(low.get('identity_frac', 0.0)) >= 0.5
    high_cold = high_med <= float(travel_eps) or float(high.get('identity_frac', 0.0)) >= 0.5
    if low_active and high_cold:
        return mid, 'high_x'
    if high_active and low_cold:
        return mid, 'low_x'
    return None


def _active_unique_neighbor_keys(
        records: Sequence[_AlignmentRecordLike],
        *,
        max_travel: float,
        peak_ratio_min: float,
) -> set[tuple[int, int]]:
    """Grid IDs with unique peaks still traveling above half the travel bar."""
    active_travel_min = float(max_travel) * 0.5
    keys: set[tuple[int, int]] = set()
    for record in records:
        ratio = finite_peak_ratio(record)
        if ratio is None or float(ratio) < float(peak_ratio_min):
            continue
        travel = float(np.linalg.norm(np.asarray(record.peak, dtype=np.float64).reshape(2)))
        if travel > active_travel_min:
            keys.add((int(record.ID[0]), int(record.ID[1])))  # type: ignore[arg-type]
    return keys


def _has_active_unique_neighbor(
        key: tuple[int, int],
        active_unique_keys: set[tuple[int, int]],
) -> bool:
    r, c = int(key[0]), int(key[1])
    for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if neighbor in active_unique_keys:
            return True
    return False


# Min unique traveling cells whose median travel must exceed half max_travel
# before FOV-wide identity branding (refuse travel≈0 locks while field still moves).
ACTIVE_UNIQUE_FIELD_MIN: int = 20


def _unique_traveling_travels(
        records: Sequence[_AlignmentRecordLike],
        *,
        min_travel: float,
        peak_ratio_min: float,
) -> NDArray[np.float64]:
    """Travel magnitudes for unique traveling peaks."""
    vals: list[float] = []
    for record in records:
        ratio = finite_peak_ratio(record)
        if ratio is None or float(ratio) < float(peak_ratio_min):
            continue
        travel = float(np.linalg.norm(np.asarray(record.peak, dtype=np.float64).reshape(2)))
        if travel >= float(min_travel):
            vals.append(travel)
    if not vals:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(vals, dtype=np.float64)


def active_unique_field_is_hot(
        records: Sequence[_AlignmentRecordLike],
        *,
        max_travel: float,
        peak_ratio_min: float = PEAK_RATIO_MIN,
        min_unique: int = ACTIVE_UNIQUE_FIELD_MIN,
        min_travel: float = 1.0,
) -> bool:
    """True when enough unique peaks still travel above half the travel bar.

    Generic field-consistency gate: do not identity-lock while the measured
    displacement field is still actively moving (healthy free-unique travel
    med ~2 px stays cold; bubble / path pairs with ~15–40 px stay hot).
    """
    travels = _unique_traveling_travels(
        records, min_travel=min_travel, peak_ratio_min=peak_ratio_min)
    if int(travels.shape[0]) < int(min_unique):
        return False
    return float(np.median(travels)) > float(max_travel) * 0.5


def unique_large_travel_raw_preserve_ids(
        records: Sequence[_AlignmentRecordLike],
        *,
        max_travel: float,
        peak_ratio_min: float = PEAK_RATIO_MIN,
        travel_frac: float = 0.5,
) -> set[tuple[int, int]]:
    """IDs of unique free cells with travel above ``travel_frac * max_travel``.

    Callers union these with soft-disc ids and coherent disc-front ids for mesh /
    anchor-smooth raw peak preserve so large unique residuals are not overwritten
    by identity-anchored smooth. Lock soft floors stay on ratio-eligible soft-disc
    only.
    """
    bar = float(max_travel) * float(travel_frac)
    ids: set[tuple[int, int]] = set()
    for record in records:
        ratio = finite_peak_ratio(record)
        if ratio is None or float(ratio) < float(peak_ratio_min):
            continue
        travel = float(np.linalg.norm(np.asarray(record.peak, dtype=np.float64).reshape(2)))
        if travel > bar:
            ids.add((int(record.ID[0]), int(record.ID[1])))  # type: ignore[arg-type]
    return ids


# Always-on coherent tear/fold front mesh preserve (no peak_ratio floor).
DISC_FRONT_MIN_CELLS: int = 6
DISC_FRONT_INLIER_COS: float = float(INLIER_COS_MIN)
DISC_FRONT_COHERENCE_MIN: float = 0.70
_DISC_FRONT_MIN_TRAVEL: float = 1.0


def _connected_components_4(keys: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Return 4-connected components covering *keys*."""
    remaining = set(keys)
    components: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(component)
    return components


def coherent_discontinuity_raw_preserve_ids(
        records: Sequence[_AlignmentRecordLike],
        discontinuity_ids: set[tuple[int, int]],
        *,
        max_travel: float,
        min_cells: int = DISC_FRONT_MIN_CELLS,
        inlier_cos_min: float = DISC_FRONT_INLIER_COS,
        coherence_min: float = DISC_FRONT_COHERENCE_MIN,
) -> set[tuple[int, int]]:
    """Active disc cells that form a spatially coherent tear/fold front.

    Mesh / anchor-smooth may keep raw peaks for these IDs even when
    ``peak_ratio < PEAK_RATIO_MIN`` (tear fronts often sit near 1.03). Isolated
    or bimodal wrap-like disc dirt stays out — this is **not** all-disc preserve.

    A cell is eligible only when it is:

    - discontinuity-tagged with travel ``> max_travel`` (active disc),
    - in a 4-connected active-disc component of size ``>= min_cells``,
    - in a component whose ``peak_direction_coherence >= coherence_min``,
    - an inlier of that component's median peak direction (unit-dot
      ``>= inlier_cos_min``).

    Lock soft floors must **not** use this set; they stay on
    ``soft_discontinuity_ids`` (``pr >= PEAK_RATIO_MIN``) only.
    """
    active = _active_disc_neighbor_keys(
        records, discontinuity_ids, max_travel=max_travel)
    if len(active) < int(min_cells):
        return set()

    peak_by_id: dict[tuple[int, int], NDArray[np.float64]] = {}
    for record in records:
        key = (int(record.ID[0]), int(record.ID[1]))  # type: ignore[arg-type]
        if key not in active:
            continue
        peak_by_id[key] = np.asarray(record.peak, dtype=np.float64).reshape(2)

    preserved: set[tuple[int, int]] = set()
    for component in _connected_components_4(active):
        if len(component) < int(min_cells):
            continue
        ordered = [key for key in component if key in peak_by_id]
        if len(ordered) < int(min_cells):
            continue
        peaks = np.asarray([peak_by_id[key] for key in ordered], dtype=np.float64)
        coherence = peak_direction_coherence(
            peaks[:, 0],
            peaks[:, 1],
            min_travel=_DISC_FRONT_MIN_TRAVEL,
        )
        if coherence < float(coherence_min):
            continue

        norms = np.linalg.norm(peaks, axis=1)
        traveling = norms >= float(_DISC_FRONT_MIN_TRAVEL)
        if not np.any(traveling):
            continue
        median_peak = np.median(peaks[traveling], axis=0)
        median_norm = float(np.linalg.norm(median_peak))
        if median_norm < float(_DISC_FRONT_MIN_TRAVEL):
            mean_unit = (peaks[traveling] / norms[traveling, None]).mean(axis=0)
            mean_norm = float(np.linalg.norm(mean_unit))
            if mean_norm <= 0.0:
                continue
            ref = mean_unit / mean_norm
        else:
            ref = median_peak / median_norm

        for key, peak, norm in zip(ordered, peaks, norms):
            if float(norm) < float(_DISC_FRONT_MIN_TRAVEL):
                continue
            unit = peak / norm
            if float(np.dot(unit, ref)) >= float(inlier_cos_min):
                preserved.add(key)
    return preserved


def _active_disc_neighbor_keys(
        records: Sequence[_AlignmentRecordLike],
        discontinuity_ids: set[tuple[int, int]],
        *,
        max_travel: float,
) -> set[tuple[int, int]]:
    """Disc-tagged cells whose peak travel still exceeds ``max_travel``."""
    if not discontinuity_ids:
        return set()
    active: set[tuple[int, int]] = set()
    for record in records:
        key = (int(record.ID[0]), int(record.ID[1]))  # type: ignore[arg-type]
        if key not in discontinuity_ids:
            continue
        travel = float(np.linalg.norm(np.asarray(record.peak, dtype=np.float64).reshape(2)))
        if travel > float(max_travel):
            active.add(key)
    return active


def field_brand_identity_suspect_ids(
        records: Sequence[_AlignmentRecordLike],
        *,
        field_mode: FieldMode,
        max_travel: float,
        travel_eps: float = 0.5,
        peak_ratio_min: float = PEAK_RATIO_MIN,
        discontinuity_ids: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]]:
    """IDs that field consistency brands IDENTITY_SUSPECT (before ZNCC).

    Brands near-zero-travel cells when:

    - a 4-connected unique neighbor is still traveling, or
    - a 4-connected **active disc** neighbor (discontinuity tag with travel
      ``> max_travel``) exists — refuses identity freeze beside a tear/fold, or
    - cold-half under ``ASYMMETRIC``, or
    - the FOV unique field is still hot (median unique travel ``> 0.5*max_travel``
      with enough unique peaks) — refuses identity locks while the displacement
      field is still active anywhere.
    """
    if len(records) == 0:
        return set()

    active_unique = _active_unique_neighbor_keys(
        records, max_travel=max_travel, peak_ratio_min=peak_ratio_min)
    active_disc = _active_disc_neighbor_keys(
        records,
        discontinuity_ids or set(),
        max_travel=max_travel,
    )
    cold: tuple[float, str] | None = None
    if field_mode == FieldMode.ASYMMETRIC:
        cold = cold_half_source_x_threshold(
            records, max_travel=max_travel, travel_eps=travel_eps)
    field_hot = active_unique_field_is_hot(
        records, max_travel=max_travel, peak_ratio_min=peak_ratio_min)

    branded: set[tuple[int, int]] = set()
    for record in records:
        key = (int(record.ID[0]), int(record.ID[1]))  # type: ignore[arg-type]
        travel = float(np.linalg.norm(np.asarray(record.peak, dtype=np.float64).reshape(2)))
        if travel >= float(travel_eps):
            continue
        if field_hot:
            branded.add(key)
            continue
        if _has_active_unique_neighbor(key, active_unique):
            branded.add(key)
            continue
        if _has_active_unique_neighbor(key, active_disc):
            branded.add(key)
            continue
        if cold is not None:
            mid, side = cold
            sx = float(np.asarray(getattr(record, 'SourcePoint'), dtype=np.float64).reshape(2)[1])
            on_cold = (sx > mid) if side == 'high_x' else (sx <= mid)
            if on_cold:
                branded.add(key)
    return branded


def classify_roles(
        records: Sequence[_AlignmentRecordLike],
        *,
        transform_cutoff: float,
        max_travel: float,
        per_record_max_travel: NDArray[np.floating] | None = None,
        soft_weight_cutoff: float | None = None,
        discontinuity_ids: set[tuple[int, int]] | None = None,
        zncc_by_id: Mapping[tuple[int, int], float] | None = None,
        low_content_ids: set[tuple[int, int]] | None = None,
        field_mode: FieldMode = FieldMode.LOCAL,
        field_suspect_ids: set[tuple[int, int]] | None = None,
        identity_zncc_min: float | None = None,
        peak_ratio_min: float = PEAK_RATIO_MIN,
        travel_eps: float = 0.5,
) -> RoleClassificationResult:
    """Classify each free record into a Role for this pass.

    Lock candidates are ``IDENTITY_SUSPECT`` when field consistency brands them
    (cold-half / active neighbor) **or** secondary ZNCC fails / is missing.
    ``field_suspect_ids`` may be precomputed to skip ZNCC extract for those IDs.
    """
    n = len(records)
    zncc_by_id = zncc_by_id or {}
    low_content_ids = low_content_ids or set()
    discontinuity_ids = discontinuity_ids or set()
    zncc_min = float(identity_zncc_min if identity_zncc_min is not None else identity_zncc_min_threshold())
    if field_suspect_ids is None:
        field_suspect_ids = field_brand_identity_suspect_ids(
            records,
            field_mode=field_mode,
            max_travel=max_travel,
            travel_eps=travel_eps,
            peak_ratio_min=peak_ratio_min,
        )

    roles: list[Role] = [Role.FREE] * n
    reasons: list[RejectReason] = [RejectReason.NONE] * n
    lock_cand = np.zeros(n, dtype=bool)
    zncc_arr = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        return RoleClassificationResult(
            roles=roles,
            reject_reasons=reasons,
            lock_candidate=lock_cand,
            zncc=zncc_arr,
            field_mode=field_mode,
            identity_zncc_min=zncc_min,
        )

    if per_record_max_travel is not None:
        travel_limits = np.asarray(per_record_max_travel, dtype=np.float64).reshape(-1)
        if travel_limits.shape[0] != n:
            raise ValueError('per_record_max_travel must match records length')
    else:
        travel_limits = np.full(n, float(max_travel), dtype=np.float64)

    weight_cutoffs = np.full(n, float(transform_cutoff), dtype=np.float64)
    if soft_weight_cutoff is not None and discontinuity_ids:
        soft = float(soft_weight_cutoff)
        for i, record in enumerate(records):
            key = tuple(record.ID)  # type: ignore[arg-type]
            if key not in discontinuity_ids:
                continue
            ratio = finite_peak_ratio(record)
            if ratio is not None and float(ratio) >= float(peak_ratio_min):
                weight_cutoffs[i] = soft

    n_peak_amb = 0
    n_low = 0
    n_zncc_eval = 0
    n_zncc_pass = 0
    n_zncc_fail = 0
    role_by_id: dict[tuple[int, int], Role] = {}

    for i, record in enumerate(records):
        key = (int(record.ID[0]), int(record.ID[1]))  # type: ignore[arg-type]
        peak = np.asarray(record.peak, dtype=np.float64).reshape(2)
        travel = float(np.linalg.norm(peak))
        weight = float(record.weight)
        ratio = finite_peak_ratio(record)

        if key in low_content_ids:
            roles[i] = Role.REJECT
            reasons[i] = RejectReason.LOW_CONTENT
            n_low += 1
            role_by_id[key] = Role.REJECT
            continue

        if is_ambiguous_peak(ratio, min_ratio=peak_ratio_min):
            roles[i] = Role.REJECT
            reasons[i] = RejectReason.PEAK_AMBIGUOUS
            n_peak_amb += 1
            role_by_id[key] = Role.REJECT
            continue

        # PC-pass (or unknown ratio — legacy-safe): check lock candidacy.
        is_cand = (weight >= float(weight_cutoffs[i])) and (travel <= float(travel_limits[i]))
        lock_cand[i] = is_cand
        if not is_cand:
            roles[i] = Role.FREE
            role_by_id[key] = Role.FREE
            continue

        # Field consistency brands identity suspects without needing ZNCC.
        if key in field_suspect_ids:
            roles[i] = Role.IDENTITY_SUSPECT
            role_by_id[key] = Role.IDENTITY_SUSPECT
            if key in zncc_by_id:
                score = float(zncc_by_id[key])
                zncc_arr[i] = score
                n_zncc_eval += 1
                if np.isfinite(score) and score >= zncc_min:
                    n_zncc_pass += 1
                else:
                    n_zncc_fail += 1
            continue

        # Secondary ZNCC at lock candidacy.
        if key in zncc_by_id:
            score = float(zncc_by_id[key])
            zncc_arr[i] = score
            n_zncc_eval += 1
            if np.isfinite(score) and score >= zncc_min:
                roles[i] = Role.LOCKABLE
                n_zncc_pass += 1
                role_by_id[key] = Role.LOCKABLE
            else:
                roles[i] = Role.IDENTITY_SUSPECT
                n_zncc_fail += 1
                role_by_id[key] = Role.IDENTITY_SUSPECT
        else:
            # Missing ZNCC → never lock (fail closed).
            roles[i] = Role.IDENTITY_SUSPECT
            n_zncc_fail += 1
            role_by_id[key] = Role.IDENTITY_SUSPECT

    n_reject = n_peak_amb + n_low
    n_free = sum(1 for r in roles if r == Role.FREE)
    n_lockable = sum(1 for r in roles if r == Role.LOCKABLE)
    n_suspect = sum(1 for r in roles if r == Role.IDENTITY_SUSPECT)

    return RoleClassificationResult(
        roles=roles,
        reject_reasons=reasons,
        lock_candidate=lock_cand,
        zncc=zncc_arr,
        field_mode=field_mode,
        n_reject=n_reject,
        n_free=n_free,
        n_lockable=n_lockable,
        n_identity_suspect=n_suspect,
        n_peak_ambiguous=n_peak_amb,
        n_low_content=n_low,
        n_lock_cand=int(np.count_nonzero(lock_cand)),
        n_zncc_eval=n_zncc_eval,
        n_zncc_pass=n_zncc_pass,
        n_zncc_fail=n_zncc_fail,
        identity_zncc_min=zncc_min,
        role_by_id=role_by_id,
    )


def exclude_reject_mesh_records(
        records: Sequence[_AlignmentRecordLike],
        roles: Sequence[Role],
        *,
        min_keep: int = 3,
) -> tuple[list, int]:
    """Drop REJECT records from the mesh input; emergency fill by lowest travel."""
    records_list = list(records)
    if not records_list:
        return records_list, 0
    if len(roles) != len(records_list):
        raise ValueError('roles must align with records')

    clear: list = []
    rejected: list = []
    for record, role in zip(records_list, roles):
        if role == Role.REJECT:
            rejected.append(record)
        else:
            clear.append(record)

    if len(clear) >= int(min_keep):
        return clear, len(rejected)

    if not rejected:
        return clear, 0

    def _travel(rec: object) -> float:
        peak = getattr(rec, 'peak', None)
        if peak is None:
            return float('inf')
        return float(np.linalg.norm(np.asarray(peak, dtype=np.float64).reshape(2)))

    rejected_sorted = sorted(rejected, key=_travel)
    need = max(0, int(min_keep) - len(clear))
    kept = clear + rejected_sorted[:need]
    dropped = len(rejected) - len(rejected_sorted[:need])
    return kept, dropped


# Re-export constants used by callers / docs.
__all__ = [
    'DEFAULT_IDENTITY_ZNCC_MIN',
    'Role',
    'RejectReason',
    'FieldMode',
    'RoleClassificationResult',
    'identity_zncc_min_threshold',
    'masked_zncc',
    'free_peak_half_stats',
    'classify_field',
    'cold_half_source_x_threshold',
    'field_brand_identity_suspect_ids',
    'unique_large_travel_raw_preserve_ids',
    'coherent_discontinuity_raw_preserve_ids',
    'DISC_FRONT_MIN_CELLS',
    'DISC_FRONT_INLIER_COS',
    'DISC_FRONT_COHERENCE_MIN',
    'active_unique_field_is_hot',
    'classify_roles',
    'exclude_reject_mesh_records',
    'PEAK_RATIO_MIN',
    'PEAK_RATIO_EARLY',
    'LOCK_FRAC_TRIGGER',
    'COHERENCE_MIN',
    'MIN_UNIQUE_PEAKS',
    'ACTIVE_UNIQUE_FIELD_MIN',
]
