"""Discontinuity tagging for sharp fold/tear warps in STOS refine."""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from nornir_imageregistration.refine_shared.runtime_config import get_runtime_config


def sharp_warps_enabled() -> bool:
    """True unless ``NORNIR_REFINE_SHARP_WARPS=0`` (default ON).

    Reads the cached config; refine entry points refresh once per pass.
    """
    return get_runtime_config().sharp_warps


def discontinuity_travel_multiplier() -> float:
    """Multiplier of ``max_travel`` for discontinuity neighbor disagreement."""
    raw = os.environ.get('NORNIR_REFINE_DISCONTINUITY_K', '').strip()
    if raw:
        try:
            return max(0.1, float(raw))
        except ValueError:
            pass
    return float(get_runtime_config().discontinuity_k)


def discontinuity_travel_relax() -> float:
    """How much larger travel is allowed for stable discontinuity cells."""
    raw = os.environ.get('NORNIR_REFINE_DISCONTINUITY_TRAVEL_MULT', '').strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return float(get_runtime_config().discontinuity_travel_mult)


def tag_discontinuities(
        records: Sequence,
        *,
        max_travel: float,
        mesh_dims: tuple[int, int] | None = None,
        stable_ids: set[tuple[int, int]] | None = None,
        discontinuity_k: float | None = None,
) -> set[tuple[int, int]]:
    """Mark cells whose raw peak disagrees with the neighbor median.

    A cell is discontinuous when:

    - ``‖peak − neighbor_median‖ > k * max_travel`` (default ``k=1.5``), and
    - if ``stable_ids`` is provided, the cell is in that set (pass-stable).

    Neighbor median uses 8-connected measured neighbors on the grid. Cells with
    fewer than two measured neighbors are never tagged.
    """
    if not sharp_warps_enabled() or not records:
        return set()

    k = float(discontinuity_k if discontinuity_k is not None else discontinuity_travel_multiplier())
    threshold = float(max_travel) * k
    if threshold <= 0:
        return set()

    peaks: dict[tuple[int, int], NDArray[np.float64]] = {}
    for rec in records:
        key = (int(rec.ID[0]), int(rec.ID[1]))
        peaks[key] = np.asarray(rec.peak, dtype=np.float64).reshape(2)

    if mesh_dims is None:
        max_r = max(r for r, _ in peaks.keys())
        max_c = max(c for _, c in peaks.keys())
        mesh_rows, mesh_cols = max_r + 1, max_c + 1
    else:
        mesh_rows, mesh_cols = int(mesh_dims[0]), int(mesh_dims[1])

    tagged: set[tuple[int, int]] = set()
    neighbor_offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    for (row, col), peak in peaks.items():
        if stable_ids is not None and (row, col) not in stable_ids:
            continue
        neighbor_peaks: list[NDArray[np.float64]] = []
        for dr, dc in neighbor_offsets:
            nr, nc = row + dr, col + dc
            if 0 <= nr < mesh_rows and 0 <= nc < mesh_cols and (nr, nc) in peaks:
                neighbor_peaks.append(peaks[(nr, nc)])
        if len(neighbor_peaks) < 2:
            continue
        neighbor_median = np.median(np.stack(neighbor_peaks, axis=0), axis=0)
        if float(np.linalg.norm(peak - neighbor_median)) > threshold:
            tagged.add((row, col))
    return tagged


def per_record_max_travel(
        records: Sequence,
        *,
        base_max_travel: float,
        discontinuity_ids: set[tuple[int, int]],
        relax_mult: float | None = None,
) -> NDArray[np.float64]:
    """Return per-record travel limits (relaxed for discontinuity cells)."""
    mult = float(relax_mult if relax_mult is not None else discontinuity_travel_relax())
    base = float(base_max_travel)
    limits = np.full(len(records), base, dtype=np.float64)
    if not discontinuity_ids or not sharp_warps_enabled():
        return limits
    for i, rec in enumerate(records):
        key = (int(rec.ID[0]), int(rec.ID[1]))
        if key in discontinuity_ids:
            limits[i] = base * mult
    return limits
