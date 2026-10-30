"""Grow STOS refine cell size when a pass finds no usable alignments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from nornir_imageregistration.refine_shared.cell_roles import RoleClassificationResult


def can_grow_cell_size_on_pass(
        *,
        pass_index: int,
        num_iterations: int,
        final_pass: bool,
) -> bool:
    """True when a later planned pass can retry with a different cell size."""
    if bool(final_pass):
        return False
    return int(pass_index) < int(num_iterations)


def pass_found_no_usable_alignments(
        role_result: "RoleClassificationResult | None",
        *,
        n_measured: int,
) -> bool:
    """True when a pass produced no FREE / LOCKABLE / IDENTITY_SUSPECT cells."""
    if int(n_measured) <= 0:
        return True
    if role_result is None:
        return True
    return (
        int(role_result.n_free)
        + int(role_result.n_lockable)
        + int(role_result.n_identity_suspect)
    ) == 0


MIN_REGISTRATIONS_TO_RESTORE_CELL_SIZE = 3


def pass_found_registrations(
        role_result: "RoleClassificationResult | None",
        *,
        n_measured: int,
        min_count: int = MIN_REGISTRATIONS_TO_RESTORE_CELL_SIZE,
) -> bool:
    """True when a pass produced enough FREE / LOCKABLE cells to resume at requested size.

    ``IDENTITY_SUSPECT`` does not count: those cells are not local registrations.
    A single ``LOCKABLE`` cell is enough; otherwise ``min_count`` FREE+LOCKABLE
    are required so one noisy peak does not drop back to a failing ROI size.
    """
    if int(n_measured) <= 0 or role_result is None:
        return False
    n_lockable = int(role_result.n_lockable)
    n_free = int(role_result.n_free)
    if n_lockable >= 1:
        return True
    return (n_free + n_lockable) >= int(min_count)


# Hard ceiling for refine ROIs. Adaptive doubling used to grow to the full
# section (~4096² on downsample-16 TEM), then batched FFT stacked every grid
# vertex as a (N, H, W) complex128 volume. Pyre's Refine dialog already stops
# at 1024; keep the library cap in sync.
MAX_REFINE_CELL_SIZE: int = 1024


def cell_size_exceeds_requested(
        cell_size: NDArray[np.integer] | Sequence[int],
        requested: NDArray[np.integer] | Sequence[int],
) -> bool:
    """True when any axis of *cell_size* is larger than *requested*."""
    current = np.asarray(cell_size, dtype=np.int64).ravel()[:2]
    want = np.asarray(requested, dtype=np.int64).ravel()[:2]
    if current.shape[0] < 2 or want.shape[0] < 2:
        raise ValueError('cell_size and requested must have two entries')
    return bool(np.any(current > want))


def cell_size_cap_from_shapes(
        source_shape: Sequence[int],
        target_shape: Sequence[int],
) -> NDArray[np.int64]:
    """Largest cell ``(y, x)`` that still fits both images and ``MAX_REFINE_CELL_SIZE``."""
    sy = int(source_shape[0])
    sx = int(source_shape[1])
    ty = int(target_shape[0])
    tx = int(target_shape[1])
    return np.asarray(
        (max(1, min(sy, ty, MAX_REFINE_CELL_SIZE)),
         max(1, min(sx, tx, MAX_REFINE_CELL_SIZE))),
        dtype=np.int64)


def clamp_cell_size_to_cap(
        cell_size: NDArray[np.integer] | Sequence[int],
        cap: NDArray[np.integer] | Sequence[int],
) -> NDArray[np.int64]:
    """Return *cell_size* clamped per-axis so it does not exceed *cap*."""
    current = np.asarray(cell_size, dtype=np.int64).ravel()[:2].copy()
    limit = np.asarray(cap, dtype=np.int64).ravel()[:2].copy()
    if current.shape[0] < 2 or limit.shape[0] < 2:
        raise ValueError('cell_size and cap must have two entries')
    current = np.maximum(current, 1)
    limit = np.maximum(limit, 1)
    return np.minimum(current, limit)


def next_cell_size_after_failure(
        cell_size: NDArray[np.integer] | Sequence[int],
        limit: NDArray[np.integer] | Sequence[int],
) -> NDArray[np.int64] | None:
    """Return doubled cell size clamped per-axis to *limit*, or ``None`` if already maxed.

    Never shrinks an axis. An axis already at or above *limit* is left unchanged;
    the other axis may still grow.
    """
    current = np.asarray(cell_size, dtype=np.int64).ravel()[:2].copy()
    cap = np.asarray(limit, dtype=np.int64).ravel()[:2].copy()
    if current.shape[0] < 2 or cap.shape[0] < 2:
        raise ValueError('cell_size and limit must have two entries')
    current = np.maximum(current, 1)
    cap = np.maximum(cap, 1)
    room = cap > current
    if not bool(np.any(room)):
        return None
    grown = current.copy()
    grown[room] = np.minimum(current[room] * 2, cap[room])
    if bool(np.all(grown == current)):
        return None
    return grown
