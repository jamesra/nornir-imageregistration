"""Measure scheduling for the trusted-mesh refine path."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration


def project_priors(
        transform: Any,
        cell_ids: Sequence[tuple[int, int]],
        source_points: Mapping[tuple[int, int], NDArray[np.floating] | Sequence[float]],
        *,
        locked_ids: set[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], NDArray[np.float64]]:
    """Project all available unlocked source points in one transform call."""
    locked = locked_ids or set()
    keys = [
        key
        for key in cell_ids
        if key not in locked and key in source_points
    ]
    if not keys:
        return {}
    points = np.asarray(
        [np.asarray(source_points[key], dtype=np.float64).reshape(2) for key in keys],
        dtype=np.float64,
    )
    mapped = nornir_imageregistration.EnsureNumpyArray(
        transform.Transform(points)
    ).astype(np.float64, copy=False).reshape(len(keys), 2)
    return {
        key: np.asarray(mapped[index], dtype=np.float64).copy()
        for index, key in enumerate(keys)
    }


def cells_whose_prior_moved(
        cell_ids: Sequence[tuple[int, int]],
        priors: Mapping[tuple[int, int], NDArray[np.floating] | Sequence[float]],
        last_prior: Mapping[tuple[int, int], NDArray[np.floating] | Sequence[float]],
        *,
        eps: float = 0.5,
        pass_index: int = 0,
        locked_ids: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Return unlocked cells that should be measured this pass.

    Pass 0 (or empty *last_prior*) returns every unlocked cell. Later passes
    return cells whose current prior moved more than *eps* since last measure.
    """
    locked_ids = locked_ids or set()
    if pass_index <= 0 or not last_prior:
        return [key for key in cell_ids if key not in locked_ids]

    todo: list[tuple[int, int]] = []
    for key in cell_ids:
        if key in locked_ids:
            continue
        if key not in last_prior:
            todo.append(key)
            continue
        if key not in priors:
            todo.append(key)
            continue
        cur = np.asarray(priors[key], dtype=np.float64).reshape(2)
        prev = np.asarray(last_prior[key], dtype=np.float64).reshape(2)
        if float(np.linalg.norm(cur - prev)) > float(eps):
            todo.append(key)
    return todo


def update_last_prior(
        last_prior: dict[tuple[int, int], NDArray[np.float64]],
        measured_ids: Sequence[tuple[int, int]],
        priors: Mapping[tuple[int, int], NDArray[np.floating] | Sequence[float]],
) -> None:
    """Record the prior used for each measured cell."""
    for key in measured_ids:
        if key not in priors:
            continue
        last_prior[key] = np.asarray(priors[key], dtype=np.float64).reshape(2).copy()
