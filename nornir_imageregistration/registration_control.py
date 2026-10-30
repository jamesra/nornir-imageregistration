"""Cooperative cancel and progress callbacks for long registration jobs."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Optional

ProgressCallback = Callable[..., None]


class RegistrationCancelled(Exception):
    """Raised when a registration job observes a set cancel event."""


def check_cancelled(cancel_event: Optional[threading.Event]) -> None:
    """Raise :class:`RegistrationCancelled` when *cancel_event* is set."""
    if cancel_event is not None and cancel_event.is_set():
        raise RegistrationCancelled()


def report_progress(
        progress_callback: Optional[ProgressCallback],
        current: int,
        total: int,
        label: str,
        preview: Any = None) -> None:
    """Invoke *progress_callback* when provided.

    *preview* is an optional extra payload (for example a per-pass transform).
    Callables that only accept ``(current, total, label)`` remain supported.
    """
    if progress_callback is None:
        return
    current_i = int(current)
    total_i = int(total)
    label_s = str(label)
    if preview is None:
        progress_callback(current_i, total_i, label_s)
        return
    try:
        progress_callback(current_i, total_i, label_s, preview)
    except TypeError:
        progress_callback(current_i, total_i, label_s)
