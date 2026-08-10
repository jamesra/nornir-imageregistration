"""Cooperative cancel and progress callbacks for long registration jobs."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Optional

ProgressCallback = Callable[[int, int, str], None]


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
        label: str) -> None:
    """Invoke *progress_callback* when provided."""
    if progress_callback is None:
        return
    progress_callback(int(current), int(total), str(label))
