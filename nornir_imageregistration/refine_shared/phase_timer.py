"""Opt-in phase timing for grid refinement passes."""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections import defaultdict

import nornir_imageregistration

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp


class RefinePhaseTimer:
    """Accumulate wall time per named phase of grid refinement (opt-in).

    Enabled by setting ``NORNIR_REFINE_PHASE_TIMING`` to a truthy value. When
    disabled, ``section`` is a no-op context manager so default runs are
    unaffected.
    """

    PHASES = ('prewarp', 'cell_extract', 'fft', 'host_sync', 'regularize', 'apply')

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            flag = os.environ.get('NORNIR_REFINE_PHASE_TIMING', '0').strip().lower()
            enabled = flag not in ('', '0', 'false', 'no', 'off')
        self.enabled = bool(enabled)
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Clear all accumulated phase totals and counts."""
        with self._lock:
            self.totals = defaultdict(float)
            self.counts = defaultdict(int)

    def snapshot(self) -> dict[str, float]:
        """Return a copy of the current cumulative per-phase totals."""
        return dict(self.totals)

    @contextlib.contextmanager
    def section(self, name: str):
        """Time the wrapped block into the *name* bucket (no-op when disabled)."""
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            if nornir_imageregistration.UsingCupy() and cp is not None:
                try:
                    cp.cuda.Device().synchronize()
                except Exception:  # pragma: no cover
                    pass
            elapsed = time.perf_counter() - start
            with self._lock:
                self.totals[name] += elapsed
                self.counts[name] += 1


_PHASE_TIMER = RefinePhaseTimer()


def get_phase_timer() -> RefinePhaseTimer:
    """Return the process-global refine phase timer."""
    return _PHASE_TIMER
