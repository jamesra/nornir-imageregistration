"""Opt-in phase timing for grid refinement passes."""

from __future__ import annotations

import contextlib
import threading
import time
from collections import defaultdict

import nornir_imageregistration
from nornir_imageregistration.refine_shared.runtime_config import _cached_config

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import nornir_imageregistration.cupy_thunk as cp


class RefinePhaseTimer:
    """Accumulate wall time per named phase of grid refinement.

    Detailed FFT/prewarp buckets stay opt-in via ``NORNIR_REFINE_PHASE_TIMING``.
    ``section_wall`` always records (no GPU sync) for coarse STOS pass summaries.
    """

    PHASES = (
        'prewarp', 'grid_build', 'approx_rigid', 'cell_extract', 'fft', 'host_sync',
        'record_assemble', 'regularize', 'apply',
        'low_content_gate', 'classify', 'zncc_secondary',
        'finalize', 'diagnostics_tables', 'diagnostics_heatmaps',
    )

    def __init__(self, enabled: bool | None = None) -> None:
        self._enabled_override: bool | None = None if enabled is None else bool(enabled)
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """True when the detailed phase buckets should record.

        Deferred to ``RefineRuntimeConfig`` rather than latched from the
        environment in ``__init__``. The process-global timer is built at import,
        so latching left a benchmark that exported ``NORNIR_REFINE_PHASE_TIMING``
        afterwards with the config reporting timing on and every bucket empty.

        Reads the cached config, not ``refresh=True``: this is consulted inside
        timed sections, and refreshing re-reads the whole environment. Callers pick
        up a late change the documented way, by refreshing the config once.

        Calls ``_cached_config`` rather than ``get_runtime_config`` to skip a frame
        and the ``refresh`` branch; sections run per vertex in the serial mesh loop.
        """
        if self._enabled_override is not None:
            return self._enabled_override
        return _cached_config().phase_timing

    @enabled.setter
    def enabled(self, value: bool | None) -> None:
        """Pin the flag, or pass ``None`` to follow the runtime config again."""
        self._enabled_override = None if value is None else bool(value)

    def reset(self) -> None:
        """Clear all accumulated phase totals and counts."""
        with self._lock:
            self.totals = defaultdict(float)
            self.counts = defaultdict(int)

    def snapshot(self) -> dict[str, float]:
        """Return a copy of the current cumulative per-phase totals."""
        return dict(self.totals)

    def add(self, name: str, elapsed: float) -> None:
        """Add an externally measured elapsed time into *name*."""
        if elapsed < 0:
            return
        with self._lock:
            self.totals[name] += float(elapsed)
            self.counts[name] += 1

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

    @contextlib.contextmanager
    def section_wall(self, name: str):
        """Always time the wrapped block (no GPU sync) for coarse STOS profiling."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                self.totals[name] += elapsed
                self.counts[name] += 1


_PHASE_TIMER = RefinePhaseTimer()


def get_phase_timer() -> RefinePhaseTimer:
    """Return the process-global refine phase timer."""
    return _PHASE_TIMER
