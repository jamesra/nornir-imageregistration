"""Consolidate NORNIR_REFINE_* env gates and pool-selection helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


def _env_flag(name: str, default: str = '') -> str:
    return os.environ.get(name, default).strip().lower()


def _is_falsey(flag: str) -> bool:
    return flag in ('0', 'false', 'no', 'off')


def _is_truthy(flag: str) -> bool:
    return flag in ('1', 'true', 'yes', 'on')


@dataclass(frozen=True)
class RefineRuntimeConfig:
    """Process-wide refinement runtime options resolved from the environment.

    Env vars are read when the config is constructed (call ``from_env()`` or
    ``get_runtime_config(refresh=True)`` to pick up mid-process changes used by
    benchmarks and tests).

    Role / content tuning (see docs/grid16_stos_cell_role_theory.md):

    - ``NORNIR_REFINE_IDENTITY_ZNCC_MIN`` — min masked ZNCC for lock-candidate
      PC-pass cells. Below → ``IDENTITY_SUSPECT`` (never lock). Unset → 0.25.
    - ``NORNIR_REFINE_LOW_CONTENT_STD_MIN`` — min ROI intensity std for alignable
      source content. Below → sticky measure-skip + ``REJECT(LOW_CONTENT)``.
      Unset → 1e-3.
    - ``NORNIR_REFINE_PASS_DIAGNOSTICS`` — write per-pass NPZ/CSV including role/zncc.
    - ``NORNIR_REFINE_PHASE_TIMING`` — detailed phase buckets (incl. classify/zncc).
    """

    phase_timing: bool
    batched_vertex_measurement: bool
    prewarp_mode: str
    tile_measure_parallel: bool
    gpu_transform: bool
    disable_prewarp_cache: bool
    mosaic_cutoff: bool
    stos_regularize: bool
    finalize_legacy: bool
    pass_diagnostics: bool
    sharp_warps: bool
    discontinuity_k: float
    discontinuity_travel_mult: float
    identity_zncc_min: float
    low_content_std_min: float

    @classmethod
    def from_env(cls) -> RefineRuntimeConfig:
        """Build a config snapshot from the current process environment."""
        phase_flag = _env_flag('NORNIR_REFINE_PHASE_TIMING', '0')
        batched = _env_flag('NORNIR_REFINE_BATCHED', '')
        if batched == '':
            batched = _env_flag('NORNIR_REFINE_BATCHED_GPU', '')
        # Default ON when unset (validated production default for mosaic).
        batched_on = True if batched == '' else not _is_falsey(batched)

        tile_flag = _env_flag('NORNIR_REFINE_TILE_PARALLEL', '')
        if _is_falsey(tile_flag):
            tile_parallel = False
        elif _is_truthy(tile_flag):
            tile_parallel = True
        else:
            tile_parallel = False

        gpu_flag = _env_flag('NORNIR_REFINE_GPU_TRANSFORM', '')
        gpu_transform = not (_is_falsey(gpu_flag) or gpu_flag == '')

        sharp_flag = _env_flag('NORNIR_REFINE_SHARP_WARPS', '')
        # Default ON when unset; only an explicit falsey disables.
        sharp_warps = True if sharp_flag == '' else not _is_falsey(sharp_flag)

        disc_k = 1.5
        raw_k = os.environ.get('NORNIR_REFINE_DISCONTINUITY_K', '').strip()
        if raw_k:
            try:
                disc_k = max(0.1, float(raw_k))
            except ValueError:
                pass

        disc_travel = 2.5
        raw_travel = os.environ.get('NORNIR_REFINE_DISCONTINUITY_TRAVEL_MULT', '').strip()
        if raw_travel:
            try:
                disc_travel = max(1.0, float(raw_travel))
            except ValueError:
                pass

        # Defaults match cell_roles.DEFAULT_IDENTITY_ZNCC_MIN /
        # cell_validity.DEFAULT_LOW_CONTENT_STD_MIN (avoid circular imports).
        identity_zncc = 0.25
        raw_zncc = os.environ.get('NORNIR_REFINE_IDENTITY_ZNCC_MIN', '').strip()
        if raw_zncc:
            try:
                identity_zncc = float(raw_zncc)
            except ValueError:
                pass

        low_content_std = 1e-3
        raw_std = os.environ.get('NORNIR_REFINE_LOW_CONTENT_STD_MIN', '').strip()
        if raw_std:
            try:
                low_content_std = max(0.0, float(raw_std))
            except ValueError:
                pass

        return cls(
            phase_timing=not (_is_falsey(phase_flag) or phase_flag == ''),
            batched_vertex_measurement=batched_on,
            prewarp_mode=_env_flag('NORNIR_REFINE_PREWARP_MODE', ''),
            tile_measure_parallel=tile_parallel,
            gpu_transform=gpu_transform,
            disable_prewarp_cache=_is_truthy(_env_flag('NORNIR_DISABLE_PREWARP_CACHE', '')),
            mosaic_cutoff=_is_truthy(_env_flag('NORNIR_REFINE_MOSAIC_CUTOFF', '')),
            stos_regularize=_is_truthy(_env_flag('NORNIR_REFINE_STOS_REGULARIZE', '')),
            finalize_legacy=_is_truthy(_env_flag('NORNIR_REFINE_FINALIZE_LEGACY', '')),
            pass_diagnostics=_is_truthy(_env_flag('NORNIR_REFINE_PASS_DIAGNOSTICS', '')),
            sharp_warps=sharp_warps,
            discontinuity_k=disc_k,
            discontinuity_travel_mult=disc_travel,
            identity_zncc_min=identity_zncc,
            low_content_std_min=low_content_std,
        )

    def prewarp_thread_dispatch_enabled(self, using_cupy: bool) -> bool:
        """Return True when per-tile prewarp should use the shared thread pool."""
        mode = self.prewarp_mode
        if 'serial' in mode:
            return False
        if 'thread' in mode:
            return True
        return using_cupy

    def prewarp_cache_enabled(self, using_cupy: bool) -> bool:
        """Return True when cross-pass prewarp caching is allowed."""
        if self.disable_prewarp_cache:
            return False
        if using_cupy:
            return 'cache' in self.prewarp_mode
        return True

    def prewarp_single_warp_coverage_enabled(self) -> bool:
        """Return True for the opt-in single-warp coverage path."""
        return 'singlewarp' in self.prewarp_mode

    def pool_for_cell_tasks(self, using_cupy: bool) -> Any:
        """Return the pool used for per-cell STOS / overlap alignment tasks."""
        import nornir_pools

        if using_cupy:
            return nornir_pools.GetGlobalSerialPool()
        return nornir_pools.GetGlobalMultithreadingPool()

    def pool_for_prewarp(self, using_cupy: bool) -> Any:
        """Return the pool used for mosaic prewarp dispatch when parallelized."""
        import nornir_pools

        if using_cupy and self.prewarp_thread_dispatch_enabled(using_cupy):
            return nornir_pools.GetGlobalThreadPool()
        if using_cupy:
            return nornir_pools.GetGlobalSerialPool()
        return nornir_pools.GetGlobalMultithreadingPool()

    def pool_for_tile_measure(self) -> Any:
        """Return the pool used when ``tile_measure_parallel`` is enabled."""
        import nornir_pools

        return nornir_pools.GetGlobalThreadPool()


@lru_cache(maxsize=1)
def _cached_config() -> RefineRuntimeConfig:
    return RefineRuntimeConfig.from_env()


def get_runtime_config(*, refresh: bool = False) -> RefineRuntimeConfig:
    """Return the process refine runtime config, optionally refreshing from env."""
    if refresh:
        _cached_config.cache_clear()
    return _cached_config()
