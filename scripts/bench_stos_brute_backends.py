#!/usr/bin/env python3
"""Benchmark AlignSections' brute section-pair registration on CuPy vs NumPy, serial vs pair-parallel.

Question answered: is ``stos_brute.SliceToSliceRigidRegistration`` (the call
``nornir_buildmanager.operations.block.__CallNornirStosBrute`` makes for every
AlignSections pair) faster on the GPU, or on CPU cores with independent pairs
registered in parallel?

Every measurement runs in fresh worker interpreters launched with ``subprocess``
(not ``multiprocessing``): ``computational_lib`` forces NumPy in any
``multiprocessing`` child, so CuPy workers must be independent processes. A pair
is registered exactly as production does: image and mask *paths*, LogPolar,
``MinOverlap=LOGPOLAR_PIPELINE_MIN_OVERLAP``, ``LargestDimension=818``,
``TestFlip=True``, ``AngleSearchRange=None``, the isotropic capture-resolution
scale as ``WarpedImageScaleFactors``, then ``nornir_pools.ReleaseStagePools()``
and ``AlignmentRecord.ToStos(...).Save(...)``.

Modes
-----
``cupy_serial``      one CuPy process, pairs one after another (production today)
``numpy_single``     one NumPy process, ``SingleThread=True``, BLAS/FFT pinned to
                     1 thread: the cost of one pair on one core
``numpy_pool``       one NumPy process, ``SingleThread=False`` (the angle pool may
                     engage); counts how many tasks reach the pool
``numpy_parallel``   W NumPy processes (1 thread each) registering independent
                     copies of a pair at once; throughput and per-pair latency
``cupy_parallel``    W CuPy processes sharing the GPU; throughput, GPU memory,
                     context-creation cost

Each worker registers one cold (warm-up) pair first, then waits at a file
barrier so parallel throughput is timed warm. ``seed_random_data`` is reset to
the same seed before every pair.

Pairs
-----
``native``  CaptureResolutionMismatch 503 (target) / 502 (source), Blob filter,
            Downsample 64 (1824 and 1887 px, like AlignSections ``-Downsample 64
            -Filters Blob -UseMasks``), masks, ``WarpedImageScaleFactors`` =
            2.143/2.176.
``scaled``  the same, with the source image and mask pre-resampled by 1.12 so the
            log-polar scale seed is far from 1.0 (the slow scale-refine path).
``leveled`` the same sections, Leveled filter at Downsample 32 (3774 px, scaled to
            818 by the registration): the well-conditioned fixture
            ``test_SliceToSliceBrute`` asserts (~176 deg, X -165, Y +90), used to
            check whether the backends agree when the answer is unambiguous.

Usage
-----
  python scripts/bench_stos_brute_backends.py --repeats 3
  python scripts/bench_stos_brute_backends.py --modes numpy_parallel --numpy-workers 4,8,16,30
  python scripts/bench_stos_brute_backends.py --modes cupy_serial --pairs native --repeats 1

Results go to stdout; ``--json`` (default under ``$TESTOUTPUTPATH``) keeps the raw
per-pair records.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_IMPORT_START_WALL: float = time.time()

import nornir_imageregistration
import nornir_pools
import numpy as np
import psutil
from nornir_imageregistration import stos_brute
from nornir_imageregistration.settings import SliceToSliceMethod
from PIL import Image

_IMPORT_DONE_WALL: float = time.time()

ALL_MODES: tuple[str, ...] = ('cupy_serial', 'numpy_single', 'numpy_pool', 'numpy_parallel', 'cupy_parallel')
PIN_ENV_VARS: tuple[str, ...] = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
TEM1_RESOLUTION: float = 2.176  # nm/px, section 503 (target)
TEM2_RESOLUTION: float = 2.143  # nm/px, section 502 (source)
SCALED_PAIR_FACTOR: float = 1.12
RANDOM_SEED: int = 0


@dataclass
class PairSpec:
    """Paths and metadata scale for one section pair, as block.py passes them."""
    name: str
    target: str
    source: str
    target_mask: str
    source_mask: str
    warped_scale: float


@dataclass
class WorkerSpec:
    """Instructions handed to one worker interpreter."""
    backend: str
    single_thread: bool
    pair: PairSpec
    count: int
    warmup: bool
    worker_id: int
    out_json: str
    scratch_dir: str
    barrier_dir: str | None = None
    n_workers: int = 1


@dataclass
class Counters:
    """Instrumentation of one registration: call counts and inclusive seconds per stage."""
    calls: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    find_best_angle_angles: int = 0
    find_best_angle_single_thread_calls: int = 0
    find_best_angle_pooled_calls: int = 0
    pool_tasks: int = 0

    def reset(self) -> None:
        """Clear every counter before the next pair."""
        self.calls.clear()
        self.seconds.clear()
        self.find_best_angle_angles = 0
        self.find_best_angle_single_thread_calls = 0
        self.find_best_angle_pooled_calls = 0
        self.pool_tasks = 0


_COUNTERS: Counters = Counters()


# ---------------------------------------------------------------- inputs

def _default_input_root() -> Path:
    """Return the CaptureResolutionMismatch fixture root under TESTINPUTPATH."""
    root = os.environ.get('TESTINPUTPATH', r'D:\nornir-testdata')
    return Path(root) / 'Images' / 'Alignment' / 'CaptureResolutionMismatch'


def _default_output_root() -> Path:
    """Return the scratch root for this benchmark under TESTOUTPUTPATH."""
    root = os.environ.get('TESTOUTPUTPATH', r'C:\Temp\Nornir')
    return Path(root) / 'bench_stos_brute_backends'


def _resample_png(src: Path, dst: Path, factor: float, nearest: bool) -> None:
    """Resample an 8-bit or 1-bit PNG by ``factor`` (nearest for masks)."""
    with Image.open(src) as im:
        mode = im.mode
        work = im.convert('L')
        size = (round(work.width * factor), round(work.height * factor))
        out = work.resize(size, Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC)
        if mode == '1':
            out = out.point([0] * 128 + [255] * 128).convert('1')
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst)


def build_pairs(input_root: Path, scratch: Path, names: list[str]) -> list[PairSpec]:
    """Build the native and/or 1.12x-scaled pair specs, creating the scaled inputs if needed."""
    def fixture(flt: str, level: str) -> tuple[Path, Path, Path, Path]:
        """Return target, source, target mask, source mask paths for one filter and level."""
        paths = (input_root / '503' / flt / 'Images' / level / f'0503_TEM_{flt}.png',
                 input_root / '502' / flt / 'Images' / level / f'0502_TEM_{flt}.png',
                 input_root / '503' / 'Mask' / 'Images' / level / '0503_TEM_Mask.png',
                 input_root / '502' / 'Mask' / 'Images' / level / '0502_TEM_Mask.png')
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f'Missing fixture {p}')
        return paths

    flt = 'Blob'
    target, source, target_mask, source_mask = fixture(flt, '064')
    warped_scale = TEM2_RESOLUTION / TEM1_RESOLUTION
    pairs: list[PairSpec] = []
    if 'native' in names:
        pairs.append(PairSpec('native', str(target), str(source), str(target_mask), str(source_mask), warped_scale))
    if 'leveled' in names:
        lt, ls, ltm, lsm = fixture('Leveled', '032')
        pairs.append(PairSpec('leveled', str(lt), str(ls), str(ltm), str(lsm), warped_scale))
    if 'scaled' in names:
        scaled_dir = scratch / 'inputs'
        scaled_source = scaled_dir / f'0502_TEM_{flt}_x{SCALED_PAIR_FACTOR:.2f}.png'
        scaled_mask = scaled_dir / f'0502_TEM_Mask_x{SCALED_PAIR_FACTOR:.2f}.png'
        if not scaled_source.exists():
            _resample_png(source, scaled_source, SCALED_PAIR_FACTOR, nearest=False)
        if not scaled_mask.exists():
            _resample_png(source_mask, scaled_mask, SCALED_PAIR_FACTOR, nearest=True)
        pairs.append(PairSpec('scaled', str(target), str(scaled_source), str(target_mask), str(scaled_mask),
                              warped_scale))
    return pairs


# ---------------------------------------------------------------- worker side

def _timed(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``fn`` so each call adds to the call count and inclusive seconds for ``name``."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _COUNTERS.calls[name] = _COUNTERS.calls.get(name, 0) + 1
            _COUNTERS.seconds[name] = _COUNTERS.seconds.get(name, 0.0) + time.perf_counter() - start

    return wrapper


def _install_instrumentation() -> None:
    """Patch stos_brute stage functions (module globals) with counting/timing wrappers.

    The wrappers keep ``__module__``/``__qualname__`` so callables submitted to the
    process pool still pickle by reference (workers run the unwrapped original).
    """
    for name in ('_find_angle_and_scale_with_logpolar', '_find_best_angle_common_random',
                 '_refine_scale_local', 'ScoreOneAngle', 'ScoreManyAnglesGpu'):
        setattr(stos_brute, name, _timed(name, getattr(stos_brute, name)))

    inner_find_best_angle = stos_brute._find_best_angle

    @wraps(inner_find_best_angle)
    def find_best_angle(*args: Any, **kwargs: Any) -> Any:
        angle_range = kwargs.get('angle_range', args[4] if len(args) > 4 else ())
        n_angles = len(angle_range)
        single = bool(kwargs.get('SingleThread', False)) or n_angles <= 1
        _COUNTERS.find_best_angle_angles += n_angles
        if single:
            _COUNTERS.find_best_angle_single_thread_calls += 1
        else:
            _COUNTERS.find_best_angle_pooled_calls += 1
        return inner_find_best_angle(*args, **kwargs)

    stos_brute._find_best_angle = _timed('_find_best_angle', find_best_angle)

    get_pool = nornir_pools.GetGlobalMultithreadingPool
    wrapped_pools: set[int] = set()

    def counting_get_pool() -> Any:
        pool = get_pool()
        if id(pool) not in wrapped_pools:
            add_task = pool.add_task

            def counted_add_task(*a: Any, **kw: Any) -> Any:
                _COUNTERS.pool_tasks += 1
                return add_task(*a, **kw)

            pool.add_task = counted_add_task  # type: ignore[method-assign]
            wrapped_pools.add(id(pool))
        return pool

    nornir_pools.GetGlobalMultithreadingPool = counting_get_pool  # type: ignore[assignment]


def _peak_rss_bytes() -> int:
    """Peak resident set of this process (peak working set on Windows)."""
    info = psutil.Process().memory_info()
    peak = getattr(info, 'peak_wset', None)
    if peak is not None:
        return int(peak)
    import resource  # POSIX only, so imported here rather than at module top
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024  # pyright: ignore[reportAttributeAccessIssue]


def _cupy_pool_bytes() -> int:
    """Bytes the CuPy default memory pool currently holds (0 when not on CuPy)."""
    if nornir_imageregistration.GetActiveComputationLib() != nornir_imageregistration.ComputationLib.cupy:
        return 0
    import cupy as cp
    return int(cp.get_default_memory_pool().total_bytes())


def _register_once(spec: WorkerSpec, index: int) -> dict[str, Any]:
    """Register spec.pair once, exactly as block.__CallNornirStosBrute does, and record the result."""
    pair = spec.pair
    _COUNTERS.reset()
    nornir_imageregistration.seed_random_data(RANDOM_SEED)
    start_wall = time.time()
    start = time.perf_counter()
    alignment = stos_brute.SliceToSliceRigidRegistration(
        target_image=pair.target,
        source_image=pair.source,
        target_mask=pair.target_mask,
        source_mask=pair.source_mask,
        AngleSearchRange=None,
        TestFlip=True,
        WarpedImageScaleFactors=pair.warped_scale,
        method=SliceToSliceMethod.LogPolar,
        MinOverlap=stos_brute.LOGPOLAR_PIPELINE_MIN_OVERLAP,
        LargestDimension=stos_brute.LOGPOLAR_PIPELINE_LARGEST_DIMENSION,
        SingleThread=spec.single_thread,
    )
    register_s = time.perf_counter() - start
    nornir_pools.ReleaseStagePools()
    stos = alignment.ToStos(pair.target, pair.source, pair.target_mask, pair.source_mask, PixelSpacing=64)
    stos.Save(str(Path(spec.scratch_dir) / f'{pair.name}_w{spec.worker_id}_{index}.stos'))
    total_s = time.perf_counter() - start
    return {
        'index': index,
        'register_s': register_s,
        'total_s': total_s,
        'start_wall': start_wall,
        'end_wall': time.time(),
        'angle': float(alignment.angle),
        'offset_yx': [float(alignment.peak[0]), float(alignment.peak[1])],
        'scale': float(alignment.scale),
        'flippedud': bool(alignment.flippedud),
        'weight': float(alignment.weight),
        'calls': dict(_COUNTERS.calls),
        'stage_seconds': {k: round(v, 4) for k, v in _COUNTERS.seconds.items()},
        'find_best_angle_angles': _COUNTERS.find_best_angle_angles,
        'find_best_angle_single_thread_calls': _COUNTERS.find_best_angle_single_thread_calls,
        'find_best_angle_pooled_calls': _COUNTERS.find_best_angle_pooled_calls,
        'pool_tasks': _COUNTERS.pool_tasks,
        'cupy_pool_bytes': _cupy_pool_bytes(),
        'rss_bytes': int(psutil.Process().memory_info().rss),
    }


def _wait_for_barrier(spec: WorkerSpec) -> float:
    """Signal ready, then block until the orchestrator creates the go file; return go wall time."""
    assert spec.barrier_dir is not None
    barrier = Path(spec.barrier_dir)
    (barrier / f'ready_{spec.worker_id}').write_text(str(os.getpid()))
    go = barrier / 'go'
    while not go.exists():
        time.sleep(0.01)
    return time.time()


def run_worker(spec_path: Path) -> None:
    """Worker entry point: set the backend, warm up, optionally barrier, register ``count`` pairs."""
    raw = json.loads(spec_path.read_text())
    raw['pair'] = PairSpec(**raw['pair'])
    spec = WorkerSpec(**raw)
    Path(spec.scratch_dir).mkdir(parents=True, exist_ok=True)

    context_s = 0.0
    if spec.backend == 'cupy':
        t = time.perf_counter()
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        import cupy as cp
        cp.zeros(1).sum().item()
        cp.cuda.Device().synchronize()
        context_s = time.perf_counter() - t
        if nornir_imageregistration.GetActiveComputationLib() != nornir_imageregistration.ComputationLib.cupy:
            raise RuntimeError('CuPy requested but the runtime probe fell back to NumPy')
    else:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

    _install_instrumentation()

    cold: dict[str, Any] | None = None
    if spec.warmup:
        cold = _register_once(spec, -1)

    go_wall: float | None = None
    if spec.barrier_dir is not None:
        go_wall = _wait_for_barrier(spec)

    records = [_register_once(spec, i) for i in range(spec.count)]

    result = {
        'backend': spec.backend,
        'worker_id': spec.worker_id,
        'pid': os.getpid(),
        'import_s': _IMPORT_DONE_WALL - _IMPORT_START_WALL,
        'import_done_wall': _IMPORT_DONE_WALL,
        'context_s': context_s,
        'cold': cold,
        'go_wall': go_wall,
        'records': records,
        'peak_rss_bytes': _peak_rss_bytes(),
        'thread_env': {k: os.environ.get(k) for k in PIN_ENV_VARS},
    }
    Path(spec.out_json).write_text(json.dumps(result, indent=1))
    nornir_pools.ClosePools()


# ---------------------------------------------------------------- orchestrator side

class ForeignLoadSampler:
    """Sample CPU used by processes outside this benchmark's process tree while a mode runs.

    The idle check only runs between modes; this records whether another job started
    mid-run, so a contended run is visible in the results instead of silently slow.
    """
    _interval_s: float
    _stop: threading.Event
    _thread: threading.Thread | None
    samples_cores: list[float]
    heavy: dict[str, float]

    def __init__(self, interval_s: float = 10.0) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None
        self.samples_cores = []
        self.heavy = {}

    def start(self) -> None:
        """Start the sampling thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Every interval, sum CPU of non-descendant processes (System Idle excluded)."""
        me = psutil.Process()
        seen: dict[int, psutil.Process] = {}
        while not self._stop.is_set():
            mine = {me.pid} | {c.pid for c in me.children(recursive=True)}
            total = 0.0
            for p in psutil.process_iter(['pid', 'name', 'cmdline']):
                if p.pid in mine or p.pid == 0:
                    continue
                proc = seen.setdefault(p.pid, p)
                try:
                    cpu = proc.cpu_percent(None)
                except psutil.Error:
                    continue
                total += cpu
                if cpu > 80.0:
                    key = f"{p.info['name']} {' '.join(p.info['cmdline'] or [])[:60]}"
                    self.heavy[key] = max(self.heavy.get(key, 0.0), cpu / 100.0)
            self.samples_cores.append(total / 100.0)
            self._stop.wait(self._interval_s)

    def stop(self) -> dict[str, Any]:
        """Stop sampling; return mean/max foreign CPU in cores and the heaviest foreign processes."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        vals = self.samples_cores[1:]  # first sample primes cpu_percent and reads 0
        heavy = dict(sorted(self.heavy.items(), key=lambda kv: -kv[1])[:5])
        return {'foreign_cores_mean': statistics.fmean(vals) if vals else None,
                'foreign_cores_max': max(vals) if vals else None, 'foreign_heavy': heavy}


class GpuSampler:
    """Poll nvidia-smi for device memory used and utilization while a mode runs."""
    _interval_ms: int
    _proc: subprocess.Popen[str] | None
    _thread: threading.Thread | None
    samples: list[tuple[float, float]]

    def __init__(self, interval_ms: int = 250) -> None:
        self._interval_ms = interval_ms
        self._proc = None
        self._thread = None
        self.samples = []

    def start(self) -> None:
        """Start nvidia-smi in loop mode and a reader thread; silently no-op without nvidia-smi."""
        try:
            self._proc = subprocess.Popen(
                ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits',
                 f'-lms={self._interval_ms}'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self._proc = None
            return
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        """Parse nvidia-smi lines into (MiB used, util %) samples."""
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 2:
                try:
                    self.samples.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass

    def stop(self) -> dict[str, float]:
        """Stop sampling; return peak/baseline memory (MiB) and mean/peak utilization (%)."""
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        if self._thread is not None:
            self._thread.join(timeout=5)
        if not self.samples:
            return {}
        mem = [s[0] for s in self.samples]
        util = [s[1] for s in self.samples]
        return {'mem_baseline_mib': mem[0], 'mem_peak_mib': max(mem),
                'util_mean_pct': statistics.fmean(util), 'util_peak_pct': max(util)}


def gpu_used_mib() -> float | None:
    """Return current device memory used in MiB from nvidia-smi, or None."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20, check=True).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def busy_report(sample_s: float = 3.0) -> tuple[float, list[str]]:
    """Return total CPU %, and descriptions of docker builds and busy foreign processes."""
    me = os.getpid()
    procs = list(psutil.process_iter(['pid', 'name', 'cmdline']))
    for p in procs:
        try:
            p.cpu_percent(None)
        except psutil.Error:
            pass
    total = psutil.cpu_percent(interval=sample_s)
    busy: list[str] = []
    for p in procs:
        try:
            if p.pid in (me, 0):  # pid 0 is the Windows System Idle Process
                continue
            cmd = ' '.join(p.info['cmdline'] or [])
            name = (p.info['name'] or '').lower()
            if 'docker' in name and (' build' in cmd or 'buildx' in cmd):
                busy.append(f'docker build pid {p.pid}')
                continue
            cpu = p.cpu_percent(None)
            if cpu > 50.0:
                busy.append(f'{name} pid {p.pid} {cpu:.0f}% {cmd[:80]}')
        except psutil.Error:
            pass
    return total, busy


def wait_for_idle(max_cpu_pct: float, poll_s: float, max_wait_s: float) -> None:
    """Block until total CPU is below ``max_cpu_pct`` and no docker build runs (or give up loudly)."""
    deadline = time.time() + max_wait_s
    while True:
        total, busy = busy_report()
        docker_build = any(b.startswith('docker build') for b in busy)
        if total <= max_cpu_pct and not docker_build:
            print(f'[idle-check] CPU {total:.1f}% - ok', flush=True)
            return
        print(f'[idle-check] CPU {total:.1f}%; busy: {busy[:8]}', flush=True)
        if time.time() > deadline:
            raise SystemExit(f'Machine still busy after {max_wait_s:.0f}s; refusing to time under contention')
        time.sleep(poll_s)


def _worker_env(backend: str, pin_threads: bool) -> dict[str, str]:
    """Environment for a worker: backend selection and optional 1-thread BLAS/FFT pinning."""
    env = dict(os.environ)
    env['NORNIR_COMPUTATIONAL_LIBRARY'] = backend
    env['PYTHONUNBUFFERED'] = '1'
    for var in PIN_ENV_VARS:
        if pin_threads:
            env[var] = '1'
        else:
            env.pop(var, None)
    return env


def launch_workers(mode: str, backend: str, single_thread: bool, pin_threads: bool, pair: PairSpec,
                   n_workers: int, count: int, out_root: Path, timeout_s: float) -> dict[str, Any]:
    """Run ``n_workers`` worker interpreters on ``pair`` and collect their JSON results.

    With more than one worker, a file barrier releases all of them together after
    each has finished its warm-up pair, so throughput is timed warm.
    """
    run_dir = out_root / 'runs' / f'{mode}_{pair.name}_w{n_workers}_{int(time.time())}'
    run_dir.mkdir(parents=True, exist_ok=True)
    barrier_dir = run_dir / 'barrier' if n_workers > 1 else None
    if barrier_dir is not None:
        barrier_dir.mkdir()
    env = _worker_env(backend, pin_threads)

    gpu_before = gpu_used_mib()
    sampler = GpuSampler()
    sampler.start()
    foreign = ForeignLoadSampler()
    foreign.start()
    procs: list[subprocess.Popen[bytes]] = []
    launch_wall = time.time()
    for wid in range(n_workers):
        spec = WorkerSpec(backend=backend, single_thread=single_thread, pair=pair, count=count, warmup=True,
                          worker_id=wid, out_json=str(run_dir / f'worker_{wid}.json'),
                          scratch_dir=str(run_dir / 'stos'),
                          barrier_dir=str(barrier_dir) if barrier_dir is not None else None,
                          n_workers=n_workers)
        spec_path = run_dir / f'spec_{wid}.json'
        spec_path.write_text(json.dumps(asdict(spec)))
        with open(run_dir / f'worker_{wid}.log', 'wb') as log:
            procs.append(subprocess.Popen([sys.executable, __file__, '--worker', str(spec_path)],
                                          env=env, stdout=log, stderr=subprocess.STDOUT))

    go_wall: float | None = None
    gpu_ready: float | None = None
    deadline = time.time() + timeout_s
    if barrier_dir is not None:
        while len(list(barrier_dir.glob('ready_*'))) < n_workers:
            if any(p.poll() not in (None, 0) for p in procs):
                break
            if time.time() > deadline:
                break
            time.sleep(0.05)
        gpu_ready = gpu_used_mib()
        go_wall = time.time()
        (barrier_dir / 'go').write_text('go')

    for p in procs:
        remaining = max(1.0, deadline - time.time())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            p.kill()
    gpu = sampler.stop()
    contention = foreign.stop()

    workers: list[dict[str, Any]] = []
    failures: list[str] = []
    for wid, p in enumerate(procs):
        out_json = run_dir / f'worker_{wid}.json'
        if p.returncode == 0 and out_json.exists():
            workers.append(json.loads(out_json.read_text()))
        else:
            failures.append(f'worker {wid} rc={p.returncode} log={run_dir / f"worker_{wid}.log"}')
    return {'mode': mode, 'backend': backend, 'single_thread': single_thread, 'pin_threads': pin_threads,
            'pair': pair.name, 'n_workers': n_workers, 'count': count, 'launch_wall': launch_wall,
            'go_wall': go_wall, 'gpu_before_mib': gpu_before, 'gpu_at_barrier_mib': gpu_ready,
            'gpu': gpu, 'contention': contention, 'workers': workers, 'failures': failures, 'run_dir': str(run_dir)}


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    """Reduce a run to per-pair latency stats, throughput, memory and result spread."""
    workers = run['workers']
    warm = [r for w in workers for r in w['records']]
    cold = [w['cold'] for w in workers if w.get('cold')]
    lat = [r['total_s'] for r in warm]
    s: dict[str, Any] = {
        'mode': run['mode'], 'pair': run['pair'], 'workers': run['n_workers'], 'pairs_timed': len(warm),
        'failures': run['failures'],
    }
    if lat:
        s.update(median_s=statistics.median(lat), min_s=min(lat), max_s=max(lat))
    if cold:
        s['cold_pair_s'] = statistics.median(c['total_s'] for c in cold)
    s['import_s'] = statistics.median(w['import_s'] for w in workers) if workers else None
    s['context_s'] = statistics.median(w['context_s'] for w in workers) if workers else None
    s['peak_rss_mib'] = max(w['peak_rss_bytes'] for w in workers) / 2 ** 20 if workers else None
    s['cupy_pool_mib'] = max((r['cupy_pool_bytes'] for r in warm), default=0) / 2 ** 20
    if run['go_wall'] is not None and warm:
        span = max(r['end_wall'] for r in warm) - run['go_wall']
        s['wall_s'] = span
        s['throughput_pairs_per_min'] = 60.0 * len(warm) / span
    elif warm:
        s['throughput_pairs_per_min'] = 60.0 / statistics.median(lat)
    gpu = run.get('gpu') or {}
    s['gpu_util_mean_pct'] = gpu.get('util_mean_pct')
    s['contention'] = run.get('contention')
    if run.get('gpu_before_mib') is not None and gpu.get('mem_peak_mib') is not None:
        s['gpu_mem_delta_peak_mib'] = gpu['mem_peak_mib'] - run['gpu_before_mib']
        s['gpu_mem_per_process_mib'] = s['gpu_mem_delta_peak_mib'] / max(1, run['n_workers'])
    if warm:
        first = warm[0]
        s['result'] = {k: first[k] for k in ('angle', 'offset_yx', 'scale', 'flippedud', 'weight')}
        s['angle_spread'] = max(r['angle'] for r in warm) - min(r['angle'] for r in warm)
        s['scale_spread'] = max(r['scale'] for r in warm) - min(r['scale'] for r in warm)
        offs = np.array([r['offset_yx'] for r in warm])
        s['offset_spread_px'] = float(np.max(np.ptp(offs, axis=0)))
        s['flip_values'] = sorted({r['flippedud'] for r in warm})
        s['pool_tasks'] = statistics.median(r['pool_tasks'] for r in warm)
        s['find_best_angle_pooled_calls'] = statistics.median(r['find_best_angle_pooled_calls'] for r in warm)
        s['find_best_angle_angles'] = statistics.median(r['find_best_angle_angles'] for r in warm)
        s['score_one_angle_calls'] = statistics.median(r['calls'].get('ScoreOneAngle', 0) for r in warm)
        stages = sorted({k for r in warm for k in r['stage_seconds']})
        s['stage_median_s'] = {k: round(statistics.median(r['stage_seconds'].get(k, 0.0) for r in warm), 3)
                               for k in stages}
    return s


def _fmt(v: Any, nd: int = 2) -> str:
    """Format numbers for the table, '-' for missing."""
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return str(v)


def print_table(summaries: list[dict[str, Any]]) -> None:
    """Print the per-mode table and the result-agreement lines."""
    cols = ('mode', 'pair', 'workers', 'pairs_timed', 'median_s', 'min_s', 'cold_pair_s',
            'throughput_pairs_per_min', 'peak_rss_mib', 'gpu_mem_per_process_mib', 'gpu_util_mean_pct',
            'context_s', 'pool_tasks', 'foreign_cores_mean')
    print('\n' + ' | '.join(cols))
    for s in summaries:
        row = dict(s, foreign_cores_mean=(s.get('contention') or {}).get('foreign_cores_mean'))
        print(' | '.join(_fmt(row.get(c)) for c in cols))
    print('\nResults (first warm record) and spread within mode:')
    for s in summaries:
        r = s.get('result')
        if r is None:
            print(f"  {s['mode']:<15} {s['pair']:<7} FAILED {s['failures']}")
            continue
        print(f"  {s['mode']:<15} {s['pair']:<7} w={s['workers']:<3} angle={r['angle']:.3f} "
              f"offset=({r['offset_yx'][0]:.2f},{r['offset_yx'][1]:.2f}) scale={r['scale']:.4f} "
              f"flip={r['flippedud']} weight={r['weight']:.4f} | spread angle={s['angle_spread']:.3f} "
              f"offset={s['offset_spread_px']:.2f}px scale={s['scale_spread']:.4f} flips={s['flip_values']}")
    print('\nStage medians (inclusive seconds, warm):')
    for s in summaries:
        if 'stage_median_s' in s:
            print(f"  {s['mode']:<15} {s['pair']:<7} w={s['workers']:<3} ScoreOneAngle calls="
                  f"{s['score_one_angle_calls']} {s['stage_median_s']}")


def _parse_int_list(text: str) -> list[int]:
    """Parse '4,8,16' into [4, 8, 16]."""
    return [int(x) for x in text.split(',') if x.strip()]


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the requested modes (or a single worker with --worker)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--modes', default=','.join(ALL_MODES), help='Comma list from ' + ','.join(ALL_MODES))
    parser.add_argument('--pairs', default='native,scaled', help='Comma list from native,scaled,leveled')
    parser.add_argument('--repeats', type=int, default=3, help='Warm pairs per serial mode (after one cold)')
    parser.add_argument('--numpy-workers', default='4,8,16,30', help='Worker counts for numpy_parallel')
    parser.add_argument('--cupy-workers', default='2,4,8', help='Worker counts for cupy_parallel')
    parser.add_argument('--parallel-pairs', default='native', help='Pairs for the parallel modes')
    parser.add_argument('--pairs-per-worker', type=int, default=2, help='Warm pairs each parallel worker registers')
    parser.add_argument('--input-root', type=Path, default=_default_input_root())
    parser.add_argument('--output-root', type=Path, default=_default_output_root())
    parser.add_argument('--json', type=Path, default=None, help='Raw results JSON (default under --output-root)')
    parser.add_argument('--max-idle-cpu', type=float, default=15.0,
                        help='Total CPU %% above which the machine counts as busy')
    parser.add_argument('--idle-wait', type=float, default=3600.0, help='Seconds to wait for an idle machine')
    parser.add_argument('--skip-idle-check', action='store_true')
    parser.add_argument('--timeout', type=float, default=3600.0, help='Per-run timeout (s)')
    args = parser.parse_args(argv)

    if args.worker is not None:
        run_worker(args.worker)
        return 0

    modes = [m for m in args.modes.split(',') if m]
    unknown = set(modes) - set(ALL_MODES)
    if unknown:
        parser.error(f'Unknown modes {sorted(unknown)}')
    out_root: Path = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    pair_names = [p for p in args.pairs.split(',') if p]
    parallel_names = [p for p in args.parallel_pairs.split(',') if p]
    pairs = {p.name: p for p in build_pairs(args.input_root, out_root, sorted(set(pair_names + parallel_names)))}
    json_path: Path = args.json or out_root / f'results_{time.strftime("%Y%m%d-%H%M%S")}.json'

    print(f'CPU: {psutil.cpu_count(logical=True)} logical / {psutil.cpu_count(logical=False)} physical, '
          f'RAM {psutil.virtual_memory().total / 2 ** 30:.0f} GiB; python {sys.version.split()[0]}', flush=True)
    for p in pairs.values():
        print(f'pair {p.name}: target={p.target} source={p.source} warped_scale={p.warped_scale:.5f}')

    plan: list[tuple[str, str, bool, bool, str, int, int]] = []
    for mode in modes:
        if mode == 'cupy_serial':
            plan += [(mode, 'cupy', True, False, n, 1, args.repeats) for n in pair_names]
        elif mode == 'numpy_single':
            plan += [(mode, 'numpy', True, True, n, 1, args.repeats) for n in pair_names]
        elif mode == 'numpy_pool':
            plan += [(mode, 'numpy', False, False, n, 1, args.repeats) for n in pair_names]
        elif mode == 'numpy_parallel':
            plan += [(mode, 'numpy', True, True, n, w, args.pairs_per_worker)
                     for n in parallel_names for w in _parse_int_list(args.numpy_workers)]
        elif mode == 'cupy_parallel':
            plan += [(mode, 'cupy', True, True, n, w, args.pairs_per_worker)
                     for n in parallel_names for w in _parse_int_list(args.cupy_workers)]

    runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for mode, backend, single, pin, pair_name, n_workers, count in plan:
        if not args.skip_idle_check:
            wait_for_idle(args.max_idle_cpu, 60.0, args.idle_wait)
        print(f'\n=== {mode} pair={pair_name} workers={n_workers} count={count} '
              f'(SingleThread={single}, BLAS/FFT threads pinned to 1: {pin})', flush=True)
        run = launch_workers(mode, backend, single, pin, pairs[pair_name], n_workers, count, out_root,
                             args.timeout)
        runs.append(run)
        summary = summarize(run)
        summaries.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k != 'stage_median_s'}, default=str), flush=True)
        json_path.write_text(json.dumps({'summaries': summaries, 'runs': runs}, indent=1, default=str))

    print_table(summaries)
    print(f'\nRaw results: {json_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
