__all__ = [
    'ComputationLib',
    'HasCupy',
    'HasCuVS',
    'UsingCupy',
    'GetActiveComputationLib',
    'SetActiveComputationLib',
    'NORNIR_COMPUTATIONAL_LIBRARY_ENV',
]

from enum import Enum
import logging
import multiprocessing
import os

# One-time setup so CUDA DLLs (e.g. curand64_*.dll) are found when CUDA_PATH is set.
# Windows does not search CUDA_PATH\bin or \bin\x64 by default; add them to the DLL search path.
def _add_cuda_bin_to_dll_path():
    cuda_path = os.environ.get("CUDA_PATH")
    if not cuda_path or not os.path.isdir(cuda_path):
        return
    try:
        add_dll = os.add_dll_directory
    except AttributeError:
        return
    for subdir in (os.path.join("bin", "x64"), "bin"):
        bin_dir = os.path.join(cuda_path, subdir)
        if os.path.isdir(bin_dir):
            try:
                add_dll(bin_dir)
            except OSError:
                pass

_logger = logging.getLogger(__name__)

# Process-local effective backend (``numpy`` / ``cupy``). Owned by Nornir; never mutates
# ``CUDA_VISIBLE_DEVICES`` (container / orchestrator may pin GPUs there).
NORNIR_COMPUTATIONAL_LIBRARY_ENV = 'NORNIR_COMPUTATIONAL_LIBRARY'
NORNIR_COMPUTATIONAL_LIBRARY_REQUESTED_ENV = 'NORNIR_COMPUTATIONAL_LIBRARY_REQUESTED'
NORNIR_POOL_WORKER_ENV = 'NORNIR_POOL_WORKER'


def _set_effective_computational_library_env(value: str) -> None:
    os.environ[NORNIR_COMPUTATIONAL_LIBRARY_ENV] = value


def _ensure_cuda_toolkit_env() -> None:
    """Point CuPy's pathfinder at the container CUDA toolkit when unset.

    Avoids the canary subprocess (``python -m cuda.pathfinder...``) that can fail
    when fork-pool workers are starting or when the debugger wraps ``sys.executable``.
    """
    if not os.environ.get('CUDA_PATH') and not os.environ.get('CUDA_HOME'):
        for candidate in ('/usr/local/cuda', '/usr/local/cuda-13', '/usr/local/cuda-13.1'):
            if os.path.isdir(candidate):
                os.environ['CUDA_PATH'] = candidate
                break
    cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
    if cuda_path and os.path.isdir(cuda_path):
        lib64 = os.path.join(cuda_path, 'lib64')
        if os.path.isdir(lib64):
            prev = os.environ.get('LD_LIBRARY_PATH', '')
            if lib64 not in prev.split(':'):
                os.environ['LD_LIBRARY_PATH'] = (
                    f"{lib64}:{prev}" if prev else lib64
                )
    _add_cuda_bin_to_dll_path()
    _patch_pathfinder_canary_subprocess()
    _preload_cuda_runtime_libs()


_cuda_runtime_preloaded = False  # type: bool
_pathfinder_canary_patched = False  # type: bool


def _patch_pathfinder_canary_subprocess() -> None:
    """Replace cuda-pathfinder's canary subprocess with in-process library lookup.

    The canary runs ``python -m cuda.pathfinder...`` which often returns empty
    stdout under debugpy or when the debugger wraps ``sys.executable``.
    """
    global _pathfinder_canary_patched
    if _pathfinder_canary_patched:
        return
    if multiprocessing.parent_process() is not None:
        return
    try:
        import functools
        from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as pf_mod
        from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
        from cuda.pathfinder._dynamic_libs.platform_loader import LOADER
        from cuda.pathfinder._dynamic_libs.search_steps import SearchContext, find_via_ctk_root

        @functools.cache
        def _resolve_inprocess(libname: str, *, timeout: float = 10.0) -> str | None:
            desc = LIB_DESCRIPTORS.get(libname)
            if desc is None:
                return None
            loaded = LOADER.check_if_already_loaded_from_elsewhere(desc, False)
            if loaded is not None:
                return str(loaded.abs_path)
            ctk_root = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
            if ctk_root and os.path.isdir(ctk_root):
                find = find_via_ctk_root(SearchContext(desc), ctk_root)
                if find is not None:
                    return str(find.abs_path)
            loaded = LOADER.load_with_system_search(desc)
            if loaded is not None:
                return str(loaded.abs_path)
            return None

        pf_mod._resolve_system_loaded_abs_path_in_subprocess.cache_clear()
        pf_mod._resolve_system_loaded_abs_path_in_subprocess = _resolve_inprocess
        try:
            import cuda.pathfinder._headers.find_nvidia_headers as _hdr_mod
            _hdr_mod._resolve_system_loaded_abs_path_in_subprocess = _resolve_inprocess
        except ImportError:
            pass
        _pathfinder_canary_patched = True
    except ImportError:
        pass


def _preload_cuda_runtime_libs() -> None:
    """Load ``cudart``/``curand`` in-process via ``CUDA_PATH`` (no canary subprocess).

    cuda-pathfinder's canary probe (``python -m cuda.pathfinder...``) can return empty
    stdout under debugpy or when fork-pool workers are starting. Preloading here
    satisfies CuPy's dynamic-library search without that subprocess.
    """
    global _cuda_runtime_preloaded
    if _cuda_runtime_preloaded:
        return
    if multiprocessing.parent_process() is not None:
        return
    if not (os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')):
        return
    try:
        from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
            load_nvidia_dynamic_lib,
        )
        load_nvidia_dynamic_lib('cudart')
        for libname in ('curand', 'cublas', 'cusparse', 'nvrtc'):
            try:
                load_nvidia_dynamic_lib(libname)
            except Exception:
                pass
        _cuda_runtime_preloaded = True
    except Exception:
        pass


class ComputationLib(Enum):
    numpy = 0
    cupy = 1

_has_cupy = False  # type: bool
_has_cuvs = False  # type: bool
_cupy_runtime_probed = False  # type: bool

# If we are in a child process, we use numpy, GPU processing currently runs on a single process
_active_lib = None if multiprocessing.parent_process() is None else ComputationLib.numpy # type: ComputationLib | None

_env_lib = os.environ.get(NORNIR_COMPUTATIONAL_LIBRARY_ENV, '').strip().lower()
if _env_lib == 'numpy':
    _active_lib = ComputationLib.numpy if _active_lib is None else _active_lib
elif _env_lib == 'cupy' and _active_lib is None:
    _active_lib = ComputationLib.cupy

if multiprocessing.parent_process() is not None:
    _active_lib = ComputationLib.numpy

_ensure_cuda_toolkit_env()

try:
    import cupy as cp
    _has_cupy = True
    _active_lib = ComputationLib.cupy if _active_lib is None else _active_lib
    # cupyx.scipy.spatial.distance.cdist checks the same imports (cuvs.distance.pairwise_distance
    # or legacy pylibraft.distance). Neighbors-only cuvs (e.g. brute_force) is not enough.
    # Import-only probe; does not touch the CUDA runtime (fork-safe until GPU ops run).
    if _has_cupy and _active_lib == ComputationLib.cupy:
        _has_cuvs = False
        try:
            import cuvs.distance as _cuvs_distance  # type: ignore[reportMissingImports]
            getattr(_cuvs_distance, "pairwise_distance")
            _has_cuvs = True
        except (ImportError, AttributeError):
            try:
                import pylibraft.distance as _pylibraft_distance  # type: ignore[reportMissingImports]
                getattr(_pylibraft_distance, "pairwise_distance")
                _has_cuvs = True
            except (ImportError, AttributeError):
                _has_cuvs = False
except ModuleNotFoundError:
    cp = None
    _has_cupy = False
    _active_lib = ComputationLib.numpy if _active_lib is None else _active_lib
except ImportError:
    cp = None
    _has_cupy = False
    _active_lib = ComputationLib.numpy if _active_lib is None else _active_lib


def _probe_cupy_runtime() -> bool:
    """Exercise required CUDA libs once; fall back to NumPy on failure.

    Deferred until CuPy is explicitly selected so fork-based CPU pools do not
    inherit a parent CUDA context initialized at import time.
    """
    global _active_lib, _cupy_runtime_probed, _cuda_runtime_preloaded, _pathfinder_canary_patched

    if _cupy_runtime_probed:
        return _active_lib == ComputationLib.cupy

    _cupy_runtime_probed = True
    if not _has_cupy or cp is None:
        return False

    _ensure_cuda_toolkit_env()

    last_exc: BaseException | None = None
    for attempt in (1, 2):
        _preload_cuda_runtime_libs()
        try:
            device_count = int(cp.cuda.runtime.getDeviceCount())
            if device_count < 1:
                raise RuntimeError("CuPy probe found zero CUDA devices")
            if not _cuda_runtime_preloaded:
                x = cp.ones((2, 2), dtype=cp.float32)
                if float(x.sum()) < 1.0:
                    raise RuntimeError("CuPy probe array sum unexpected")
            return True
        except Exception as exc:
            last_exc = exc
            if attempt == 1:
                _pathfinder_canary_patched = False
                _cuda_runtime_preloaded = False
                continue
            break

    _active_lib = ComputationLib.numpy
    return False


def ConfigureForkPoolWorker() -> None:
    """Force NumPy backend in fork-based pool workers.

    Fork-server workers inherit the server's ``_active_lib`` (often CuPy) without
    re-running module import. Call from ``nornir_pools.init_pool_process`` so
    tile/transform work never triggers CUDA in a forked worker.

    Does not modify ``NORNIR_COMPUTATIONAL_LIBRARY`` (the build's requested
    backend); marks the worker with ``NORNIR_POOL_WORKER=1`` instead.
    """
    global _active_lib
    if multiprocessing.parent_process() is None:
        return
    _active_lib = ComputationLib.numpy
    os.environ[NORNIR_POOL_WORKER_ENV] = '1'


def HasCuVS() -> bool:
    """Return True if the GPU distance stack needed by cupyx ``cdist`` is available.

    Matches ``cupyx.scipy.spatial.distance`` soft dependencies: ``cuvs.distance``
    or legacy ``pylibraft.distance``. CuPy must be active.

    :return: True if pairwise distance primitives are importable, False otherwise.
    """
    return _has_cupy and _has_cuvs

def HasCupy() -> bool:
    """Return True if the cupy package is available on this system.

    :return: True if cupy can be imported, False otherwise.
    """
    return _has_cupy

def UsingCupy() -> bool:
    """Return True if the active computation library is cupy.

    :return: True when active lib is cupy, False when numpy.
    """
    return _active_lib == ComputationLib.cupy

def TryInitCupyContext() -> bool:
    """Initialize the CuPy GPU context so later calls do not block on first use.

    Useful to avoid timeouts in tests. No-op when not using CuPy.

    :return: True if CuPy is active and context was used, False otherwise.
    :raises Exception: Propagated from CuPy/CUDA if GPU initialization or allocations fail.
    """

    if UsingCupy():
        if cp is None:
            return False
        if not _probe_cupy_runtime():
            return False
        result = cp.random.random((2, 2))
        result = cp.array((1, 2, 3))
        return True

    return False


def SetActiveComputationLib(lib: ComputationLib) -> None:
    """Set the active computation backend (numpy or cupy).

    Updates ``NORNIR_COMPUTATIONAL_LIBRARY`` for this process only. Does not
    modify ``CUDA_VISIBLE_DEVICES``; GPU visibility remains under container /
    orchestrator control.

    :param lib: ComputationLib.numpy or ComputationLib.cupy.
    :raises ModuleNotFoundError: If cupy is requested but not available.
    :raises RuntimeError: If cupy is requested from a child process.
    """
    global _active_lib, _cupy_runtime_probed

    if lib == ComputationLib.cupy and not _has_cupy:
        raise ModuleNotFoundError("Cupy is not available")

    if lib == ComputationLib.cupy and multiprocessing.parent_process() is not None:
        raise RuntimeError("Cupy untested in a child process")

    if lib == ComputationLib.cupy:
        if _cupy_runtime_probed and _active_lib == ComputationLib.cupy:
            _set_effective_computational_library_env('cupy')
            return

        _ensure_cuda_toolkit_env()
        _active_lib = ComputationLib.cupy
        if not _probe_cupy_runtime():
            _active_lib = ComputationLib.numpy
            _set_effective_computational_library_env('numpy')
            _logger.warning(
                "CuPy runtime probe failed; using NumPy backend "
                "(NORNIR_COMPUTATIONAL_LIBRARY=%r, CUDA_VISIBLE_DEVICES=%r). "
                "Assemble will use TilesToImageParallel (CPU) instead of GPU TilesToImage. "
                "Nornir does not modify CUDA_VISIBLE_DEVICES; fix GPU visibility in the "
                "container or shell if CuPy was requested.",
                os.environ.get(NORNIR_COMPUTATIONAL_LIBRARY_ENV),
                os.environ.get('CUDA_VISIBLE_DEVICES'))
            return
        _set_effective_computational_library_env('cupy')
        return

    _active_lib = lib
    _set_effective_computational_library_env('numpy')


def GetActiveComputationLib() -> ComputationLib:
    """Return the currently active computation library (numpy or cupy).

    :return: ComputationLib.numpy or ComputationLib.cupy.
    """
    global _active_lib
    return _active_lib # type: ignore
