__all__ = ['ComputationLib', 'HasCupy', 'HasCuVS', 'UsingCupy', 'GetActiveComputationLib', 'SetActiveComputationLib']

from enum import Enum
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

_add_cuda_bin_to_dll_path()

class ComputationLib(Enum):
    numpy = 0
    cupy = 1

_has_cupy = False  # type: bool
_has_cuvs = False  # type: bool

# If we are in a child process, we use numpy, GPU processing currently runs on a single process
_active_lib = None if multiprocessing.parent_process() is None else ComputationLib.numpy # type: ComputationLib | None

try:
    import cupy as cp
    _has_cupy = True
    _active_lib = ComputationLib.cupy if _active_lib is None else _active_lib
    # Probe required CUDA libs at startup; if any are missing (e.g. cublasLt, curand), fall back to numpy
    if _active_lib == ComputationLib.cupy:
        try:
            a = cp.array([[1.0, 0.0], [0.0, 1.0]])
            cp.linalg.inv(a)
            cp.random.standard_normal((2, 2))  # triggers curand*.dll load
        except Exception:
            _active_lib = ComputationLib.numpy
    # cupyx.scipy.spatial.distance.cdist checks the same imports (cuvs.distance.pairwise_distance
    # or legacy pylibraft.distance). Neighbors-only cuvs (e.g. brute_force) is not enough.
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
    """
    
    if UsingCupy():
        if cp is None:
            return False
        result = cp.random.random((2, 2))
        result = cp.array((1, 2, 3))
        return True
    
    return False
    

def SetActiveComputationLib(lib: ComputationLib) -> None:
    """Set the active computation backend (numpy or cupy).

    :param lib: ComputationLib.numpy or ComputationLib.cupy.
    :raises ModuleNotFoundError: If cupy is requested but not available.
    :raises RuntimeError: If cupy is requested from a child process.
    """
    global _active_lib

    if lib == ComputationLib.cupy and not _has_cupy:
        raise ModuleNotFoundError("Cupy is not available")

    if lib == ComputationLib.cupy and multiprocessing.parent_process() is not None:
        raise RuntimeError("Cupy untested in a child process")

    _active_lib = lib


def GetActiveComputationLib() -> ComputationLib:
    """Return the currently active computation library (numpy or cupy).

    :return: ComputationLib.numpy or ComputationLib.cupy.
    """
    global _active_lib
    return _active_lib # type: ignore