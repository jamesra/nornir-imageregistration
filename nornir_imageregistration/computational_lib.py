__all__ = ['ComputationLib', 'HasCupy', 'GetActiveComputationalLib', 'SetActiveComputationalLib']

from enum import Enum

class ComputationLib(Enum):
    numpy = 0
    cupy = 1

_has_cupy = False # type: bool
_active_lib = ComputationLib.numpy # type: ComputationLib

try:
    import cupy as cp
    _has_cupy = True
except ImportError:
    cp = None
    _has_cupy = False
except ModuleNotFoundError:
    cp = None
    _has_cupy = False

def HasCupy() -> bool:
    """Return true if cupy is available"""
    return cp is None

def SetActiveComputationalLib(lib: ComputationLib):
    """Set the active computational library"""
    global _active_lib

    if lib == ComputationLib.cupy and not _has_cupy:
        raise ModuleNotFoundError("Cupy is not available")

    _active_lib = lib


def GetActiveComputationalLib() -> ComputationLib:
    """Get the active computational library"""
    global _active_lib
    return _active_lib
