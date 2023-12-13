__all__ = ['ComputationLib', 'HasCupy', 'UsingCupy', 'GetActiveComputationLib', 'SetActiveComputationLib']

from enum import Enum
import multiprocessing
import numpy as np

class ComputationLib(Enum):
    numpy = 0
    cupy = 1

_has_cupy = False # type: bool

# If we are in a child process, we use numpy, GPU processing currently runs on a single process
_active_lib = None if multiprocessing.parent_process() is None else ComputationLib.numpy # type: ComputationLib

try:
    import cupy as cp
    _has_cupy = True
    _active_lib = ComputationLib.cupy if _active_lib is None else _active_lib
except ImportError:
    cp = None
    _has_cupy = False
    _active_lib = ComputationLib.numpy if _active_lib is None else _active_lib
except ModuleNotFoundError:
    cp = None
    _has_cupy = False
    _active_lib = ComputationLib.numpy if _active_lib is None else _active_lib

def HasCupy() -> bool:
    """Return true if cupy is available"""
    return cp is None

def UsingCupy() -> bool:
    return _active_lib == ComputationLib.cupy

def SetActiveComputationLib(lib: ComputationLib):
    """Set the active computational library"""
    global _active_lib

    if lib == ComputationLib.cupy and not _has_cupy:
        raise ModuleNotFoundError("Cupy is not available")

    if lib == ComputationLib.cupy and multiprocessing.parent_process() is not None:
        raise RuntimeError("Cupy untested in a child process")

    _active_lib = lib


def GetActiveComputationLib() -> ComputationLib:
    """Get the active computational library"""
    global _active_lib
    return _active_lib
