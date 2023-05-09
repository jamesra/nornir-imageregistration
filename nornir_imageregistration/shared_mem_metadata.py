from typing import NamedTuple
from numpy.typing import DTypeLike
from multiprocessing.shared_memory import SharedMemory

class Shared_Mem_Metadata(NamedTuple):
    name: str
    shape: tuple
    dtype: DTypeLike
    readonly: bool
    shared_memory: SharedMemory
