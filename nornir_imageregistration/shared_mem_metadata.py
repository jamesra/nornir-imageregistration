from typing import NamedTuple
from numpy.typing import DTypeLike

class Shared_Mem_Metadata(NamedTuple):
    name: str
    shape: tuple
    dtype: DTypeLike
    readonly: bool
