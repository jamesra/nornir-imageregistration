from typing import TypeAlias

from numpy.typing import NDArray

from nornir_imageregistration.mmap_metadata import memmap_metadata
from nornir_imageregistration.shared_mem_metadata import Shared_Mem_Metadata

ImageLike: TypeAlias = NDArray | str | memmap_metadata | Shared_Mem_Metadata
