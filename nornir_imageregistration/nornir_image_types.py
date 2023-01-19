from numpy.typing import NDArray
from nornir_imageregistration.mmap_metadata import memmap_metadata

ImageLike = NDArray | str | memmap_metadata
