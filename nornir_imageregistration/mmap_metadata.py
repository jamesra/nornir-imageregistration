from numpy.typing import NDArray, DTypeLike
import tempfile
import numpy as np 

class memmap_metadata(object):
    """meta-data for a memmap array"""
    _path: str
    _shape: NDArray
    _dtype: DTypeLike
    _mode: str | None

    @property
    def path(self) -> str:
        return self._path

    @property
    def shape(self) -> NDArray:
        return self._shape

    @property
    def dtype(self) -> DTypeLike:
        return self._dtype

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str | None):
        # Default to copy-on-write
        if value is None:
            self._mode = 'c'
            return

        if not isinstance(value, str):
            raise ValueError("Mode must be a string and one of the allowed memmap mode strings, 'r','r+','w+','c'")

        self._mode = value

    def __init__(self, path: str, shape: NDArray, dtype: DTypeLike, mode: str | None = None):
        self._path = path
        self._shape = shape
        self._dtype = dtype
        self._mode = None
        self.mode = mode


def CreateTemporaryReadonlyMemmapFile(npArray: NDArray) -> memmap_metadata:
    with tempfile.NamedTemporaryFile(suffix='.memmap', delete=False) as hFile:
        TempFullpath = hFile.name
        hFile.close()
    memImage = np.memmap(TempFullpath, dtype=npArray.dtype, shape=npArray.shape, mode='w+')
    memImage[:] = npArray[:]
    memImage.flush()
    del memImage
    # np.save(TempFullpath, npArray)
    return memmap_metadata(path=TempFullpath, shape=npArray.shape, dtype=npArray.dtype)
