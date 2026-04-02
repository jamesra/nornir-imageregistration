nornir-imageregistration
======================

Optional: CuVS
--------------
The ``gpu_cuvs`` extra enables GPU-accelerated nearest-neighbor search in transforms (when CuPy is active). Install with ``pip install nornir-imageregistration[gpu_cuvs]``. CuVS wheels are provided by NVIDIA for **Linux only** and supported Python versions (e.g. 3.10–3.13 on PyPI). On other platforms or without the extra, the package uses scipy for nearest-neighbor with no change to the public API.
