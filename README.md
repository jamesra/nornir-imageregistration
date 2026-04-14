# nornir-imageregistration

Core image registration algorithms for aligning 2D images into larger mosaics and 3D volumes.

## Documentation

- **Full manual and API (umbrella):** [https://nornir.github.io/](https://nornir.github.io/)
- **This package:** [Packages — nornir-imageregistration](https://nornir.github.io/packages/nornir_imageregistration.html)
- **API reference:** [`nornir_imageregistration` module](https://nornir.github.io/api/nornir_imageregistration.html)

## Optional: CuVS

The `gpu_cuvs` extra enables GPU-accelerated nearest-neighbor search in transforms (when CuPy is active). Install with `pip install nornir-imageregistration[gpu_cuvs]`. CuVS wheels are provided by NVIDIA for **Linux only** and supported Python versions (e.g. 3.10–3.13 on PyPI). On other platforms or without the extra, the package uses scipy for nearest-neighbor with no change to the public API.
