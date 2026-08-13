# nornir-imageregistration

Core image registration algorithms for aligning 2D images into larger mosaics and 3D volumes.

## Documentation

- **Full manual and API (umbrella):** [https://nornir.github.io/](https://nornir.github.io/)
- **This package:** [Packages — nornir-imageregistration](https://nornir.github.io/packages/nornir_imageregistration.html)
- **API reference:** [`nornir_imageregistration` module](https://nornir.github.io/api/nornir_imageregistration.html)

## Optional: CuVS

The `gpu_cuvs` extra enables GPU pairwise `cdist` (via cupyx) whenever inputs are already CuPy arrays. Nearest-neighbor search still uses scipy `cKDTree` below 4096 points: CuVS brute-force is O(N²) and slower than a 2D tree at typical mesh sizes. At 4096 points or more the index switches to CuVS (`NORNIR_CUVS_NN_MIN_POINTS` overrides the gate). Install with `pip install nornir-imageregistration[gpu_cuvs]`. CuVS wheels are provided by NVIDIA for **Linux only** (`cuvs-cu13` with CUDA 13 / `cupy-cuda13x`). Headless `nornir:dev` and `nornir:cupy` images install `cuvs-cu13` next to CuPy. On other platforms or without the extra, the package uses scipy for nearest-neighbor with no change to the public API.
