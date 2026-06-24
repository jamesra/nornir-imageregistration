# Mosaic Refine Grid - GPU Performance Assessment

Assessment-only findings for `RefineGridMosaic` / `_refine_tileset`
(`nornir_imageregistration/local_distortion_correction.py`). No algorithm or
parallelism changes were made; this phase adds an opt-in phase timer plus a
benchmark harness and records where time goes, so a follow-up GPU refactor can
be evidence-driven.

## How to reproduce

- Harness: `scripts/microbench_mosaic_refine.py`
  - `python scripts/microbench_mosaic_refine.py --backend both --cell-sweep 64 96 128 --repeats 2 --iterations 10`
  - `--backend both` runs each backend in an isolated subprocess; the `numpy`
    subprocess is launched with `CUDA_VISIBLE_DEVICES=""` so its fork-based
    prewarp pool cannot inherit a CUDA context (and it reflects a CPU-only host).
  - For a direct single-backend `--backend numpy` run on a GPU box, set
    `CUDA_VISIBLE_DEVICES=""` yourself, otherwise the prewarp fork pool crashes
    with `cudaErrorInitializationError`.
- Per-phase split: add `--phase-timing` (sets `NORNIR_REFINE_PHASE_TIMING=1`).
  `_refine_tileset` then logs a per-pass and total breakdown of
  `prewarp / cell_extract / fft / host_sync / regularize / apply`. Under CuPy
  each section synchronizes the device so timings reflect kernel completion.
- Transfer boundaries: `scripts/audit_host_device_transfers.py` (static scan).

## Test configuration

- Hardware: NVIDIA RTX 4500 Ada Generation; AMD Ryzen 9 9950X (16C / 32T).
- Fixture: `RC2_4Square_Assembled` section 0690, translated mosaic, L4 tiles
  (`Leveled/TilePyramid/004`), 20 tiles.
- Params: `mesh=8x8`, `image_scale=0.25`, `displacement_threshold=0.5`,
  `iterations=10` (converges in 4-5 passes).

## Result 1: CPU is ~2.2x faster than GPU as currently architected

Best wall time over 2 repeats (warm pool); `cells` = total FFT cells measured
across all passes; `cells/s` = `cells / wall`.

| backend | cell    | passes | best wall (s) | cells/s |
|---------|---------|--------|---------------|---------|
| numpy   | 64x64   | 4      | 2.56          | 87.8    |
| numpy   | 96x96   | 5      | 3.07          | 89.4    |
| numpy   | 128x128 | 5      | 3.35          | 82.7    |
| cupy    | 64x64   | 4      | 5.70          | 39.5    |
| cupy    | 96x96   | 5      | 7.39          | 37.1    |
| cupy    | 128x128 | 5      | 7.97          | 34.8    |

`numpy / cupy` best-wall ratio is ~0.45: the GPU path is roughly **2.2x slower**
than the threaded CPU path on this workstation-class GPU. GPU throughput is
nearly flat (~35-40 cells/s) regardless of cell size, which is the signature of
a launch/synchronization-bound workload rather than a compute-bound one.

## Result 2: where the time goes (per-phase split, 96x96, 5 passes, 274 cells)

| phase        | numpy           | cupy             |
|--------------|-----------------|------------------|
| prewarp      | dominant (pool) | 2.174 s (29.7%)  |
| cell_extract | 0.015 s         | 0.529 s (7.2%)   |
| fft          | 0.276 s (2.6%)  | 4.585 s (62.6%)  |
| host_sync    | 0.001 s         | 0.004 s (0.1%)   |
| regularize   | 0.014 s         | 0.021 s (0.3%)   |
| apply        | 0.003 s         | 0.013 s          |

Per-cell FFT-stage cost: **~16.7 ms/cell on GPU vs ~1.0 ms/cell on CPU** (274
cells). The 96x96 cells are far too small to amortize GPU kernel launch and the
per-cell device synchronization, so the GPU spends ~17x longer per cell.

Note on the numpy `prewarp` bucket: it wraps the multiprocess prewarp pool's
`wait_completion()`, so it captures pool dispatch/startup wall time and is the
CPU path's dominant cost. The CPU vertex FFT loop itself is cheap (0.276 s).

## Result 3: the GPU hotspot is find_peak, not the FFT

cProfile of one CuPy run (5 passes, 274 cells):

| function (phasecorrelation.py)     | ncalls | cumtime (s) | per-call |
|------------------------------------|--------|-------------|----------|
| `find_offset`                      | 274    | 4.147       | 15.1 ms  |
| `find_peak`                        | 274    | 3.034       | 11.1 ms  |
| `image_phase_correlation` (fft2)   | 274    | 0.881       | 3.2 ms   |
| `_prewarp_tile_for_grid_refine`    | 20     | 2.239       | 112 ms   |

The dominant per-cell GPU cost is **`find_peak`** (fftshift + overlap-mask +
percentile + argmax), not `fft2`/`ifft2`. Any batching effort must batch the
peak-detection stage, not only the transform.

## Result 4: per-cell host sync count

Static audit flags the boundary at `local_distortion_correction.py:650`:
`EnsureNumpyArray(record.peak)` inside `_measure_grid_vertex_displacements`.
Runtime: one `.get()` per measured vertex (~274 per 5-pass run). The payload is
tiny (a 2-element peak), so the bytes are negligible, but it forces a device
sync per cell, which serializes the otherwise-asynchronous GPU pipeline. The
structural problem is the serial per-cell pattern, not the transfer volume.

## STOS reference (different algorithm, shared FFT core)

`scripts/microbench_stos_refinement.py` (Grid8 stos, 3 iterations, CuPy):
~35-61 s per run. STOS refinement (`RefineTransform` / `AttemptAlignPoint`) runs
a brute-force rigid angle search per grid cell, far heavier than the mosaic
single-translation FFT per vertex, but it uses the **same** `phasecorrelation`
core and the **same** serial-under-CuPy per-cell dispatch. A batching framework
built for the mosaic vertex loop is the natural unification point for both.

## Ranked architecture hypotheses for the follow-up

1. **Batch vertex cells into one GPU FFT + batched peak detection.** Stack the
   per-vertex `(N, h, w)` cells and run `fft2`/`ifft2` and (critically) the
   `find_peak` stage over the whole batch, eliminating per-cell launch/sync.
   Targets the 60%+ `fft`/`find_peak` bucket. Evidence: 274 serial cells at
   ~17 ms vs ~1 ms on CPU; `find_peak` is ~11 ms of that. *Unifies with STOS.*
2. **Vectorize `find_peak` across cells** (fftshift, overlap masking,
   percentile, argmax as batched ops). This is the actual GPU hotspot and is a
   prerequisite for hypothesis 1 paying off. Evidence: `find_peak` cum 3.03 s of
   `find_offset`'s 4.15 s.
3. **Remove the per-cell host sync.** Accumulate peaks on-device and do a single
   `.get()` per tile/neighbor instead of per vertex. Evidence: ~274 `.get()`
   per run, each forcing a device sync. (Enables 1 and 2.)
4. **Re-enable parallel / stream-overlapped prewarp under CuPy.** Prewarp is
   forced serial under CuPy (2.17 s, ~30%) while the CPU path threads it.
   Evidence: `_prewarp_all_tiles_for_grid_refine` serial branch when
   `UsingCupy()`. Medium priority; secondary to the vertex loop.
5. **GPU-resident regularization via `cupyx.scipy.ndimage`.** Currently
   negligible (0.02 s on CPU SciPy), but once 1-3 keep shifts on-device it
   avoids a device->host hop before blending. Low priority until then.
6. **Shared batched cell-extract + phase-correlation helper for mosaic and
   STOS.** Both paths extract cells and call the same FFT core serially; a
   common batched primitive is the unification target.

## Recommendation

Prototype hypotheses 1-3 together (batched GPU FFT + batched `find_peak` +
single sync); they are inseparable and target the bucket that makes the GPU
slower than CPU today. Defer 4-5 until after. If a batched GPU path cannot beat
the CPU's ~85 cells/s, keep the CPU backend as the default for mosaic grid
refine on workstation-class GPUs and reserve GPU effort for the heavier STOS
brute-force search, where per-cell work is large enough to amortize the GPU.

## Batched GPU prototype (follow-up results)

Implemented hypotheses 1-3 as an opt-in prototype:
[`batched_phase_correlation.py`](../nornir_imageregistration/batched_phase_correlation.py)
(`batched_image_phase_correlation`, `batched_find_offset`, vectorized
`batched_find_peak` = masked argmax + local center-of-mass) and
`_measure_grid_vertex_displacements_batched` in `local_distortion_correction.py`,
gated behind `NORNIR_REFINE_BATCHED_GPU` + `UsingCupy()` (serial remains the
default). The batched peak finder replaces the connected-component `find_peak`
with a vectorized argmax + local centroid so the whole `(N, h, w)` stack is one
FFT and one host transfer.

Box: NVIDIA RTX 4500 Ada Generation / AMD Ryzen 9 9950X (16C/32T). Grid690
fixture, cell 96x96, mesh 8x8, 5 passes, 274 cells.

### Throughput (warm repeat, best of 2)

| backend          | wall (s) | cells/s | vs CPU |
|------------------|----------|---------|--------|
| numpy (CPU)      | 3.08     | 89.0    | 1.00x  |
| cupy serial      | 8.31     | 33.0    | 0.37x  |
| cupy **batched** | 3.40     | 80.7    | 0.91x  |

The batched path is a **2.4x** speedup over serial CuPy and reaches ~0.91x of
warm CPU - parity, not yet a win.

### Phase split: serial vs batched CuPy (total over 5 passes, s)

| phase        | cupy serial | cupy batched | change         |
|--------------|-------------|--------------|----------------|
| prewarp      | 2.013 (26%) | 2.038 (75%)  | unchanged      |
| cell_extract | 0.586 (8%)  | 0.395 (15%)  | ~1.5x faster   |
| **fft**      | 5.028 (66%) | 0.211 (8%)   | **~24x faster**|
| host_sync    | 0.004       | 0.028        | (n 274 -> 54)  |
| regularize   | 0.019       | 0.020        | unchanged      |
| apply        | 0.016       | 0.014        | unchanged      |

The batched primitive did exactly what the assessment predicted: it collapsed
the dominant `fft`/`find_peak` bucket from 5.03 s to 0.21 s. The binding cost is
now **prewarp** (2.04 s, 75% of the GPU refine time), which is forced serial
under CuPy (hypothesis 4).

### Parity vs golden (`scripts/compare_refine_peakfinder.py`)

| metric (working-res px)        | serial | batched | limit |
|--------------------------------|--------|---------|-------|
| mean target delta vs golden    | 1.852  | 1.880   | 2.2   |
| seam MAE (mean / max)          | 0.086 / 0.117 | 0.086 / 0.117 | 35 |
| batched-vs-serial target delta | -      | 0.221   | 1.0   |

Both paths converge in 5 passes. The batched peak finder stays within the golden
bound and is essentially identical to serial at the seams.

### Decision: GO - promoted to the production default under CuPy

- **GO** on the batched vertex path: it is a 2.4x GPU win, passes the golden
  parity gate, and removes the FFT/`find_peak` bottleneck entirely.
- **Promoted to production:** `_use_batched_gpu_vertex_measurement()` now
  defaults the batched path **on** whenever `UsingCupy()`. The NumPy path is
  unchanged (always serial). Set `NORNIR_REFINE_BATCHED_GPU=0` to force the
  legacy serial path for A/B or fallback.
- **CPU-vs-batched verification** (`scripts/verify_cpu_vs_batched.py`, isolated
  subprocesses): mean target-point delta **0.226 px**, max **0.253 px** (limits
  1.0 / 3.0); CPU vs golden 1.857, batched vs golden 1.880. The two backends
  produce the same registration within sub-pixel tolerance.
- **Follow-ups (not blocking the default):** promote `batched_find_offset` into
  a shared primitive and unify with STOS (hypothesis 6); prewarp is now the
  binding cost (75%) so a warp-work reduction (not dispatch tuning - see the
  contingency below) is the next lever to push GPU clearly past CPU.

See the [prewarp contingency](#prewarp-contingency-follow-up) below.

## Prewarp contingency (follow-up)

Because the batched path made **prewarp** the binding GPU cost (2.04 s, 75%),
plan Step 7's contingency was triggered: prototype lower-overhead prewarp
dispatch and a CuPy-safe cross-pass cache behind an opt-in flag
(`NORNIR_REFINE_PREWARP_MODE`, default `serial` = current behavior). Tokens
`thread` (dispatch warps on the shared thread pool even under CuPy) and `cache`
(allow the cross-pass prewarp cache under CuPy) can be combined.

### Results (Grid690, cell 96x96, 5 passes, 274 cells)

| config                       | cold cells/s | warm cells/s | prewarp (s) | note |
|------------------------------|--------------|--------------|-------------|------|
| cupy batched, `serial`       | 71.6         | 80.7         | 2.04        | baseline |
| cupy batched, `thread`       | 69.2         | 81.0         | ~2.0        | neutral |
| cupy batched, `cache`        | 35.9         | 49.6         | 3.69        | **worse** |
| numpy `serial` (multiproc)   | 19.6         | 63.4*        | -           | cold = pool spawn |
| numpy `thread`               | 67.1         | 62.1         | -           | no spawn cost |

\* CPU warm throughput is noisy across runs (seen 63-89 cells/s depending on
system load); treat CPU vs GPU as a near-tie at warm steady state.

### Findings

- **CuPy `thread` dispatch: neutral.** The warp (`_TransformImageUsingCoords`,
  nearest-neighbor `map_coordinates`) is GPU-bound and serialized on the single
  device; host tile-load is page-cache-warm. Overlapping load/coord compute with
  threads yields no measurable gain (80.7 -> 81.0 cells/s). The prewarp cost is
  **intrinsic GPU warp compute**, not dispatch overhead - so hypothesis 4
  (re-enable parallel prewarp) does not pay off as a dispatch change.
- **CuPy `cache`: regression, no-go.** Enabling the cross-pass cache under CuPy
  drove prewarp *up* (2.04 -> 3.69 s) and throughput down (80.7 -> 49.6 cells/s):
  in grid refine every tile's lattice moves each pass so the cache never hits,
  while caching device arrays across passes adds GPU memory-pool pressure and
  sync. This confirms the original reason the CuPy cache was disabled.
- **CPU `thread`: helps cold-start only.** Replacing the multiprocess pool with
  the thread pool cut the cold first-pass from ~14 s to ~4 s (no process spawn /
  re-import), with warm throughput unchanged within noise. Useful for short
  refine jobs; not a steady-state win.

### Decision

- **Keep prewarp default (`serial`) unchanged** on both CuPy and CPU.
- **`cache` mode: reject** under CuPy (regression confirmed).
- **`thread` mode: retain as opt-in** for its CPU cold-start benefit; no default
  change.
- To actually beat warm CPU on GPU, the next lever is **reducing the warp work
  itself** (batched / region-limited warp that re-renders only the moved
  neighborhood rather than the full tile every pass), not dispatch tuning. That
  is out of scope for this phase and is the recommended follow-up if GPU mosaic
  refine is pursued as the default.
