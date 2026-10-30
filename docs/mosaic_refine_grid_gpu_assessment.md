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
- **Promoted to production:** the batched gate (later generalized to
  `_use_batched_vertex_measurement()`, see the equalization section below)
  defaults the batched path **on** whenever `UsingCupy()`. The NumPy path was
  unchanged at this point (always serial). Set `NORNIR_REFINE_BATCHED=0`
  (or the legacy `NORNIR_REFINE_BATCHED_GPU=0`) to force the legacy serial path.
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

## Warp + transfer optimization (follow-up results)

Two changes targeting the prewarp bucket and residual per-cell syncs.

Box: NVIDIA RTX 4500 Ada / AMD Ryzen 9 9950X. Grid690, cell 96x96, mesh 8x8,
5 passes. Warm repeat (best of 2).

### Change 1 - Batched cell-validity reduction (default, output-neutral)

`_extract_refinement_cell` previously called `float(xp.count_nonzero(...))` per
cell - a device->host scalar sync on the order of vertices x neighbors x tiles x
passes. The batched vertex path now extracts all candidate cells with
`_extract_refinement_cell_and_mask` (no per-cell sync) and reduces validity in
one batched `count_nonzero` + a single `.get()`.

This is **byte-identical** to the prior output (verified: CPU-vs-batched mean
0.226 px and golden 1.857/1.880 reproduced to 4 decimals), and on its own:

| metric        | before | after (batched validity) |
|---------------|--------|---------------------------|
| `cell_extract`| 0.40 s | 0.089 s                   |
| cells/s       | 80.7   | **95.9**                  |

That alone moves the batched GPU path **past warm CPU (~89)** with no output
change. It is the new default.

### Change 2 - Single-warp prewarp (opt-in: `NORNIR_REFINE_PREWARP_MODE=singlewarp`)

Prewarp warped each tile twice/pass: the image, then a `ones`-image to build the
validity mask. Investigation showed the legacy coverage mask is a **latent
artifact**: `_TransformImageUsingCoords` clips its output to the source min/max,
so the warped ones-image background is raised to 1.0 and `coverage > 0.999` is
**uniformly True** for interior tiles - the per-pixel coverage was effectively
unused. The true coverage is exactly the warp's scatter targets, so it can be
returned for free from the single image warp (`return_valid_mask=True` on
`_TransformImageUsingCoords`), eliminating the second `map_coordinates`.

Because this replaces the degenerate all-true mask with the real coverage, it
**changes registration output slightly**, so it is gated behind
`NORNIR_REFINE_PREWARP_MODE=singlewarp` (default keeps the exact two-warp
behavior).

| config (batched)            | prewarp (s) | cells/s | golden delta | cpu-vs-batched |
|-----------------------------|-------------|---------|--------------|----------------|
| default (two-warp)          | 2.18        | 95.9    | 1.880        | 0.226          |
| `singlewarp`                | 1.40        | **126.6** | 2.005 (<2.2) | 0.181          |

`singlewarp` parity: golden 2.005 px (< 2.2 gate), seam MAE 0.089/0.117 (< 35),
CPU-vs-batched mean 0.181 / max 0.238 px, batched-vs-serial < 1.0. All pass; the
~0.12 px shift from the legacy 1.88 is the coverage-bug fix.

### Host/device transfer audit (current state)

- **Prewarp coord generation** (per tile per pass): the grid `InverseTransform`
  (`GridWithRBFFallback`) runs on CPU, forcing a D2H of the full target ROI grid
  and an H2D re-upload of the filtered coords. This is the largest remaining
  avoidable trip; it is folded into the `prewarp` bucket. Fixing it needs a
  GPU-native grid inverse transform - high effort, deferred.
- **Per-cell `count_nonzero`**: removed (Change 1, batched reduction).
- **Per-vertex peak `.get()`**: already removed by the batched FFT path (one
  `.get()` per neighbor).
- **Overlap mask**: uploaded once per geometry and cached on device.

### Decision

- **Ship Change 1 (batched validity) as the default** - output-neutral and
  alone clears warm CPU (95.9 vs ~89 cells/s).
- **Ship Change 2 (`singlewarp`) as opt-in** (1.4x CPU at 126.6 cells/s) since it
  alters registration output; promote to default only with explicit sign-off
  (it is arguably a correctness improvement, well within the golden 2.2 px gate).
- **Next lever:** GPU-native grid `InverseTransform` to remove the per-pass
  coord D2H/H2D round-trip, which now dominates the prewarp bucket.

## GPU inverse transform (opt-in: `NORNIR_REFINE_GPU_TRANSFORM`)

The "next lever" above, now implemented and measured.

### Prize measurement (Step 0)

A temporary sub-timer around `write_to_target_roi_coords` (the prewarp coord
generation) and a decomposition of its internals (device-synced, 20 tile
prewarps on Grid690) showed:

| coordgen sub-step    | time   | share |
|----------------------|--------|-------|
| `GetROICoords`       | 0.019 s| 1.6 % |
| `InverseTransform`   | 0.717 s| 58.3 %|
| `InvalidIndices`+mask| 0.494 s| 40.1 %|
| **total coordgen**   | 1.230 s| 100 % |

Coordgen is ~53-60 % of the whole `prewarp` bucket, and ~98 % of it
(`InverseTransform` + mask) is host-round-trip cost: the CPU
`GridWithRBFFallback.InverseTransform` forces the H x W ROI grid to host (SciPy
LinearND) and returns NumPy, which then makes the downstream `InvalidIndices`
mask a NumPy-mask-on-CuPy-array boundary. Material - proceed.

### Implementation

- `ConvertTransformToGridTransform` gained `prefer_gpu: bool = False`; when true
  under CuPy with cupyx `LinearNDInterpolator` available it builds
  `GridWithRBFFallback_GPUComponent` (mirrors `factory.ParseGridTransform`).
  Default false keeps every other caller (STOS, tests) on the CPU class.
- `_refine_gpu_transform_enabled()` gates the flag (requires `UsingCupy()` +
  cupyx LinearND) and is wired only from `_initialize_tile_grid_transforms`.
- `RefineGridMosaic` does not resample at the end (the refined grid is the
  output), so the finalize loop now downconverts a GPU-component transform back
  to the CPU `GridWithRBFFallback` (`PopulateTargetPoints` brings the points to
  host). Verified: with the flag on, all output transforms are
  `GridWithRBFFallback` with NumPy `TargetPoints` and `ToITKString()` succeeds -
  the saved mosaic stays host-backed.

### Degenerate-triangulation fallback (the Step 3 watch item)

cupyx `LinearNDInterpolator` is stricter than SciPy. The **initial** refine grid
is a perfect regular lattice, which is Delaunay-degenerate, so cupyx raises and
the documented SciPy Qhull fallback fires - but only on **pass 1** (4 tiles).
After pass 1 perturbs the vertices, cupyx triangulates fine. Counter
instrumentation over a full run:

| interpolator build outcome                  | count | share |
|---------------------------------------------|-------|-------|
| cupyx success (on-device)                   | 16    | 80 %  |
| cupyx -> SciPy fallback (pass-1 lattice)    | 4     | 20 %  |
| forced SciPy                                | 0     | 0 %   |

The fallback is logged (not silent) and bounded to the first pass; the on-device
inverse engages for the large majority of work.

### Throughput (Grid690, cell 96x96, mesh 8x8, batched; warm best-of-2)

| config (batched)              | prewarp (warm) | wall (warm) | passes | cells/s (warm) | vs baseline |
|-------------------------------|----------------|-------------|--------|----------------|-------------|
| default (CPU transform)       | 1.81 s         | 2.62 s      | 5      | 104.8          | -           |
| `NORNIR_REFINE_GPU_TRANSFORM` | 0.75 s         | 1.39 s      | 4      | **164.5**      | +57 %       |
| `singlewarp`                  | 1.53 s         | 2.25 s      | 5      | 111.9          | +7 %        |
| GPU transform + `singlewarp`  | 0.64 s         | 1.34 s      | 4      | **168.3**      | +61 %       |

(Absolute cells/s runs hotter than the earlier sections' table due to
session/thermal variance; the relative deltas are the signal.) The GPU transform
is the dominant lever: prewarp **-58 %**, cells/s **+57 %**. It stacks with
`singlewarp` (combined prewarp -65 %). The pass count drops 5 -> 4 because the
cupyx-vs-SciPy LinearND difference shifts the convergence trajectory.

### Parity (Step 3)

Output-shifting (cupyx LinearND is the same algorithm as SciPy but not
bit-identical), so gated and validated:

| gate                            | flag OFF | flag ON   | limit  | result |
|---------------------------------|----------|-----------|--------|--------|
| golden mean delta (px)          | 1.880    | **1.353** | <= 2.2 | PASS   |
| seam max MAE                    | 0.117    | 0.116     | < 35   | PASS   |
| batched-vs-serial mean (px)     | -        | 0.771     | <= 1.0 | PASS   |
| cpu(numpy)-vs-gpu-xform mean(px)| -        | 1.37      | <= 1.0 | over*  |
| cpu(numpy)-vs-gpu-xform max (px)| -        | 2.44      | <= 3.0 | PASS   |

`PARITY VERDICT: PASS` (golden gate). *The cpu-vs-gpu-xform mean is above
`verify_cpu_vs_batched`'s 1.0 limit, which was calibrated for the byte-near
batched-only change; both legs are inside the authoritative golden gate. The GPU-transform output is actually **closer to the C++
golden** (1.353) than the default path (1.880) and than the CPU/serial reference
(1.774). The direct CPU(numpy)-vs-GPU-transform mean is 1.37 px (max 2.44):
above `verify_cpu_vs_batched`'s 1.0 mean limit (which was calibrated for the
byte-near batched-only change), but both legs are well within the authoritative
golden gate and the GPU path is the more accurate of the two.

### Transfer audit update

- **Prewarp coord generation**: the per-pass `InverseTransform` D2H/H2D and the
  NumPy-mask-on-CuPy boundary are eliminated for ~80 % of builds (all but the
  pass-1 degenerate lattice). This was the largest remaining avoidable trip.
- Final mosaic serialization stays CPU-backed (downconvert in finalize loop).

### Decision

- **Ship `NORNIR_REFINE_GPU_TRANSFORM` as opt-in** (default off). It is the
  single biggest refine lever measured (prewarp -58 %, cells/s +57 %) and parity
  PASSES the golden gate (and is closer to golden than the default), but it is
  output-shifting, so per the benchmark-driven default policy it stays opt-in
  pending explicit per-flag sign-off rather than auto-flipping the default.
- Promotion recommendation (for sign-off): the change is arguably a net
  accuracy improvement (golden 1.880 -> 1.353) and stacks with `singlewarp`. A
  per-tile worst-case check on extreme-warp tiles (max delta 2.44 px here) should
  accompany any default flip, consistent with the `singlewarp` promotion gate.
- Out of scope (unchanged): forward-transform GPU residency, RBF continuous
  fallback parity, STOS wiring; pass-1 degeneracy could later be avoided with a
  tiny lattice joggle to push that last 20 % on-device.

## Equalizing the CPU and GPU paths (RPC3/0601, ds4)

Earlier phases optimized one backend at a time, so each path had a lever the
other was denied: the **GPU** had the batched vertex FFT but a serial prewarp;
the **CPU** had a parallel prewarp pool and the cross-pass cache but only the
serial per-vertex loop. This phase gave both paths the same opportunities,
validated parity, and flipped only the defaults the benchmarks justified.

Benchmarked on a full production-resolution section: RPC3 section **0601**,
4x-downsampled tiles (`Leveled/TilePyramid/004`, 1024x1024), `--image-scale
0.25`, cell 128x128, mesh 11x11, 5 iterations
([scripts/microbench_mosaic_refine.py](../scripts/microbench_mosaic_refine.py);
see the `nornir-realworld-refine-benchmark` skill). Box: NVIDIA RTX 4500 Ada /
AMD Ryzen 9 9950X. `cells/s` is the backend-fair metric (the batched peak
algorithm converges in 5 passes / 53 347 cells vs the CPU-serial 4 passes /
42 778, so wall time alone is not comparable).

### Phase 1 head-to-head

| backend | path                         | passes | wall (s) | cells/s | vs its old default |
|---------|------------------------------|--------|----------|---------|--------------------|
| numpy   | serial (old CPU default)     | 4      | 92.2     | 463.8   | -                  |
| numpy   | **batched** (new opt-in)     | 5      | 81.0     | **659.0** | **+42 %**        |
| cupy    | serial prewarp (old default) | 5      | 106.6    | 500.4   | -                  |
| cupy    | **thread prewarp** (new default) | 5  | 85.8     | **621.8** | **+24 %**        |

Both backends improved materially once each got the other's lever. After
equalization the CPU-batched path is the throughput leader (659) and the GPU
thread-prewarp path is a close second (622); the two are within ~6 %.

### What changed in the gates

- **Batched vertex measurement generalized to NumPy.** The gate is now the
  backend-agnostic `_use_batched_vertex_measurement()` reading
  `NORNIR_REFINE_BATCHED` (legacy `NORNIR_REFINE_BATCHED_GPU` honored as an
  alias). Default **on** for both backends (NumPy promoted after sign-off; it is
  output-shifting: argmax+centroid vs connected-component `find_peak`). Parity:
  CPU-serial vs CPU-batched mean **0.226 px** / max **0.253** (limits 1.0 / 3.0);
  vs golden CPU-serial 1.857, CPU-batched 1.880 (< 2.2); functional + legacy
  tests pass under the batched default.
- **Thread prewarp is now the CuPy default.** `_prewarp_thread_dispatch_enabled()`
  defaults thread dispatch on under CuPy (`serial` token opts out); CPU keeps its
  multiprocess pool unless `thread` is requested. It is **output-neutral**:
  cupy serial-prewarp vs thread-prewarp is byte-identical (mean/max **0.0**), so
  it auto-flips per the output-neutral policy. The earlier Grid690 "thread is
  neutral" finding does **not** contradict this: Grid690 has 4 tiles, so there is
  almost no host-side CPU `InverseTransform` work to overlap; a real section
  (~130 tiles/pass) has enough that overlapping it with warp kernels saves ~20 %.

### Phase 2 - GPU prewarp cache: rejected (no benefit)

`NORNIR_REFINE_PREWARP_MODE=cache` under CuPy (composes with the new thread
default) was **slower** at section scale: 89.1 s / 598 cells/s vs thread-only
85.8 s / 622. In grid refine most tiles get a nonzero applied shift every pass,
so the lattice revision changes and the cache misses; the bookkeeping costs more
than it saves. Parity is clean (thread vs thread+cache byte-identical, 0.0/0.0 -
the historical "cache shifts output under CuPy" caveat no longer reproduces),
but with no speed prize there is **no default flip and no sign-off needed**. The
mode stays opt-in.

### Phase 3 - Tile-level measurement parallelism: implemented, kept off

The per-tile measurement loop was refactored into `_measure_tile_grid_update`
(pure w.r.t. shared state) and can dispatch tiles across the shared thread pool
via `NORNIR_REFINE_TILE_PARALLEL`. `_RefinePhaseTimer` was made thread-safe (a
lock; it is still a no-op when timing is disabled). It is **output-neutral**
(off vs on byte-identical on both numpy and cupy, 0.0/0.0), but a **regression**
on both backends:

| backend | tile-parallel off | tile-parallel on |
|---------|-------------------|------------------|
| numpy (batched) | 81.0 s / 659 cells/s | 94.8 s / 563 |
| cupy (thread)   | 85.8 s / 622 cells/s | 161.0 s / 331 |

CPU oversubscribes (the batched FFT/BLAS is already internally multithreaded);
CuPy serializes kernels on the default stream while adding thread/memory-pool
contention. It stays **default-off**, retained as an opt-in hook for future
CUDA-stream (GPU) or single-thread-BLAS (CPU) experiments.

### Net outcome

| flag                         | backend | default after this phase | classification |
|------------------------------|---------|--------------------------|----------------|
| `NORNIR_REFINE_BATCHED`      | CuPy    | on                       | output-shifting (validated) |
| `NORNIR_REFINE_BATCHED`      | NumPy   | **on (new, signed off)** | output-shifting (validated) |
| `NORNIR_REFINE_PREWARP_MODE=thread` | CuPy | **on (new)**          | output-neutral  |
| `NORNIR_REFINE_PREWARP_MODE=cache`  | CuPy | off (no benefit)       | output-neutral  |
| `NORNIR_REFINE_TILE_PARALLEL`| both    | off (regression)         | output-neutral  |

Two defaults flipped: the CuPy thread prewarp (output-neutral, +24 %) and -
after explicit sign-off - the NumPy batched vertex path (output-shifting,
+42 %). The batched path now defaults on for **both** backends; set
`NORNIR_REFINE_BATCHED=0` to fall back to the legacy serial peak finder. Parity
for the NumPy flip: CPU-serial vs CPU-batched 0.226 px mean / 0.253 max
(limits 1.0 / 3.0), golden 1.857 -> 1.880 (< 2.2), functional + legacy-parity
tests pass under the batched default.

## Related comparison

For STOS vs mosaic safeguard differences, shared `refine_shared` helpers, and
env gates such as `NORNIR_REFINE_MOSAIC_CUTOFF` / `NORNIR_REFINE_STOS_REGULARIZE`,
see [grid_refine_stos_vs_mosaic.md](grid_refine_stos_vs_mosaic.md).
