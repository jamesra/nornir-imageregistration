# Grid refinement: STOS vs mosaic

Comparison of slice-to-slice (`RefineTransform` / `RefineStosFile`) and mosaic
(`RefineGridMosaic` / `_refine_tileset`) grid refinement. Both live under
`nornir_imageregistration` and share phase-correlation primitives, but use
different iterative drivers.

## Entry points

| Path | Build entry | Core refine |
|------|-------------|-------------|
| Mosaic | `registration.GridTransform` (`Mosaic` pipeline) | `RefineGridMosaic` → `_refine_tileset` |
| STOS | `block.StosGridRefine` / `RefineInvoker` (`RefineSectionAlignment`) | `RefineStosFile` → `RefineTransform` |

Shared helpers live in `nornir_imageregistration.refine_shared`.

## Algorithm summary

**Mosaic:** prewarp tiles into mosaic space → translation-only phase correlation
at each mesh vertex vs overlapping neighbors → spatial regularization
(median / gap-fill / Gaussian) → mass-weighted blend → add shifts to grid target
points. Stops when max displacement ≤ threshold or a pass fails to improve.

**STOS:** grid on source image → per-cell brute rigid registration (optional
rotation) → `estimate_cutoff` on registration weights → rebuild
`MeshWithRBFFallback` → finalize/lock stable cells → optional
`TryToImproveAlignments`. Final pass freezes inclusion cutoff to the first-pass
value.

## Safeguard matrix

### Present in STOS, not mosaic (by default)

| Safeguard | STOS | Mosaic note |
|-----------|------|-------------|
| `estimate_cutoff` outlier rejection | Always | Opt-in via `NORNIR_REFINE_MOSAIC_CUTOFF=1` |
| Per-cell finalization / locking | Always | Vertices may keep drifting |
| First-pass cutoff frozen on final pass | Always | N/A (no transform rebuild) |
| `TryToImproveAlignments` | Always | N/A |
| Explicit tissue masks | `min_unmasked_area` (default 0.49) | Warp coverage only |
| Rotation search | Optional angles | Translation-only FFT |
| Checksum incremental skip | `RefineInvoker` | `TransformRefineOrchestrator` + `IsInputTransformMatched` |

### Present in mosaic, not STOS (by default)

| Safeguard | Mosaic | STOS note |
|-----------|--------|-----------|
| Spatial displacement regularization | Always | Opt-in via `NORNIR_REFINE_STOS_REGULARIZE=1` |
| Gap-fill for unmeasured vertices | Always | Omits failed cells |
| Dual stop (threshold + no improvement) | Always | Iteration / finalization based |
| Atomic pass apply | Always | Rebuild is intentional |
| Batched FFT measurement | Default on (`NORNIR_REFINE_BATCHED`) | Translation-only cells can use shared batched helper |

## Overlap / validity thresholds

| Parameter | Mosaic default | STOS default | Meaning |
|-----------|----------------|--------------|---------|
| `min_overlap` | **0.25** | — | Fraction of valid (covered) pixels in a prewarped cell |
| `min_alignment_overlap` | — | **0.5** | Min ROI overlap for rigid registration |
| `min_unmasked_area` | — | **0.49** | Fraction of tissue mask that must be unmasked |

These differ because mosaic cells are sampled after warp (coverage already
restricts the domain) while STOS cells are cropped from full images with
optional binary masks. Prefer documenting the difference over forcing a single
number; callers may still override per pipeline.

## Failure handling

| Path | On refine failure |
|------|-------------------|
| Mosaic `GridTransform` | Clean partial output and re-raise |
| STOS `__RunPythonGridRefinementCmd` | Log error; copy input `.stos` only when `NORNIR_STOS_REFINE_FALLBACK=1` (or legacy debug-off path disabled by default) |

## Runtime configuration

See `nornir_imageregistration.refine_shared.RefineRuntimeConfig` for
`NORNIR_REFINE_*` env gates (batched measurement, prewarp mode, mosaic cutoff,
STOS regularize, phase timing, GPU transform, tile parallelism).

## Related docs

- [Mosaic refine grid GPU assessment](mosaic_refine_grid_gpu_assessment.md)
