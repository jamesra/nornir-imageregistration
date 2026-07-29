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
| Spatial displacement regularization | Always | Locked-anchor gap-fill once ``anchor_smooth_min_locks`` met (default 3); optional all-measured via ``NORNIR_REFINE_STOS_REGULARIZE=1`` |
| Gap-fill for unmeasured vertices | Always | Locked-anchor mesh gap-fill when enough locks; early passes omit failed cells |
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
| STOS `__RunPythonGridRefinementCmd` | Write sibling `*.unrefined.stos` (scaled input, else identity) for Pyre, leave official output absent, and re-raise |

## STOS finalize / lock policy

`RefineTransform` locks cells via `refine_shared.finalize` when **all** of these hold:

- pass index ≥ `min_finalize_pass` (default **2**)
- ‖peak‖ ≤ `max_travel_for_finalization`
- weight ≥ transform-inclusion cutoff (same inflection bar as mesh inclusion)
- peak stable for `finalize_stability_passes` consecutive passes (default **2**, ε ≈ 0.5 px)

After each mesh update, locks that disagree with the transform prediction by more
than `max_travel * finalize_unlock_travel_multiplier` (default **1.5**) are unlocked.
Set the multiplier to **0** to disable unlock.

Rollback: `NORNIR_REFINE_FINALIZE_LEGACY=1` restores distance-primary locking with
a 2% weight floor (pre-fix behavior).

### Anchor-smooth mesh (committed)

Once `len(finalized_points) >= anchor_smooth_min_locks` (default **3**), mesh
rebuild and final output use `regularize_displacements` seeded **only** from locked
cells. Raw phase-correlation peaks still drive measurement and finalize; un-lockable
cells receive gap-filled smoothed peaks for triangulation. Early passes with fewer
locks keep the travel-filter + weight-cutoff mesh path.

Settings on `GridRefinement`: `anchor_smooth_min_locks`, `anchor_smooth_median_radius`
(default **1**, same as mosaic).

### Manual regression checklist

When validating against real data (optional CI when `INPUT_NORNIR_DATA` is set):

1. **Composite seam pair** — re-run refine-grid with `SavePlots=True`; locks should
   not appear along a vertical seam until weights are strong and peaks stable;
   Composite view should not show a left/right color discontinuity.
2. **RC2 TEM pair** — e.g. Brute64→Grid for `1453-1452` or a known `1024` section;
   compare overlay / displacement RMS along the former seam; pass logs should show
   fewer early locks and any unlock events in the problem region.

Unit coverage: `tests/test_refine_finalize_gate.py`.

## Runtime configuration

See `nornir_imageregistration.refine_shared.RefineRuntimeConfig` for
`NORNIR_REFINE_*` env gates (batched measurement, prewarp mode, mosaic cutoff,
STOS regularize, phase timing, GPU transform, tile parallelism,
`NORNIR_REFINE_FINALIZE_LEGACY`).

## Related docs

- [Mosaic refine grid GPU assessment](mosaic_refine_grid_gpu_assessment.md)
