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
| Batched FFT measurement | Default on (`NORNIR_REFINE_BATCHED`) | Default on (`NORNIR_REFINE_BATCHED_STOS`; independent of mosaic gate) |

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

### Trusted mesh

STOS refine builds the field only from LOCKED fixed points and PROVISIONAL
movable points. UNTRUSTED records never enter mesh construction. Later passes
measure unlocked cells only when their prior moved, and refinement stops when
the trusted tier set is unchanged. The former anchor-smooth, discontinuity
raw-preserve, FieldMode, best-effort, residual translation, and sparse-preserve
paths are retired.

### Pass diagnostics

`NORNIR_REFINE_PASS_DIAGNOSTICS=1` writes per-pass
`refine_passNN_diagnostics.npz` / `.csv` under `outputDir`. Columns include
`peak_ratio`, **`role`**, **`reject_reason`**, **`zncc`**, **`lock_candidate`**,
and optional **`source_content`**. Heatmaps (weight, travel, residual, lock mask,
raw−smoothed delta, discontinuity, peak_ratio, role, zncc) are written only when
`SavePlots=True`.

See [`grid16_stos_cell_role_theory.md`](grid16_stos_cell_role_theory.md) for tier
semantics. Cross-link: `NORNIR_REFINE_PHASE_TIMING=1` adds
`classify` / `zncc_secondary` / `low_content_gate` phase buckets.

### Cell Role tuning envs

| Env | Default | Meaning / effect |
|-----|---------|------------------|
| `NORNIR_REFINE_IDENTITY_ZNCC_MIN` | unset (off) | **Legacy** optional absolute ZNCC floor; when set, also required for LOCKABLE. Not the primary lock gate. |
| `NORNIR_REFINE_LOW_CONTENT_STD_MIN` | `1e-3` | Min source ROI intensity std. Below → sticky measure-skip + `REJECT(LOW_CONTENT)` for that grid ID for the rest of the refine. |

Unset means use the code default for `LOW_CONTENT`; for `IDENTITY_ZNCC_MIN`, unset means the absolute floor is **disabled**. Read sites:
`refine_shared/runtime_config.py`, `cell_roles.identity_zncc_min_threshold`,
`cell_validity.low_content_std_min_threshold`.

ZNCC decoy rings scale with cell size relative to the 128 px reference so
prominence geometry matches across CellArea 128/256/512. Decoy sigma uses a
fixed ZNCC-unit floor (not `1/sqrt(N)`).

### Manual regression checklist

When validating against real data (optional CI when `INPUT_NORNIR_DATA` is set):

1. **Composite seam pair** — re-run refine-grid with `SavePlots=True`; locks should
   not appear along a vertical seam until weights are strong and peaks stable;
   Composite view should not show a left/right color discontinuity.
2. **RC2 TEM pair** — e.g. Brute64→Grid for `1453-1452` or a known `1024` section;
   compare overlay / displacement RMS along the former seam; pass logs should show
   fewer early locks and any unlock events in the problem region.
3. **Tier fixtures** — force re-refine **240-241**, **241-242**, and one healthy
   pair with `NORNIR_REFINE_PASS_DIAGNOSTICS=1`; check tier evidence and require
   the healthy lock fraction not to drop from its recorded baseline.

Unit coverage: `tests/test_refine_finalize_gate.py`, `tests/test_cell_roles.py`.

## Runtime configuration

See `nornir_imageregistration.refine_shared.RefineRuntimeConfig` for
`NORNIR_REFINE_*` env gates (batched measurement, prewarp mode, mosaic cutoff,
STOS regularize, phase timing, GPU transform, tile parallelism,
`NORNIR_REFINE_FINALIZE_LEGACY`, `NORNIR_REFINE_PASS_DIAGNOSTICS`,
`NORNIR_REFINE_SHARP_WARPS`, `NORNIR_REFINE_DISCONTINUITY_K`,
`NORNIR_REFINE_DISCONTINUITY_TRAVEL_MULT`,
`NORNIR_REFINE_IDENTITY_ZNCC_MIN`, `NORNIR_REFINE_LOW_CONTENT_STD_MIN`).

## Related docs

- [Mosaic refine grid GPU assessment](mosaic_refine_grid_gpu_assessment.md)
- [RC2 Grid16 refine baseline](grid16_rc2_refine_baseline.md)
- [Grid16 failure modes (240-241 / 241-242)](grid16_rc2_refine_failure_modes.md)
- [STOS trusted-cell tier theory](grid16_stos_cell_role_theory.md)
