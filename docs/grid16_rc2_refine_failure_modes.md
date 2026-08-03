# Grid16 refine failure modes (240-241 / 241-242 / 252-254)

Companion to [`grid16_rc2_refine_baseline.md`](grid16_rc2_refine_baseline.md).
Dirt / false-peak Phase 3 is **PASS** for healthy tissue.

**Canonical policy:** [`grid16_stos_cell_role_theory.md`](grid16_stos_cell_role_theory.md)
(Role + FieldMode). Recovery below is **generic** (lock scarcity / field
consistency / raw-preserve) — no pair-name or center-ROI specials.

## Track A — coherent unique residual → `TranslateFixed`

### Observed (classic coherent cluster)

| Signal | Value |
|--------|--------|
| Composite | Uniform translation fringe |
| Unique peak coherence (all unique) | can be ~0.66 with outliers |
| Dominant cluster coherence | ~0.93 after inlier filter |
| Unique median peak | ≈ (−0.1, +37) px |

### Observed (wrap-peak soup — Aug 2026 240-241 rerun)

| Signal | Value |
|--------|--------|
| Unique count | ~39–49 (≪ `MIN_UNIQUE_PEAKS=50`) |
| Unique directions | opposing ±~60 px axis modes |
| Track A | **skips** (logs `n_unique` / `n_inliers` / reason) |

### Implementation

Module: [`coherent_residual.py`](../nornir_imageregistration/refine_shared/coherent_residual.py).

| Constant | Value |
|----------|--------|
| `LOCK_FRAC_TRIGGER` | 0.05 |
| `COHERENCE_MIN` | 0.85 |
| `MIN_UNIQUE_PEAKS` | 50 |
| `INLIER_COS_MIN` | 0.5 (~60° of median direction) |
| `MIN_MESH_ABS_AFTER_RESIDUAL` | 100 |
| `MIN_MESH_FRAC_AFTER_RESIDUAL` | 0.05 |

Hook: [`RefineTransform`](../nornir_imageregistration/local_distortion_correction.py) after measure —
at most once: inlier median → `TranslateFixed` → remasure. Does **not** accept wrap
±cell_size peaks as Track A inliers.

### Preserve post-residual transform

When Track A / global FOV succeeds but the remasure pass travel/REJECT-filters the
mesh down to a handful of points (e.g. ~14 of ~6000), rebuilding
`MeshWithRBFFallback` from that set **discards** the dense translated control
set and the Composite fringe returns. If `residual_applied` ∧ `lock_frac < 0.05`
∧ `n_mesh < max(ABS, frac·n_grid)`, keep `stosTransform` (post-`TranslateFixed`)
instead of the sparse mesh — mid-pass, end-of-pass, and final. Do **not** force
wrap unique peaks into the mesh.

**Final gate `n_grid`:** must be the last-pass **FOV** measured size
(`len(alignment_points) + len(finalized_points)`), not
`len(final_control_records)`. Using the sparse control-set size (~12) makes
`lock_frac` look healthy (e.g. 2/14) so preserve skips and a 12-point rebuild
undoes TranslateFixed before `ConvertTransformToGridTransform` densifies it
(Grid16 240-241).

### Global FOV fallback (pathological low-unique)

When `lock_frac < 0.05` and Track A skips for scarce/incoherent unique peaks,
run **downsampled whole-FOV** phase correlation under the current transform
(`estimate_global_fov_residual_translation`, `max_dim=512`), then `TranslateFixed`
once and remasure. Healthy lock fractions never enter.

## Identity freeze / center bubble → field branding + mesh raw-preserve

### Observed (241-242 center)

| Metric | Center | Note |
|--------|--------|------|
| Unique | high | Tissue *can* register |
| Free travel med | ~13–15 px | Above settled-lock ~2 px bar |
| Mid-x locks | all identity | Freeze pose for smooth |
| `raw_vs_smooth_delta` | ≈ travel | Smooth erases residual |

Do **not** globally raise finalize travel/weight bars (healthy free-unique already
~2 px / weight ~3.3).

### Implementation

1. **FOV-hot unique field:** when ≥20 unique peaks have median travel
   `> 0.5 * max_travel`, brand all `travel < eps` cells `IDENTITY_SUSPECT`
   (plus existing neighbor / ASYMMETRIC cold-half rules).
2. **Mesh raw-preserve:** `FREE` cells with `pr ≥ PEAK_RATIO_MIN` and travel
   `> 0.5 * max_travel` keep raw peaks under anchor-smooth (union with
   **soft-disc** only — ratio-eligible discontinuity tags) so residuals can
   move the mesh until travel falls under the settled-lock bar.

Modules: [`cell_roles.py`](../nornir_imageregistration/refine_shared/cell_roles.py),
[`anchor_smooth.py`](../nornir_imageregistration/refine_shared/anchor_smooth.py).

## Tear / disc front (252-254)

### Observed

| Signal | Value |
|--------|--------|
| Locks | healthy ~34% |
| Disc cells (stable) | ~75–86/pass when mesh does not raw-preserve ambiguous disc |
| Soft eligible `pr ≥ 1.20` | ~0–10 of tagged disc (med `pr≈1.03` across gap) |
| Right disc `raw_vs_smooth` | ~40 px when soft-starved — peaks erased |

### Failed experiment (do not repeat)

Raw-preserving **all** discontinuity tags (including ambiguous wrap-like peaks)
caused a feedback loop on 252-254: disc **78→489**, travel p90 **3.5→19**,
top-left mesh wave / pockets. Ambiguous disc must **not** enter mesh raw-preserve
unless they form a spatially coherent active front.

### Implementation

| Concern | Policy |
|---------|--------|
| Mesh / anchor-smooth | Raw-preserve `soft_discontinuity_ids` ∪ unique large-travel ∪ **`coherent_discontinuity_raw_preserve_ids`** |
| Lock soft floors | `soft_discontinuity_ids` (`pr ≥ PEAK_RATIO_MIN`) only |
| Field identity | Brand `IDENTITY_SUSPECT` when `travel < eps` next to an **active disc** neighbor (disc tag with travel `> max_travel`) |

**Coherent disc-front** (`coherent_discontinuity_raw_preserve_ids`): active disc
cells (travel `> max_travel`) in a 4-connected component of size ≥6 whose peak
directions agree (`peak_direction_coherence ≥ 0.70`, unit-dot to component median
≥ `INLIER_COS_MIN`). No `peak_ratio` floor — tear fronts often sit near `pr≈1.03`.
Isolated dirt and bimodal wrap clusters stay out. Lock policy for `pr < 1.20`
unchanged (`REJECT` / never soft floor).

### Still open

- Fail/flag stos when final lock fraction stays near zero after recovery
- Full Phase-4 ZNCC on every cell (secondary ZNCC remains lock-candidate only)
