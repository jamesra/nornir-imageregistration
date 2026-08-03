# STOS Grid refine: cell Role and FieldMode theory

Single classification model for Grid16 (and STOS grid refine generally).
Replaces layered specials (soft-disc-only-unique, amb-exclude, early-lock,
Track A residual, Track B identity-neighbor) with **per-cell Role** plus
**pass-level FieldMode**.

See also: [`grid16_rc2_refine_baseline.md`](grid16_rc2_refine_baseline.md),
[`grid16_rc2_refine_failure_modes.md`](grid16_rc2_refine_failure_modes.md),
[`grid_refine_stos_vs_mosaic.md`](grid_refine_stos_vs_mosaic.md).

## Evidence regimes

| Regime | Evidence | Behavior |
|--------|----------|----------|
| Healthy | Final locks ~33–37% | `LOCAL`; normal mesh + lock |
| Dirt | Ambiguous peaks | `REJECT(PEAK_AMBIGUOUS)` out of mesh/locks |
| Path **240-241** | Low locks + coherent unique **or** wrap-peak soup | Track A / global FOV `TranslateFixed`; **preserve** transform if mesh collapses |
| Asymmetric / bubble **241-242** | Identity ring + unique high-travel free | FOV-hot / neighbor `IDENTITY_SUSPECT`; unique raw-preserve in mesh |
| Tear **252-254** | Disc front with ambiguous `pr`; wrong regional locks | Disc-neighbor identity brand; mesh raw-preserve soft-disc ∪ unique ∪ **coherent disc-front** (not all disc — feedback loop) |

## Per-cell Role

Classify each free cell once after measure (constants:
`PEAK_RATIO_MIN=1.20`, `PEAK_RATIO_EARLY=1.50`).

| Role | Definition | Mesh | Lock |
|------|------------|------|------|
| `REJECT` | `pr < 1.20` this pass, or low content | no raw peak | never |
| `FREE` | PC-pass; not yet lock-candidate (travel only; weight bar off) | yes (bars) | no |
| `LOCKABLE` | lock-candidate ∧ PC-pass ∧ (not field-suspect) ∧ ZNCC-pass | yes | yes when stable |
| `IDENTITY_SUSPECT` | lock-candidate ∧ (field suspect **or** ZNCC-fail) | may stay free | **never** |

Decision order: low-content / ambiguous → `REJECT`; else if not lock-candidate →
`FREE`; else **field consistency** (FOV-hot unique field, cold-half under
`ASYMMETRIC`, active unique neighbor, or **active disc neighbor**) →
`IDENTITY_SUSPECT`; else ZNCC pass → `LOCKABLE`, fail → `IDENTITY_SUSPECT`.

**Registration-weight inflection is diagnostic-only.** RefineTransform still
computes `estimate_registration_weight_cutoff` for logs/plots, but mesh inclusion,
lock candidacy, and finalize use `transform_cutoff=-inf`. Active gates are travel,
peak-ratio / low-content `REJECT`, field branding, and secondary masked ZNCC.

**Absolute ZNCC alone cannot detect false identity locks** when ROIs already look
correlated under a bad local prediction (241-242 med ZNCC ~0.75). Field consistency
is the primary gate for that failure; ZNCC remains a useful secondary when intensity
truly disagrees. Do not raise `IDENTITY_ZNCC_MIN` to paper over field failures.

Secondary **masked ZNCC** runs only for PC-pass lock candidates that field rules
have not already branded suspect (skips redundant ROI re-extract). Threshold default
in code; override with `NORNIR_REFINE_IDENTITY_ZNCC_MIN`.

### REJECT reasons

| Reason | Signal | Sticky? |
|--------|--------|---------|
| `PEAK_AMBIGUOUS` | known `pr < 1.20` | **No** — remasure next pass |
| `LOW_CONTENT` | source ROI std/MAD below min | **Yes** for measure (source cached) |

Source-`LOW_CONTENT` cells are never remasured for the rest of that refine;
control points are placed by the mesh / anchor-smooth, not by a registration peak.
Moving-only flat with structured source may remasure after the transform improves.

Unknown `peak_ratio` (`None`) is not peak-ambiguous (legacy-safe).

## FieldMode

| Mode | Trigger | Action |
|------|---------|--------|
| `LOCAL` | default | Roles only |
| `RIGID_RESIDUAL` | lock_frac `< 0.05` and unique coherent **inliers** | once: `TranslateFixed(median)`; remasure |
| (global FOV) | lock_frac `< 0.05` and Track A skips (scarce/incoherent unique) | once: downsampled FOV PC → `TranslateFixed`; remasure |
| `ASYMMETRIC` | free-peak travel medians disagree across low-x / high-x | cold-half identity → `IDENTITY_SUSPECT` |

Track A inlier filter: unique traveling peaks within unit-dot ≥ `INLIER_COS_MIN`
(0.5 ≈ 60°) of the preliminary median direction; coherence + median on inliers only.
Does **not** lower `COHERENCE_MIN` without inliers. Wrap-like opposing ±cell_size
peaks are **not** Track A inliers — use global FOV recovery instead.

`ASYMMETRIC` half stats use **free measured peaks only**. Field branding also fires
when the FOV unique field is **hot** (enough unique peaks with median travel
`> 0.5 * max_travel`), a 4-connected unique neighbor is still traveling, or a
4-connected **active disc** neighbor (discontinuity tag with travel `> max_travel`)
exists — refuses identity freeze beside a tear/fold front.

**Mesh raw-preserve vs lock soft floors (disc):**

| Concern | Set |
|---------|-----|
| Anchor-smooth / mesh raw peaks | `soft_discontinuity_ids` ∪ unique large-travel (`pr ≥ PEAK_RATIO_MIN`, travel `> 0.5*max_travel`) ∪ **`coherent_discontinuity_raw_preserve_ids`** (active disc cluster ≥6, direction coherence ≥0.70, inlier cos ≥0.5; **no** `pr` floor) |
| Lock travel/weight soft floors | `soft_discontinuity_ids` only (`disc ∩ pr ≥ PEAK_RATIO_MIN`) |

Do **not** raw-preserve all tagged discontinuities: ambiguous wrap-like disc peaks
(`pr≈1.03`) caused a disc-count feedback loop (252-254: disc 78→489) unless they
form a spatially coherent active front. Isolated / bimodal wrap dirt stays out of
mesh preserve and remains `REJECT` for locks.

After Track A / global FOV, if locks stay scarce and the mesh collapses below
`max(MIN_MESH_ABS_AFTER_RESIDUAL, MIN_MESH_FRAC_AFTER_RESIDUAL·n_grid)`,
**keep** the post-`TranslateFixed` transform instead of rebuilding from the
sparse survivors.

Do not raise finalize travel/weight FOV-wide.

## Mapping from old gates

| Old | New |
|-----|-----|
| Soft-disc ∩ unique | `FREE`/`LOCKABLE` ∩ disc ∩ unique |
| `exclude_ambiguous_mesh_records` | drop `REJECT` from mesh |
| Early-lock (`pr ≥ 1.50`) | `LOCKABLE` with stability=1 |
| Track A coherent residual | `FieldMode.RIGID_RESIDUAL` (+ inliers) |
| Track B identity_neighbor | obsolete; field consistency + ZNCC → `IDENTITY_SUSPECT` |

## Instrumentation

**Always-on pass log:** `field_mode`, role histogram, `lock_cand` / `zncc_*`
funnel, `source_low_content_skip`, `classify_s` / `zncc_s`. Track A logs
success (`n_unique` / `n_inliers` / coherence) or **skip reason**; global FOV
recovery logs when attempted.

**`NORNIR_REFINE_PASS_DIAGNOSTICS=1`:** NPZ/CSV columns `role`, `reject_reason`,
`zncc`, `lock_candidate` (and optional `source_content`). Lifetime
`refine_cell_history.npz` (travel / role / peak_ratio / zncc per cell per pass).
Heatmaps and history polylines need `SavePlots`.

**`NORNIR_REFINE_PHASE_TIMING=1`:** buckets `classify`, `zncc_secondary`,
`low_content_gate`, `approx_rigid` (logged at end of each pass so wall buckets
include Role/ZNCC work).

## Tuning env overrides

| Env | Default | Effect |
|-----|---------|--------|
| `NORNIR_REFINE_IDENTITY_ZNCC_MIN` | code constant | Below → `IDENTITY_SUSPECT`; above → may `LOCKABLE` (when not field-suspect) |
| `NORNIR_REFINE_LOW_CONTENT_STD_MIN` | code constant | Source below → sticky measure-skip + `REJECT(LOW_CONTENT)` |

Unset means use the code default. See Runtime configuration in
[`grid_refine_stos_vs_mosaic.md`](grid_refine_stos_vs_mosaic.md).
