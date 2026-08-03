# RC2 Grid16 refine baseline (log-derived)

Durable reference for how STOS **Grid16** refine behaves on volume
``/storage4/RC2``. Use this when answering questions about lock rates,
discontinuity tagging, travel vs weight rejection, or calibrating false-peak
gates. Prefer this file over re-parsing the multi-MB build log unless the log
has been extended with new pairs.

## Provenance

| Field | Value |
|-------|--------|
| Volume | `/storage4/RC2` |
| Pipeline | `RefineSectionAlignment` → output group **Grid16** |
| Source log | `/storage4/RC2/log-16.31.26_04.16.txt` |
| Capture | Stats parsed from refine segments in that log (2026-07-31) |
| Machine-readable twin | [`grid16_rc2_refine_baseline.json`](grid16_rc2_refine_baseline.json) |
| Pair attribution | Segments bounded by `Refining .../Grid16/Automatic/<pair>_...stos` |

Notes from the same run:

- `NORNIR_REFINE_PASS_DIAGNOSTICS` / `SavePlots` was **off** — no
  `refine_pass*_diagnostics.npz` under Grid16; Phase 1 `peak_ratio` was
  computed in-process but not archived.
- Section **0400** errors: cannot generate downsample-16 Leveled images
  (separate data issue).
- Many mappings correctly **skipped** when Grid16 output already matched the
  input transform checksum.

Related earlier diagnostic runs (not this log): Grid32 pair **156-157** and
**255-256** under `/storage4/RC2/TEM/Grid32/refine_diagnostics/` (copied from
`/tmp/refine_diag_*`).

## Two regimes

| Regime | Lock fraction (final pass) | Sharp-warp disc max | Dominant finalize reject |
|--------|----------------------------|---------------------|---------------------------|
| Healthy (n=23) | median **~31.6%** (range ~29–36%) | typically tens–hundreds (some ~1k) | **weight** ≫ travel |
| Pathological (n=6) | **&lt;10%** (often &lt;1%) | **~4k–5.3k** cells | **travel** ≫ weight |

Classification used here: pathological if ``max_disc > 2000`` **or**
``final_lock_fraction < 0.05``.

### Healthy pattern

Pass 1–2: zero locks (``min_finalize_pass`` / stability). Pass 3: large lock
burst (~1k+). Settle near ~1.4k / ~4.5k on typical pairs. Discontinuity tags
stay modest. Finalize rejects are mostly **below weight cutoff**.

### Pathological pattern

Discontinuity tags explode early (majority of cells). Mesh **travel** drops
thousands of free points. Locks crawl (tens to low hundreds). Soft
discontinuity travel/weight floors then apply to many cells whose peaks are
likely false — unlike the sparse 2–3 disc tags on the older Grid32 156-157
diagnostic refine.

## Pathological pairs (this log)

| Pair | Final locked / N | Lock % | max disc | max travel-drop | max rej weight | max rej travel | Pass 2 deferred | Complete |
|------|------------------|--------|----------|-----------------|----------------|----------------|-----------------|----------|
| 248-249 | 3 / 1073 | 0.28% | 950 | 771 | 117 | 731 | 165 | yes |
| 269-270 | 23 / 4549 | 0.51% | 4046 | 3658 | 571 | 3395 | 893 | yes |
| 261-262 | 50 / 6198 | 0.81% | 5284 | 4725 | 743 | 4432 | 1169 | yes |
| 240-241 | 54 / 6319 | 0.85% | 5334 | 5043 | 656 | 4511 | 1127 | no (in progress at parse) |
| 262-263 | 292 / 6169 | 4.73% | 5040 | 4343 | 806 | 4116 | 1364 | yes |
| 260-261 | 420 / 4952 | 8.48% | 3879 | 3721 | 694 | 3530 | 1186 | yes |

## Healthy pairs (this log)

| Pair | Final locked / N | Lock % | max disc | max rej weight | max rej travel | Pass 2 deferred |
|------|------------------|--------|----------|----------------|----------------|-----------------|
| 272-273 | 1502 / 5084 | 29.54% | 1131 | 3145 | 793 | — |
| 259-260 | 934 / 3146 | 29.69% | 688 | 2032 | 478 | — |
| 270-271 | 1379 / 4479 | 30.79% | 90 | 3122 | 28 | — |
| 265-266 | 1387 / 4461 | 31.09% | 65 | 3112 | 14 | — |
| 256-257 | 1349 / 4323 | 31.21% | 118 | 3000 | 70 | — |
| 258-260 | 1297 / 4134 | 31.37% | 251 | 2880 | 179 | — |
| 266-267 | 1353 / 4312 | 31.38% | 117 | 2993 | 64 | — |
| 251-252 | 1328 / 4224 | 31.44% | 399 | 2896 | 268 | — |
| 268-269 | 1439 / 4575 | 31.45% | 459 | 3056 | 316 | — |
| 292-293 | 1410 / 4481 | 31.47% | 299 | 3099 | 225 | — |
| 250-251 | 1384 / 4376 | 31.63% | 200 | 3038 | 135 | — |
| 267-268 | 1440 / 4550 | 31.65% | 121 | 3161 | 40 | — |
| 257-258 | 1411 / 4454 | 31.68% | 247 | 3083 | 171 | — |
| 271-273 | 1404 / 4422 | 31.75% | 598 | 2969 | 407 | — |
| 252-254 | 1434 / 4514 | 31.77% | 545 | 3040 | 368 | — |
| 255-256 | 1437 / 4506 | 31.89% | 318 | 3083 | 214 | — |
| 273-274 | 1452 / 4551 | 31.91% | 46 | 3141 | 13 | — |
| 254-255 | 1417 / 4386 | 32.31% | 423 | 2980 | 310 | — |
| 249-250 | 163 / 503 | 32.41% | 5 | 348 | 12 | — |
| 245-246 | 2243 / 6588 | 34.05% | 315 | 4483 | 221 | — |
| 243-245 | 3077 / 8680 | 35.45% | 1486 | 5618 | 1035 | — |
| 241-242 | 2213 / 6189 | 35.76% | 686 | 3843 | 490 | — |
| 246-248 | 2299 / 6316 | 36.40% | 1060 | 4080 | 743 | — |

## Calibration anchors (false-peak / sharp-warp work)

| Role | Pair | Why |
|------|------|-----|
| Healthy | **255-256**, **292-293**, **254-255** | Typical ~32% lock, weight-dominated rejects, modest disc |
| Pathological | **261-262**, **269-270**, **240-241** | Disc explosion + travel-dominated failure |
| Earlier fold study | Grid32 **156-157** | Sparse disc tags (2–3) in dedicated diagnostic refine — different failure shape than Grid16 pathological |

## Implications (short)

1. On healthy Grid16 tissue, ~⅓ of cells lock; the rest fail **weight** first — false-peak uniqueness should unlock earlier locks on high-`peak_ratio` cells without flooding low-weight junk.
2. On pathological pairs, discontinuity soft-gates currently fire on thousands of cells; **gate / exclude low `peak_ratio` before disc soft floors** or false peaks get relaxed travel/weight.
3. Re-runs that need NPZ/CSV tables set ``NORNIR_REFINE_PASS_DIAGNOSTICS=1``;
   heatmaps also require ``SavePlots``.

## Post–Phase-3 corpus (log-24 / log-43)

| Field | Value |
|-------|--------|
| Logs | `/storage4/RC2/log-24.01.26_03.24.txt`, `/storage4/RC2/log-43.01.26_05.43.txt` |
| Diagnostics | `/storage4/RC2/TEM/Grid16/refine_diagnostics/` (NPZ+CSV; heatmaps off unless `SavePlots`) |
| Dirt visual | **PASS** — Composite dirt blob no longer scatters free yellow points |
| `PEAK_RATIO_MIN` / `EARLY` | **1.20** / **1.50** accepted for healthy + dirt |
| Recovery Track A/B | **Always-on** — coherent residual `TranslateFixed` when locks &lt;5% + coherent unique peaks; finalize rejects identity locks next to active unique neighbors (see [`grid16_rc2_refine_failure_modes.md`](grid16_rc2_refine_failure_modes.md)) |

### Timing (log-32 CuPy; approx_rigid vectorized)

Source: `/storage4/RC2/log-32.02.26_10.32.txt` — 119 Grid16 pairs (CuPy phase totals),
before vectorizing ``ApproximateRigidTransformBySourcePoints``.

| Metric | log-32 (pre-vectorize) |
|--------|------------------------|
| `pair_wall_s` median | **~362 s** (~6 min) |
| Phase share **`approx_rigid`** | **~58%** (median ~212 s/pair) |
| Phase share `cell_extract` | ~14% |
| Phase share `grid_build` | ~11% |
| Phase share **`fft`** | **~1%** |
| `diagnostics_heatmaps_s` | always **0** |

Do **not** assume FFT is the bottleneck — batched STOS FFT is already tiny.
The former ~17–19 s/pass gap under `measure_s` was **approx_rigid**: one batched
``Transform`` of ring samples, then **N×** ``scipy.spatial.transform.Rotation.align_vectors``.
That loop is now a vectorized host Kabsch batch
(``EstimateRigidComponentsFromControlPointsBatched``). Secondary CuPy wins:
skip full-image ``xp.asarray`` when already on-device; batch ``low_content_gate`` std
without forcing a full-mosaic device upload.

**Spot-check (183-184, CuPy, ``NORNIR_REFINE_PHASE_TIMING=1``):**

| Metric | log-32 baseline | after change |
|--------|-----------------|--------------|
| `pair_wall_s` | 363.3 | **335.2** |
| `approx_rigid` | 234.5 (62%) | **151.8 (48%)** |
| `cell_extract` | 55.2 | 60.5 |
| Final locks | 2352/5962 (~39%) | **2307/5962 (~39%)** |

FOV first-pass ``approx_rigid`` ~8.8 s (was ~18.8 s). Remaining ``approx_rigid``
cost is largely ``CenteredSimilarity2DTransform`` construction / inverse matrices
per cell (ROI path still consumes a transform list).

### Healthy lock band (re-refined pairs)

Final locks stay ~**33–37%** (e.g. 183-184 37.4%, 242-243 32.8%, 241-242 35.2%).

### Failure exemplars (visual + NPZ)

#### 240-241 — coherent translation (path)

- Composite: uniform magenta/green offset (global translation).
- log-43 re-refine: final **16 / 6210 (~0.26%)** locks; pass 1–2 locked **0**.
- Unique cells (`peak_ratio ≥ 1.20`, ~2.9%) have **coherent** peak direction
  (`|mean unit vector| ≈ 0.92`) with mean peak ≈ `(-2.8, +42.7)` px and travel med ~47.
- More uniqueness gating will not invent locks; recovery needs a **coherent residual
  rigid/affine** (or explicit fail/flag when lock fraction is near zero).

#### 241-242 — asymmetric local distortion

- Composite: **left** well registered; **right** still fringing (the case Grid refine should fix).
- Final lock **35%**, but **all locks are identity** (`travel < 0.5`).
- Low-`source_x` half: ~21% locks, travel med ~16, many high-travel free peaks
  (centroid ~`(2522, 5606)`).
- High-`source_x` half: ~50% locks, travel med ~0 (centroid of identity locks ~`(5126, 3627)`).
- Axis read: Composite **right ≈ high `source_x`** identity-locked; **left ≈ low `source_x`**
  still carrying free corrections the mesh can apply. Hypothesis: identity locks
  **freeze a wrong field** on the bad side.

## Cell Role theory

Lock / mesh policy is defined by the unified Role + FieldMode model in
[`grid16_stos_cell_role_theory.md`](grid16_stos_cell_role_theory.md). Track A/B
recovery specials are absorbed there (`RIGID_RESIDUAL`, `IDENTITY_SUSPECT` via
PC-pass ∧ ZNCC-fail).

## Updating this baseline

When a newer Grid16 refine log is available, re-parse with the same
``Refining .../Automatic/<pair>`` segmentation, refresh the JSON twin, and bump
the provenance table above. Do not mix Grid32 and Grid16 lock fractions without
labeling downsample.
