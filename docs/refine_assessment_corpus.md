# Refine-grid assessment corpus

How to judge whether a trusted-mesh (or other refine) change **helped** or **broke** a `.stos` pair without re-registering a full volume.

## Layers

| Layer | Where | Role |
|-------|--------|------|
| Source volume `.stos` | TEM / RC2 / etc. | Discovery only; rank with `ScoreStosGroupQuality` |
| Strain-crop fixtures | `TESTINPUTPATH/refine_fixtures/` | Expandable mini-`.stos` + gold + manifests |
| Catalog SQLite | `TESTINPUTPATH/refine_fixtures/catalog.sqlite` | Inventory, tags, certified cells, **adopted** bests |
| Runs SQLite | `TESTOUTPUTPATH/refine_runs.sqlite` | Every A/B attempt (local; not git-tracked) |
| `best.json` | Per fixture dir | Git-diffable snapshot written **only on adopt** |
| CI subset | `build_ci_testdata.py` → release zip | Tiny crops that unit/integration tests need |

Full-volume `RefineSectionAlignment` is a **late production gate**, not the discovery method.

## Gold vs living best

- **Gold** (Manual / synthetic): where the field *should* be. Certified cells = unique (`peak_ratio`) + ZNCC-prominent + within `max_travel` of Manual (or synthetic warp).
- **Living best** (adopted scores): best *measurement quality* achieved after an accepted change. Metric **vector**, not a single scalar:
  - Maximize: crop `pair_zncc`, unique-fraction (and min over passes), median/p10 `peak_ratio` on certified cells, ZNCC prominence on certified cells
  - Band: lock fraction (healthy Grid16 ~29–36%; do not mix Grid32 without labeling)
  - Flag only: quality flag still on/off for characterization fixtures

**Never auto-update best from every A/B run.** Use `nornir-adopt-refine-scores` after you accept a change.

## CLI tools

| Script | Entry point | Purpose |
|--------|-------------|---------|
| `import_refine_fixture.py` | `nornir-import-refine-fixture` | Crop from source `.stos` (+ optional Manual / diagnostics), register in catalog |
| `tag_fixture.py` | `nornir-tag-fixture` | Confirm or suggest tags |
| `ab_refine_fixtures.py` | `nornir-ab-refine-fixtures` | Baseline vs `NORNIR_REFINE_TRUSTED_MESH=1`; write runs DB + HTML/CSV |
| `adopt_refine_scores.py` | `nornir-adopt-refine-scores` | Copy a run into catalog bests + export `best.json` |

Pytest **never** writes the catalog or runs DB.

## Tags

Many-to-many (`suggested` vs `confirmed`). Visual tags (`tear`, `fold`, …) need human confirm. Suggestable tags (`high-relative-distortion`, `identity-freeze`, …) may come from metrics. Confirmed tags are copied into `manifest.json`.

## Trusted-mesh flag

Set `NORNIR_REFINE_TRUSTED_MESH=1` to run the flagged path: mesh from LOCKED + PROVISIONAL only, measure cells whose prior moved, stop when the trusted set is unchanged, skip Track A/B / best-effort / anchor-smooth. Final lock fraction below `LOCK_FRAC_TRIGGER` writes a `.quality_flag` sidecar next to the output `.stos` and logs `QUALITY FLAG`.

Phase 2 (retire Track A/B code) waits until the fixture A/B table has **no unexpected broken** rows.

## Adding Grid32 / non-TEM later

Manifest / catalog fields `group`, `channel`, and `downsample` are the extension point. Keep separate expected lock bands per downsample; do not mix Grid16 and Grid32 in one band check.
