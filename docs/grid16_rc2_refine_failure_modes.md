# Grid16 trusted-mesh fixture expectations

Historical refine failure modes are now fixture expectations for the trusted
mesh. The implementation must not recover them with field-wide translations,
branding, smoothing, raw-preserve exceptions, or emergency mesh filling.

For every fixture, report final lock fraction, unique-fraction for every pass,
control-point delta, wall time, and output pair ZNCC. A missing `peak_ratio` is
non-lockable. A quality flag is required when final lock fraction is below
`LOCK_FRAC_TRIGGER`.

## Required pairs

| Pair | Historical failure | Trusted-mesh expectation |
|---|---|---|
| 1215-1214 | Track B accepted a 0.95 whole-FOV peak and shifted the section 2654–3510 px | no field-wide translation; only unique, prominent cells enter the mesh |
| 240-241 | coherent residual and absolute sparse-preserve thresholds depended on grid size | trust is decided per cell; no absolute point-count floor |
| 241-242 | identity bubble caused field branding and anchor smoothing | identity cells lock only from their own evidence and neighbour agreement |
| 252-254 | raw-preserved discontinuities produced a 78→489 feedback front | ambiguous front cells stay UNTRUSTED; agreement may promote only unique, prominent cells |
| 228-229 | high-weight, high-travel outliers bent the mesh | weight alone never grants mesh membership |
| 953-952 | best-effort admitted ambiguous peaks to thicken a sparse mesh | ambiguous peaks remain UNTRUSTED even when the trusted set is small |
| 183-184 | healthy control | final lock fraction must not drop; uniqueness must remain stable or improve |

## September 2026 A/B

Grid16 production settings were cell size 256, spacing 192, and a cap of ten
passes. Baseline artifacts are under
`TESTOUTPUTPATH/refine_grid_baseline_current`; trusted artifacts and the complete
per-pass comparison are under
`TESTOUTPUTPATH/refine_grid_baseline_trusted_mesh_final`.

| Pair | Final lock fraction | Final unique fraction | Output pair ZNCC |
|---|---:|---:|---:|
| 1215-1214 | 0.841 → 0.962 | 0.981 → 0.996 | 0.0464 → 0.0457 |
| 240-241 | 0.962 → 0.997 | 0.994 → 0.999 | 0.5821 → 0.5791 |
| 241-242 | 0.291 → 0.999 | 0.609 → 0.999 | 0.5210 → 0.6651 |
| 252-254 | 0.922 → 0.992 | 0.987 → 0.999 | 0.4860 → 0.4810 |
| 228-229 | 0.961 → 0.994 | 0.998 → 0.999 | 0.6839 → 0.6843 |
| 953-952 | 0.818 → 0.978 | 0.928 → 0.991 | 0.1194 → 0.1202 |
| 183-184 | 0.823 → 0.993 | 0.908 → 0.997 | 0.2337 → 0.2286 |

The healthy pair's lock and uniqueness fractions improved. Small pair-ZNCC
changes on already aligned pairs are retained as characterization data rather
than used as permission to admit untrusted cells.

## Regression rule

Any follow-on change must name one trust question, identify the existing
inventory step that answers it, and show why that answer is insufficient on a
named fixture. New constants must scale with the grid, pass distribution,
`max_travel`, or cell size. Untrusted records must never reach
`_build_mesh_transform_or_keep`.
