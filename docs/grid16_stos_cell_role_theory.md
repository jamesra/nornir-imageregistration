# STOS grid refine: trusted-cell tiers

Grid refine uses one rule: **only trusted cells shape the field**. A cell earns
trust from its own unique phase-correlation peak and ZNCC prominence, then from
agreement with trusted neighbours. Missing `peak_ratio` fails closed. Where
fewer than three cells are trusted, refinement keeps the input transform.

See also [`refine_grid_step_inventory.md`](refine_grid_step_inventory.md) and
[`grid16_rc2_refine_failure_modes.md`](grid16_rc2_refine_failure_modes.md).

## Tiers

| Tier | Required evidence | Mesh behavior | Remeasurement |
|---|---|---|---|
| `LOCKED` | finite `peak_ratio >= PEAK_RATIO_MIN`, ZNCC prominence, finalization travel/convergence | fixed control point | unlock-stale check |
| `PROVISIONAL` | the same measurement evidence plus agreement with trusted neighbours; when no locks exist, membership in a mutually consistent 4-connected seed cluster | movable control point | when its prior moves |
| `UNTRUSTED` | missing or failed evidence | excluded | when the field under it moves |

Agreement tolerance is relative to local support. Three or more locks within two
grid hops use `max_travel_for_finalization`; one or two use the midpoint of that
distance and the cell half-size; a no-lock seed is judged by cluster direction
and travel relative to the cluster median.

The seed cluster has no permission to bypass either uniqueness or ZNCC. Cell-size
growth may be attempted once when no trusted set exists; if that also finds
nothing trusted, the input field stands.

## Per-pass loop

1. Build the grid and remove finalized, masked, out-of-bounds, and sticky
   low-content cells.
2. Measure every unlocked cell on pass 1. On later passes, measure only cells
   whose mapped prior moved beyond the finalization stability epsilon.
3. Require a finite unique peak and ZNCC prominence. Assign tiers and demote
   provisional cells that no longer agree with locks within two hops.
4. Stop when the trusted tier snapshot is unchanged or the measurement todo is
   empty.
5. Build the field from LOCKED fixed points and PROVISIONAL movable points.
   UNTRUSTED records never reach `_build_mesh_transform_or_keep`.
6. Unlock stale finalized points, finalize newly converged LOCKABLE cells, and
   repeat up to `num_iterations`.

`num_iterations` is a cap, not a request to run empty passes.

## Retired field proxies

The following mechanisms no longer participate in `RefineTransform`:

- coherent and whole-FOV residual `TranslateFixed` recovery, including remeasure
  revert;
- discontinuity raw-preserve and coherent-front exceptions;
- `FieldMode` branding of identity cells;
- best-effort promotion of ambiguous peaks;
- anchor smoothing;
- travel/REJECT mesh fallback;
- absolute sparse-mesh preserve thresholds;
- final anchor smoothing, preserve, and nudge.

These mechanisms tried to limit damage after untrusted records entered the mesh.
The tier boundary removes that failure mode directly.

## Evidence and observability

`NORNIR_REFINE_PASS_DIAGNOSTICS=1` writes per-pass NPZ/CSV data. Acceptance uses
both final lock fraction and unique-fraction-over-passes; healthy 183-184 must
not lose lock fraction. A final lock fraction below `LOCK_FRAC_TRIGGER` emits a
quality flag.

`NORNIR_REFINE_PHASE_TIMING=1` reports phase timing. Pyre receives pass progress
with the current trusted-mesh todo count and may preview pass transforms.

The former `NORNIR_REFINE_TRUSTED_MESH` feature flag was retired after the
September 2026 fixture A/B. Trusted mesh is now the only STOS refine path.
