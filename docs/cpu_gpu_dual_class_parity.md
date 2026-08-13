# CPU / GPU dual-class transform parity

## Problem

Several transform types exist as parallel **host** and **device** classes
(`ControlPointBase` / `ControlPointBase_GPUComponent`, `Landmark` /
`Landmark_GPU`, `Triangulation` / `Triangulation_GPUComponent`). Overnight
bug review found asymmetric fixes: epsilon ignored on one side, FlipWarped
restoring only X, duplicate detection comparing enumerate indices, and
`EnsurePointsAre4xN` mismatches between NumPy and CuPy setters.

## Contract

1. **Shared helpers first.** Prefer module-level helpers
   (`GroupControlPointIndicesByPosition`, `EnsurePointsAre4xN_*`) used by
   both backends. Backend classes should stay thin shells over shared logic.
2. **`xp = cp.get_array_module(points)`** for mutate / dedupe / Flip that
   receive a concrete array. Do **not** use process-wide `UsingCupy()` to
   decide output type when an input array is present (see
   `.cursor/rules/Numpy-CuPy-compatibility.mdc`).
3. **Same semantics on both backends** for:
   - `FindDuplicates` / `RemoveDuplicateControlPoints` (3-decimal rounded
     fixed-space `(y, x)`)
   - `FindDuplicateFixedPoints(..., epsilon=)` (`distance <= epsilon`)
   - `Flip` (target and source **X** about each space’s vertical midline)
   - `FlipWarped` (source **X** about `flip_center`; restore **both** axes
     after negation)

## Parity checklist (tests)

| Operation | Host entry | Device entry | Test |
|-----------|------------|--------------|------|
| Group / dedupe indices | `ControlPointBase.*` | same helpers on CuPy arrays | `test_rbf_precompute_and_duplicates.py` |
| `RemoveDuplicateControlPoints` | NumPy points | CuPy points | same + `test_cpu_gpu_controlpoint_parity.py` |
| `Flip` | `ControlPointBase.Flip` | `ControlPointBase_GPUComponent.Flip` | `test_cpu_gpu_controlpoint_parity.py` |
| `FlipWarped` | `Triangulation` / `Landmark` | GPU counterparts | `test_cpu_gpu_controlpoint_parity.py` |

## Follow-up (optional)

Consolidate one dual pair so FlipWarped lives in a shared helper called by
both Landmark and Triangulation CPU/GPU shells.
