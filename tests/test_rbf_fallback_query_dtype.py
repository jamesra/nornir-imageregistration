"""
The RBF fallback query dtype policy must be one policy, applied at every call site.

Before this change the policy was duplicated four times and had drifted:

    CPU  Transform         line 206  unconditional downcast to float32
    CPU  InverseTransform  line 241  guard only, so float64 survived
    GPU  Transform         line 539  unconditional line commented out, guard only
    GPU  InverseTransform  line 579  guard only

So the CPU class disagreed with itself between forward and inverse, and CPU
disagreed with GPU for identical points. Measured CPU-vs-GPU gap on the fallback
path was 0.0078 px at a 100k-pixel section extent, exactly one float32 quantum.

float32 is kept deliberately: the RBF extrapolation is 12-170% faster in single
precision and the error amplification through the RBF is 1.00x, so the cost is one
quantum of the query coordinate (0.008 px at 100k extent) and nothing more.

Note the continuous transform returns float64 regardless of query dtype, so the
policy is only observable in what the fallback *receives*. These tests therefore
assert on the query handed to the continuous transform, not on the result dtype.
"""

from __future__ import annotations

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration.transforms import gridwithrbffallback as mod
from nornir_imageregistration.transforms.gridwithrbffallback import GridWithRBFFallback

EXTENT = 100_000.0
GRID_N = 5


@pytest.fixture(autouse=True)
def _host_backend():
    nornir_imageregistration.SetActiveComputationLib(
        nornir_imageregistration.ComputationLib.numpy)


def _build_grid(extent=EXTENT, n=GRID_N):
    from nornir_imageregistration.grid_subdivision import ITKGridDivision
    grid = ITKGridDivision(
        source_shape=np.asarray((extent, extent), dtype=np.float64),
        cell_size=np.asarray((extent / n, extent / n), dtype=np.float64),
        grid_dims=np.asarray((n, n), dtype=np.int64))
    src = np.asarray(grid.SourcePoints, dtype=np.float64)
    tgt = src.copy()
    # A mild non-affine warp, so the RBF fallback has something real to extrapolate.
    tgt[:, 0] += 7.25 + 2e-5 * src[:, 1]
    tgt[:, 1] += -3.75 + 3e-5 * src[:, 0]
    grid.TargetPoints = tgt
    return grid


def _build_cpu():
    return GridWithRBFFallback(_build_grid())


def _outside_points(n=64, dtype=np.float64):
    """Points well outside the grid, so the discrete transform fails and the RBF runs."""
    rng = np.random.default_rng(0xC05B005)
    pts = rng.random((n, 2)) * EXTENT + EXTENT * 1.5
    return np.asarray(pts, dtype=dtype)


class _QuerySpy:
    """Records the dtype of every query handed to the continuous fallback."""

    def __init__(self, transform):
        self.dtypes: list[np.dtype] = []
        self._continuous = transform._continuous_transform
        self._real_forward = self._continuous.Transform
        self._real_inverse = self._continuous.InverseTransform

    def __enter__(self):
        def forward(points, **kwargs):
            self.dtypes.append(np.asarray(points).dtype)
            return self._real_forward(points, **kwargs)

        def inverse(points, **kwargs):
            self.dtypes.append(np.asarray(points).dtype)
            return self._real_inverse(points, **kwargs)

        self._continuous.Transform = forward
        self._continuous.InverseTransform = inverse
        return self

    def __exit__(self, *exc):
        self._continuous.Transform = self._real_forward
        self._continuous.InverseTransform = self._real_inverse
        return False


def test_fallback_actually_runs_for_these_points():
    """Guards the tests below from becoming vacuous if the points stop extrapolating."""
    t = _build_cpu()
    with _QuerySpy(t) as spy:
        t.Transform(_outside_points())
    assert spy.dtypes, 'no fallback query was issued, so the dtype tests prove nothing'


@pytest.mark.parametrize('direction', ['forward', 'inverse'])
def test_float64_query_is_downcast(direction):
    """The documented policy: float64 in, float32 to the RBF."""
    t = _build_cpu()
    pts = _outside_points(dtype=np.float64)

    with _QuerySpy(t) as spy:
        if direction == 'forward':
            t.Transform(pts)
        else:
            t.InverseTransform(pts)

    assert spy.dtypes
    assert all(d == np.float32 for d in spy.dtypes), \
        f'{direction} handed the RBF {spy.dtypes}, expected all float32'


def test_forward_and_inverse_agree_on_the_policy():
    """The original complaint: the CPU class was inconsistent with itself."""
    t = _build_cpu()
    pts = _outside_points(dtype=np.float64)

    with _QuerySpy(t) as forward_spy:
        t.Transform(pts)
    with _QuerySpy(t) as inverse_spy:
        t.InverseTransform(pts)

    assert set(forward_spy.dtypes) == set(inverse_spy.dtypes)


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float16])
def test_non_float32_input_is_coerced(dtype):
    """Integer and half-precision queries must not reach the RBF unconverted.

    Small magnitudes here: float16 tops out at 65504, so section-scale coordinates
    would overflow the cast before reaching the transform. Negative values are still
    outside the grid, which starts at the origin, so the fallback still runs.
    """
    t = _build_cpu()
    rng = np.random.default_rng(5)
    pts = np.asarray(-rng.random((64, 2)) * 4000.0 - 1000.0, dtype=dtype)

    with _QuerySpy(t) as spy:
        t.Transform(pts)

    assert spy.dtypes
    assert all(d == np.float32 for d in spy.dtypes)


def test_float32_input_passes_through_without_a_copy():
    """An already-float32 query should not be reallocated."""
    pts = _outside_points(dtype=np.float32)

    returned = mod._as_fallback_query_points(pts)

    assert returned is pts


def test_helper_preserves_values_within_float32_range():
    pts = np.array([[1.5, -2.25], [1024.0625, 4096.5]], dtype=np.float64)

    out = mod._as_fallback_query_points(pts)

    assert out.dtype == np.float32
    np.testing.assert_array_equal(np.asarray(out, dtype=np.float64), pts)


def test_all_live_call_sites_use_the_shared_helper():
    """Prevents the policy from drifting back into per-site copies.

    Only the two live classes are checked. GridWithRBFInterpolator_CPU/_GPU still
    carry their own copies, but they are dead code: nothing constructs them, the
    only references are commented out in factory.py, and their Transform methods
    still contain unconditional debug print() calls.
    """
    import inspect

    live_methods = [
        mod.GridWithRBFFallback.Transform,
        mod.GridWithRBFFallback.InverseTransform,
        mod.GridWithRBFFallback_GPUComponent.Transform,
        mod.GridWithRBFFallback_GPUComponent.InverseTransform,
    ]

    for method in live_methods:
        source = inspect.getsource(method)
        assert '_as_fallback_query_points' in source, \
            f'{method.__qualname__} no longer routes through the shared policy'
        assert 'dtype=np.float32' not in source, \
            f'{method.__qualname__} reintroduced an inline dtype policy'


def test_accuracy_cost_is_one_quantum_and_not_amplified():
    """The justification for keeping float32: the RBF does not amplify the rounding.

    The continuous transform's first call differs from later calls, so it is settled
    before measuring; otherwise that state, not dtype, dominates the comparison.
    """
    t = _build_cpu()
    cont = t._continuous_transform
    pts = _outside_points(n=500, dtype=np.float64)
    pts32 = np.asarray(pts, dtype=np.float32)

    for _ in range(2):
        cont.Transform(pts)
        cont.Transform(pts32)

    out64 = np.asarray(cont.Transform(pts), dtype=np.float64)
    out32 = np.asarray(cont.Transform(pts32), dtype=np.float64)

    quantum = np.abs(pts - pts32.astype(np.float64)).max()
    error = np.abs(out64 - out32).max()

    assert quantum > 0, 'these coordinates are float32-exact, so the test is vacuous'
    # Allow generous slack; the point is that this is O(1) and not O(100).
    assert error <= quantum * 4, \
        f'RBF amplified the float32 quantum {error / quantum:.1f}x, ' \
        f'which would make the downcast unsafe'
