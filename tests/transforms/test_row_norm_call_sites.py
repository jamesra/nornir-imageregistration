"""Pin the row-wise Euclidean norm used by the blend, scale-estimate and layout-tension sites.

BlendTransformsWithParams, BlendTransformsIterativelyWithParams and EstimateScale each take
``sqrt(sum(x ** 2, axis=1))`` of an Nx2 point difference. These tests hold that quantity
bit-identical to the hand-rolled expression on NumPy and CuPy, so routing the sites through
:func:`nornir_imageregistration.array_distance` cannot change a single bit of their output.
"""

import unittest
from typing import Literal
from unittest import mock

import nornir_imageregistration
import nornir_imageregistration.transforms
import nornir_imageregistration.transforms.utils as blend_utils
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from nornir_imageregistration.transforms import MeshWithRBFFallback
from nornir_imageregistration.transforms.converters import EstimateScale
from nornir_imageregistration.transforms.utils import (
    BlendTransformsIterativelyWithParams,
    BlendTransformsWithParams,
    LinearBlendParams,
)


def _row_norm(xp, points):
    """The hand-rolled expression the call sites used before sharing array_distance."""
    return xp.sqrt(xp.sum(points ** 2, axis=1))


def _point_rows(dtype, width: Literal[32, 64]) -> st.SearchStrategy:
    """Nx2 arrays spanning sub-pixel to mosaic-scale magnitudes (hundreds of thousands of px)."""
    return hnp.arrays(dtype=dtype,
                      shape=st.tuples(st.integers(1, 64), st.just(2)),
                      elements=st.floats(-5e5, 5e5, allow_nan=False, allow_infinity=False, width=width))


def _scattered_points(seed: int, count: int, magnitude: float) -> np.ndarray:
    """Count random Nx2 float64 points within +/- magnitude; distinct with probability one."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-magnitude, magnitude, size=(count, 2))


def _grid_mesh_and_linear(seed: int, deviation: float):
    """A 4x4 grid mesh whose targets deviate randomly from a rigid translation."""
    rng = np.random.default_rng(seed)
    ys, xs = np.meshgrid(np.arange(4) * 100.0, np.arange(4) * 100.0, indexing='ij')
    source = np.column_stack((ys.ravel(), xs.ravel()))
    offset = np.array([3.0, -7.0])
    target = source + offset + rng.normal(0.0, deviation, size=source.shape)
    mesh = MeshWithRBFFallback(np.hstack((target, source)))
    linear = nornir_imageregistration.transforms.RigidTranslation(target_offset=offset)
    return mesh, linear


class TestArrayDistanceMatchesTheRowNorm(unittest.TestCase):
    """array_distance must equal the hand-rolled row norm bit for bit, dtype included."""

    @settings(deadline=None, max_examples=200)
    @given(points=st.one_of(_point_rows(np.float64, 64), _point_rows(np.float32, 32)))
    def test_numpy(self, points: np.ndarray) -> None:
        with np.errstate(under='ignore'):  # tiny float32 squares underflow identically on both sides
            expected = _row_norm(np, points)
            actual = nornir_imageregistration.array_distance(points)
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertTrue(np.array_equal(expected, actual))

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "CuPy not available")
    @settings(deadline=None, max_examples=50)
    @given(points=st.one_of(_point_rows(np.float64, 64), _point_rows(np.float32, 32)))
    def test_cupy(self, points: np.ndarray) -> None:
        cp = nornir_imageregistration.cp
        device_points = cp.asarray(points)
        expected = _row_norm(cp, device_points)
        actual = nornir_imageregistration.array_distance(device_points)
        self.assertIs(cp.get_array_module(actual), cp)
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertTrue(bool(cp.array_equal(expected, actual)))


class TestEstimateScale(unittest.TestCase):
    """EstimateScale is the ratio of summed centred row norms, target over source."""

    @settings(deadline=None, max_examples=100)
    @given(seed=st.integers(0, 2 ** 32 - 1),
           count=st.integers(2, 200),
           magnitude=st.sampled_from([1e-2, 1.0, 1e3, 5e5]),
           scale=st.floats(0.1, 10.0))
    def test_matches_the_hand_rolled_ratio_exactly(self, seed: int, count: int,
                                                   magnitude: float, scale: float) -> None:
        source = _scattered_points(seed, count, magnitude)
        target = source * scale + _scattered_points(seed + 1, count, magnitude * 1e-3)
        centered_source = source - np.mean(source, axis=0)
        centered_target = target - np.mean(target, axis=0)
        expected = float(np.sum(_row_norm(np, centered_target)) / np.sum(_row_norm(np, centered_source)))
        self.assertEqual(expected, EstimateScale(source, target))

    def test_a_known_scale(self) -> None:
        source = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        self.assertAlmostEqual(2.5, EstimateScale(source, source * 2.5 + 40.0))


class TestBlendTravelDistances(unittest.TestCase):
    """The travel distances fed to the blend weights are the linear-to-target row norms."""

    @settings(deadline=None, max_examples=25)
    @given(seed=st.integers(0, 2 ** 32 - 1),
           deviation=st.sampled_from([0.5, 10.0, 80.0]))
    def test_one_shot_distances(self, seed: int, deviation: float) -> None:
        mesh, linear = _grid_mesh_and_linear(seed, deviation)
        params = LinearBlendParams.from_kwargs(min_blend=0.05, travel_limit=100.0)
        with mock.patch.object(blend_utils, '_travel_blend_weights',
                               wraps=blend_utils._travel_blend_weights) as weights_spy:
            BlendTransformsWithParams(mesh, linear, params)

        self.assertEqual(1, weights_spy.call_count)
        distances = weights_spy.call_args.args[0]
        linear_points = np.asarray(linear.Transform(mesh.SourcePoints))
        expected = _row_norm(np, linear_points - np.asarray(mesh.TargetPoints))
        self.assertTrue(np.array_equal(expected, distances))

    @settings(deadline=None, max_examples=25)
    @given(seed=st.integers(0, 2 ** 32 - 1),
           deviation=st.sampled_from([0.5, 10.0, 80.0]))
    def test_iterative_distances_and_stopping_point(self, seed: int, deviation: float) -> None:
        """Each iteration's distances are row norms, and the loop stops on the max row-norm step.

        A first run with zero tolerances records every step's movement; tolerances placed just
        above and just below one recorded movement must stop the loop exactly there and one
        iteration later, which only holds if movement is the max row norm of the step.
        """
        mesh, linear = _grid_mesh_and_linear(seed, deviation)
        iterations = 6

        def run(reblend_tolerance: float):
            params = LinearBlendParams.from_kwargs(min_blend=0.05, travel_limit=100.0,
                                                   reblend_iterations=iterations,
                                                   reblend_tolerance=reblend_tolerance,
                                                   reblend_weight_tolerance=0.0)
            with mock.patch.object(blend_utils, '_travel_blend_weights',
                                   wraps=blend_utils._travel_blend_weights) as weights_spy, \
                    mock.patch.object(blend_utils, '_blend_target_points',
                                      wraps=blend_utils._blend_target_points) as blend_spy:
                BlendTransformsIterativelyWithParams(mesh, linear, params)
            return weights_spy, blend_spy

        weights_spy, blend_spy = run(0.0)
        self.assertEqual(iterations, blend_spy.call_count)
        movements = []
        for weights_call, blend_call in zip(weights_spy.call_args_list, blend_spy.call_args_list):
            target_points, linear_points, _ = blend_call.args
            expected = _row_norm(np, linear_points - target_points)
            self.assertTrue(np.array_equal(expected, weights_call.args[0]))
            new_target_points = blend_utils._blend_target_points(*blend_call.args)
            movements.append(float(np.max(_row_norm(np, new_target_points - target_points))))

        def expected_calls(reblend_tolerance: float) -> int:
            """Iterations the loop runs: through the first step moving less than the tolerance."""
            for i, step in enumerate(movements):
                if step < reblend_tolerance:
                    return i + 1
            return iterations

        for movement in movements[:3]:
            for reblend_tolerance in (np.nextafter(movement, np.inf), movement):
                _, blend_spy = run(float(reblend_tolerance))
                self.assertEqual(expected_calls(float(reblend_tolerance)), blend_spy.call_count,
                                 f"reblend_tolerance={reblend_tolerance!r}")


if __name__ == '__main__':
    unittest.main()
