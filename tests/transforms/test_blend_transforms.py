"""Tests for linear blend and iterative re-blend of control-point transforms."""

import unittest
import warnings
from typing import cast

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import nornir_imageregistration.transforms
from nornir_imageregistration.transforms import MeshWithRBFFallback
from nornir_imageregistration.transforms.addition import (
  AddTransformsWithLinearCorrection,
)
from nornir_imageregistration.transforms.base import IControlPoints
from nornir_imageregistration.transforms.converters import (
  ConvertControlPointsToRigidTransformForBlend,
  EstimateRigidComponentsFromControlPoints,
)
from nornir_imageregistration.transforms.utils import (
  DEFAULT_MAX_BLEND_WEIGHT,
  DEFAULT_REBLEND_ITERATIONS,
  DEFAULT_REBLEND_TOLERANCE,
  DEFAULT_REBLEND_WEIGHT_TOLERANCE,
  BlendTransforms,
  BlendTransformsIteratively,
  BlendWithLinear,
  LinearBlendParams,
  _as_numpy_points,
  _blend_transforms,
  _blend_transforms_iteratively,
  _travel_blend_weights,
  estimate_inverse_map_y_correlation,
  resolve_effective_max_blend,
)


def _mesh_from_points(source_points: np.ndarray, target_points: np.ndarray) -> MeshWithRBFFallback:
  point_pairs = np.hstack((target_points, source_points))
  return MeshWithRBFFallback(point_pairs)


class TestTravelBlendWeights(unittest.TestCase):
  def test_smoothstep_has_no_half_limit_dead_zone(self) -> None:
    travel_limit = 100.0
    distances = np.array([0.0, 25.0, 49.0, 50.0, 51.0, 75.0, 100.0])
    weights = _travel_blend_weights(distances, travel_limit, min_blend=0.0)

    self.assertGreater(weights[1], 0.0)
    self.assertLess(weights[-1], 1.0)
    self.assertLessEqual(float(np.max(weights)), DEFAULT_MAX_BLEND_WEIGHT)

  def test_min_blend_floor_is_applied(self) -> None:
    weights = _travel_blend_weights(np.array([0.0]), travel_limit=100.0, min_blend=0.05)
    self.assertAlmostEqual(float(weights[0]), 0.05)

  def test_uniform_min_blend_without_travel_limit(self) -> None:
    source = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    target = source + np.array([[20.0, 0.0], [0.0, 20.0], [-20.0, 0.0], [0.0, -20.0]])
    nonlinear = _mesh_from_points(source, target)
    linear = nornir_imageregistration.transforms.RigidTranslation(target_offset=np.zeros(2))
    blended = BlendTransforms(nonlinear, linear, min_blend=0.01)
    for idx in range(source.shape[0]):
      move = float(np.linalg.norm(_as_numpy_points(blended.TargetPoints[idx]) - target[idx]))
      expected = 0.01 * float(np.linalg.norm(_as_numpy_points(linear.Transform(source[idx:idx + 1])[0]) - target[idx]))
      self.assertAlmostEqual(move, expected, places=5)

  def test_max_blend_caps_travel_weights(self) -> None:
    distances = np.array([0.0, 100.0, 200.0])
    weights = _travel_blend_weights(distances,
                                    travel_limit=100.0,
                                    min_blend=0.01,
                                    max_blend=0.25)
    self.assertAlmostEqual(float(weights[0]), 0.01)
    self.assertAlmostEqual(float(weights[-1]), 0.25)

  def test_deprecated_linear_factor_alias(self) -> None:
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always', DeprecationWarning)
      weights = _travel_blend_weights(np.array([0.0]), travel_limit=100.0, linear_factor=0.05)
    self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
    self.assertAlmostEqual(float(weights[0]), 0.05)


class TestResolveEffectiveMaxBlend(unittest.TestCase):
  def test_defaults_to_min_blend_without_travel_limit(self) -> None:
    self.assertAlmostEqual(resolve_effective_max_blend(0.01, None, None), 0.01)

  def test_defaults_to_nine_tenths_with_travel_limit(self) -> None:
    self.assertAlmostEqual(resolve_effective_max_blend(0.01, 512.0, None), DEFAULT_MAX_BLEND_WEIGHT)

  def test_explicit_max_blend_is_preserved(self) -> None:
    self.assertAlmostEqual(resolve_effective_max_blend(0.01, 512.0, 0.25), 0.25)


class TestBlendTransforms(unittest.TestCase):
  def test_uniform_min_blend_moves_toward_rigid(self) -> None:
    source = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    target = source + np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0], [0.0, -5.0]])
    nonlinear = _mesh_from_points(source, target)
    linear = nornir_imageregistration.transforms.RigidTranslation(target_offset=np.array([1.0, 1.0]))

    blended = BlendTransforms(nonlinear, linear, min_blend=0.2)
    blended_targets = _as_numpy_points(blended.TargetPoints)
    for idx in range(source.shape[0]):
      self.assertLess(
        float(np.linalg.norm(blended_targets[idx] - target[idx])),
        float(np.linalg.norm(linear.Transform(source[idx:idx + 1])[0] - target[idx])),
      )


class TestBlendTransformsIteratively(unittest.TestCase):
  def test_reblend_converges_without_full_rigid_snap(self) -> None:
    source = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    target = source.copy()
    target[0] += np.array([80.0, 0.0])
    nonlinear = _mesh_from_points(source, target)
    linear = nornir_imageregistration.transforms.RigidTranslation(target_offset=np.zeros(2))

    one_shot = BlendTransforms(nonlinear, linear, travel_limit=100.0, min_blend=0.05)
    iterated = BlendTransformsIteratively(nonlinear,
                                          linear,
                                          travel_limit=100.0,
                                          min_blend=0.05,
                                          reblend_iterations=8,
                                          reblend_tolerance=0.01)

    one_shot_move = float(np.linalg.norm(_as_numpy_points(one_shot.TargetPoints[0]) - target[0]))
    iterated_move = float(np.linalg.norm(_as_numpy_points(iterated.TargetPoints[0]) - target[0]))
    self.assertGreater(one_shot_move, 0.0)
    self.assertGreater(iterated_move, one_shot_move)
    self.assertGreater(
        float(np.linalg.norm(_as_numpy_points(iterated.TargetPoints[0]) - linear.Transform(source[0:1])[0])),
        0.0)

  def test_low_deviation_points_move_less_than_outlier(self) -> None:
    source = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    target = source.copy()
    target[0] += np.array([80.0, 0.0])
    target[1] += np.array([2.0, 0.0])
    nonlinear = _mesh_from_points(source, target)
    linear = nornir_imageregistration.transforms.RigidTranslation(target_offset=np.zeros(2))

    blended = BlendTransformsIteratively(nonlinear,
                                         linear,
                                         travel_limit=100.0,
                                         min_blend=0.05,
                                         reblend_iterations=8,
                                         reblend_tolerance=0.01)
    outlier_move = float(np.linalg.norm(_as_numpy_points(blended.TargetPoints[0]) - target[0]))
    normal_move = float(np.linalg.norm(_as_numpy_points(blended.TargetPoints[1]) - target[1]))
    self.assertGreater(outlier_move, normal_move)

  def test_blend_preserves_y_flipped_orientation(self) -> None:
    source = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0],
                       [50.0, 0.0], [50.0, 100.0], [0.0, 50.0], [100.0, 50.0]])
    linear = nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
        target_offset=np.array([5.0, -3.0]),
        source_rotation_center=np.array([50.0, 50.0]),
        angle=0.4,
        scalar=1.02,
        flip_ud=True,
    )
    target = np.array([linear.Transform(point.reshape(1, 2))[0] for point in source])
    target[0] += np.array([12.0, -8.0])
    target[4] += np.array([-6.0, 4.0])
    nonlinear = _mesh_from_points(source, target)

    mesh_corr = estimate_inverse_map_y_correlation(nonlinear)
    self.assertLess(mesh_corr, -0.9)

    blended = BlendWithLinear(nonlinear,
                              travel_limit=100.0,
                              min_blend=0.05,
                              reblend_iterations=8,
                              reblend_tolerance=0.5)
    blended_corr = estimate_inverse_map_y_correlation(blended)
    self.assertLess(blended_corr, 0.0)
    self.assertEqual(np.sign(mesh_corr), np.sign(blended_corr))


class TestLinearBlendParams(unittest.TestCase):
  """Packing kwargs into LinearBlendParams matches direct construction and blend outputs."""

  def test_from_kwargs_coalesces_linear_factor(self) -> None:
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always', DeprecationWarning)
      params = LinearBlendParams.from_kwargs(linear_factor=0.05, travel_limit=100.0)
    self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
    self.assertEqual(params.min_blend, 0.05)
    self.assertEqual(params.travel_limit, 100.0)

  def test_from_kwargs_none_tolerances_become_defaults(self) -> None:
    params = LinearBlendParams.from_kwargs(
        min_blend=0.1,
        reblend_tolerance=None,
        reblend_weight_tolerance=None)
    self.assertEqual(params.reblend_tolerance, DEFAULT_REBLEND_TOLERANCE)
    self.assertEqual(params.reblend_weight_tolerance, DEFAULT_REBLEND_WEIGHT_TOLERANCE)

  def test_default_reblend_iterations_matches_iterative_api(self) -> None:
    self.assertEqual(LinearBlendParams().reblend_iterations, DEFAULT_REBLEND_ITERATIONS)
    self.assertEqual(
        LinearBlendParams.from_kwargs().reblend_iterations, DEFAULT_REBLEND_ITERATIONS)

  def test_kwargs_and_params_paths_match_target_points(self) -> None:
    source = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    target = source.copy()
    target[0] += np.array([80.0, 0.0])
    nonlinear = _mesh_from_points(source, target)
    linear = nornir_imageregistration.transforms.RigidTranslation(target_offset=np.zeros(2))
    params = LinearBlendParams.from_kwargs(
        min_blend=0.05,
        travel_limit=100.0,
        reblend_iterations=8,
        reblend_tolerance=0.01)
    via_kwargs = cast(IControlPoints, BlendTransformsIteratively(
        nonlinear, linear, min_blend=0.05, travel_limit=100.0,
        reblend_iterations=8, reblend_tolerance=0.01))
    via_params = cast(IControlPoints, _blend_transforms_iteratively(nonlinear, linear, params))
    np.testing.assert_allclose(
        _as_numpy_points(via_kwargs.TargetPoints),
        _as_numpy_points(via_params.TargetPoints),
        atol=0.0,
        rtol=0.0)

  @given(
      min_blend=st.one_of(st.none(), st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)),
      max_blend=st.one_of(st.none(), st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)),
      travel_limit=st.one_of(st.none(), st.floats(1.0, 1000.0, allow_nan=False, allow_infinity=False)),
      reblend_iterations=st.integers(1, 8),
      reblend_tolerance=st.one_of(st.none(), st.floats(0.001, 2.0, allow_nan=False, allow_infinity=False)),
      reblend_weight_tolerance=st.one_of(
          st.none(), st.floats(0.001, 1.0, allow_nan=False, allow_infinity=False)),
  )
  @settings(max_examples=40, deadline=None)
  def test_from_kwargs_matches_direct_dataclass(
          self,
          min_blend: float | None,
          max_blend: float | None,
          travel_limit: float | None,
          reblend_iterations: int,
          reblend_tolerance: float | None,
          reblend_weight_tolerance: float | None) -> None:
    packed = LinearBlendParams.from_kwargs(
        min_blend=min_blend,
        max_blend=max_blend,
        travel_limit=travel_limit,
        reblend_iterations=reblend_iterations,
        reblend_tolerance=reblend_tolerance,
        reblend_weight_tolerance=reblend_weight_tolerance)
    expected = LinearBlendParams(
        min_blend=min_blend,
        max_blend=max_blend,
        travel_limit=travel_limit,
        reblend_iterations=reblend_iterations,
        reblend_tolerance=(
            DEFAULT_REBLEND_TOLERANCE if reblend_tolerance is None else reblend_tolerance),
        reblend_weight_tolerance=(
            DEFAULT_REBLEND_WEIGHT_TOLERANCE
            if reblend_weight_tolerance is None
            else reblend_weight_tolerance))
    self.assertEqual(packed, expected)
    self.assertEqual(packed.effective_max_blend,
                     resolve_effective_max_blend(min_blend, travel_limit, max_blend))

  def test_one_shot_params_path_matches_kwargs(self) -> None:
    source = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    target = source + np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0], [0.0, -5.0]])
    nonlinear = _mesh_from_points(source, target)
    linear = nornir_imageregistration.transforms.RigidTranslation(
        target_offset=np.array([1.0, 1.0]))
    params = LinearBlendParams.from_kwargs(min_blend=0.2)
    via_kwargs = cast(IControlPoints, BlendTransforms(nonlinear, linear, min_blend=0.2))
    via_params = cast(IControlPoints, _blend_transforms(nonlinear, linear, params))
    np.testing.assert_allclose(
        _as_numpy_points(via_kwargs.TargetPoints),
        _as_numpy_points(via_params.TargetPoints),
        atol=0.0,
        rtol=0.0)

  def test_add_transforms_still_accepts_loose_kwargs(self) -> None:
    source = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    a_to_b = _mesh_from_points(source, source + 1.0)
    b_to_c = nornir_imageregistration.transforms.RigidTranslation(
        target_offset=np.array([2.0, 3.0]))
    out = cast(IControlPoints, AddTransformsWithLinearCorrection(
        b_to_c, a_to_b, min_blend=0.2, travel_limit=50.0, reblend_iterations=2))
    self.assertTrue(np.isfinite(_as_numpy_points(out.TargetPoints)).all())


class TestDegenerateRigidFitForBlend(unittest.TestCase):
  """Collapsed meshes must not abort LinearBlend via align_vectors SVD failure."""

  def test_coincident_points_with_override_are_translation_only(self) -> None:
    source = np.full((4, 2), 10.0)
    target = np.full((4, 2), 25.0)
    components = EstimateRigidComponentsFromControlPoints(
        target, source, reflected_override=False)
    self.assertEqual(components.angle, 0.0)
    self.assertEqual(components.scale, 1.0)
    np.testing.assert_allclose(components.translation, np.array([15.0, 15.0]), atol=1e-6)

  def test_coincident_points_without_override_still_report_colinear(self) -> None:
    source = np.full((4, 2), 10.0)
    target = np.full((4, 2), 25.0)
    with self.assertRaises(ValueError) as raised:
      EstimateRigidComponentsFromControlPoints(target, source)
    self.assertIn('colinear', str(raised.exception).lower())

  def test_blend_converter_survives_collapsed_mesh(self) -> None:
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target = np.full((4, 2), 50.0)
    mesh = _mesh_from_points(source, target)
    rigid = ConvertControlPointsToRigidTransformForBlend(mesh)
    predicted = np.asarray(rigid.Transform(source))
    self.assertEqual(predicted.shape, source.shape)
    self.assertTrue(np.isfinite(predicted).all())

  def test_blend_with_linear_survives_near_collapsed_targets(self) -> None:
    source = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    target = np.array([[10.0, 10.0], [10.2, 10.0], [10.0, 10.2], [10.2, 10.2]])
    mesh = _mesh_from_points(source, target)
    blended = BlendWithLinear(mesh, min_blend=0.05, travel_limit=100.0)
    self.assertTrue(np.isfinite(_as_numpy_points(blended.TargetPoints)).all())


if __name__ == '__main__':
  unittest.main()
