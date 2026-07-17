"""Tests for linear blend and iterative re-blend of control-point transforms."""

import unittest

import numpy as np

import nornir_imageregistration.transforms
from nornir_imageregistration.transforms import MeshWithRBFFallback
from nornir_imageregistration.transforms.utils import (
    DEFAULT_MAX_BLEND_WEIGHT,
    _travel_blend_weights,
    BlendTransforms,
    BlendTransformsIteratively,
)


def _mesh_from_points(source_points: np.ndarray, target_points: np.ndarray) -> MeshWithRBFFallback:
  point_pairs = np.hstack((target_points, source_points))
  return MeshWithRBFFallback(point_pairs)


class TestTravelBlendWeights(unittest.TestCase):
  def test_smoothstep_has_no_half_limit_dead_zone(self) -> None:
    travel_limit = 100.0
    distances = np.array([0.0, 25.0, 49.0, 50.0, 51.0, 75.0, 100.0])
    weights = _travel_blend_weights(distances, travel_limit, linear_factor=0.0)

    self.assertGreater(weights[1], 0.0)
    self.assertLess(weights[-1], 1.0)
    self.assertLessEqual(float(np.max(weights)), DEFAULT_MAX_BLEND_WEIGHT)

  def test_linear_factor_floor_is_applied(self) -> None:
    weights = _travel_blend_weights(np.array([0.0]), travel_limit=100.0, linear_factor=0.05)
    self.assertAlmostEqual(float(weights[0]), 0.05)


class TestBlendTransforms(unittest.TestCase):
  def test_uniform_linear_factor_moves_toward_rigid(self) -> None:
    source = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    target = source + np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0], [0.0, -5.0]])
    nonlinear = _mesh_from_points(source, target)
    linear = nornir_imageregistration.transforms.RigidTranslation(target_offset=np.array([1.0, 1.0]))

    blended = BlendTransforms(nonlinear, linear, linear_factor=0.2)
    blended_targets = blended.TargetPoints
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

    one_shot = BlendTransforms(nonlinear, linear, travel_limit=100.0, linear_factor=0.05)
    iterated = BlendTransformsIteratively(nonlinear,
                                          linear,
                                          travel_limit=100.0,
                                          linear_factor=0.05,
                                          reblend_iterations=8,
                                          reblend_tolerance=0.01)

    one_shot_move = float(np.linalg.norm(one_shot.TargetPoints[0] - target[0]))
    iterated_move = float(np.linalg.norm(iterated.TargetPoints[0] - target[0]))
    self.assertGreater(one_shot_move, 0.0)
    self.assertGreater(iterated_move, one_shot_move)
    self.assertGreater(float(np.linalg.norm(iterated.TargetPoints[0] - linear.Transform(source[0:1])[0])),
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
                                         linear_factor=0.05,
                                         reblend_iterations=8,
                                         reblend_tolerance=0.01)
    outlier_move = float(np.linalg.norm(blended.TargetPoints[0] - target[0]))
    normal_move = float(np.linalg.norm(blended.TargetPoints[1] - target[1]))
    self.assertGreater(outlier_move, normal_move)


if __name__ == '__main__':
  unittest.main()
