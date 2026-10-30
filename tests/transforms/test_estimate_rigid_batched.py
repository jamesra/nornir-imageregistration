"""Parity tests: batched rigid estimation vs scalar EstimateRigidComponentsFromControlPoints."""

from __future__ import annotations

import math
import unittest

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import nornir_imageregistration
from nornir_imageregistration.transforms import CenteredSimilarity2DTransform
from nornir_imageregistration.transforms.converters import (
    EstimateRigidComponentsFromControlPoints,
    EstimateRigidComponentsFromControlPointsBatched,
    RigidComponentsToCenteredSimilarityTransform,
)


def _angle_diff(a: float, b: float) -> float:
    return float(abs((a - b + math.pi) % (2 * math.pi) - math.pi))


class TestEstimateRigidBatched(unittest.TestCase):
    """Batched estimator must match scalar mapping and components."""

    def test_batch_matches_scalar_explicit_translate_scale_rotate(self) -> None:
        rng = np.random.default_rng(0)
        source = rng.normal(size=(12, 9, 2)) * 10.0 + 50.0
        target = np.empty_like(source)
        for i in range(source.shape[0]):
            t = CenteredSimilarity2DTransform(
                target_offset=rng.normal(size=2) * 4.0,
                source_rotation_center=source[i].mean(axis=0),
                angle=float(rng.uniform(-math.pi, math.pi)),
                scalar=float(rng.uniform(0.9, 1.1)),
                flip_ud=bool(i % 3 == 0),
            )
            target[i] = np.asarray(t.Transform(source[i]))

        batch = EstimateRigidComponentsFromControlPointsBatched(source, target)
        self.assertEqual(len(batch), source.shape[0])
        for i in range(source.shape[0]):
            scalar = EstimateRigidComponentsFromControlPoints(
                target_points=target[i], source_points=source[i])
            self.assertEqual(batch[i].reflected, scalar.reflected)
            self.assertAlmostEqual(batch[i].scale, scalar.scale, places=10)
            self.assertLess(_angle_diff(batch[i].angle, scalar.angle), 1e-9)
            bt = RigidComponentsToCenteredSimilarityTransform(batch[i])
            st_t = RigidComponentsToCenteredSimilarityTransform(scalar)
            err = np.max(np.abs(np.asarray(bt.Transform(source[i])) - np.asarray(st_t.Transform(source[i]))))
            self.assertLess(float(err), 1e-6)

    @given(
        angle=st.floats(min_value=-math.pi + 1e-3, max_value=math.pi - 1e-3, allow_nan=False, allow_infinity=False),
        scale=st.floats(min_value=0.85, max_value=1.15, allow_nan=False, allow_infinity=False),
        flip=st.booleans(),
        offset=arrays(dtype=np.float64, shape=(2,), elements=st.floats(-8.0, 8.0, allow_nan=False, allow_infinity=False)),
        source=arrays(
            dtype=np.float64,
            shape=(9, 2),
            elements=st.floats(-20.0, 20.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=40, deadline=None)
    def test_batch_vs_scalar_property(self, angle: float, scale: float, flip: bool,
                                      offset: np.ndarray, source: np.ndarray) -> None:
        source = np.asarray(source, dtype=np.float64) + np.array([40.0, 40.0])
        # Avoid degenerate rings (near-colinear after transform) by requiring spread.
        if float(np.linalg.norm(source.std(axis=0))) < 1e-3:
            return
        t = CenteredSimilarity2DTransform(
            target_offset=np.asarray(offset, dtype=np.float64),
            source_rotation_center=source.mean(axis=0),
            angle=float(angle),
            scalar=float(scale),
            flip_ud=bool(flip),
        )
        target = np.asarray(t.Transform(source))
        try:
            batch = EstimateRigidComponentsFromControlPointsBatched(
                source[None, ...], target[None, ...])[0]
            scalar = EstimateRigidComponentsFromControlPoints(
                target_points=target, source_points=source)
        except ValueError:
            # Colinear rejection is shared; treat as out-of-domain for the property.
            return
        self.assertEqual(batch.reflected, scalar.reflected)
        self.assertAlmostEqual(batch.scale, scalar.scale, places=8)
        self.assertLess(_angle_diff(batch.angle, scalar.angle), 1e-7)
        bt = RigidComponentsToCenteredSimilarityTransform(batch)
        st_t = RigidComponentsToCenteredSimilarityTransform(scalar)
        err = np.max(np.abs(np.asarray(bt.Transform(source)) - np.asarray(st_t.Transform(source))))
        self.assertLess(float(err), 1e-5)

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "requires CuPy")
    def test_scalar_cupy_stays_on_device_and_matches_numpy(self) -> None:
        import cupy as cupy_mod

        rng = np.random.default_rng(1)
        source = rng.normal(size=(9, 2)) * 10.0 + 50.0
        t = CenteredSimilarity2DTransform(
            target_offset=np.array([3.0, -2.0]),
            source_rotation_center=source.mean(axis=0),
            angle=0.4,
            scalar=1.05,
            flip_ud=False,
        )
        target = np.asarray(t.Transform(source))
        host = EstimateRigidComponentsFromControlPoints(target, source)
        device = EstimateRigidComponentsFromControlPoints(
            cupy_mod.asarray(target), cupy_mod.asarray(source))
        self.assertIsInstance(device.translation, cupy_mod.ndarray)
        self.assertIsInstance(device.source_rotation_center, cupy_mod.ndarray)
        np.testing.assert_allclose(
            nornir_imageregistration.EnsureNumpyArray(device.translation),
            nornir_imageregistration.EnsureNumpyArray(host.translation), atol=1e-9)
        np.testing.assert_allclose(
            nornir_imageregistration.EnsureNumpyArray(device.source_rotation_center),
            nornir_imageregistration.EnsureNumpyArray(host.source_rotation_center),
            atol=1e-9)
        self.assertAlmostEqual(device.scale, host.scale, places=10)
        self.assertLess(_angle_diff(device.angle, host.angle), 1e-9)

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "requires CuPy")
    def test_translation_only_cupy_stays_on_device(self) -> None:
        import cupy as cupy_mod

        source = np.ones((5, 2), dtype=np.float64) * 10.0
        target = source + np.array([2.0, 3.0])
        host = EstimateRigidComponentsFromControlPoints(
            target, source, reflected_override=False)
        device = EstimateRigidComponentsFromControlPoints(
            cupy_mod.asarray(target), cupy_mod.asarray(source),
            reflected_override=False)
        self.assertIsInstance(device.translation, cupy_mod.ndarray)
        np.testing.assert_allclose(
            nornir_imageregistration.EnsureNumpyArray(device.translation),
            nornir_imageregistration.EnsureNumpyArray(host.translation), atol=1e-9)
        self.assertAlmostEqual(device.scale, 1.0)
        self.assertAlmostEqual(device.angle, 0.0)


if __name__ == '__main__':
    unittest.main()
