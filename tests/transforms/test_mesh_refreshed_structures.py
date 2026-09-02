"""Snapshot/install of mesh Delaunay interpolators and RBF must not mutate until apply."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st

import nornir_imageregistration
from nornir_imageregistration import HasCupy, interactive_edit
from nornir_imageregistration.transforms.meshwithrbffallback import (
    MeshWithRBFFallback,
    MeshWithRBFFallback_GPUComponent,
)
from nornir_imageregistration.transforms.one_way_rbftransform import OneWayRBFWithLinearCorrection
from nornir_imageregistration.transforms.triangulation import (
    Triangulation,
    apply_barycentric_stencil,
    barycentric_sample_delaunay,
    barycentric_stencil_delaunay,
    barycentric_weights_in_triangle,
)


def _identity_mesh() -> MeshWithRBFFallback:
    points = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 32.0, 0.0, 32.0],
            [32.0, 0.0, 32.0, 0.0],
            [32.0, 32.0, 32.0, 32.0],
        ],
        dtype=np.float64,
    )
    return MeshWithRBFFallback(points)


class TestMeshRefreshedStructures(unittest.TestCase):
    """Off-UI bundle construction must leave the live mesh untouched until apply."""

    def test_build_does_not_mutate_live_structures(self) -> None:
        mesh = _identity_mesh()
        mesh.InitializeDataStructures()
        live_rbf = mesh._ForwardRBFInstance
        live_fixed = mesh._fixedtri
        self.assertIsNotNone(live_rbf)
        self.assertIsNotNone(live_fixed)

        new_target = np.asarray(mesh.TargetPoints[0], dtype=np.float64) + np.array((5.0, -3.0))
        mesh.UpdateTargetPointsByIndex(0, new_target)
        self.assertIs(mesh._ForwardRBFInstance, live_rbf)
        self.assertIsNone(mesh._fixedtri)
        self.assertTrue(mesh._continuous_stale)

        bundle = mesh.build_refreshed_continuous()
        self.assertIs(mesh._ForwardRBFInstance, live_rbf)
        self.assertIsNone(mesh._fixedtri)
        self.assertIsNot(bundle.forward_rbf, live_rbf)
        self.assertIsNotNone(bundle.fixedtri)

        mesh.apply_refreshed_continuous(bundle)
        self.assertIs(mesh._ForwardRBFInstance, bundle.forward_rbf)
        self.assertIs(mesh._fixedtri, bundle.fixedtri)
        self.assertFalse(mesh._continuous_stale)

    def test_stale_forward_uses_source_delaunay_without_qhull(self) -> None:
        mesh = _identity_mesh()
        _ = mesh.source_space_trianglulation
        _ = mesh.target_space_trianglulation
        warped_before = mesh._warpedtri
        fixed_before = mesh._fixedtri
        new_target = np.asarray(mesh.TargetPoints[0], dtype=np.float64) + np.array((4.0, 2.0))
        interactive_edit.begin()
        try:
            mesh.UpdateTargetPointsByIndex(0, new_target)
            self.assertIs(mesh._warpedtri, warped_before)
            self.assertIs(mesh._fixedtri, fixed_before)
            with patch("scipy.spatial.Delaunay", side_effect=AssertionError("Qhull")):
                mapped = mesh.Transform(np.asarray(mesh.SourcePoints), extrapolate=False)
                mesh.InverseTransform(np.asarray(mesh.TargetPoints), extrapolate=False)
        finally:
            interactive_edit.end()
        np.testing.assert_allclose(mapped[0], new_target, rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(
            mapped[1:], np.asarray(mesh.TargetPoints[1:]), rtol=1e-6, atol=1e-5)
        self.assertIsNone(mesh._ForwardInterpolator)
        self.assertIsNone(mesh._ForwardRBFInstance)

    def test_discrete_forward_reuses_host_target_snapshot(self) -> None:
        mesh = _identity_mesh()
        _ = mesh.source_space_trianglulation
        query = np.asarray(mesh.SourcePoints, dtype=np.float64)
        new_target = np.asarray(mesh.TargetPoints[0], dtype=np.float64) + np.array((3.0, 1.0))
        interactive_edit.begin()
        try:
            mesh.UpdateTargetPointsByIndex(0, new_target)
            mesh.Transform(query, extrapolate=False)
            copies_after_first = mesh._host_target_copy_count
            mesh.Transform(query, extrapolate=False)
        finally:
            interactive_edit.end()
        self.assertEqual(copies_after_first, 1)
        self.assertEqual(mesh._host_target_copy_count, 1)

    @unittest.skipUnless(HasCupy(), "requires CuPy")
    def test_gpu_discrete_forward_reuses_host_target_snapshot(self) -> None:
        import scipy.spatial

        from nornir_imageregistration.transforms import utils as transform_utils

        mesh = MeshWithRBFFallback_GPUComponent(np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 32.0, 0.0, 32.0],
                [32.0, 0.0, 32.0, 0.0],
                [32.0, 32.0, 32.0, 32.0],
            ],
            dtype=np.float64,
        ))
        mesh._warpedtri = scipy.spatial.Delaunay(
            transform_utils.host_copy_points(mesh.SourcePoints), incremental=False)
        query = np.asarray(
            nornir_imageregistration.EnsureNumpyArray(mesh.SourcePoints), dtype=np.float64)
        new_target = np.asarray(
            nornir_imageregistration.EnsureNumpyArray(mesh.TargetPoints[0]), dtype=np.float64) + np.array((2.0, 4.0))
        interactive_edit.begin()
        try:
            mesh.UpdateTargetPointsByIndex(0, new_target)
            mesh.Transform(query, extrapolate=False)
            copies_after_first = mesh._host_target_copy_count
            mesh.Transform(query, extrapolate=False)
        finally:
            interactive_edit.end()
        self.assertEqual(copies_after_first, 1)
        self.assertEqual(mesh._host_target_copy_count, 1)

    @unittest.skipUnless(HasCupy(), "requires CuPy")
    def test_gpu_build_fans_host_structures_to_thread_pool(self) -> None:
        """#192: host Delaunay work must overlap GPU RBF solves via the pool."""
        from unittest.mock import MagicMock

        import nornir_imageregistration.transforms.meshwithrbffallback as mesh_mod

        mesh = MeshWithRBFFallback_GPUComponent(np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 32.0, 0.0, 32.0],
                [32.0, 0.0, 32.0, 0.0],
                [32.0, 32.0, 32.0, 32.0],
            ],
            dtype=np.float64,
        ))
        recorded: list[str] = []

        def fake_get_pool():
            pool = MagicMock()

            def add_task(name, fn, *args, **kwargs):
                recorded.append(str(name))
                result = fn(*args, **kwargs)
                task = MagicMock()
                task.wait_return.return_value = result
                return task

            pool.add_task.side_effect = add_task
            return pool

        with patch.object(mesh_mod.nornir_pools, "GetGlobalThreadPool", side_effect=fake_get_pool):
            bundle = mesh.build_refreshed_continuous()

        self.assertTrue(
            any("host" in name.lower() for name in recorded),
            msg=f"expected host structure task, got {recorded!r}",
        )
        self.assertIsNotNone(bundle.forward_rbf)
        self.assertIsNotNone(bundle.warpedtri)

    def test_remove_duplicates_false_skips_collapse(self) -> None:
        mesh = _identity_mesh()
        n_before = int(np.asarray(mesh.TargetPoints).shape[0])
        with patch(
                "nornir_imageregistration.transforms.triangulation.Triangulation.RemoveDuplicateControlPoints",
                side_effect=AssertionError("dedupe when remove_duplicates=False"),
        ):
            mesh.UpdateTargetPointsByIndex(
                0,
                np.asarray(mesh.TargetPoints[0], dtype=np.float64) + (1.0, 2.0),
                remove_duplicates=False,
            )
        self.assertEqual(int(np.asarray(mesh.TargetPoints).shape[0]), n_before)

    def test_remove_duplicates_true_calls_collapse(self) -> None:
        mesh = _identity_mesh()
        with patch(
                "nornir_imageregistration.transforms.triangulation.Triangulation.RemoveDuplicateControlPoints",
                wraps=Triangulation.RemoveDuplicateControlPoints,
        ) as mock_collapse:
            mesh.UpdateTargetPointsByIndex(
                0, np.asarray(mesh.TargetPoints[0], dtype=np.float64) + (1.0, 2.0))
            mock_collapse.assert_called_once()

    def test_remove_duplicates_false_keeps_coincident_row(self) -> None:
        mesh = _identity_mesh()
        n_before = int(np.asarray(mesh.TargetPoints).shape[0])
        onto = np.asarray(mesh.TargetPoints[1], dtype=np.float64)
        index = mesh.UpdateTargetPointsByIndex(0, onto, remove_duplicates=False)
        self.assertEqual(int(index), 0)
        self.assertEqual(int(np.asarray(mesh.TargetPoints).shape[0]), n_before)

    def test_remove_duplicates_true_collapses_coincident_row(self) -> None:
        mesh = _identity_mesh()
        n_before = int(np.asarray(mesh.TargetPoints).shape[0])
        onto = np.asarray(mesh.TargetPoints[1], dtype=np.float64)
        mesh.UpdateTargetPointsByIndex(0, onto)
        self.assertLess(int(np.asarray(mesh.TargetPoints).shape[0]), n_before)


class TestBarycentricStencil(unittest.TestCase):
    """Cached Delaunay stencils must match sample and restencil without Qhull."""

    def test_stencil_at_control_points_reproduces_target_points(self) -> None:
        mesh = _identity_mesh()
        source = np.asarray(mesh.SourcePoints, dtype=np.float64)
        target = np.asarray(mesh.TargetPoints, dtype=np.float64)
        delaunay = mesh.source_space_trianglulation
        indices, weights = barycentric_stencil_delaunay(source, delaunay)
        mapped = apply_barycentric_stencil(indices, weights, target)
        np.testing.assert_allclose(mapped, target, rtol=1e-6, atol=1e-5)

    def test_stencil_matches_barycentric_sample(self) -> None:
        mesh = _identity_mesh()
        queries = np.array(
            [
                [0.0, 0.0],
                [8.0, 8.0],
                [16.0, 16.0],
                [24.0, 8.0],
                [31.0, 31.0],
            ],
            dtype=np.float64,
        )
        delaunay = mesh.source_space_trianglulation
        target = np.asarray(mesh.TargetPoints, dtype=np.float64)
        indices, weights = barycentric_stencil_delaunay(queries, delaunay)
        mapped = apply_barycentric_stencil(indices, weights, target)
        sampled = barycentric_sample_delaunay(queries, delaunay, target)
        np.testing.assert_allclose(mapped, sampled, rtol=1e-9, atol=1e-9)

    @example(dy=0.0, dx=0.0)
    @example(dy=5.0, dx=-3.0)
    @given(
        dy=st.floats(-8.0, 8.0, allow_nan=False, allow_infinity=False),
        dx=st.floats(-8.0, 8.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_target_move_stencil_blend_matches_sample(self, dy: float, dx: float) -> None:
        mesh = _identity_mesh()
        queries = np.array(
            [
                [0.0, 0.0],
                [8.0, 8.0],
                [16.0, 16.0],
                [24.0, 8.0],
            ],
            dtype=np.float64,
        )
        delaunay = mesh.source_space_trianglulation
        indices, weights = barycentric_stencil_delaunay(queries, delaunay)
        new_target = np.asarray(mesh.TargetPoints[0], dtype=np.float64) + np.array((dy, dx), dtype=np.float64)
        mesh.UpdateTargetPointsByIndex(0, new_target)
        target = np.asarray(mesh.TargetPoints, dtype=np.float64)
        blended = apply_barycentric_stencil(indices, weights, target)
        sampled = barycentric_sample_delaunay(queries, delaunay, target)
        np.testing.assert_allclose(blended, sampled, rtol=1e-6, atol=1e-5)

    def test_centroid_weights_are_inside_one_third(self) -> None:
        a = np.array([0.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 10.0], dtype=np.float64)
        c = np.array([10.0, 0.0], dtype=np.float64)
        query = (a + b + c) / 3.0
        weights, inside = barycentric_weights_in_triangle(query, a, b, c)
        self.assertTrue(bool(inside[0]))
        np.testing.assert_allclose(weights[0], np.full(3, 1.0 / 3.0), rtol=1e-9, atol=1e-9)

    def test_source_move_still_inside_without_find_simplex(self) -> None:
        mesh = _identity_mesh()
        query = np.array([[8.0, 8.0]], dtype=np.float64)
        delaunay = mesh.source_space_trianglulation
        indices, weights = barycentric_stencil_delaunay(query, delaunay)
        i0, i1, i2 = (int(v) for v in indices[0])
        interactive_edit.begin()
        try:
            mesh.UpdateSourcePointsByIndex(
                0, np.asarray(mesh.SourcePoints[0], dtype=np.float64) + np.array((0.5, 0.4)))
            source = np.asarray(mesh.SourcePoints, dtype=np.float64)
            with patch.object(delaunay, "find_simplex", side_effect=AssertionError("find_simplex")):
                new_weights, inside = barycentric_weights_in_triangle(
                    query, source[i0], source[i1], source[i2])
            self.assertTrue(bool(inside[0]))
            self.assertFalse(np.allclose(new_weights[0], weights[0]))
        finally:
            interactive_edit.end()

    def test_source_move_outside_triangle_does_not_call_qhull(self) -> None:
        mesh = _identity_mesh()
        query = np.array([[8.0, 8.0]], dtype=np.float64)
        delaunay = mesh.source_space_trianglulation
        indices, _weights = barycentric_stencil_delaunay(query, delaunay)
        i0, i1, i2 = (int(v) for v in indices[0])
        interactive_edit.begin()
        try:
            mesh.UpdateSourcePointsByIndex(0, np.array([30.0, 30.0], dtype=np.float64))
            source = np.asarray(mesh.SourcePoints, dtype=np.float64)
            with patch("scipy.spatial.Delaunay", side_effect=AssertionError("Qhull")):
                with patch.object(delaunay, "find_simplex", side_effect=AssertionError("find_simplex")):
                    _new_weights, inside = barycentric_weights_in_triangle(
                        query, source[i0], source[i1], source[i2])
            self.assertFalse(bool(inside[0]))
        finally:
            interactive_edit.end()


class TestMeshStaleRbfFallback(unittest.TestCase):
    """Outside-hull Transform uses the last installed RBF until apply replaces it."""

    def test_initialized_transform_discrete_inside_rbf_outside(self) -> None:
        mesh = _identity_mesh()
        mesh.InitializeDataStructures()
        inside = np.array([[16.0, 16.0]], dtype=np.float64)
        outside = np.array([[-20.0, -20.0]], dtype=np.float64)
        discrete_inside = mesh.Transform(inside, extrapolate=False)
        filled_inside = mesh.Transform(inside, extrapolate=True)
        np.testing.assert_allclose(filled_inside, discrete_inside, rtol=1e-6, atol=1e-5)
        self.assertFalse(np.isnan(np.asarray(discrete_inside)).any())
        discrete_outside = mesh.Transform(outside, extrapolate=False)
        self.assertTrue(np.isnan(np.asarray(discrete_outside)).any())
        filled_outside = mesh.Transform(outside, extrapolate=True)
        self.assertFalse(np.isnan(np.asarray(filled_outside)).any())
        with patch(
                "nornir_imageregistration.transforms.meshwithrbffallback.OneWayRBFWithLinearCorrection",
                side_effect=AssertionError("RBF construct"),
        ):
            mesh.Transform(outside, extrapolate=False)
            mesh.Transform(inside, extrapolate=False)

    @example(dy=0.0, dx=0.0)
    @example(dy=5.0, dx=-3.0)
    @given(
        dy=st.floats(-8.0, 8.0, allow_nan=False, allow_infinity=False),
        dx=st.floats(-8.0, 8.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_update_keeps_stale_rbf_for_oob(self, dy: float, dx: float) -> None:
        mesh = _identity_mesh()
        mesh.InitializeDataStructures()
        live_rbf = mesh._ForwardRBFInstance
        self.assertIsNotNone(live_rbf)
        new_target = np.asarray(mesh.TargetPoints[0], dtype=np.float64) + np.array((dy, dx))
        mesh.UpdateTargetPointsByIndex(0, new_target)
        self.assertIs(mesh._ForwardRBFInstance, live_rbf)
        self.assertTrue(mesh._continuous_stale)
        outside = np.array([[-25.0, 50.0]], dtype=np.float64)
        with patch.object(
                OneWayRBFWithLinearCorrection,
                "PrecomputeWeights",
                side_effect=AssertionError("RBF solve"),
        ):
            with patch(
                    "nornir_imageregistration.transforms.meshwithrbffallback.OneWayRBFWithLinearCorrection",
                    side_effect=AssertionError("RBF construct"),
            ):
                mapped = mesh.Transform(outside, extrapolate=True)
        self.assertFalse(np.isnan(np.asarray(mapped)).any())
        self.assertIs(mesh._ForwardRBFInstance, live_rbf)

    def test_apply_refreshed_continuous_replaces_instance(self) -> None:
        mesh = _identity_mesh()
        mesh.InitializeDataStructures()
        live_rbf = mesh._ForwardRBFInstance
        mesh.UpdateTargetPointsByIndex(
            0, np.asarray(mesh.TargetPoints[0], dtype=np.float64) + np.array((4.0, -2.0)))
        self.assertIs(mesh._ForwardRBFInstance, live_rbf)
        bundle = mesh.build_refreshed_continuous()
        self.assertIsNot(bundle.forward_rbf, live_rbf)
        mesh.apply_refreshed_continuous(bundle)
        self.assertIs(mesh._ForwardRBFInstance, bundle.forward_rbf)
        self.assertFalse(mesh._continuous_stale)

    def test_barycentric_nan_outside_nans_oob_keeps_inside(self) -> None:
        mesh = _identity_mesh()
        delaunay = mesh.source_space_trianglulation
        target = np.asarray(mesh.TargetPoints, dtype=np.float64)
        queries = np.array(
            [
                [16.0, 16.0],
                [-10.0, -10.0],
            ],
            dtype=np.float64,
        )
        filled = barycentric_sample_delaunay(queries, delaunay, target)
        nanned = barycentric_sample_delaunay(queries, delaunay, target, nan_outside=True)
        np.testing.assert_allclose(nanned[0], filled[0], rtol=1e-9, atol=1e-9)
        self.assertFalse(np.isnan(filled[1]).any())
        self.assertTrue(np.isnan(nanned[1]).all())
