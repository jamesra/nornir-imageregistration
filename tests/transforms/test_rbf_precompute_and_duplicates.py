"""Tests for control-point duplicate grouping and RBF weight precompute."""

from __future__ import annotations

import unittest

import hypothesis.strategies as st
import numpy as np
from hypothesis import example, given, settings
from hypothesis.extra.numpy import arrays

from nornir_imageregistration.transforms.controlpointbase import (
    ControlPointBase,
    ControlPointBase_GPUComponent,
    ControlPointsHaveDuplicatePositions,
    GroupControlPointIndicesByPosition,
)
from nornir_imageregistration.transforms.meshwithrbffallback import MeshWithRBFFallback
from nornir_imageregistration.transforms.one_way_rbftransform import OneWayRBFWithLinearCorrection

try:
    from tests.transforms import TranslateTransformPoints
except ImportError:
    from transforms import TranslateTransformPoints  # type: ignore[no-redef]


def _maybe_cupy():
    try:
        import cupy as cp
    except ImportError:
        return None
    if getattr(cp, "__name__", "") == "nornir_imageregistration.cupy_thunk":
        return None
    return cp


class TestGroupControlPointIndicesByPosition(unittest.TestCase):
    def test_group_includes_singletons_and_duplicates(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 2.0, 2.0],
                [0.0, 0.0, 3.0, 3.0],  # dup of row 0
                [2.0, 2.0, 4.0, 4.0],
                [1.0004, 1.0004, 5.0, 5.0],  # rounds to same as row 1
            ],
            dtype=np.float32,
        )
        groups = GroupControlPointIndicesByPosition(points)
        self.assertEqual(groups, [[0, 2], [1, 4], [3]])

    def test_empty_and_single_point(self) -> None:
        self.assertEqual(GroupControlPointIndicesByPosition(np.empty((0, 2), dtype=np.float32)), [])
        self.assertEqual(
            GroupControlPointIndicesByPosition(np.array([[1.5, 2.5]], dtype=np.float32)),
            [[0]],
        )

    def test_group_accepts_cupy_arrays(self) -> None:
        cp = _maybe_cupy()
        if cp is None:
            self.skipTest("CuPy not available")
        points = cp.asarray(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=cp.float32,
        )
        groups = GroupControlPointIndicesByPosition(points)
        self.assertEqual(groups, [[0, 2], [1]])

    def test_numpy_and_cupy_groups_match(self) -> None:
        cp = _maybe_cupy()
        if cp is None:
            self.skipTest("CuPy not available")
        host = np.array(
            [
                [10.0, 20.0],
                [30.0, 40.0],
                [10.0001, 20.0001],
                [50.0, 60.0],
                [30.0, 40.0],
            ],
            dtype=np.float32,
        )
        self.assertEqual(
            GroupControlPointIndicesByPosition(host),
            GroupControlPointIndicesByPosition(cp.asarray(host)),
        )


class TestControlPointsHaveDuplicatePositions(unittest.TestCase):
    def test_no_duplicates(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        self.assertFalse(ControlPointsHaveDuplicatePositions(points))

    def test_exact_duplicates(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        self.assertTrue(ControlPointsHaveDuplicatePositions(points))

    def test_duplicates_after_rounding(self) -> None:
        points = np.array([[1.0, 1.0], [1.0004, 1.0004]], dtype=np.float32)
        self.assertTrue(ControlPointsHaveDuplicatePositions(points, decimals=3))
        self.assertFalse(ControlPointsHaveDuplicatePositions(points, decimals=4))

    def test_empty_and_single_are_not_duplicates(self) -> None:
        self.assertFalse(ControlPointsHaveDuplicatePositions(np.empty((0, 2), dtype=np.float32)))
        self.assertFalse(ControlPointsHaveDuplicatePositions(np.array([[1.0, 2.0]], dtype=np.float32)))

    def test_cupy_matches_numpy(self) -> None:
        cp = _maybe_cupy()
        if cp is None:
            self.skipTest("CuPy not available")
        unique = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        dups = np.array([[0.0, 0.0], [1.0, 2.0], [0.0001, 0.0001]], dtype=np.float32)
        self.assertEqual(
            ControlPointsHaveDuplicatePositions(unique),
            ControlPointsHaveDuplicatePositions(cp.asarray(unique)),
        )
        self.assertEqual(
            ControlPointsHaveDuplicatePositions(dups),
            ControlPointsHaveDuplicatePositions(cp.asarray(dups)),
        )
        self.assertFalse(ControlPointsHaveDuplicatePositions(cp.asarray(unique)))
        self.assertTrue(ControlPointsHaveDuplicatePositions(cp.asarray(dups)))

    @given(n=st.integers(min_value=2, max_value=24))
    @settings(max_examples=40, deadline=None)
    def test_unique_rows_have_no_duplicates_until_injected(self, n: int) -> None:
        # Distinct after decimals=3 rounding.
        points = np.column_stack(
            (
                np.arange(n, dtype=np.float64),
                np.arange(n, dtype=np.float64) * 0.1,
            )
        )
        self.assertFalse(ControlPointsHaveDuplicatePositions(points))
        with_dup = np.vstack((points, points[0:1]))
        self.assertTrue(ControlPointsHaveDuplicatePositions(with_dup))


_finite32 = st.floats(
    min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False, width=32)


def _as_host(arr) -> np.ndarray:
    getter = getattr(arr, "get", None)
    if callable(getter):
        return np.asarray(getter())
    return np.asarray(arr)


def _keep_first_oracle(pts: np.ndarray, decimals: int = 3) -> np.ndarray:
    yx = np.asarray(pts[:, 0:2], dtype=np.float64)
    scale = 10.0 ** decimals
    rounded = np.around(yx, decimals=decimals)
    qy = np.rint(rounded[:, 0] * scale).astype(np.int64)
    qx = np.rint(rounded[:, 1] * scale).astype(np.int64)
    seen: set[tuple[int, int]] = set()
    keep: list[int] = []
    for i in range(int(pts.shape[0])):
        key = (int(qy[i]), int(qx[i]))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return pts[np.asarray(keep, dtype=np.intp)]


class TestDuplicateGrouping(unittest.TestCase):
    def test_find_duplicates_returns_only_multi_member_groups(self) -> None:
        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [2.0, 2.0],
            ],
            dtype=np.float32,
        )
        dups = ControlPointBase.FindDuplicates(points)
        self.assertEqual(dups, [[0, 2]])

    def test_remove_duplicate_control_points_keeps_first(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 20.0, 20.0],
                [0.0, 0.0, 30.0, 30.0],
            ],
            dtype=np.float32,
        )
        cleaned = ControlPointBase.RemoveDuplicateControlPoints(points)
        np.testing.assert_array_equal(
            cleaned,
            np.array(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [1.0, 1.0, 20.0, 20.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_remove_duplicate_empty_and_single(self) -> None:
        empty = np.empty((0, 4), dtype=np.float32)
        self.assertEqual(ControlPointBase.RemoveDuplicateControlPoints(empty).shape[0], 0)
        one = np.array([[1.5, 2.5, 3.5, 4.5]], dtype=np.float32)
        np.testing.assert_array_equal(ControlPointBase.RemoveDuplicateControlPoints(one), one)

    def test_remove_duplicate_drops_nan_rows(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [np.nan, 0.0, 2.0, 2.0],
                [1.0, 1.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        )
        cleaned = ControlPointBase.RemoveDuplicateControlPoints(points)
        np.testing.assert_array_equal(
            cleaned,
            np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 3.0, 3.0]], dtype=np.float32),
        )

    def test_remove_duplicate_rounds_to_three_decimals(self) -> None:
        points = np.array(
            [
                [1.0, 1.0, 10.0, 10.0],
                [1.0004, 1.0004, 20.0, 20.0],
            ],
            dtype=np.float32,
        )
        cleaned = ControlPointBase.RemoveDuplicateControlPoints(points)
        self.assertEqual(cleaned.shape[0], 1)
        np.testing.assert_array_equal(cleaned[0], points[0])

    @given(
        yx=arrays(
            dtype=np.float32,
            shape=st.integers(min_value=2, max_value=24).map(lambda n: (n, 2)),
            elements=_finite32,
        ),
        dup_src=st.integers(min_value=0, max_value=23),
        dup_dst=st.integers(min_value=0, max_value=23),
    )
    @example(
        yx=np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        dup_src=0,
        dup_dst=2,
    )
    @example(
        yx=np.array([[-2.5, -3.25], [1.0, 0.0], [-2.5, -3.25]], dtype=np.float32),
        dup_src=0,
        dup_dst=2,
    )
    @settings(max_examples=40, deadline=None)
    def test_remove_dups_keeps_first_numpy_matches_cupy(
            self,
            yx: np.ndarray,
            dup_src: int,
            dup_dst: int) -> None:
        n = int(yx.shape[0])
        yx = np.array(yx, copy=True)
        yx[int(dup_dst) % n] = yx[int(dup_src) % n]
        pts = np.hstack((yx, yx + 100.0)).astype(np.float32)
        host_out = ControlPointBase.RemoveDuplicateControlPoints(pts)
        np.testing.assert_array_equal(host_out, _keep_first_oracle(pts))
        cp = _maybe_cupy()
        if cp is None:
            return
        device_out = ControlPointBase.RemoveDuplicateControlPoints(cp.asarray(pts))
        gpu_out = ControlPointBase_GPUComponent.RemoveDuplicateControlPoints(cp.asarray(pts))
        np.testing.assert_allclose(_as_host(device_out), host_out, rtol=0, atol=1e-5)
        np.testing.assert_allclose(_as_host(gpu_out), host_out, rtol=0, atol=1e-5)


class TestCreateBetaMatrixDuplicates(unittest.TestCase):
    def test_create_beta_matrix_rejects_duplicates(self) -> None:
        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        with self.assertRaises(ValueError):
            OneWayRBFWithLinearCorrection.CreateBetaMatrix(
                points, OneWayRBFWithLinearCorrection.DefaultBasisFunction
            )

    def test_create_beta_matrix_accepts_unique_points(self) -> None:
        points = TranslateTransformPoints[:, 2:].astype(np.float32)
        beta = OneWayRBFWithLinearCorrection.CreateBetaMatrix(
            points, OneWayRBFWithLinearCorrection.DefaultBasisFunction
        )
        self.assertEqual(beta.shape, (len(points) + 3, len(points) + 3))


class TestMeshRBFPrecompute(unittest.TestCase):
    def test_initialize_data_structures_precomputes_weights(self) -> None:
        mesh = MeshWithRBFFallback(TranslateTransformPoints.astype(np.float32))
        mesh.InitializeDataStructures()
        self.assertIsNotNone(mesh._ForwardRBFInstance)
        self.assertIsNotNone(mesh._ReverseRBFInstance)
        self.assertIsNotNone(mesh._ForwardRBFInstance._weights)
        self.assertIsNotNone(mesh._ReverseRBFInstance._weights)

    def test_precompute_weights_on_one_way_rbf(self) -> None:
        warped = TranslateTransformPoints[:, 2:].astype(np.float32)
        fixed = TranslateTransformPoints[:, 0:2].astype(np.float32)
        rbf = OneWayRBFWithLinearCorrection(warped, fixed)
        self.assertIsNone(rbf._weights)
        rbf.PrecomputeWeights()
        self.assertIsNotNone(rbf._weights)

    def test_extrapolate_lazily_solves_rbf_weights(self) -> None:
        """extrapolate=True must work without a prior InitializeDataStructures call."""
        mesh = MeshWithRBFFallback(TranslateTransformPoints.astype(np.float32))
        outside = np.array([[1e6, 1e6]], dtype=np.float32)
        _ = mesh.Transform(outside, extrapolate=False)
        self.assertIsNone(mesh._ForwardRBFInstance)
        result = mesh.Transform(outside, extrapolate=True)
        self.assertIsNotNone(mesh._ForwardRBFInstance)
        self.assertIsNotNone(mesh._ForwardRBFInstance._weights)
        result_np = np.asarray(result)
        self.assertFalse(np.isnan(result_np).any())


if __name__ == '__main__':
    unittest.main()
