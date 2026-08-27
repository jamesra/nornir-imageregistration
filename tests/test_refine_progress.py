"""Unit tests for STOS grid refine dashboard progress helpers."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.computational_lib import HasCupy
from nornir_imageregistration.refine_shared.progress import (
    RefineGridProgressReporter,
    count_initial_grid_points,
    publish_stos_refine_files_progress,
    report_pass_transform,
    snapshot_transform_for_preview,
)


class TestRefineGridProgress(unittest.TestCase):
    """Coverage for refine_shared.progress MQTT helpers."""

    def _make_settings(self) -> nornir_imageregistration.settings.GridRefinement:
        """Build a small GridRefinement context for grid-count tests."""
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        shape = (128, 128)
        image = np.linspace(0.0, 1.0, shape[0] * shape[1], dtype=np.float32).reshape(shape)
        stats = nornir_imageregistration.ImageStats.Create(image)
        return nornir_imageregistration.settings.GridRefinement(
            target_image=image,
            source_image=image.copy(),
            target_image_stats=stats,
            source_image_stats=stats,
            cell_size=np.asarray((32, 32), dtype=np.int32),
            grid_spacing=np.asarray((32, 32), dtype=np.int32),
            single_thread_processing=True,
        )

    def test_count_initial_grid_points_positive(self) -> None:
        """Masked grid refinement on uniform images yields a positive cell count."""
        settings = self._make_settings()
        transform = nornir_imageregistration.transforms.rigid.RigidTranslation(
            (0.0, 0.0))
        count = count_initial_grid_points(transform, settings)
        self.assertGreater(count, 0)

    def test_refine_grid_progress_reporter_publishes_pass_and_locked(self) -> None:
        """Reporter emits pass and locked iterate_progress events with expected depths."""
        reporter = RefineGridProgressReporter(5, 20, depth_base=1)
        with mock.patch("nornir_shared.prettyoutput.publish_run_event") as publish:
            reporter.on_pass_start(2)
            reporter.on_pass_locked(2, locked_count=7, active_count=18)
            reporter.on_complete()

        self.assertGreaterEqual(publish.call_count, 3)
        pass_call = publish.call_args_list[0]
        self.assertEqual(pass_call.args[0], "iterate_progress")
        self.assertEqual(pass_call.kwargs["current"], 2)
        self.assertEqual(pass_call.kwargs["total"], 5)
        self.assertEqual(pass_call.kwargs["depth"], 1)
        self.assertEqual(pass_call.kwargs["track_id"], "stos_refine:passes")

        locked_call = publish.call_args_list[1]
        self.assertEqual(locked_call.kwargs["current"], 7)
        self.assertEqual(locked_call.kwargs["total"], 20)
        self.assertEqual(locked_call.kwargs["depth"], 2)
        self.assertEqual(locked_call.kwargs["track_id"], "stos_refine:locked")
        self.assertIn("7/20", locked_call.kwargs["label"])
        self.assertIn("7/18 active", locked_call.kwargs["label"])

    def test_publish_stos_refine_files_progress_skips_zero_total(self) -> None:
        """Outer file-loop progress is not published when there is no work."""
        with mock.patch("nornir_shared.prettyoutput.publish_run_event") as publish:
            publish_stos_refine_files_progress(current=0, total=0, label="test")
        publish.assert_not_called()

    def test_publish_stos_refine_files_progress_depth_zero(self) -> None:
        """Outer file-loop progress uses depth 0 and the files track id."""
        with mock.patch("nornir_shared.prettyoutput.publish_run_event") as publish:
            publish_stos_refine_files_progress(current=1, total=3, label="StosGridRefine → Grid")
        publish.assert_called_once_with(
            "iterate_progress",
            current=1,
            total=3,
            depth=0,
            track_id="stos_refine:files",
            label="StosGridRefine → Grid",
        )


class TestPassTransformPreview(unittest.TestCase):
    def _mesh(self) -> nornir_imageregistration.transforms.MeshWithRBFFallback:
        points = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 10.0],
            [10.0, 0.0, 10.0, 0.0],
            [10.0, 10.0, 12.0, 11.0],
        ], dtype=np.float64)
        return nornir_imageregistration.transforms.MeshWithRBFFallback(points)

    def test_snapshot_transform_is_independent(self) -> None:
        original = self._mesh()
        snapshot = snapshot_transform_for_preview(original)
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNot(snapshot, original)
        mutated = nornir_imageregistration.EnsureNumpyArray(original.points).copy()
        mutated[3, 2] = 99.0
        original.points = mutated
        snap_points = nornir_imageregistration.EnsureNumpyArray(snapshot.points)
        self.assertNotAlmostEqual(float(snap_points[3, 2]), 99.0)

    def test_report_pass_transform_sends_preview_copy(self) -> None:
        seen: list[tuple[int, int, str, object]] = []
        original = self._mesh()
        report_pass_transform(
            lambda c, t, label, preview=None: seen.append((c, t, label, preview)),
            original,
            2,
            5,
        )
        self.assertEqual(len(seen), 1)
        current, total, label, preview = seen[0]
        self.assertEqual((current, total), (2, 5))
        self.assertIn("updating view", label)
        self.assertIsNot(preview, original)

    @unittest.skipUnless(HasCupy(), "requires CuPy")
    def test_snapshot_cupy_mesh_is_host_cpu_mesh(self) -> None:
        """GUI preview must not keep worker-thread CuPy control points."""
        import cupy as cupy_mod

        host = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 10.0],
            [10.0, 0.0, 10.0, 0.0],
            [10.0, 10.0, 12.0, 11.0],
        ], dtype=np.float64)
        original = nornir_imageregistration.transforms.MeshWithRBFFallback_GPUComponent(
            cupy_mod.asarray(host))
        snapshot = snapshot_transform_for_preview(original)
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsInstance(
            snapshot, nornir_imageregistration.transforms.MeshWithRBFFallback)
        self.assertNotIsInstance(
            snapshot, nornir_imageregistration.transforms.MeshWithRBFFallback_GPUComponent)
        snap_points = snapshot.points
        self.assertFalse(isinstance(snap_points, cupy_mod.ndarray))
        self.assertIsInstance(np.asarray(snap_points), np.ndarray)
        mutated = nornir_imageregistration.EnsureNumpyArray(original.points).copy()
        mutated[3, 2] = 99.0
        original.points = cupy_mod.asarray(mutated)
        snap_after = nornir_imageregistration.EnsureNumpyArray(snapshot.points)
        self.assertNotAlmostEqual(float(snap_after[3, 2]), 99.0)

    def test_report_pass_transform_skips_three_arg_callback(self) -> None:
        seen: list[tuple[int, int, str]] = []
        report_pass_transform(
            lambda c, t, label: seen.append((c, t, label)),
            self._mesh(),
            1,
            3,
            label="custom",
        )
        self.assertEqual(seen, [(1, 3, "custom")])

    def test_refine_transform_reports_pass_preview_inside_loop(self) -> None:
        """Multi-pass refine previews from the same RefineTransform call."""
        import inspect

        from nornir_imageregistration.local_distortion_correction import RefineTransform

        source = inspect.getsource(RefineTransform)
        self.assertIn("report_pass_transform", source)
        self.assertIn("while i <= settings.num_iterations", source)
        self.assertGreater(source.count("report_pass_transform("), 1)


if __name__ == "__main__":
    unittest.main()
