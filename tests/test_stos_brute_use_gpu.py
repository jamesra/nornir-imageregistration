"""CPU override for slice-to-slice FFT scoring (Spacebar queue)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.alignment_record import AlignmentRecord
from nornir_imageregistration.computational_lib import ComputationLib
from nornir_imageregistration import local_distortion_correction
from nornir_imageregistration import stos_brute


class TestUseGpuScoringOverride(unittest.TestCase):
    """``use_gpu=False`` must not follow the process-wide CuPy backend."""

    def test_use_cupy_for_scoring_false_ignores_process_lib(self) -> None:
        with patch(
                "nornir_imageregistration.GetActiveComputationLib",
                return_value=ComputationLib.cupy,
        ):
            self.assertFalse(stos_brute._use_cupy_for_scoring(False))
            self.assertTrue(stos_brute._use_cupy_for_scoring(None))
            self.assertTrue(stos_brute._use_cupy_for_scoring(True))

    def test_find_best_angle_use_gpu_false_skips_gpu_multi_angle(self) -> None:
        rng = np.random.default_rng(0)
        source = rng.random((32, 32)).astype(np.float32)
        target = np.roll(source, 2, axis=0)
        source_stats = nornir_imageregistration.ImageStats.CalcStats(source)
        target_stats = nornir_imageregistration.ImageStats.CalcStats(target)
        with patch(
                "nornir_imageregistration.GetActiveComputationLib",
                return_value=ComputationLib.cupy,
        ):
            with patch("nornir_imageregistration.stos_brute.ScoreManyAnglesGpu") as gpu_score:
                record = stos_brute._find_best_angle(
                    source_image=source,
                    target_image=target,
                    source_stats=source_stats,
                    target_stats=target_stats,
                    angle_range=[0.0, 1.0],
                    min_overlap=0.5,
                    SingleThread=True,
                    use_gpu=False,
                )
        gpu_score.assert_not_called()
        self.assertIsNotNone(record)
        self.assertGreater(float(record.weight), 0.0)

    def test_attempt_align_point_forwards_use_gpu(self) -> None:
        roi = np.ones((16, 16), dtype=np.float32)
        with patch(
                "nornir_imageregistration.local_distortion_correction.ApproximateRigidTransformByTargetPoints",
                return_value=[MagicMock()],
        ):
            with patch(
                    "nornir_imageregistration.local_distortion_correction.BuildAlignmentROIs",
                    return_value=(roi, roi),
            ):
                with patch(
                        "nornir_imageregistration.local_distortion_correction.is_alignable_cell",
                        return_value=True,
                ):
                    with patch(
                            "nornir_imageregistration.stos_brute.SliceToSliceRigidRegistration",
                            return_value=AlignmentRecord(peak=(0.0, 0.0), weight=1.0, angle=0.0),
                    ) as rigid:
                        local_distortion_correction.AttemptAlignPoint(
                            transform=MagicMock(),
                            targetImage=np.ones((32, 32), dtype=np.float32),
                            sourceImage=np.ones((32, 32), dtype=np.float32),
                            target_image_stats=None,
                            source_image_stats=None,
                            target_controlpoint=np.array([8.0, 8.0]),
                            alignmentArea=np.array([16.0, 16.0]),
                            anglesToSearch=np.array([0.0]),
                            use_gpu=False,
                        )
        rigid.assert_called_once()
        self.assertIs(rigid.call_args.kwargs["use_gpu"], False)

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "requires CuPy")
    def test_find_best_angle_gpu_skips_shared_memory(self) -> None:
        """Multi-angle GPU scoring must keep live device arrays, not POSIX shm."""
        import cupy as cupy_mod

        rng = np.random.default_rng(0)
        source = rng.random((32, 32)).astype(np.float32)
        target = np.roll(source, 2, axis=0)
        source_stats = nornir_imageregistration.ImageStats.CalcStats(source)
        target_stats = nornir_imageregistration.ImageStats.CalcStats(target)
        dummy = AlignmentRecord(peak=(0.0, 0.0), weight=1.0, angle=0.0)
        previous = nornir_imageregistration.GetActiveComputationLib()
        nornir_imageregistration.SetActiveComputationLib(ComputationLib.cupy)
        try:
            with patch(
                    "nornir_imageregistration.npArrayToSharedArray",
                    side_effect=AssertionError(
                        "GPU angle search must not copy to shared memory"),
            ) as shm:
                with patch(
                        "nornir_imageregistration.stos_brute.ScoreManyAnglesGpu",
                        return_value=[dummy],
                ) as gpu_score:
                    with patch("nornir_pools.GetGlobalMultithreadingPool") as pool:
                        record = stos_brute._find_best_angle(
                            source_image=source,
                            target_image=target,
                            source_stats=source_stats,
                            target_stats=target_stats,
                            angle_range=[0.0, 1.0],
                            min_overlap=0.5,
                            SingleThread=False,
                            use_gpu=True,
                        )
        finally:
            nornir_imageregistration.SetActiveComputationLib(previous)
        shm.assert_not_called()
        pool.assert_not_called()
        gpu_score.assert_called_once()
        self.assertIs(record, dummy)
        kwargs = gpu_score.call_args.kwargs
        self.assertIsInstance(kwargs["target_original"], cupy_mod.ndarray)
        self.assertIsInstance(kwargs["source_original"], cupy_mod.ndarray)

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "requires CuPy")
    def test_find_best_angle_gpu_two_angles_scores_on_device(self) -> None:
        rng = np.random.default_rng(0)
        source = rng.random((32, 32)).astype(np.float32)
        target = np.roll(source, 2, axis=0)
        source_stats = nornir_imageregistration.ImageStats.CalcStats(source)
        target_stats = nornir_imageregistration.ImageStats.CalcStats(target)
        previous = nornir_imageregistration.GetActiveComputationLib()
        nornir_imageregistration.SetActiveComputationLib(ComputationLib.cupy)
        try:
            record = stos_brute._find_best_angle(
                source_image=source,
                target_image=target,
                source_stats=source_stats,
                target_stats=target_stats,
                angle_range=[0.0, 5.0],
                min_overlap=0.5,
                SingleThread=False,
                use_gpu=True,
            )
        finally:
            nornir_imageregistration.SetActiveComputationLib(previous)
        self.assertIsNotNone(record)
        self.assertGreater(float(record.weight), 0.0)
