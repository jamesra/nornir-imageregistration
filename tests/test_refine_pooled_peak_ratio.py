"""Regression tests: the pooled refine paths must carry `peak_ratio` forward.

`peak_ratio` is the primary/2nd-peak uniqueness score that every ambiguity gate
keys on. Dropping it fails *open* -- `is_ambiguous_peak(None)` is False -- so a
missing kwarg silently disables the false-peak rejection rather than raising.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from nornir_imageregistration import local_distortion_correction as ldc
from nornir_imageregistration.alignment_record import (
    AlignmentRecord,
    EnhancedAlignmentRecord,
)
from nornir_imageregistration.refine_shared.peak_ratio_gates import (
    PEAK_RATIO_MIN,
    is_ambiguous_peak,
)


class _FakeTask:
    """Stand-in for a pool task that returns a pre-baked alignment record."""

    ID: int
    key: tuple[int, int]
    _record: AlignmentRecord

    def __init__(self, record: AlignmentRecord) -> None:
        self._record = record

    def wait_return(self) -> AlignmentRecord:
        return self._record


class _FakePool:
    """Pool that hands back queued records instead of running the callable."""

    _records: list[AlignmentRecord]
    _next: int

    def __init__(self, records: list[AlignmentRecord]) -> None:
        self._records = records
        self._next = 0

    def add_task(self, _name: str, _func: Any, **_kwargs: Any) -> _FakeTask:
        task = _FakeTask(self._records[self._next])
        self._next += 1
        return task


def _alignment_record(peak_ratio: float | None) -> AlignmentRecord:
    return AlignmentRecord(
        peak=np.asarray((1.0, 2.0), dtype=np.float64),
        weight=7.0,
        angle=0.0,
        flipped_ud=False,
        peak_ratio=peak_ratio,
    )


def _enhanced_record(key: tuple[int, int], *, weight: float,
                     peak_ratio: float | None) -> EnhancedAlignmentRecord:
    point = np.asarray((float(key[0]) * 10.0, float(key[1]) * 10.0), dtype=np.float64)
    return EnhancedAlignmentRecord(
        ID=key,
        TargetPoint=point.copy(),
        SourcePoint=point.copy(),
        peak=np.zeros(2, dtype=np.float64),
        weight=weight,
        angle=0.0,
        flipped_ud=False,
        peak_ratio=peak_ratio,
    )


class TestPooledRefinePeakRatio(unittest.TestCase):
    """`_RefinePointsForTwoImages` pooled branch must not drop `peak_ratio`."""

    def _run_pooled(self, peak_ratios: list[float | None]) -> list[EnhancedAlignmentRecord]:
        keys = [(0, i) for i in range(len(peak_ratios))]
        points = np.asarray([[0.0, float(i) * 10.0] for i in range(len(keys))],
                            dtype=np.float64)
        records = [_alignment_record(r) for r in peak_ratios]

        settings = SimpleNamespace(
            single_thread_processing=False,
            cupy_processing=False,
            cell_size=np.asarray((64, 64), dtype=np.int32),
            angles_to_search=[0.0],
            min_alignment_overlap=0.5,
            target_image=None,
            source_image=None,
            target_image_meta=None,
            source_image_meta=None,
            target_image_stats=None,
            source_image_stats=None,
            ring_scale_fraction_max=None,
            ring_angle_max_degrees=None,
            ring_allow_flip_change=False,
        )

        pool = _FakePool(records)
        with mock.patch.object(ldc, 'get_runtime_config',
                               return_value=SimpleNamespace(
                                   pool_for_cell_tasks=lambda _cupy: pool)), \
                mock.patch.object(ldc, '_use_batched_vertex_measurement',
                                  return_value=False), \
                mock.patch.object(ldc, 'ApproximateRigidTransformBySourcePoints',
                                  return_value=[None] * len(keys)), \
                mock.patch('nornir_imageregistration.in_debug_mode',
                           return_value=False):
            return ldc._RefinePointsForTwoImages(
                transform=None,  # type: ignore[arg-type]
                keys=keys,
                sourcePoints=points,
                targetPoints=points,
                settings=settings)  # type: ignore[arg-type]

    def test_pooled_path_preserves_peak_ratio(self) -> None:
        out = self._run_pooled([1.05, 3.5])
        self.assertEqual(len(out), 2)
        self.assertEqual([r.peak_ratio for r in out], [1.05, 3.5])

    def test_pooled_ambiguous_peak_still_gated(self) -> None:
        """An ambiguous measurement must remain visible to the ambiguity gate."""
        ambiguous = PEAK_RATIO_MIN * 0.5
        out = self._run_pooled([ambiguous])
        self.assertTrue(is_ambiguous_peak(out[0].peak_ratio))

    def test_pooled_path_preserves_unmeasured_peak_ratio(self) -> None:
        out = self._run_pooled([None])
        self.assertIsNone(out[0].peak_ratio)


class TestTryToImproveAlignmentsPeakRatio(unittest.TestCase):
    """The finalized-record rebuild must carry `peak_ratio` across re-evaluation."""

    def _run(self, refined_weight: float) -> EnhancedAlignmentRecord:
        key = (0, 0)
        original = _enhanced_record(key, weight=5.0, peak_ratio=4.0)
        refined = _enhanced_record(key, weight=refined_weight, peak_ratio=1.01)
        settings = SimpleNamespace(max_travel_for_finalization=100.0)

        with mock.patch.object(ldc, '_RefinePointsForTwoImages',
                               return_value=[refined]):
            output, _improved = ldc.TryToImproveAlignments(
                transform=None,  # type: ignore[arg-type]
                alignment_records={key: original},
                settings=settings)  # type: ignore[arg-type]
        return output[key]

    def test_keeps_refined_peak_ratio_when_refinement_wins(self) -> None:
        self.assertEqual(self._run(refined_weight=9.0).peak_ratio, 1.01)

    def test_keeps_original_peak_ratio_when_refinement_loses(self) -> None:
        self.assertEqual(self._run(refined_weight=1.0).peak_ratio, 4.0)


if __name__ == '__main__':
    unittest.main()
