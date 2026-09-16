"""Tests for trusted-mesh prior projection and scheduling."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import local_distortion_correction as ldc
from nornir_imageregistration.refine_shared.measure_schedule import (
    cells_whose_prior_moved,
    project_priors,
    update_last_prior,
)
from nornir_imageregistration.transforms.rigid import RigidTranslation


class _CountingTransform:
    """Affine stand-in that records Transform batch sizes."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def Transform(self, points: np.ndarray) -> np.ndarray:
        array = np.asarray(points, dtype=np.float64)
        self.batch_sizes.append(int(array.shape[0]))
        return array + np.asarray((2.0, -3.0))


def test_project_priors_vectorizes_and_preserves_ids() -> None:
    """Unlocked prior projection uses one transform call with scalar-path parity."""
    transform = _CountingTransform()
    ids = [(0, 0), (0, 1), (1, 0)]
    source = {
        (0, 0): np.asarray((1.0, 2.0)),
        (0, 1): np.asarray((3.0, 4.0)),
        (1, 0): np.asarray((5.0, 6.0)),
    }
    projected = project_priors(
        transform,
        ids,
        source,
        locked_ids={(0, 1)},
    )
    assert transform.batch_sizes == [2]
    assert list(projected) == [(0, 0), (1, 0)]
    np.testing.assert_allclose(projected[(0, 0)], [3.0, -1.0])
    np.testing.assert_allclose(projected[(1, 0)], [7.0, 3.0])


def test_vectorized_projection_drives_existing_schedule() -> None:
    """Batched priors produce the same movement decisions as scalar values."""
    transform = _CountingTransform()
    ids = [(0, 0), (0, 1)]
    source = {
        (0, 0): np.asarray((1.0, 2.0)),
        (0, 1): np.asarray((3.0, 4.0)),
    }
    priors = project_priors(transform, ids, source)
    last = {
        (0, 0): np.asarray((3.0, -1.0)),
        (0, 1): np.asarray((4.0, 1.0)),
    }
    assert cells_whose_prior_moved(ids, priors, last, eps=0.5, pass_index=2) == [(0, 1)]


class TestMeasureSchedule(unittest.TestCase):
    def test_pass_zero_returns_all_unlocked(self) -> None:
        ids = [(0, 0), (0, 1), (1, 0)]
        todo = cells_whose_prior_moved(
            ids, {}, {}, pass_index=0, locked_ids={(0, 1)})
        self.assertEqual(todo, [(0, 0), (1, 0)])

    def test_later_pass_only_moved_priors(self) -> None:
        ids = [(0, 0), (0, 1), (1, 0)]
        last = {
            (0, 0): np.array([10.0, 10.0]),
            (0, 1): np.array([10.0, 20.0]),
            (1, 0): np.array([20.0, 10.0]),
        }
        priors = {
            (0, 0): np.array([10.1, 10.0]),  # within eps
            (0, 1): np.array([12.0, 20.0]),  # moved
            (1, 0): np.array([20.0, 10.0]),  # unchanged
        }
        todo = cells_whose_prior_moved(
            ids, priors, last, eps=0.5, pass_index=2, locked_ids=set())
        self.assertEqual(todo, [(0, 1)])

    def test_update_last_prior_copies(self) -> None:
        last: dict[tuple[int, int], np.ndarray] = {}
        priors = {(0, 0): np.array([1.0, 2.0])}
        update_last_prior(last, [(0, 0)], priors)
        priors[(0, 0)][0] = 99.0
        self.assertEqual(float(last[(0, 0)][0]), 1.0)

    def test_grid_subset_matches_same_cells_from_full_measurement(self) -> None:
        image = np.ones((128, 128), dtype=np.float32)
        stats = nornir_imageregistration.ImageStats.Create(image)
        settings = nornir_imageregistration.settings.GridRefinement(
            source_image=image,
            target_image=image,
            source_image_stats=stats,
            target_image_stats=stats,
            source_mask=np.ones(image.shape, dtype=bool),
            target_mask=np.ones(image.shape, dtype=bool),
            cell_size=np.array((32, 32), dtype=np.int32),
            grid_spacing=np.array((32, 32), dtype=np.int32),
            min_unmasked_area=0.5,
            single_thread_processing=True,
        )
        transform = RigidTranslation(
            target_offset=np.array((3.0, -2.0), dtype=np.float64))
        calls: list[tuple[list[tuple[int, int]], np.ndarray, np.ndarray]] = []

        def capture(
                _transform: object,
                keys: list[tuple[int, int]],
                source_points: np.ndarray,
                target_points: np.ndarray,
                _settings: object,
                **_kwargs: object,
        ) -> list:
            calls.append((list(keys), source_points.copy(), target_points.copy()))
            return []

        with mock.patch.object(ldc, '_RefinePointsForTwoImages', side_effect=capture):
            ldc._RefineGridPointsForTwoImages(
                transform, finalized={}, settings=settings)
            full_keys, full_source, full_target = calls[-1]
            selected = {full_keys[1], full_keys[-2]}
            ldc._RefineGridPointsForTwoImages(
                transform,
                finalized={},
                settings=settings,
                measure_cell_ids=selected,
            )

        subset_keys, subset_source, subset_target = calls[-1]
        expected_indices = [i for i, key in enumerate(full_keys) if key in selected]
        self.assertEqual(subset_keys, [full_keys[i] for i in expected_indices])
        np.testing.assert_array_equal(
            subset_source, full_source[expected_indices])
        np.testing.assert_array_equal(
            subset_target, full_target[expected_indices])


if __name__ == '__main__':
    unittest.main()
