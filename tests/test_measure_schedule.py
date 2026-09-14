"""Unit tests for trusted-mesh measure scheduling."""

from __future__ import annotations

import unittest

import numpy as np

from nornir_imageregistration.refine_shared.measure_schedule import (
    cells_whose_prior_moved,
    update_last_prior,
)


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


if __name__ == '__main__':
    unittest.main()
