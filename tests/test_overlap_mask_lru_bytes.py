"""Regression for #196: overlap-mask LRU insert stays O(1) in entry count."""
from __future__ import annotations

import unittest
from collections import OrderedDict
from unittest.mock import patch

import numpy as np

import nornir_imageregistration.overlapmasking as om


class TestOverlapMaskLruRunningBytes(unittest.TestCase):
    def setUp(self) -> None:
        om.clear_overlap_mask_caches()

    def tearDown(self) -> None:
        om.clear_overlap_mask_caches()

    def test_insert_loop_is_linear_in_nbytes_calls(self) -> None:
        """#196: each put must not re-sum every cached entry."""
        cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        running = [0]
        # Large budget so nothing evicts; old code still summed all entries each put.
        budget = 10 * 1024 * 1024
        n_inserts = 80
        tile = np.ones((32, 32), dtype=bool)

        with patch.object(om, '_array_nbytes', wraps=om._array_nbytes) as counted:
            for i in range(n_inserts):
                om._lru_put(cache, (i,), tile.copy(), budget, running_bytes=running)

        # New entry + (optional) replace accounting: O(1) per insert, not O(n) sum.
        self.assertLessEqual(counted.call_count, n_inserts * 2)
        self.assertEqual(len(cache), n_inserts)
        self.assertEqual(running[0], n_inserts * int(tile.nbytes))

    def test_stats_match_running_counter(self) -> None:
        mask = om.GetOverlapMask((64, 64), (64, 64), (128, 128), 0.1, 0.9)
        self.assertIsNotNone(mask)
        stats = om.overlap_mask_cache_stats()
        self.assertEqual(stats['host_entries'], 1)
        self.assertEqual(stats['host_bytes'], int(mask.nbytes))


if __name__ == '__main__':
    unittest.main()
