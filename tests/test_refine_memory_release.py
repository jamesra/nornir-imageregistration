"""Unit tests for refinement memory release policy (#186)."""

from __future__ import annotations

import inspect
import unittest
from unittest import mock

from nornir_imageregistration import local_distortion_correction as ldc


class TestReleaseRefinementWorkerMemory(unittest.TestCase):
    """Between-pass release must not forfeit allocator reuse."""

    def test_reclaim_caches_false_skips_gc_and_pool_free(self) -> None:
        with mock.patch.object(ldc.gc, "collect") as collect:
            ldc._release_refinement_worker_memory(reclaim_caches=False)
            collect.assert_not_called()
            ldc._release_refinement_worker_memory(reclaim_caches=True)
            collect.assert_called()

    def test_refine_tileset_pass_loop_skips_cache_reclaim(self) -> None:
        source = inspect.getsource(ldc._refine_tileset)
        self.assertIn(
            "_release_refinement_worker_memory(reclaim_caches=False)",
            source,
        )
        # End-of-refine still reclaims aggressively (default True).
        self.assertGreaterEqual(source.count("_release_refinement_worker_memory("), 2)


if __name__ == "__main__":
    unittest.main()
