"""A failed alignment must not leave a spring in the layout (#122).

Both failure handlers in ``_FindTileOffsets`` called ``layout.RemoveOverlap`` and *also* built a
stage-position ``AlignmentRecord`` with weight 0.  The trailing ``if offset is not None`` block
then ran and called ``SetOffset``, re-adding the spring that had just been removed, so the
removal was dead and a zero-weight spring reached relaxation.

The pool is stubbed so a task can fail on demand; the alternative is an image pair that provokes
a genuine FloatingPointError out of phase correlation, which would test the correlation code
rather than this error handling.
"""

import unittest

import numpy as np

import nornir_imageregistration
import nornir_pools
from nornir_imageregistration import arrange_mosaic
from nornir_imageregistration.layout import Layout, LayoutPosition


class _Task:
    def __init__(self, result):
        self._result = result

    def wait_return(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _Pool:
    """Runs nothing; hands back whatever the test queued, in submission order."""

    def __init__(self, results):
        self._results = list(results)
        self._next = 0

    def add_task(self, name, func, *args, **kwargs):
        task = _Task(self._results[self._next])
        self._next += 1
        return task

    def wait_completion(self):
        return


class _Rect:
    def __init__(self, center):
        self.Center = np.asarray(center, dtype=np.float64)
        self.Area = 1.0


class _Tile:
    def __init__(self, ID, center):
        self.ID = ID
        self.ImagePath = f"tile_{ID}.png"
        self.FixedBoundingBox = _Rect(center)


class _Overlap(nornir_imageregistration.tile_overlap.TileOverlap):
    """A real TileOverlap with its fields set directly.

    Subclassed rather than duck-typed because Layout._parameter_to_offset_IDs dispatches on
    isinstance; the base constructor would require tile images this test has no use for. Setting
    the private fields keeps every property the code under test reads on the real implementation.
    """

    def __init__(self, A, B, offset, feature_scores=(1.0, 1.0)):
        self._Tiles = (A, B)
        self._offset = np.asarray(offset, dtype=np.float64)
        self._imageScale = 1.0
        self._scaled_overlapping_source_rects = (None, None)
        self._overlapping_source_rects = (None, None)
        self._overlapping_target_rect = None
        self._feature_scores = (1.0, 1.0)
        self._normalized_feature_scores = feature_scores
        self._overlap = None


def _run(results, overlaps, use_feature_score=False):
    """Drive _FindTileOffsets with a stubbed pool and return the resulting layout."""
    pool = _Pool(results)
    saved = (nornir_pools.GetGlobalSerialPool, nornir_pools.GetGlobalMultithreadingPool)
    nornir_pools.GetGlobalSerialPool = lambda: pool
    nornir_pools.GetGlobalMultithreadingPool = lambda: pool
    try:
        return arrange_mosaic._FindTileOffsets(list(overlaps),
                                               excess_scalar=1.0,
                                               image_to_source_space_scale=1.0,
                                               use_feature_score=use_feature_score)
    finally:
        (nornir_pools.GetGlobalSerialPool,
         nornir_pools.GetGlobalMultithreadingPool) = saved


def _three_tile_chain():
    """Tiles 0-1-2 in a row, so one overlap can fail while another succeeds."""
    tiles = [_Tile(0, (0, 0)), _Tile(1, (0, 100)), _Tile(2, (0, 200))]
    return tiles, [_Overlap(tiles[0], tiles[1], (0, 100)),
                   _Overlap(tiles[1], tiles[2], (0, 100))]


def _record(peak, weight):
    return nornir_imageregistration.AlignmentRecord(peak=np.asarray(peak, dtype=np.float64),
                                                    weight=weight)


class TestASuccessfulAlignment(unittest.TestCase):
    """The working path must be untouched."""

    def test_a_spring_is_created(self):
        tiles, overlaps = _three_tile_chain()
        layout = _run([_record((0, 100), 0.8), _record((0, 100), 0.6)], overlaps)
        self.assertTrue(layout.ContainsOffset((0, 1)))
        self.assertTrue(layout.ContainsOffset((1, 2)))

    def test_the_weight_is_carried_through(self):
        tiles, overlaps = _three_tile_chain()
        layout = _run([_record((0, 100), 0.8), _record((0, 100), 0.6)], overlaps)
        weights = layout.nodes[0].OffsetArray[:, LayoutPosition.iOffsetWeight]
        self.assertAlmostEqual(0.8, float(weights[0]))


class TestAFailedAlignment(unittest.TestCase):
    """The reported bug: the removal was undone by a re-add."""

    def _failed_then_ok(self, error):
        tiles, overlaps = _three_tile_chain()
        return _run([error, _record((0, 100), 0.6)], overlaps)

    def test_a_floating_point_error_leaves_no_spring(self):
        layout = self._failed_then_ok(FloatingPointError("overlap is one colour"))
        self.assertFalse(layout.ContainsOffset((0, 1)))

    def test_a_value_error_leaves_no_spring(self):
        layout = self._failed_then_ok(ValueError("could not find overlap"))
        self.assertFalse(layout.ContainsOffset((0, 1)))

    def test_the_surviving_overlap_is_unaffected(self):
        """Only the failed pair loses its spring."""
        for error in (FloatingPointError("boom"), ValueError("boom")):
            with self.subTest(error=type(error).__name__):
                layout = self._failed_then_ok(error)
                self.assertTrue(layout.ContainsOffset((1, 2)))

    def test_no_zero_weight_spring_reaches_the_layout(self):
        """A zero weight is the signature of the re-added spring."""
        for error in (FloatingPointError("boom"), ValueError("boom")):
            with self.subTest(error=type(error).__name__):
                layout = self._failed_then_ok(error)
                for ID, node in layout.nodes.items():
                    if node.IsIsolated:
                        continue
                    weights = np.asarray(node.OffsetArray[:, LayoutPosition.iOffsetWeight])
                    self.assertFalse(np.any(weights == 0),
                                     f"node {ID} carries a zero-weight offset: {weights}")

    def test_the_nodes_still_exist(self):
        """Only the spring is dropped; the tiles remain placeable at stage position."""
        layout = self._failed_then_ok(ValueError("boom"))
        for ID in (0, 1, 2):
            with self.subTest(node=ID):
                self.assertTrue(layout.Contains(ID))

    def test_every_overlap_failing_leaves_no_springs(self):
        tiles, overlaps = _three_tile_chain()
        layout = _run([ValueError("boom"), FloatingPointError("boom")], overlaps)
        self.assertFalse(layout.ContainsOffset((0, 1)))
        self.assertFalse(layout.ContainsOffset((1, 2)))


class TestWhyAZeroWeightSpringMatters(unittest.TestCase):
    """Justifies treating this as a bug rather than a cosmetic leftover."""

    def test_normalize_offset_weights_is_skewed_by_a_zero(self):
        """One zero drags the weight minimum down and rescales every other weight.

        Weights 0.5 and 0.9 alone rescale to 0.0 and 1.0. Adding a third offset at weight 0
        moves the minimum to 0, so the 0.5 offset now lands mid-range instead of at the floor --
        a real offset's pull in the relaxation changed because an unrelated overlap failed.
        """
        without = self._normalized_weight_of_the_weaker_offset(include_zero=False)
        with_zero = self._normalized_weight_of_the_weaker_offset(include_zero=True)
        self.assertAlmostEqual(0.0, without)
        self.assertNotAlmostEqual(without, with_zero,
                                  msg="a zero-weight offset should change the rescaling")

    def _normalized_weight_of_the_weaker_offset(self, include_zero: bool) -> float:
        layout = Layout()
        for ID, position in ((0, (0, 0)), (1, (0, 100)), (2, (0, 200))):
            layout.CreateNode(ID, np.asarray(position, dtype=np.float64))
        layout.SetOffset(0, 1, np.asarray((0.0, 100.0)), 0.5)
        layout.SetOffset(1, 2, np.asarray((0.0, 100.0)), 0.9)
        if include_zero:
            layout.SetOffset(0, 2, np.asarray((0.0, 200.0)), 0.0)
        nornir_imageregistration.layout.NormalizeOffsetWeights(layout)
        node = layout.nodes[0]
        iRow = np.flatnonzero(node.ConnectedIDs == 1)[0]
        return float(node.OffsetArray[iRow, LayoutPosition.iOffsetWeight])

    def test_component_building_does_not_skip_a_zero_weight(self):
        """Only NaN is skipped, so a zero-weight spring still welds two components."""
        layout = Layout()
        for ID, position in ((0, (0, 0)), (1, (0, 100))):
            layout.CreateNode(ID, np.asarray(position, dtype=np.float64))
        layout.SetOffset(0, 1, np.asarray((0.0, 100.0)), 0.0)
        components = nornir_imageregistration.layout.BuildLayoutWithHighestWeightsFirst(layout)
        self.assertEqual(1, len(components))
        self.assertTrue(components[0].ContainsOffset((0, 1)))


class TestTheGuardIsReachable(unittest.TestCase):
    """The 'if offset is not None' block exists only to be skipped by these handlers."""

    def test_both_handlers_can_leave_offset_none(self):
        import inspect
        source = inspect.getsource(arrange_mosaic._FindTileOffsets)
        self.assertIn('offset = None', source)
        self.assertIn('if offset is not None:', source)

    def test_no_handler_builds_a_zero_weight_alignment_record(self):
        import inspect
        source = inspect.getsource(arrange_mosaic._FindTileOffsets)
        self.assertNotIn('AlignmentRecord(peak=t.tile_overlap.scaled_offset, weight=0)', source)


if __name__ == '__main__':
    unittest.main()
