"""Coverage-mask skipping in ImageToTilesGenerator -- review #232.

``GenerateOptimizedTiles`` hands the assembled mask to ``ImageToTilesGenerator`` as
``coverage_mask`` so that grid cells with no tile over them are not emitted at all. The
IDOC tests used to assert that every cell of the grid arrived, which asserted against that
design: the IDOC mosaic is rotated inside its axis-aligned bounding box, so its four grid
corners are outside the section and assemble to zero covered pixels.

Those tests need the large IDOC dataset and ~14s each. These pin the same contract on
arrays in milliseconds, so a regression in the skip rule is caught without the dataset.
"""

import numpy as np
import pytest

import nornir_imageregistration


TILE = np.asarray((4, 4), dtype=np.int64)
GRID = np.asarray((3, 3), dtype=np.int64)


def _source() -> np.ndarray:
    """A 12x12 image whose every pixel is non-zero, so only the mask can cause a skip."""
    return np.arange(1, (12 * 12) + 1, dtype=np.float32).reshape(12, 12)


def _cells(**kwargs) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for (r, c, _img) in
            nornir_imageregistration.ImageToTilesGenerator(source_image=_source(),
                                                           tile_size=TILE,
                                                           grid_shape=GRID,
                                                           **kwargs)}


ALL_CELLS = {(r, c) for r in range(3) for c in range(3)}


class TestWithoutAMaskEveryCellArrives:
    def test_all_nine_cells_are_yielded(self):
        assert _cells() == ALL_CELLS

    def test_an_all_true_mask_is_the_same_as_no_mask(self):
        assert _cells(coverage_mask=np.ones((12, 12), dtype=bool)) == ALL_CELLS

    def test_the_yield_order_is_row_then_column(self):
        got = [(int(r), int(c)) for (r, c, _i) in
               nornir_imageregistration.ImageToTilesGenerator(
                   source_image=_source(), tile_size=TILE, grid_shape=GRID)]
        assert got == [(r, c) for r in range(3) for c in range(3)]


class TestAMaskSkipsExactlyTheUncoveredCells:
    @staticmethod
    def _mask_covering(cells: set[tuple[int, int]]) -> np.ndarray:
        mask = np.zeros((12, 12), dtype=bool)
        for (r, c) in cells:
            mask[r * 4:(r + 1) * 4, c * 4:(c + 1) * 4] = True
        return mask

    def test_the_four_corners_can_be_skipped(self):
        """The IDOC failure shape: a rotated section leaves the grid corners uncovered."""
        corners = {(0, 0), (0, 2), (2, 0), (2, 2)}
        covered = ALL_CELLS - corners
        assert _cells(coverage_mask=self._mask_covering(covered)) == covered

    def test_an_all_false_mask_yields_nothing(self):
        assert _cells(coverage_mask=np.zeros((12, 12), dtype=bool)) == set()

    @pytest.mark.parametrize('kept', [{(1, 1)}, {(0, 0)}, {(2, 2)},
                                      {(0, 1), (1, 0)}, ALL_CELLS - {(1, 1)}])
    def test_only_the_covered_cells_arrive(self, kept):
        assert _cells(coverage_mask=self._mask_covering(kept)) == kept

    def test_the_image_content_of_a_kept_cell_is_unaffected_by_the_mask(self):
        """Masking decides *whether* a tile is emitted, never what is in it."""
        source = _source()
        unmasked = {(int(r), int(c)): img.copy() for (r, c, img) in
                    nornir_imageregistration.ImageToTilesGenerator(
                        source_image=source, tile_size=TILE, grid_shape=GRID)}
        kept = {(1, 1), (0, 2)}
        for (r, c, img) in nornir_imageregistration.ImageToTilesGenerator(
                source_image=source, tile_size=TILE, grid_shape=GRID,
                coverage_mask=self._mask_covering(kept)):
            np.testing.assert_array_equal(unmasked[(int(r), int(c))], img)


class TestCoverageIsAnyPixelNotMostPixels:
    """A cell clipped by the section edge must still be emitted, however little it holds."""

    @pytest.mark.parametrize('n_true', [1, 2, 7, 15])
    def test_a_partly_covered_cell_is_still_yielded(self, n_true):
        mask = np.zeros((12, 12), dtype=bool)
        flat = mask[4:8, 4:8].reshape(-1)
        flat[:n_true] = True
        mask[4:8, 4:8] = flat.reshape(4, 4)
        assert _cells(coverage_mask=mask) == {(1, 1)}

    def test_zero_true_pixels_is_the_only_thing_that_skips(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[5, 5] = True
        assert _cells(coverage_mask=mask) == {(1, 1)}
        mask[5, 5] = False
        assert _cells(coverage_mask=mask) == set()


class TestTheOffsetAppliesToSkippedGridsToo:
    def test_coord_offset_shifts_the_reported_cell(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[4:8, 4:8] = True
        assert _cells(coverage_mask=mask, coord_offset=np.array([10, 20])) == {(11, 21)}

    def test_the_offset_does_not_move_which_cells_are_skipped(self):
        """The mask is indexed in image space, so an offset renames rather than reselects."""
        mask = np.zeros((12, 12), dtype=bool)
        mask[0:4, 8:12] = True
        offset = np.array([5, 5])
        assert _cells(coverage_mask=mask) == {(0, 2)}
        assert _cells(coverage_mask=mask, coord_offset=offset) == {(5, 7)}


class TestAMaskSmallerThanTheGridIsPaddedAsUncovered:
    def test_a_mask_short_of_the_required_shape_is_zero_padded(self):
        """Padding with False, not True, keeps a short mask from inventing coverage."""
        mask = np.ones((6, 6), dtype=bool)
        got = _cells(coverage_mask=mask)
        assert got == {(0, 0), (0, 1), (1, 0), (1, 1)}
