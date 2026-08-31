"""CenteredGridDivision must handle a grid axis that holds a single cell (#251).

The overage adjustment ramps a correction across each axis by cell position.  With one cell on
an axis there is no spread to ramp across, and the ramp used to evaluate 0/0 -- raising
FloatingPointError under this package's numpy error state.  These tests pin the fixed behaviour
and the convention it follows: coordinate 0 is never adjusted.
"""

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import grid_subdivision


CELL = np.asarray((64, 64), dtype=np.int32)


def _grid(shape, cell_size=CELL):
    return nornir_imageregistration.CenteredGridDivision(np.asarray(shape, np.int64),
                                                         cell_size=cell_size)


class TestThePremise(unittest.TestCase):
    """The conditions that made this a crash rather than a quiet NaN."""

    def test_the_package_raises_on_invalid_arithmetic(self):
        """0/0 is an error here, not a NaN, so the bug was a hard failure."""
        err = np.geterr()
        self.assertEqual('raise', err['invalid'])

    def test_a_single_cell_axis_has_a_zero_coordinate_max(self):
        """This is the divisor that was zero."""
        for shape, expected_zero_axes in (((64, 64), [0, 1]),
                                          ((64, 256), [0]),
                                          ((256, 64), [1]),
                                          ((256, 256), [])):
            with self.subTest(shape=shape):
                dims = nornir_imageregistration.TileGridShape(np.asarray(shape, np.int64), CELL)
                coords = grid_subdivision.build_coords_array(dims)
                coord_max = np.max(coords, 0)
                zero_axes = [i for i, v in enumerate(coord_max) if v == 0]
                self.assertEqual(expected_zero_axes, zero_axes)


class TestTheSingleCellCasesConstruct(unittest.TestCase):
    """Every shape that used to raise now builds a grid."""

    def test_an_image_of_exactly_one_cell(self):
        """The shape reported in #251."""
        g = _grid((64, 64))
        self.assertEqual(1, g.SourcePoints.shape[0])
        np.testing.assert_array_equal(np.asarray((1, 1)), np.asarray(g.grid_dims))

    def test_an_image_smaller_than_one_cell(self):
        """Here the overage is non-zero, so the adjustment actually has something to apply."""
        g = _grid((30, 30))
        self.assertEqual(1, g.SourcePoints.shape[0])

    def test_a_long_thin_image_one_cell_wide(self):
        """Only one axis is degenerate; the other must still be ramped normally."""
        for shape in ((64, 256), (256, 64), (40, 256), (256, 40)):
            with self.subTest(shape=shape):
                g = _grid(shape)
                self.assertEqual(4, g.SourcePoints.shape[0])

    def test_no_source_point_is_nan(self):
        """The failure mode was an invalid value, so guard against it returning as a NaN."""
        for shape in ((64, 64), (30, 30), (64, 256), (256, 64), (40, 256), (256, 40)):
            with self.subTest(shape=shape):
                points = np.asarray(_grid(shape).SourcePoints)
                self.assertTrue(np.all(np.isfinite(points)),
                                f"non-finite source points for {shape}: {points}")


class TestTheAdjustmentConvention(unittest.TestCase):
    """A lone point is coordinate 0, and coordinate 0 is never adjusted."""

    def test_a_lone_point_sits_at_the_cell_centre(self):
        """No adjustment means the point stays where the un-adjusted grid put it."""
        for shape in ((64, 64), (30, 30)):
            with self.subTest(shape=shape):
                np.testing.assert_array_almost_equal(np.asarray((32.0, 32.0)),
                                                     np.asarray(_grid(shape).SourcePoints[0]))

    def test_coordinate_zero_is_unadjusted_when_the_axis_has_spread(self):
        """The convention a lone point is being made consistent with."""
        for shape in ((96, 96), (65, 65), (256, 256)):
            with self.subTest(shape=shape):
                np.testing.assert_array_almost_equal(np.asarray((32.0, 32.0)),
                                                     np.asarray(_grid(shape).SourcePoints[0]))

    def test_the_degenerate_axis_does_not_disturb_the_healthy_one(self):
        """A one-cell-wide image must ramp its long axis exactly as a square image does."""
        thin = np.asarray(_grid((64, 256)).SourcePoints)
        square = np.asarray(_grid((256, 256)).SourcePoints)
        # Column positions of the thin image's four points, against the first row of the square.
        np.testing.assert_array_almost_equal(square[:4, 1], thin[:, 1])

    def test_the_ramped_axis_is_unchanged_by_the_fix(self):
        """Guard the multi-cell path against regression; these values predate the fix."""
        g = _grid((96, 96))
        np.testing.assert_array_almost_equal(np.asarray((80.0, 80.0)),
                                             np.asarray(g.SourcePoints[-1]))


class TestTheSiblingClass(unittest.TestCase):
    """ITKGridDivision never had the pattern; confirm it is still fine."""

    def test_itk_grid_division_handles_one_cell(self):
        g = nornir_imageregistration.ITKGridDivision(np.asarray((64, 64), np.int64), cell_size=CELL)
        self.assertTrue(np.all(np.isfinite(np.asarray(g.SourcePoints))))


if __name__ == '__main__':
    unittest.main()
