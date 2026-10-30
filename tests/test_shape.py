"""Shape is a NumPy (y, x) pair."""

import numpy as np

from nornir_imageregistration.type_info import Shape


def test_shape_indexes_y_then_x() -> None:
    tile = Shape.from_xy(x=16, y=8)
    assert tile == Shape(y=8, x=16)
    assert tile[0] == 8
    assert tile[1] == 16
    assert tile[-1] == 16
    assert tuple(tile) == (8, 16)
    grid = np.asarray((2, 3), dtype=np.int64)
    assert np.array_equal(grid * np.asarray(tile), np.asarray((16, 48)))
