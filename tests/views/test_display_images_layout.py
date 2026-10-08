"""Unit tests for grayscale display grid/title layout helpers."""

import unittest

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from nornir_imageregistration.views.display_images import (
    ShowGrayscale,
    _GridLayoutDims,
    _TitleLayoutDims,
)


class TestDisplayImagesLayoutDims(unittest.TestCase):
    def test_grid_layout_dims_none_and_single_array(self) -> None:
        self.assertEqual(_GridLayoutDims(None), (0, 0))
        image = np.zeros((4, 4), dtype=np.float32)
        self.assertEqual(_GridLayoutDims(image), (1, 1))

    def test_grid_layout_dims_nested_ragged_rows(self) -> None:
        a = np.zeros((2, 2))
        b = np.zeros((2, 2))
        c = np.zeros((2, 2))
        self.assertEqual(_GridLayoutDims([a, b, c]), (3, 1))
        self.assertEqual(_GridLayoutDims([[a, b], [c]]), (2, 2))
        self.assertEqual(_GridLayoutDims([[a], [b, c]]), (2, 2))

    def test_title_layout_dims_string_and_nested(self) -> None:
        self.assertEqual(_TitleLayoutDims("single"), (1, 1))
        self.assertEqual(_TitleLayoutDims(["A", "B", "C"]), (3, 1))
        self.assertEqual(_TitleLayoutDims([["A", "B"], ["C"]]), (2, 2))
        self.assertEqual(_TitleLayoutDims([["A"], ["BC"]]), (2, 1))

    def test_show_grayscale_rejects_mismatched_title_layout(self) -> None:
        image = np.zeros((8, 8))
        with self.assertRaises(ValueError) as ctx:
            ShowGrayscale(
                [[image, image], [image]],  # type: ignore[arg-type]
                title="mismatch",
                image_titles=["A", "B", "C"],
                PassFail=True,
            )
        self.assertIn("layout of image titles must match", str(ctx.exception))


class TestDisplayImagesLayoutHypothesis(unittest.TestCase):
    @given(
        col_counts=st.lists(
            st.integers(min_value=1, max_value=4),
            min_size=1,
            max_size=6,
        ),
    )
    @settings(max_examples=40, deadline=None)
    def test_matching_nested_grid_and_title_dims(self, col_counts: list[int]) -> None:
        rows: list[list[np.ndarray]] = []
        title_rows: list[list[str]] = []
        for width in col_counts:
            rows.append([np.zeros((2, 2)) for _ in range(width)])
            title_rows.append([f"t{i}" for i in range(width)])
        self.assertEqual(_GridLayoutDims(rows), _TitleLayoutDims(title_rows))
        self.assertEqual(_GridLayoutDims(rows)[1], max(col_counts))

    @given(n=st.integers(min_value=1, max_value=12))
    @settings(max_examples=30, deadline=None)
    def test_flat_list_grid_matches_title_list(self, n: int) -> None:
        images = [np.zeros((2, 2)) for _ in range(n)]
        titles = [f"img{i}" for i in range(n)]
        self.assertEqual(_GridLayoutDims(images), (n, 1))
        self.assertEqual(_TitleLayoutDims(titles), (n, 1))


if __name__ == "__main__":
    unittest.main()
