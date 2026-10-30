"""Tests for masked grid control-point spring relaxation."""

from __future__ import annotations

import unittest

import numpy as np
from hypothesis import given, settings as hyp_settings, strategies as st

import nornir_imageregistration
from nornir_imageregistration.grid_subdivision import (
    cell_unmasked_fractions,
    classify_fixed_grid_points,
)
from nornir_imageregistration.grid_layout_relax import (
    create_grid_spacing_layout,
    four_adjacent_indices,
    propagate_masked_grid_positions,
    set_local_similarity_rest_offsets,
)
from nornir_imageregistration.layout import Layout


class TestCellUnmaskedFractions(unittest.TestCase):
    @hyp_settings(max_examples=100, deadline=None)
    @given(
        height=st.integers(min_value=1, max_value=64),
        width=st.integers(min_value=1, max_value=64),
        cell_h=st.integers(min_value=1, max_value=32),
        cell_w=st.integers(min_value=1, max_value=32),
        center_y=st.floats(min_value=-32, max_value=96, allow_nan=False, allow_infinity=False),
        center_x=st.floats(min_value=-32, max_value=96, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_direct_counts_match_crop_image(
            self,
            height: int,
            width: int,
            cell_h: int,
            cell_w: int,
            center_y: float,
            center_x: float,
            seed: int,
    ) -> None:
        mask = np.random.default_rng(seed).random((height, width)) >= 0.5
        point = np.array([[center_y, center_x]], dtype=np.float64)
        actual = cell_unmasked_fractions(mask, point, (cell_h, cell_w))[0]
        origin = point[0] - (np.asarray((cell_h, cell_w), dtype=np.float64) / 2.0)
        cropped = nornir_imageregistration.CropImage(
            mask,
            int(origin[1]),
            int(origin[0]),
            cell_w,
            cell_h,
            cval=False,
        )
        assert cropped is not None
        expected = float(np.count_nonzero(cropped)) / float(cell_h * cell_w)
        self.assertEqual(actual, expected)

    def test_full_and_empty_cells(self) -> None:
        mask = np.zeros((64, 64), dtype=bool)
        mask[16:48, 16:48] = True
        points = np.array([
            [32.0, 32.0],  # fully inside tissue
            [0.0, 0.0],   # outside / empty
        ], dtype=np.float64)
        fractions = cell_unmasked_fractions(mask, points, cell_size=(16, 16))
        self.assertAlmostEqual(fractions[0], 1.0)
        self.assertAlmostEqual(fractions[1], 0.0)

    def test_partial_cell(self) -> None:
        mask = np.zeros((32, 32), dtype=bool)
        mask[:, :16] = True
        points = np.array([[16.0, 16.0]], dtype=np.float64)
        fractions = cell_unmasked_fractions(mask, points, cell_size=(16, 16))
        self.assertGreater(fractions[0], 0.4)
        self.assertLess(fractions[0], 0.6)

    def test_classify_requires_both_sides(self) -> None:
        source = np.array([[10.0, 10.0], [10.0, 30.0]], dtype=np.float64)
        target = source.copy()
        source_mask = np.ones((40, 40), dtype=bool)
        target_mask = np.ones((40, 40), dtype=bool)
        target_mask[:, 20:] = False
        fixed = classify_fixed_grid_points(
            source, target, cell_size=(8, 8), min_unmasked=0.25,
            source_mask=source_mask, target_mask=target_mask)
        self.assertTrue(fixed[0])
        self.assertFalse(fixed[1])

    def test_missing_mask_counts_as_unmasked(self) -> None:
        points = np.array([[5.0, 5.0]], dtype=np.float64)
        fixed = classify_fixed_grid_points(
            points, points, cell_size=(4, 4), min_unmasked=0.25,
            source_mask=None, target_mask=None)
        self.assertTrue(fixed[0])

    def test_cached_source_ok_skips_source_crop(self) -> None:
        from nornir_imageregistration.grid_subdivision import source_ok_for_grid_points

        source = np.array([[10.0, 10.0], [10.0, 30.0]], dtype=np.float64)
        target = source.copy()
        source_mask = np.ones((40, 40), dtype=bool)
        source_ok = source_ok_for_grid_points(source, (8, 8), 0.25, source_mask)
        # Target: first ok, second masked
        target_mask = np.ones((40, 40), dtype=bool)
        target_mask[:, 20:] = False
        fixed = classify_fixed_grid_points(
            source, target, cell_size=(8, 8), min_unmasked=0.25,
            source_mask=None,  # would be all-true without source_ok
            target_mask=target_mask,
            source_ok=source_ok)
        self.assertTrue(fixed[0])
        self.assertFalse(fixed[1])
        # source_ok alone can pin both if target is fully open
        fixed_open = classify_fixed_grid_points(
            source, target, cell_size=(8, 8), min_unmasked=0.25,
            source_ok=np.array([True, False]), target_mask=None)
        self.assertTrue(fixed_open[0])
        self.assertFalse(fixed_open[1])


class TestGridLayoutRelax(unittest.TestCase):
    def _identity_lattice(self, rows: int, cols: int, spacing: float = 10.0) -> np.ndarray:
        points = np.zeros((rows * cols, 2), dtype=np.float64)
        for r in range(rows):
            for c in range(cols):
                points[r * cols + c] = (r * spacing, c * spacing)
        return points

    def test_four_adjacent_corners(self) -> None:
        self.assertEqual(sorted(four_adjacent_indices(0, 3, 3)), [1, 3])
        self.assertEqual(sorted(four_adjacent_indices(4, 3, 3)), [1, 3, 5, 7])

    def test_fixed_nodes_do_not_move(self) -> None:
        rows, cols = 3, 3
        spacing = 10.0
        points = self._identity_lattice(rows, cols, spacing)
        fixed = np.ones(rows * cols, dtype=bool)
        fixed[4] = False  # center free
        # Move a fixed neighbor (seed index 1 = top-middle) rightward
        points[1, 1] += 5.0
        updated = propagate_masked_grid_positions(
            points, fixed, seed_indices=[1],
            grid_dims=(rows, cols), grid_spacing=(spacing, spacing))
        for i in range(rows * cols):
            if fixed[i]:
                np.testing.assert_allclose(updated[i], points[i])
        # Free center should have moved toward resolving tension with seed 1
        self.assertFalse(np.allclose(updated[4], points[4]))

    def test_visit_once_chain(self) -> None:
        rows, cols = 1, 4
        spacing = 10.0
        points = self._identity_lattice(rows, cols, spacing)
        # indices 0 fixed, 1-3 free; move seed 0
        fixed = np.array([True, False, False, False])
        points[0, 1] += 8.0
        updated = propagate_masked_grid_positions(
            points, fixed, seed_indices=[0],
            grid_dims=(rows, cols), grid_spacing=(spacing, spacing))
        # Every free node should have been pulled at least once along the chain
        self.assertGreater(updated[1, 1], points[1, 1])
        self.assertGreater(updated[2, 1], points[2, 1])
        self.assertGreater(updated[3, 1], points[3, 1])
        # Seed (fixed) stays put
        np.testing.assert_allclose(updated[0], points[0])

    def test_no_fixed_anchors_leaves_seed_unchanged(self) -> None:
        rows, cols = 2, 2
        spacing = 10.0
        points = self._identity_lattice(rows, cols, spacing)
        fixed = np.zeros(rows * cols, dtype=bool)
        original = points.copy()
        # Seed a free node; neighbors still relax once from that seed
        points[0, 1] += 5.0
        updated = propagate_masked_grid_positions(
            points, fixed, seed_indices=[0],
            grid_dims=(rows, cols), grid_spacing=(spacing, spacing))
        # Seed stays at its pre-wave position (seeds are not re-relaxed)
        np.testing.assert_allclose(updated[0], points[0])
        # Neighbors of the seed do move
        self.assertFalse(np.allclose(updated[1], original[1]))

    def test_progress_every_n_and_cancel_keeps_partial(self) -> None:
        rows, cols = 1, 12
        spacing = 10.0
        points = self._identity_lattice(rows, cols, spacing)
        fixed = np.zeros(rows * cols, dtype=bool)
        fixed[0] = True
        points[0, 1] += 5.0
        seen: list[int] = []

        def on_progress(partial: np.ndarray) -> None:
            seen.append(int(np.count_nonzero(~np.isclose(partial, points))))

        cancel_after = {"n": 0}

        def should_cancel() -> bool:
            cancel_after["n"] += 1
            return cancel_after["n"] > 5

        updated = propagate_masked_grid_positions(
            points, fixed, seed_indices=[0],
            grid_dims=(rows, cols), grid_spacing=(spacing, spacing),
            should_cancel=should_cancel,
            on_progress=on_progress,
            progress_every=2,
        )
        self.assertGreaterEqual(len(seen), 1)
        # Cancelled mid-wave: not all free nodes necessarily moved, but some did.
        self.assertFalse(np.allclose(updated, points))
        # Progress callbacks used stride 2; cancel after 5 loop iterations ⇒ few posts.
        self.assertLessEqual(len(seen), 6)

        rows, cols = 2, 2
        points = self._identity_lattice(rows, cols)
        fixed = np.ones(rows * cols, dtype=bool)
        points[0, 0] += 3.0
        updated = propagate_masked_grid_positions(
            points, fixed, seed_indices=[0],
            grid_dims=(rows, cols), grid_spacing=(10.0, 10.0))
        np.testing.assert_allclose(updated, points)

    def test_rotated_rest_offsets_from_fixed_pairs(self) -> None:
        rows, cols = 3, 3
        spacing = 10.0
        # Identity lattice then rotate 90° about origin: (y,x) -> (x, -y)
        base = self._identity_lattice(rows, cols, spacing)
        angle = np.pi / 2.0
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        # Apply rotation in (Y,X) as column vector
        points = (rot @ base.T).T
        fixed = np.ones(rows * cols, dtype=bool)
        # Leave rightmost column free
        for r in range(rows):
            fixed[r * cols + (cols - 1)] = False

        layout = create_grid_spacing_layout(points, (rows, cols), (spacing, spacing))
        set_local_similarity_rest_offsets(
            layout, points, fixed, (rows, cols), (spacing, spacing))

        # A free right-edge spring from col1->col2 should match a fixed right spring
        # from the same row (col0->col1), not axis-aligned spacing.
        i_fixed = 0 * cols + 0
        j_fixed = 0 * cols + 1
        expected = points[j_fixed] - points[i_fixed]
        i_free = 0 * cols + 1
        j_free = 0 * cols + 2
        got = layout.nodes[i_free].GetOffset(j_free)
        np.testing.assert_allclose(got, expected, atol=1e-6)
        # Must not be the axis-aligned fallback
        self.assertFalse(np.allclose(got, np.array([0.0, spacing])))

    @hyp_settings(deadline=None, max_examples=25)
    @given(angle_deg=st.floats(min_value=-45.0, max_value=45.0, allow_nan=False, allow_infinity=False))
    def test_rest_vector_magnitude_matches_spacing_under_rotation(self, angle_deg: float) -> None:
        rows, cols = 3, 3
        spacing = 12.0
        base = self._identity_lattice(rows, cols, spacing)
        angle = np.deg2rad(angle_deg)
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        points = (rot @ base.T).T
        fixed = np.array([True, True, False,
                          True, True, False,
                          True, True, False], dtype=bool)
        layout = create_grid_spacing_layout(points, (rows, cols), (spacing, spacing))
        set_local_similarity_rest_offsets(
            layout, points, fixed, (rows, cols), (spacing, spacing))
        rest = layout.nodes[1].GetOffset(2)
        self.assertAlmostEqual(float(np.linalg.norm(rest)), spacing, places=5)


class TestLayoutRelaxNode(unittest.TestCase):
    def test_relax_node_and_fixed_ids(self) -> None:
        layout = Layout()
        layout.CreateNode(0, np.array([0.0, 0.0]))
        layout.CreateNode(1, np.array([0.0, 12.0]))
        layout.SetOffset(0, 1, offset=np.array([0.0, 10.0]), weight=1.0)
        before = layout.GetPosition(0).copy()
        Layout.RelaxNodes(layout, fixed_ids={0})
        np.testing.assert_allclose(layout.GetPosition(0), before)
        self.assertFalse(np.allclose(layout.GetPosition(1), [0.0, 12.0]))

        layout2 = Layout()
        layout2.CreateNode(0, np.array([0.0, 0.0]))
        layout2.CreateNode(1, np.array([0.0, 12.0]))
        layout2.SetOffset(0, 1, offset=np.array([0.0, 10.0]), weight=1.0)
        Layout.RelaxNode(layout2, 1)
        np.testing.assert_allclose(layout2.GetPosition(1), [0.0, 10.0])


if __name__ == '__main__':
    unittest.main()
