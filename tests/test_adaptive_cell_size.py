"""Grow STOS refine cell size when a pass finds no usable alignments."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st

from nornir_imageregistration.local_distortion_correction import (
    _grow_refine_cell_size_after_failure,
    _restore_refine_cell_size_after_success,
)
from nornir_imageregistration.refine_shared.adaptive_cell_size import (
    MIN_REGISTRATIONS_TO_RESTORE_CELL_SIZE,
    can_grow_cell_size_on_pass,
    cell_size_cap_from_shapes,
    cell_size_exceeds_requested,
    next_cell_size_after_failure,
    pass_found_no_usable_alignments,
    pass_found_registrations,
)
from nornir_imageregistration.refine_shared.cell_roles import (
    FieldMode,
    RoleClassificationResult,
)
from nornir_imageregistration.refine_shared.source_content_cache import SourceContentCache


def _role_counts(
        *,
        n_free: int = 0,
        n_lockable: int = 0,
        n_identity_suspect: int = 0,
        n_reject: int = 8,
) -> RoleClassificationResult:
    return RoleClassificationResult(
        roles=[],
        reject_reasons=[],
        lock_candidate=np.zeros(0, dtype=bool),
        zncc=np.zeros(0, dtype=np.float64),
        field_mode=FieldMode.LOCAL,
        n_reject=n_reject,
        n_free=n_free,
        n_lockable=n_lockable,
        n_identity_suspect=n_identity_suspect,
    )


class TestPassFoundNoUsableAlignments(unittest.TestCase):
    def test_empty_measure_is_failure(self) -> None:
        self.assertTrue(pass_found_no_usable_alignments(None, n_measured=0))

    def test_all_reject_is_failure(self) -> None:
        self.assertTrue(
            pass_found_no_usable_alignments(_role_counts(n_reject=12), n_measured=12))

    def test_free_cell_is_usable(self) -> None:
        self.assertFalse(
            pass_found_no_usable_alignments(
                _role_counts(n_free=1, n_reject=11), n_measured=12))

    def test_identity_suspect_is_usable(self) -> None:
        self.assertFalse(
            pass_found_no_usable_alignments(
                _role_counts(n_identity_suspect=2, n_reject=10), n_measured=12))

    def test_lockable_is_usable(self) -> None:
        self.assertFalse(
            pass_found_no_usable_alignments(
                _role_counts(n_lockable=1, n_reject=11), n_measured=12))


class TestPassFoundRegistrations(unittest.TestCase):
    def test_empty_is_not_registration(self) -> None:
        self.assertFalse(pass_found_registrations(None, n_measured=0))

    def test_all_reject_is_not_registration(self) -> None:
        self.assertFalse(
            pass_found_registrations(_role_counts(n_reject=12), n_measured=12))

    def test_identity_suspect_alone_does_not_restore(self) -> None:
        self.assertFalse(
            pass_found_registrations(
                _role_counts(n_identity_suspect=4, n_reject=8), n_measured=12))

    def test_one_free_is_not_enough(self) -> None:
        self.assertFalse(
            pass_found_registrations(
                _role_counts(n_free=1, n_reject=11), n_measured=12))

    def test_min_free_cells_restore(self) -> None:
        self.assertTrue(
            pass_found_registrations(
                _role_counts(
                    n_free=MIN_REGISTRATIONS_TO_RESTORE_CELL_SIZE, n_reject=9),
                n_measured=12))

    def test_one_lockable_is_enough(self) -> None:
        self.assertTrue(
            pass_found_registrations(
                _role_counts(n_lockable=1, n_reject=11), n_measured=12))


class TestCellSizeExceedsRequested(unittest.TestCase):
    def test_equal_is_not_above(self) -> None:
        self.assertFalse(cell_size_exceeds_requested((128, 128), (128, 128)))

    def test_any_axis_above(self) -> None:
        self.assertTrue(cell_size_exceeds_requested((256, 128), (128, 128)))
        self.assertTrue(cell_size_exceeds_requested((128, 256), (128, 128)))

    @given(
        cy=st.integers(min_value=1, max_value=512),
        cx=st.integers(min_value=1, max_value=512),
        ry=st.integers(min_value=1, max_value=512),
        rx=st.integers(min_value=1, max_value=512),
    )
    @example(cy=256, cx=256, ry=128, rx=128)
    @example(cy=128, cx=128, ry=128, rx=128)
    @settings(max_examples=60, deadline=None)
    def test_matches_per_axis_greater(
            self, cy: int, cx: int, ry: int, rx: int) -> None:
        self.assertEqual(
            cell_size_exceeds_requested((cy, cx), (ry, rx)),
            (cy > ry) or (cx > rx))


class TestNextCellSizeAfterFailure(unittest.TestCase):
    def test_doubles_when_under_cap(self) -> None:
        grown = next_cell_size_after_failure((32, 32), (256, 256))
        assert grown is not None
        np.testing.assert_array_equal(grown, np.asarray((64, 64)))

    def test_clamps_to_cap(self) -> None:
        grown = next_cell_size_after_failure((128, 128), (200, 180))
        assert grown is not None
        np.testing.assert_array_equal(grown, np.asarray((200, 180)))

    def test_none_when_already_at_cap(self) -> None:
        self.assertIsNone(next_cell_size_after_failure((200, 180), (200, 180)))

    def test_grows_only_axes_with_room(self) -> None:
        grown = next_cell_size_after_failure((100, 200), (150, 200))
        assert grown is not None
        np.testing.assert_array_equal(grown, np.asarray((150, 200)))

    def test_does_not_shrink_oversize_axis(self) -> None:
        self.assertIsNone(next_cell_size_after_failure((300, 300), (200, 200)))

    def test_cap_from_shapes_uses_min_axes(self) -> None:
        cap = cell_size_cap_from_shapes((400, 300), (350, 500))
        np.testing.assert_array_equal(cap, np.asarray((350, 300)))

    def test_cap_from_shapes_clamps_to_1024_on_large_images(self) -> None:
        """A 4096² section must not grow ROIs to the full shared image."""
        from nornir_imageregistration.refine_shared.adaptive_cell_size import MAX_REFINE_CELL_SIZE

        cap = cell_size_cap_from_shapes((4096, 4096), (4100, 4080))
        np.testing.assert_array_equal(
            cap, np.asarray((MAX_REFINE_CELL_SIZE, MAX_REFINE_CELL_SIZE)))

    def test_doubling_stops_at_1024_on_4096_image(self) -> None:
        cap = cell_size_cap_from_shapes((4096, 4096), (4096, 4096))
        self.assertIsNone(next_cell_size_after_failure((1024, 1024), cap))
        grown = next_cell_size_after_failure((512, 512), cap)
        assert grown is not None
        np.testing.assert_array_equal(grown, np.asarray((1024, 1024)))

    def test_clamp_cell_size_to_cap(self) -> None:
        from nornir_imageregistration.refine_shared.adaptive_cell_size import clamp_cell_size_to_cap

        np.testing.assert_array_equal(
            clamp_cell_size_to_cap((4096, 4096), (1024, 1024)),
            np.asarray((1024, 1024)))
        np.testing.assert_array_equal(
            clamp_cell_size_to_cap((256, 256), (1024, 1024)),
            np.asarray((256, 256)))

    @given(
        y=st.integers(min_value=1, max_value=512),
        x=st.integers(min_value=1, max_value=512),
        cap_y=st.integers(min_value=1, max_value=1024),
        cap_x=st.integers(min_value=1, max_value=1024),
    )
    @example(y=32, x=32, cap_y=256, cap_x=256)
    @example(y=128, x=128, cap_y=200, cap_x=180)
    @example(y=200, x=180, cap_y=200, cap_x=180)
    @settings(max_examples=80, deadline=None)
    def test_grown_size_never_shrinks_and_respects_cap(
            self, y: int, x: int, cap_y: int, cap_x: int) -> None:
        current = np.asarray((y, x), dtype=np.int64)
        cap = np.asarray((cap_y, cap_x), dtype=np.int64)
        grown = next_cell_size_after_failure(current, cap)
        if grown is None:
            self.assertFalse(np.any(cap > current))
            return
        self.assertTrue(bool(np.all(grown >= current)))
        self.assertTrue(bool(np.all(grown <= np.maximum(current, cap))))
        self.assertTrue(bool(np.any(grown > current)))


class TestCanGrowOnPass(unittest.TestCase):
    def test_blocks_final_pass(self) -> None:
        self.assertFalse(
            can_grow_cell_size_on_pass(
                pass_index=3, num_iterations=5, final_pass=True))

    def test_blocks_last_planned_index(self) -> None:
        self.assertFalse(
            can_grow_cell_size_on_pass(
                pass_index=5, num_iterations=5, final_pass=False))

    def test_allows_earlier_pass(self) -> None:
        self.assertTrue(
            can_grow_cell_size_on_pass(
                pass_index=1, num_iterations=5, final_pass=False))


class TestGrowRefineCellSizeHelper(unittest.TestCase):
    def test_doubles_and_clears_source_cache(self) -> None:
        settings = SimpleNamespace(
            cell_size=np.asarray((32, 32), dtype=np.int64),
            num_iterations=4,
            source_image=np.zeros((256, 256)),
            target_image=np.zeros((256, 256)),
        )
        cache = SourceContentCache(min_std=1.0)
        cache.remember((0, 0), 0.1)
        self.assertTrue(cache.is_low_content((0, 0)))
        self.assertTrue(
            _grow_refine_cell_size_after_failure(
                settings, cache, pass_index=1, final_pass=False))  # type: ignore[arg-type]
        np.testing.assert_array_equal(settings.cell_size, np.asarray((64, 64)))
        self.assertFalse(cache.is_low_content((0, 0)))
        self.assertEqual(cache.as_mapping(), {})

    def test_false_when_at_cap(self) -> None:
        settings = SimpleNamespace(
            cell_size=np.asarray((128, 128), dtype=np.int64),
            num_iterations=4,
            source_image=np.zeros((128, 128)),
            target_image=np.zeros((128, 128)),
        )
        cache = SourceContentCache()
        self.assertFalse(
            _grow_refine_cell_size_after_failure(
                settings, cache, pass_index=1, final_pass=False))  # type: ignore[arg-type]
        np.testing.assert_array_equal(settings.cell_size, np.asarray((128, 128)))

    def test_false_when_at_1024_cap_on_large_image(self) -> None:
        settings = SimpleNamespace(
            cell_size=np.asarray((1024, 1024), dtype=np.int64),
            num_iterations=4,
            source_image=SimpleNamespace(shape=(4096, 4096)),
            target_image=SimpleNamespace(shape=(4096, 4096)),
        )
        cache = SourceContentCache()
        self.assertFalse(
            _grow_refine_cell_size_after_failure(
                settings, cache, pass_index=1, final_pass=False))  # type: ignore[arg-type]
        np.testing.assert_array_equal(settings.cell_size, np.asarray((1024, 1024)))

    def test_false_on_final_pass(self) -> None:
        settings = SimpleNamespace(
            cell_size=np.asarray((32, 32), dtype=np.int64),
            num_iterations=4,
            source_image=np.zeros((256, 256)),
            target_image=np.zeros((256, 256)),
        )
        cache = SourceContentCache()
        self.assertFalse(
            _grow_refine_cell_size_after_failure(
                settings, cache, pass_index=1, final_pass=True))  # type: ignore[arg-type]


class TestRestoreRefineCellSizeHelper(unittest.TestCase):
    def test_restores_requested_and_clears_cache(self) -> None:
        requested = np.asarray((32, 32), dtype=np.int64)
        settings = SimpleNamespace(
            cell_size=np.asarray((128, 128), dtype=np.int64),
            num_iterations=4,
        )
        cache = SourceContentCache(min_std=1.0)
        cache.remember((0, 0), 0.1)
        self.assertTrue(
            _restore_refine_cell_size_after_success(
                settings, cache, requested,  # type: ignore[arg-type]
                pass_index=2, final_pass=False))
        np.testing.assert_array_equal(settings.cell_size, requested)
        self.assertFalse(cache.is_low_content((0, 0)))
        self.assertEqual(cache.as_mapping(), {})

    def test_false_when_already_at_requested(self) -> None:
        requested = np.asarray((32, 32), dtype=np.int64)
        settings = SimpleNamespace(
            cell_size=requested.copy(),
            num_iterations=4,
        )
        cache = SourceContentCache()
        self.assertFalse(
            _restore_refine_cell_size_after_success(
                settings, cache, requested,  # type: ignore[arg-type]
                pass_index=2, final_pass=False))

    def test_false_on_final_pass(self) -> None:
        requested = np.asarray((32, 32), dtype=np.int64)
        settings = SimpleNamespace(
            cell_size=np.asarray((128, 128), dtype=np.int64),
            num_iterations=4,
        )
        cache = SourceContentCache()
        self.assertFalse(
            _restore_refine_cell_size_after_success(
                settings, cache, requested,  # type: ignore[arg-type]
                pass_index=2, final_pass=True))
        np.testing.assert_array_equal(settings.cell_size, np.asarray((128, 128)))


class TestSourceContentCacheClear(unittest.TestCase):
    def test_clear_drops_sticky_low_content(self) -> None:
        cache = SourceContentCache(min_std=1.0)
        cache.remember((1, 2), 0.2)
        cache.clear()
        self.assertFalse(cache.is_low_content((1, 2)))
        self.assertIsNone(cache.get((1, 2)))


if __name__ == '__main__':
    unittest.main()
