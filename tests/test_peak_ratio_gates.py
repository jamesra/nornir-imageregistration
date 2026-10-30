"""Unit tests for refine_shared.peak_ratio_gates host-side helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from nornir_imageregistration.refine_shared.peak_ratio_gates import (
    PEAK_RATIO_EARLY,
    PEAK_RATIO_MIN,
    ambiguous_record_ids,
    exclude_ambiguous_mesh_records,
    finite_peak_ratio,
    is_ambiguous_peak,
    is_early_lock_ratio,
    soft_discontinuity_ids,
)


def _record(
        key: tuple[int, int],
        *,
        peak_ratio: object = 1.4,
        peak: tuple[float, float] | None = (0.1, 0.0),
) -> SimpleNamespace:
    return SimpleNamespace(ID=key, peak_ratio=peak_ratio, peak=peak)


class TestFinitePeakRatio(unittest.TestCase):
    """Stored peak_ratio parsing on alignment records."""

    def test_missing_attribute_returns_none(self) -> None:
        self.assertIsNone(finite_peak_ratio(SimpleNamespace()))

    def test_none_value_returns_none(self) -> None:
        self.assertIsNone(finite_peak_ratio(_record((0, 0), peak_ratio=None)))

    def test_non_numeric_returns_none(self) -> None:
        self.assertIsNone(finite_peak_ratio(_record((0, 0), peak_ratio="not-a-float")))

    def test_non_finite_returns_none(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assertIsNone(finite_peak_ratio(_record((0, 0), peak_ratio=value)))

    def test_valid_float_returned(self) -> None:
        self.assertEqual(finite_peak_ratio(_record((0, 0), peak_ratio=1.25)), 1.25)
        self.assertEqual(finite_peak_ratio(_record((0, 0), peak_ratio="1.5")), 1.5)


class TestAmbiguousAndEarlyLock(unittest.TestCase):
    """Hard-reject floor and early-lock bar."""

    def test_ambiguous_when_none_or_below_min(self) -> None:
        self.assertTrue(is_ambiguous_peak(None))
        self.assertTrue(is_ambiguous_peak(PEAK_RATIO_MIN - 1e-9))
        self.assertFalse(is_ambiguous_peak(PEAK_RATIO_MIN))
        self.assertFalse(is_ambiguous_peak(PEAK_RATIO_MIN + 0.01))

    def test_early_lock_only_when_finite_at_bar(self) -> None:
        self.assertFalse(is_early_lock_ratio(None))
        self.assertFalse(is_early_lock_ratio(PEAK_RATIO_EARLY - 1e-9))
        self.assertTrue(is_early_lock_ratio(PEAK_RATIO_EARLY))
        self.assertTrue(is_early_lock_ratio(PEAK_RATIO_EARLY + 1.0))


class TestAmbiguousRecordIds(unittest.TestCase):
    """Grid IDs flagged for missing or low peak_ratio."""

    def test_collects_ambiguous_ids(self) -> None:
        records = [
            _record((0, 0), peak_ratio=1.6),
            _record((0, 1), peak_ratio=1.05),
            _record((1, 0), peak_ratio=None),
        ]
        self.assertEqual(
            ambiguous_record_ids(records),
            {(0, 1), (1, 0)},
        )


class TestSoftDiscontinuityIds(unittest.TestCase):
    """Discontinuity cells eligible for relaxed travel."""

    def test_only_finite_at_or_above_min_in_disc_set(self) -> None:
        records = [
            _record((0, 0), peak_ratio=2.0),
            _record((0, 1), peak_ratio=1.05),
            _record((1, 0), peak_ratio=PEAK_RATIO_MIN),
            _record((1, 1), peak_ratio=None),
        ]
        disc = {(0, 1), (1, 0), (1, 1), (2, 2)}
        self.assertEqual(soft_discontinuity_ids(records, disc), {(1, 0)})

    def test_empty_discontinuity_returns_empty(self) -> None:
        records = [_record((0, 0), peak_ratio=2.0)]
        self.assertEqual(soft_discontinuity_ids(records, set()), set())


class TestExcludeAmbiguousMeshRecordsBoundaries(unittest.TestCase):
    """Mesh filter keeps emergency fill travel ordering."""

    def test_empty_input(self) -> None:
        kept, dropped = exclude_ambiguous_mesh_records([], min_keep=3)
        self.assertEqual(kept, [])
        self.assertEqual(dropped, 0)

    def test_missing_peak_uses_infinite_travel_in_emergency(self) -> None:
        records = [
            _record((0, 0), peak_ratio=1.05, peak=(1.0, 0.0)),
            _record((0, 1), peak_ratio=1.05, peak=(2.0, 0.0)),
            _record((1, 0), peak_ratio=1.05, peak=None),
        ]
        kept, dropped = exclude_ambiguous_mesh_records(records, min_keep=2)
        self.assertEqual(len(kept), 2)
        self.assertEqual({r.ID for r in kept}, {(0, 0), (0, 1)})
        self.assertEqual(dropped, 1)


if __name__ == "__main__":
    unittest.main()
