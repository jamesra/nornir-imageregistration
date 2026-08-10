"""Tests for cooperative registration cancel and progress helpers."""

from __future__ import annotations

import threading
import unittest

import numpy as np

from nornir_imageregistration.image_stats import ImageStats
from nornir_imageregistration.registration_control import (
    RegistrationCancelled,
    check_cancelled,
    report_progress,
)
from nornir_imageregistration.stos_brute import _find_best_angle


class TestRegistrationControl(unittest.TestCase):
    def test_check_cancelled_raises_when_set(self) -> None:
        event = threading.Event()
        event.set()
        with self.assertRaises(RegistrationCancelled):
            check_cancelled(event)

    def test_check_cancelled_noop_when_unset_or_none(self) -> None:
        check_cancelled(None)
        check_cancelled(threading.Event())

    def test_report_progress_invokes_callback(self) -> None:
        seen: list[tuple[int, int, str]] = []
        report_progress(lambda c, t, label: seen.append((c, t, label)), 2, 5, "step")
        self.assertEqual(seen, [(2, 5, "step")])
        report_progress(None, 1, 1, "ignored")


class TestFindBestAngleCancel(unittest.TestCase):
    def test_find_best_angle_aborts_when_cancelled(self) -> None:
        rng = np.random.default_rng(0)
        source = rng.random((32, 32), dtype=np.float32)
        target = rng.random((32, 32), dtype=np.float32)
        source_stats = ImageStats.CalcStats(source)
        target_stats = ImageStats.CalcStats(target)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(RegistrationCancelled):
            _find_best_angle(
                source_image=source,
                target_image=target,
                source_stats=source_stats,
                target_stats=target_stats,
                angle_range=[-2.0, 0.0, 2.0],
                min_overlap=0.5,
                SingleThread=True,
                cancel_event=cancel,
            )


if __name__ == "__main__":
    unittest.main()
