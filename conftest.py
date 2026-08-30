"""
Pytest: enable headless figure output before any test imports nornir_imageregistration.

``nornir_imageregistration`` calls ``matplotlib.use("Agg" if is_headless() else "qtAgg")``
at import time. Set ``NORNIR_HEADLESS`` here so pytest loads this module before
collecting tests, avoiding GUI windows and writing PNG artifacts instead.

This file lives at the package root (not under ``tests/``) so pytest loads it before
collecting modules under ``tests/``.

Override for interactive debugging: ``NORNIR_HEADLESS=0 pytest ...``
"""

from __future__ import annotations

import os

os.environ.setdefault("NORNIR_HEADLESS", "1")

import pytest


@pytest.fixture(autouse=True)
def _deterministic_padding_noise():
    """Give every test the same padding noise, whatever order it runs in.

    ``pad_image_for_phase_correlation`` fills padding with noise so phase
    correlation has no hard edge to lock onto. That draw is now reproducible per
    run, but the generator still advances between calls, so a test's noise would
    otherwise depend on how many tests ran before it. Reseeding here makes a test
    behave the same alone as in a suite -- the rotating failures in
    ``test_SliceToSliceBrute`` were this.
    """
    import nornir_imageregistration

    nornir_imageregistration.seed_random_data()
    yield
