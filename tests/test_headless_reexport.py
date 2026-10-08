"""``nornir_imageregistration.headless.is_headless`` is the shared rule, not a copy of it."""

from __future__ import annotations

import nornir_shared.headless

import nornir_imageregistration.headless


def test_is_headless_is_the_nornir_shared_function() -> None:
    assert nornir_imageregistration.headless.is_headless is nornir_shared.headless.is_headless
    assert "is_headless" in nornir_imageregistration.headless.__all__
