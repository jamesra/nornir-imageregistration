"""
Pytest: enable headless figure output before any test imports nornir_imageregistration.

``nornir_imageregistration`` calls ``matplotlib.use("Agg" if is_headless() else "qtAgg")``
at import time. Set ``NORNIR_HEADLESS`` here so pytest loads this module before
collecting tests, avoiding GUI windows and writing PNG artifacts instead.

This file lives at the package root (not under ``test/``) so it is not imported as
``test.conftest``, which would register the stdlib-style package name ``test`` and
break sibling projects (e.g. nornir-buildmanager) that also use a ``test`` package.

Override for interactive debugging: ``NORNIR_HEADLESS=0 pytest ...``
"""

from __future__ import annotations

import os

os.environ.setdefault("NORNIR_HEADLESS", "1")
