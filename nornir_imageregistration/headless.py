"""
Headless / agent-friendly plotting: save figures to PNG instead of opening UI windows.

Detection (any condition is enough):
- ``NORNIR_HEADLESS`` is ``1``, ``true``, ``yes``, or ``on`` (case-insensitive).
- Non-Windows host with no ``DISPLAY`` (typical Linux CI / Docker agents).

PNG artifacts from ``save_figure_to_png_artifact`` go under
``<test_output_root>/_plot_artifacts`` when ``TEST_OUTPUT_DIR`` or
``TESTOUTPUTPATH`` is set (same root as other imageregistration test outputs);
otherwise the system temp directory.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid

__all__ = [
    "is_headless",
    "artifact_png_path",
    "inspect_png_output",
    "save_figure_to_png_artifact",
    "save_current_pyplot_figure",
]


def is_headless() -> bool:
    flag = os.environ.get("NORNIR_HEADLESS", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if sys.platform != "win32" and not os.environ.get("DISPLAY"):
        return True
    return False


def _headless_plot_artifact_base() -> str:
    """Root directory for headless matplotlib PNGs (``_plot_artifacts`` under test output)."""
    root = (
        os.environ.get("TEST_OUTPUT_DIR", "").strip()
        or os.environ.get("TESTOUTPUTPATH", "").strip()
    )
    if root:
        return os.path.join(root, "_plot_artifacts")
    return tempfile.gettempdir()


def artifact_png_path(prefix: str = "nornir-ir") -> str:
    base = _headless_plot_artifact_base()
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{prefix}-{os.getpid()}-{uuid.uuid4().hex}.png")


def inspect_png_output(path: str) -> None:
    """Raise AssertionError if ``path`` is not a readable, non-empty PNG."""
    from PIL import Image

    if not os.path.isfile(path):
        raise AssertionError(f"headless PNG artifact missing: {path}")
    if os.path.getsize(path) < 32:
        raise AssertionError(f"headless PNG artifact too small: {path}")

    with Image.open(path) as im:
        im.load()
        w, h = im.size
    if w < 1 or h < 1:
        raise AssertionError(f"headless PNG has invalid dimensions: {path}")


def save_figure_to_png_artifact(fig, *, tag: str = "fig", dpi: int = 150) -> str:
    """Save ``fig`` to a unique PNG path, close it, and validate the file."""
    import matplotlib.pyplot as plt

    path = artifact_png_path(tag)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    inspect_png_output(path)
    return path


def save_current_pyplot_figure(*, tag: str = "plt", dpi: int = 150) -> str:
    """Save the current pyplot figure, close it, and validate the PNG."""
    import matplotlib.pyplot as plt

    return save_figure_to_png_artifact(plt.gcf(), tag=tag, dpi=dpi)
