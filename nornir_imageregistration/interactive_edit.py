"""Process-wide interactive edit scope for transform UIs (e.g. Pyre point drag).

While depth > 0, transform models may soft-invalidate derived structures instead of
rebuilding them on every mouse-move.
"""

from __future__ import annotations

_depth = 0


def begin() -> None:
    global _depth
    _depth += 1


def end() -> None:
    global _depth
    if _depth > 0:
        _depth -= 1


def in_progress() -> bool:
    return _depth > 0
