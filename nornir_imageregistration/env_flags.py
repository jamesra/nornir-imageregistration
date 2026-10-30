"""Normalize boolean environment-flag tokens used across imageregistration.

Stdlib only so callers can resolve flags without importing CuPy, matplotlib,
or the package ``__init__`` that selects a plotting backend.

Two token sets exist on purpose:

- ``BASIC_*`` — ``1``/``true``/``yes`` and ``0``/``false``/``no``; used by
  assemble prefetch, grid-extrapolate, and inverse-scipy gates (``on``/``off``
  are unknown there).
- ``TRUTHY_TOKENS`` / ``FALSEY_TOKENS`` — also accept ``on`` / ``off``. Used by
  refine runtime config, ``NORNIR_PROFILE``, ``NORNIR_LOG_GPU_MEM``, and
  ``NORNIR_ASSEMBLE_SERIAL_GPU`` (not every assemble flag is BASIC).
"""

from __future__ import annotations

import os

__all__ = [
    "BASIC_FALSEY_TOKENS",
    "BASIC_TRUTHY_TOKENS",
    "FALSEY_TOKENS",
    "TRUTHY_TOKENS",
    "env_flag",
    "env_is_truthy",
    "is_basic_falsey",
    "is_basic_truthy",
    "is_falsey",
    "is_truthy",
]

BASIC_TRUTHY_TOKENS = frozenset({"1", "true", "yes"})
BASIC_FALSEY_TOKENS = frozenset({"0", "false", "no"})
TRUTHY_TOKENS = BASIC_TRUTHY_TOKENS | {"on"}
FALSEY_TOKENS = BASIC_FALSEY_TOKENS | {"off"}


def env_flag(name: str, default: str = "") -> str:
    """Return the env var stripped and lowercased, or *default* when unset."""
    return os.environ.get(name, default).strip().lower()


def is_truthy(flag: str) -> bool:
    """Return True when *flag* is in ``TRUTHY_TOKENS`` (already normalized)."""
    return flag in TRUTHY_TOKENS


def is_falsey(flag: str) -> bool:
    """Return True when *flag* is in ``FALSEY_TOKENS`` (already normalized)."""
    return flag in FALSEY_TOKENS


def is_basic_truthy(flag: str) -> bool:
    """Return True for ``BASIC_TRUTHY_TOKENS`` (no ``on``)."""
    return flag in BASIC_TRUTHY_TOKENS


def is_basic_falsey(flag: str) -> bool:
    """Return True for ``BASIC_FALSEY_TOKENS`` (no ``off``)."""
    return flag in BASIC_FALSEY_TOKENS


def env_is_truthy(name: str, default: str = "") -> bool:
    """Return True when the named env var's normalized value is in ``TRUTHY_TOKENS``."""
    return is_truthy(env_flag(name, default))
