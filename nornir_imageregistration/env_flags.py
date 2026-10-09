"""Normalize boolean environment-flag tokens used across imageregistration.

Stdlib only so callers can resolve flags without importing CuPy, matplotlib,
or the package ``__init__`` that selects a plotting backend.
"""

from __future__ import annotations

import os

__all__ = [
    "FALSEY_TOKENS",
    "TRUTHY_TOKENS",
    "env_flag",
    "env_is_falsey",
    "env_is_truthy",
    "is_falsey",
    "is_truthy",
]

TRUTHY_TOKENS = frozenset({"1", "true", "yes", "on"})
FALSEY_TOKENS = frozenset({"0", "false", "no", "off"})


def env_flag(name: str, default: str = "") -> str:
    """Return the env var stripped and lowercased, or *default* when unset."""
    return os.environ.get(name, default).strip().lower()


def is_truthy(flag: str) -> bool:
    """Return True when *flag* is a known truthy token (already normalized)."""
    return flag in TRUTHY_TOKENS


def is_falsey(flag: str) -> bool:
    """Return True when *flag* is a known falsey token (already normalized)."""
    return flag in FALSEY_TOKENS


def env_is_truthy(name: str, default: str = "") -> bool:
    """Return True when the named env var's normalized value is truthy."""
    return is_truthy(env_flag(name, default))


def env_is_falsey(name: str, default: str = "") -> bool:
    """Return True when the named env var's normalized value is falsey."""
    return is_falsey(env_flag(name, default))
