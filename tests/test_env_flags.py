"""Unit tests for the shared env-flag truthy/falsey helpers."""

from __future__ import annotations

import os
import unittest

from hypothesis import given
from hypothesis import strategies as st

from nornir_imageregistration.env_flags import (
    BASIC_FALSEY_TOKENS,
    BASIC_TRUTHY_TOKENS,
    FALSEY_TOKENS,
    TRUTHY_TOKENS,
    env_flag,
    env_is_truthy,
    is_basic_falsey,
    is_basic_truthy,
    is_falsey,
    is_truthy,
)


class TestEnvFlags(unittest.TestCase):
    def test_refine_truthy_tokens(self) -> None:
        for token in ("1", "true", "yes", "on", "TRUE", " Yes ", "ON"):
            with self.subTest(token=token):
                self.assertTrue(is_truthy(token.strip().lower()))

    def test_refine_falsey_tokens(self) -> None:
        for token in ("0", "false", "no", "off", "FALSE", " No ", "OFF"):
            with self.subTest(token=token):
                self.assertTrue(is_falsey(token.strip().lower()))

    def test_basic_tokens_exclude_on_off(self) -> None:
        self.assertFalse(is_basic_truthy("on"))
        self.assertFalse(is_basic_falsey("off"))
        self.assertTrue(is_basic_truthy("yes"))
        self.assertTrue(is_basic_falsey("no"))

    def test_empty_and_unknown_are_neither(self) -> None:
        for token in ("", "maybe", "2", "enabled"):
            with self.subTest(token=token):
                self.assertFalse(is_truthy(token))
                self.assertFalse(is_falsey(token))
                self.assertFalse(is_basic_truthy(token))
                self.assertFalse(is_basic_falsey(token))

    def test_env_flag_strips_and_lowercases(self) -> None:
        key = "NORNIR_TEST_ENV_FLAG_NORMALIZE"
        old = os.environ.get(key)
        try:
            os.environ[key] = "  YeS  "
            self.assertEqual(env_flag(key), "yes")
            os.environ.pop(key, None)
            self.assertEqual(env_flag(key, "On"), "on")
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

    def test_env_is_truthy(self) -> None:
        key = "NORNIR_TEST_ENV_FLAG_BOOL"
        old = os.environ.get(key)
        try:
            os.environ[key] = "on"
            self.assertTrue(env_is_truthy(key))
            os.environ[key] = "off"
            self.assertFalse(env_is_truthy(key))
            os.environ.pop(key, None)
            self.assertFalse(env_is_truthy(key))
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

    @given(st.sampled_from(sorted(TRUTHY_TOKENS)))
    def test_truthy_tokens_property(self, token: str) -> None:
        self.assertTrue(is_truthy(token))
        self.assertFalse(is_falsey(token))

    @given(st.sampled_from(sorted(FALSEY_TOKENS)))
    def test_falsey_tokens_property(self, token: str) -> None:
        self.assertTrue(is_falsey(token))
        self.assertFalse(is_truthy(token))

    @given(st.sampled_from(sorted(BASIC_TRUTHY_TOKENS)))
    def test_basic_truthy_tokens_property(self, token: str) -> None:
        self.assertTrue(is_basic_truthy(token))
        self.assertTrue(is_truthy(token))

    @given(st.sampled_from(sorted(BASIC_FALSEY_TOKENS)))
    def test_basic_falsey_tokens_property(self, token: str) -> None:
        self.assertTrue(is_basic_falsey(token))
        self.assertTrue(is_falsey(token))

    @given(
        st.text(
            alphabet=st.characters(blacklist_categories=("Cs",)),
            max_size=16,
        ).filter(lambda s: s.strip().lower() not in TRUTHY_TOKENS | FALSEY_TOKENS)
    )
    def test_unknown_tokens_are_neither(self, token: str) -> None:
        normalized = token.strip().lower()
        self.assertFalse(is_truthy(normalized))
        self.assertFalse(is_falsey(normalized))
        self.assertFalse(is_basic_truthy(normalized))
        self.assertFalse(is_basic_falsey(normalized))


if __name__ == "__main__":
    unittest.main()
