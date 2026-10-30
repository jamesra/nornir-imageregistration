"""Regression for #117 / C05-B011: interpolator OOM must not become silent NaNs."""
from __future__ import annotations

import types
import unittest

import numpy as np

from nornir_imageregistration.transforms import gridtransform as gt


class _BoomInterpolator:
    def __init__(self, error: BaseException):
        self._error = error

    def __call__(self, points):
        raise self._error


def _make_transform(*, scipy: bool, error: BaseException):
    return types.SimpleNamespace(
        __class__=type('FakeGrid', (), {}),
        _scipy_inverse_interp=scipy,
        InverseInterpolator=_BoomInterpolator(error),
        _InverseInterpolator=_BoomInterpolator(error),
        TargetPoints=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        SourcePoints=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    )


class TestLinearNdFallbackExceptions(unittest.TestCase):
    def test_memory_error_propagates_from_inverse_fallback(self):
        transform = _make_transform(scipy=True, error=MemoryError('oom'))
        with self.assertRaises(MemoryError):
            gt._inverse_transform_with_linear_nd_fallback(
                transform,
                np.array([[0.1, 0.1]], dtype=np.float64),
                scipy_flag_attr='_scipy_inverse_interp',
                interpolator_property='InverseInterpolator',
                output_dtype=np.float64,
            )
        self.assertIsNotNone(transform._InverseInterpolator)

    def test_runtime_error_propagates_from_forward_fallback(self):
        transform = types.SimpleNamespace(
            __class__=type('FakeTri', (), {}),
            _scipy_forward_interp=True,
            ForwardInterpolator=_BoomInterpolator(RuntimeError('dtype boom')),
            _ForwardInterpolator=_BoomInterpolator(RuntimeError('dtype boom')),
            SourcePoints=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            TargetPoints=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        )
        with self.assertRaises(RuntimeError):
            gt._forward_transform_with_linear_nd_fallback(
                transform,
                np.array([[0.1, 0.1]], dtype=np.float64),
                scipy_flag_attr='_scipy_forward_interp',
                interpolator_property='ForwardInterpolator',
                output_dtype=np.float32,
            )

    def test_non_retryable_value_error_still_soft_fails(self):
        transform = _make_transform(scipy=True, error=ValueError('QH6214 qhull input error'))
        out = gt._inverse_transform_with_linear_nd_fallback(
            transform,
            np.array([[0.1, 0.1]], dtype=np.float64),
            scipy_flag_attr='_scipy_inverse_interp',
            interpolator_property='InverseInterpolator',
            output_dtype=np.float64,
        )
        self.assertTrue(np.all(np.isnan(out)))
        self.assertIsNone(transform._InverseInterpolator)


if __name__ == '__main__':
    unittest.main()
