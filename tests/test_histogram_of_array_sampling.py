"""Tests for HistogramOfArray sampling under NumPy and CuPy."""

from __future__ import annotations

import numpy as np
import pytest

from nornir_imageregistration.image_stats import (
    ApproximateHistogramOfArray,
    HistogramOfArray,
    even_histogram_stride,
)


def test_histogram_of_array_even_num_samples() -> None:
    """num_samples uses even ravel stride and yields a populated histogram."""
    image = np.linspace(0, 255, 10_000, dtype=np.uint8).reshape(100, 100)
    hist = HistogramOfArray(
        image, bpp=8, num_bins=256, min_val=0, max_val=255, num_samples=1_000)
    assert hist.NumSamples > 0
    assert sum(hist.Bins) == hist.NumSamples
    # Even stride ~10 keeps about 1000 of 10000 pixels.
    assert 900 <= hist.NumSamples <= 1100


def test_histogram_of_array_cupy_with_num_samples() -> None:
    """CuPy inputs with num_samples no longer fail via device index into NumPy."""
    cupy = pytest.importorskip('cupy')
    image = (cupy.arange(10_000, dtype=cupy.uint16).reshape(100, 100) % 256).astype(cupy.uint8)
    hist = HistogramOfArray(
        image, bpp=8, num_bins=256, min_val=0, max_val=255, num_samples=2_000)
    assert hist.NumSamples > 0
    assert sum(hist.Bins) == hist.NumSamples


def test_even_histogram_stride_targets_one_to_five_percent() -> None:
    stride = even_histogram_stride(0.02)
    fraction = 1.0 / float(stride * stride)
    assert 0.01 <= fraction <= 0.05


def test_approximate_histogram_of_array_non_empty() -> None:
    image = np.linspace(0, 255, 10_000, dtype=np.uint8).reshape(100, 100)
    hist = ApproximateHistogramOfArray(image, sample_fraction=0.02)
    assert hist is not None
    assert hist.NumSamples > 0
    assert hist.NumBins == 256
    assert sum(hist.Bins) > 0


def test_approximate_histogram_of_array_empty() -> None:
    assert ApproximateHistogramOfArray(np.zeros((0, 0), dtype=np.uint8)) is None


def test_approximate_histogram_of_array_cupy() -> None:
    cupy = pytest.importorskip('cupy')
    image = cupy.arange(256, dtype=cupy.uint8).reshape(16, 16)
    hist = ApproximateHistogramOfArray(image, sample_fraction=0.05)
    assert hist is not None
    assert hist.NumSamples > 0
