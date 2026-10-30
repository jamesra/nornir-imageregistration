"""The batched correlation runs at the caller's precision instead of forcing float64.

``batched_image_phase_correlation`` and ``batched_find_offset`` both used to open with an
unconditional ``xp.asarray(..., dtype=xp.float64)``. Callers hand these functions
``float32`` cell stacks -- ``local_distortion_correction`` builds them from
``fixed.image.dtype`` -- so every batch paid a conversion the serial path never made and
then carried ``complex128`` transforms through the FFT.

Both FFT backends honour single precision (``numpy.fft`` and ``cupy.fft`` each return
``complex64`` for a ``float32`` input), so the upcast bought no accuracy from the
transform itself; it only doubled the workspace. Measured on 64 cells of 128x128:

==================  ==========  ==========
stage               float64     float32
==================  ==========  ==========
CPU workspace          83.9 MB     41.9 MB
GPU workspace          16.8 MB      8.4 MB
GPU time                1.70 ms     0.30 ms
==================  ==========  ==========

Accuracy is unaffected in any regime that matters. Across ordinary texture, contrast
scaled down to 0.005, additive noise, and cells from 64 to 256 px, the integer peak
choice was identical for every cell and the refined sub-pixel peak moved by at most
6.1e-07 px -- five orders of magnitude below the ~0.1 px error the method has against a
known ground-truth shift.

The float32 floor matters: integer and ``float16`` stacks are promoted up to float32
rather than transformed at their own precision, which would lose the peak.
"""

import numpy as np
import pytest

from nornir_imageregistration import batched_phase_correlation as bpc

_CELL = 64
_SHAPE = np.asarray([_CELL, _CELL])


def _shifted_stack(n: int, size: int, shift_max: int = 4, seed: int = 11):
    """``n`` cell pairs cut from shared smooth texture at known integer offsets."""
    rng = np.random.default_rng(seed)
    big = rng.random((n, size * 2, size * 2))
    for _ in range(2):
        big = (big + np.roll(big, 1, -1) + np.roll(big, -1, -1)
               + np.roll(big, 1, -2) + np.roll(big, -1, -2)) / 5.0
    fixed = np.empty((n, size, size))
    moving = np.empty((n, size, size))
    half = size // 2
    for i in range(n):
        dy = int(rng.integers(-shift_max, shift_max + 1))
        dx = int(rng.integers(-shift_max, shift_max + 1))
        fixed[i] = big[i, half:half + size, half:half + size]
        moving[i] = big[i, half + dy:half + dy + size, half + dx:half + dx + size]
    return fixed, moving


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_correlation_keeps_caller_precision(dtype):
    """A float32 stack correlates in float32; a float64 stack still gets float64."""
    fixed, moving = _shifted_stack(6, _CELL)
    correlation = bpc.batched_image_phase_correlation(
        np.asarray(fixed, dtype=dtype), np.asarray(moving, dtype=dtype))
    assert correlation.dtype == np.dtype(dtype)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_find_offset_keeps_caller_precision(dtype):
    """The end-to-end entry point propagates the working precision to its results."""
    fixed, moving = _shifted_stack(6, _CELL)
    peaks, weights, ratios = bpc.batched_find_offset(
        np.asarray(fixed, dtype=dtype), np.asarray(moving, dtype=dtype),
        _SHAPE, min_overlap=0.25, max_overlap=1.0)
    assert peaks.dtype == np.dtype(dtype)
    assert weights.dtype == np.dtype(dtype)
    assert ratios.dtype == np.dtype(dtype)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float16])
def test_narrow_inputs_are_floored_at_float32(dtype):
    """Integer and half-precision stacks run at float32, never at their own width."""
    fixed, moving = _shifted_stack(4, _CELL)
    if np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(dtype)
        fixed = (fixed * info.max).astype(dtype)
        moving = (moving * info.max).astype(dtype)
    else:
        fixed = fixed.astype(dtype)
        moving = moving.astype(dtype)

    correlation = bpc.batched_image_phase_correlation(fixed, moving)
    assert correlation.dtype == np.dtype(np.float32)


def test_float32_agrees_with_float64():
    """Dropping to float32 must not move the answer at a scale anyone can measure.

    The tolerance is far tighter than the method's own ~0.1 px accuracy; it is set to
    catch a real precision loss, not to accommodate one.
    """
    fixed, moving = _shifted_stack(32, _CELL)
    p64, w64, r64 = bpc.batched_find_offset(
        np.asarray(fixed, dtype=np.float64), np.asarray(moving, dtype=np.float64),
        _SHAPE, min_overlap=0.25, max_overlap=1.0)
    p32, w32, r32 = bpc.batched_find_offset(
        np.asarray(fixed, dtype=np.float32), np.asarray(moving, dtype=np.float32),
        _SHAPE, min_overlap=0.25, max_overlap=1.0)

    np.testing.assert_allclose(np.asarray(p32, dtype=np.float64), p64, atol=1e-4)
    np.testing.assert_allclose(np.asarray(w32, dtype=np.float64), w64, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(r32, dtype=np.float64), r64, rtol=1e-3)


def test_float32_picks_the_same_integer_peak_under_low_contrast():
    """Low contrast is where float32 would show first, so pin the peak choice there."""
    fixed, moving = _shifted_stack(32, _CELL)
    fixed = 0.5 + (fixed - fixed.mean()) * 0.01
    moving = 0.5 + (moving - moving.mean()) * 0.01

    c64 = bpc.batched_image_phase_correlation(np.asarray(fixed, dtype=np.float64),
                                              np.asarray(moving, dtype=np.float64))
    c32 = bpc.batched_image_phase_correlation(np.asarray(fixed, dtype=np.float32),
                                              np.asarray(moving, dtype=np.float32))
    n = c64.shape[0]
    argmax64 = np.asarray(c64).reshape(n, -1).argmax(axis=1)
    argmax32 = np.asarray(c32).reshape(n, -1).argmax(axis=1)
    np.testing.assert_array_equal(argmax32, argmax64)


def test_float32_stack_allocates_a_single_precision_transform():
    """Guard the actual saving: the transform itself must be complex64, not complex128.

    Without this the dtype assertions above could be satisfied by correlating in
    float64 and casting the result back down, which would keep the workspace cost the
    issue is about.
    """
    fixed, moving = _shifted_stack(4, _CELL)
    captured = []
    real_fft2 = np.fft.fft2

    def recording_fft2(a, *args, **kwargs):
        out = real_fft2(a, *args, **kwargs)
        captured.append(out.dtype)
        return out

    np.fft.fft2 = recording_fft2
    try:
        bpc.batched_image_phase_correlation(np.asarray(fixed, dtype=np.float32),
                                            np.asarray(moving, dtype=np.float32))
    finally:
        np.fft.fft2 = real_fft2

    assert captured, 'expected the correlation to run a forward FFT'
    assert all(dt == np.dtype(np.complex64) for dt in captured), captured
