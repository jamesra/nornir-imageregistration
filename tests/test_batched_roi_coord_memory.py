"""
Batched ROI sampling must not build homogeneous coordinate arrays or stack copies.

`_sample_source_rois_batched` used to expand every chunk's target grid to
homogeneous `(n, HW, 3)` float64, matmul it against the full 3x3 inverse maps, and
discard the resulting third column. Those two arrays -- not the `(N, H, W)` ROI
stacks, which are comparatively tiny -- set the peak of this function. It now
applies the 2x2 linear part and the translation column directly, which is
arithmetically the same affine map.

These tests pin the coordinate math against an explicit homogeneous reference and
pin the allocation shapes so the discarded-column form cannot come back.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy

import nornir_imageregistration
from nornir_imageregistration import local_distortion_correction as ldc

CELL_H = 16
CELL_W = 16
IMAGE_SIZE = 128


def _source_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)


def _botlefts(count: int) -> np.ndarray:
    lin = np.linspace(20, IMAGE_SIZE - 20 - CELL_H, count)
    return np.asarray([(v, v) for v in lin], dtype=np.float64)


def _inverse_matrices(count: int, *, angle: float = 0.0,
                      shift: tuple[float, float] = (0.5, -0.25)) -> np.ndarray:
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    matrices = np.zeros((count, 3, 3), dtype=np.float64)
    for i in range(count):
        matrices[i] = ((cos_a, -sin_a, shift[0]),
                       (sin_a, cos_a, shift[1]),
                       (0.0, 0.0, 1.0))
    return matrices


def _homogeneous_reference_coords(botlefts: np.ndarray,
                                  matrices: np.ndarray) -> np.ndarray:
    """The original homogeneous formulation, kept as the arithmetic reference."""
    relative = nornir_imageregistration.assemble.GetROICoords(
        (0.0, 0.0), (CELL_H, CELL_W), xp=np)
    relative = np.asarray(relative, dtype=np.float32)
    botlefts_dev = botlefts.astype(np.float32, copy=False)
    samples_per_cell = CELL_H * CELL_W
    count = botlefts.shape[0]

    write = relative[None, :, :] + botlefts_dev[:, None, :]
    ones = np.ones((count, samples_per_cell, 1), dtype=np.float64)
    homog = np.concatenate((write.astype(np.float64, copy=False), ones), axis=2)
    return np.matmul(homog, np.swapaxes(matrices, -1, -2))[:, :, :2]


def _affine_coords(botlefts: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    """The shipped formulation, mirroring `_sample_source_rois_batched`."""
    relative = nornir_imageregistration.assemble.GetROICoords(
        (0.0, 0.0), (CELL_H, CELL_W), xp=np)
    relative = np.asarray(relative, dtype=np.float32)
    botlefts_dev = botlefts.astype(np.float32, copy=False)

    write = relative[None, :, :] + botlefts_dev[:, None, :]
    linear_t = np.swapaxes(matrices[:, :2, :2], -1, -2)
    coords = np.matmul(write.astype(np.float64, copy=False), linear_t)
    coords += matrices[:, :2, 2][:, None, :]
    return coords


@pytest.mark.parametrize('angle', [0.0, 0.15, -0.4, np.pi / 3])
def test_affine_coords_match_homogeneous_reference(angle):
    """Dropping the homogeneous column is arithmetically identical."""
    botlefts = _botlefts(5)
    matrices = _inverse_matrices(5, angle=angle)
    np.testing.assert_array_equal(
        _affine_coords(botlefts, matrices),
        _homogeneous_reference_coords(botlefts, matrices))


def _sample(source, matrices, botlefts, *, oob_cval=np.nan):
    pure = np.asarray(
        [bool(np.allclose(m[:2, :2], np.eye(2))) for m in matrices], dtype=bool)
    return ldc._sample_source_rois_batched(
        source, matrices, pure, botlefts, CELL_H, CELL_W, np, scipy,
        oob_cval=oob_cval)


@pytest.mark.parametrize('angle', [0.0, 0.2, -0.35])
def test_sampled_values_are_finite_and_in_source_range(angle):
    """Sampling still clips into the source range and keeps the stack shape."""
    source = _source_image()
    count = 4
    sampled = _sample(source, _inverse_matrices(count, angle=angle), _botlefts(count))

    assert sampled.shape == (count, CELL_H, CELL_W)
    assert sampled.dtype == source.dtype
    finite = sampled[np.logical_not(np.isnan(sampled))]
    assert finite.size > 0
    assert finite.min() >= source.min()
    assert finite.max() <= source.max()


class _AllocSpy:
    """Record shapes and dtypes passed to one numpy allocator."""

    def __init__(self, name: str):
        self.name = name
        self.calls: list[tuple[tuple[int, ...], np.dtype | None]] = []
        self._real = getattr(np, name)

    def __enter__(self):
        setattr(np, self.name, self._wrapped)
        return self

    def __exit__(self, *exc):
        setattr(np, self.name, self._real)
        return False

    def _wrapped(self, shape, *args, **kwargs):
        as_tuple = (shape,) if isinstance(shape, int) else tuple(
            int(v) for v in np.atleast_1d(np.asarray(shape)))
        dtype = kwargs.get('dtype', args[0] if args else None)
        self.calls.append((as_tuple, np.dtype(dtype) if dtype is not None else None))
        return self._real(shape, *args, **kwargs)

    def float64_column_of_ones(self) -> int:
        """Count (HW, 1) float64 allocations, the homogeneous companion column."""
        return sum(1 for shape, dtype in self.calls
                   if len(shape) == 2 and shape[-1] == 1 and dtype == np.float64)


def test_no_homogeneous_ones_column_allocated():
    """No (n, HW, 3) float64 companion array is built for the matmul."""
    source = _source_image()
    botlefts = _botlefts(4)
    matrices = _inverse_matrices(4, angle=0.3)

    with _AllocSpy('ones') as ones_spy:
        _sample(source, matrices, botlefts)

    assert ones_spy.float64_column_of_ones() == 0, (
        'a homogeneous coordinate column was allocated; the affine form should '
        'apply the 2x2 linear part and translation directly')


def test_chunked_sampling_does_not_concatenate(monkeypatch):
    """Chunks are written into the final stack rather than concatenated."""
    source = _source_image()
    count = 6
    botlefts = _botlefts(count)
    matrices = _inverse_matrices(count, angle=0.1)

    # Force several chunks: budget of two cells' worth of samples.
    monkeypatch.setattr(ldc, '_batched_roi_sample_budget',
                        lambda h, w: 2 * h * w)
    calls: list[int] = []
    real_concatenate = np.concatenate

    def counting_concatenate(*args, **kwargs):
        calls.append(1)
        return real_concatenate(*args, **kwargs)

    monkeypatch.setattr(np, 'concatenate', counting_concatenate)
    multi_chunk = _sample(source, matrices, botlefts)

    assert multi_chunk.shape == (count, CELL_H, CELL_W)
    assert not calls, 'chunk results were concatenated instead of written in place'


@pytest.mark.parametrize('cells_per_chunk', [1, 2, 3, 4, 5])
def test_sampled_values_do_not_depend_on_chunk_size(monkeypatch, cells_per_chunk):
    """Chunking is a memory knob and must not change a single sampled value.

    The sample AABB and its cubic prefilter used to be computed per chunk. With
    ``order=3`` the prefilter's boundary conditions depend on the extent it runs over,
    so a cell's values depended on which other cells shared its chunk -- measured at up
    to 8e-02, roughly 8% of full intensity range, against a float32 epsilon of ~1e-07.
    The crop and prefilter now run once over the union of all cells' samples.

    Equality here is exact, not approximate: every chunking samples the same
    prefiltered array, so the arithmetic is identical rather than merely close.
    """
    source = _source_image()
    count = 6
    botlefts = _botlefts(count)
    matrices = _inverse_matrices(count, angle=0.1)

    single_chunk = _sample(source, matrices, botlefts)
    monkeypatch.setattr(ldc, '_batched_roi_sample_budget',
                        lambda h, w: cells_per_chunk * h * w)
    chunked = _sample(source, matrices, botlefts)

    np.testing.assert_array_equal(chunked, single_chunk)


def test_chunk_independence_holds_for_pure_translation(monkeypatch):
    """Pure-translation cells skip the coordinate rounding, so cover them separately."""
    source = _source_image()
    count = 6
    botlefts = _botlefts(count)
    matrices = _inverse_matrices(count, angle=0.0)

    single_chunk = _sample(source, matrices, botlefts)
    monkeypatch.setattr(ldc, '_batched_roi_sample_budget', lambda h, w: 2 * h * w)
    np.testing.assert_array_equal(_sample(source, matrices, botlefts), single_chunk)


def test_union_bounds_cover_every_cell(monkeypatch):
    """The corner-derived union AABB must not clip any cell's samples.

    ``_global_sample_bounds`` takes the four corners of each cell's sample grid rather
    than reducing over the full coordinate set, which is exact for an affine map but
    would silently crop cells if the corner logic were wrong. Compare against bounds
    reduced from every coordinate.
    """
    count = 6
    botlefts = _botlefts(count)
    for angle in (0.0, 0.1, -0.4):
        matrices = _inverse_matrices(count, angle=angle)
        coords = _affine_coords(botlefts, matrices).reshape(-1, 2)
        expected_lo = np.floor(coords.min(axis=0)).astype(np.int64)
        expected_hi = np.ceil(coords.max(axis=0)).astype(np.int64) + 1

        relative = np.asarray(nornir_imageregistration.assemble.GetROICoords(
            (0.0, 0.0), (CELL_H, CELL_W), xp=np), dtype=np.float32)
        linear_t = np.swapaxes(matrices[:, :2, :2], -1, -2)
        offset = matrices[:, :2, 2]
        y0, x0, y1, x1 = ldc._global_sample_bounds(
            relative, botlefts.astype(np.float32), linear_t, offset,
            IMAGE_SIZE, IMAGE_SIZE, np)

        assert (y0, x0) == (max(0, int(expected_lo[0])), max(0, int(expected_lo[1]))), angle
        assert (y1, x1) == (min(IMAGE_SIZE, int(expected_hi[0])),
                            min(IMAGE_SIZE, int(expected_hi[1]))), angle


def test_all_cells_off_image_returns_fill():
    """A lattice entirely outside the source yields the fill value, not a crash."""
    source = _source_image()
    count = 3
    botlefts = np.full((count, 2), 10_000.0)
    matrices = _inverse_matrices(count, angle=0.0)

    sampled = _sample(source, matrices, botlefts, oob_cval=np.nan)
    assert sampled.shape == (count, CELL_H, CELL_W)
    assert np.all(np.isnan(sampled))


def test_prefilter_runs_once_regardless_of_chunk_count(monkeypatch):
    """One spline_filter for the whole batch, not one per chunk."""
    source = _source_image()
    count = 6
    botlefts = _botlefts(count)
    matrices = _inverse_matrices(count, angle=0.1)

    calls: list[int] = []
    real_filter = scipy.ndimage.spline_filter

    def counting_filter(*args, **kwargs):
        calls.append(1)
        return real_filter(*args, **kwargs)

    monkeypatch.setattr(scipy.ndimage, 'spline_filter', counting_filter)
    monkeypatch.setattr(ldc, '_batched_roi_sample_budget', lambda h, w: h * w)
    _sample(source, matrices, botlefts)

    assert len(calls) == 1, f'expected a single prefilter, got {len(calls)}'


def test_crop_stack_does_not_hold_every_crop():
    """Target crops are copied into the stack, not collected and stacked."""
    target = _source_image(7)
    botlefts = _botlefts(5)

    calls: list[int] = []
    real_stack = np.stack

    def counting_stack(*args, **kwargs):
        calls.append(1)
        return real_stack(*args, **kwargs)

    np.stack = counting_stack  # type: ignore[assignment]
    try:
        stack = ldc._crop_target_rois_batched(target, botlefts, CELL_H, CELL_W, None, np)
    finally:
        np.stack = real_stack  # type: ignore[assignment]

    assert stack.shape == (5, CELL_H, CELL_W)
    assert not calls, 'crops were accumulated and stacked'

    for i, (yo, xo) in enumerate(botlefts.astype(np.int64)):
        expected = nornir_imageregistration.CropImage(
            target, int(xo), int(yo), CELL_W, CELL_H, cval=False, image_stats=None)
        np.testing.assert_array_equal(stack[i], expected)


def test_crop_stack_empty_lattice():
    """A zero-cell lattice yields an empty stack rather than raising."""
    target = _source_image(7)
    stack = ldc._crop_target_rois_batched(
        target, np.zeros((0, 2), dtype=np.float64), CELL_H, CELL_W, None, np)
    assert stack.shape == (0, CELL_H, CELL_W)
    assert stack.dtype == target.dtype
