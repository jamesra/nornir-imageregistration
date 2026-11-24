"""crop_source_cell_stds_batched: bounded chunks give the same stds as one whole-grid stack.

RC3-v2 267-268 Grid16 (2026-11-24): a doubled 1024 px cell over ~1280 cells asked CuPy for one 10.7 GB block.
"""

from __future__ import annotations

import numpy as np
import pytest

from nornir_imageregistration.refine_shared import source_content_cache


def _reference_stds(image: np.ndarray, points: np.ndarray, cell: tuple[int, int]) -> np.ndarray:
    """Per-cell std as the unchunked code computed it: crop with zeros outside, NaN as 0, < 2 finite -> 0."""
    h, w = cell
    out = []
    for y, x in points:
        bot, left = int(np.floor(y - h / 2.0)), int(np.floor(x - w / 2.0))
        crop = np.zeros((h, w), dtype=np.float64)
        y0, x0 = max(bot, 0), max(left, 0)
        y1, x1 = min(bot + h, image.shape[0]), min(left + w, image.shape[1])
        if y1 > y0 and x1 > x0:
            crop[y0 - bot:y1 - bot, x0 - left:x1 - left] = image[y0:y1, x0:x1]
        finite = np.isfinite(crop)
        out.append(np.std(np.where(finite, crop, 0.0)) if finite.sum() >= 2 else 0.0)
    return np.asarray(out)


@pytest.mark.parametrize('chunk_bytes', [1, 16 * 16 * 8 * 3, 256 * 1024 * 1024])
def test_chunked_stds_match_reference(monkeypatch: pytest.MonkeyPatch, chunk_bytes: int) -> None:
    rng = np.random.default_rng(7)
    image = rng.random((120, 100)).astype(np.float32)
    image[10:20, 30:40] = np.nan
    points = np.array([[8.0, 8.0], [60.0, 50.0], [15.0, 35.0], [115.0, 95.0], [-30.0, -30.0], [70.5, 20.25],
                       [40.0, 90.0]])
    monkeypatch.setattr(source_content_cache, '_STD_CHUNK_BYTES', chunk_bytes)
    stds = source_content_cache.crop_source_cell_stds_batched(image, points, (16, 16))
    assert stds.shape == (len(points),)
    np.testing.assert_allclose(stds, _reference_stds(image, points, (16, 16)), rtol=1e-6, atol=1e-9)


def test_empty_points_return_empty() -> None:
    stds = source_content_cache.crop_source_cell_stds_batched(np.zeros((10, 10)), np.zeros((0, 2)), (4, 4))
    assert stds.shape == (0,)
