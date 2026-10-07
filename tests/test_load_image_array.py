"""Raw Pillow pixels, with no mask and no extrema fill."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from nornir_imageregistration.pillow_helpers import load_image_array


def test_load_image_array_matches_pillow_pixels(tmp_path: Path) -> None:
    path = tmp_path / "tile.png"
    expected = np.arange(12, dtype=np.uint8).reshape(3, 4)
    Image.fromarray(expected, mode="L").save(path)
    loaded = load_image_array(str(path))
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, expected)
