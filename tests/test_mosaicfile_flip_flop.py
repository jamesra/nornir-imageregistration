"""Golden fixtures for MosaicFile.Write Flip/Flop and Utah idoc invert."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from nornir_imageregistration.files.mosaicfile import MosaicFile
from nornir_imageregistration.transforms.factory import LoadTransform


def _tile_yx_from_mosaic(path: str) -> dict[str, tuple[float, float]]:
    """Parse mosaic rigid translations as (Y, X) per tile basename."""
    mosaic = MosaicFile.Load(path)
    assert mosaic is not None
    out: dict[str, tuple[float, float]] = {}
    for name, transform_string in mosaic.ImageToTransformString.items():
        transform = LoadTransform(transform_string)
        offset = np.asarray(transform.target_offset, dtype=np.float64).reshape(2)
        out[os.path.basename(name)] = (float(offset[0]), float(offset[1]))
    return out


class TestMosaicFileFlipFlop(unittest.TestCase):
    """Lock Flip=negate Y, Flop=negate X, and Utah Flip=not Flip."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="mosaic-flip-")
        self.entries = {
            "a.png": (10.0, 20.0),  # (X, Y) as MosaicFile.Write expects
            "b.png": (30.0, 40.0),
        }
        self.image_size = (100, 100)

    def _write(self, name: str, *, flip: bool, flop: bool) -> str:
        path = os.path.join(self._tmpdir, name)
        MosaicFile.Write(
            path,
            self.entries,
            Flip=flip,
            Flop=flop,
            ImageSize=self.image_size,
            Downsample=1,
        )
        return path

    def test_flip_negates_y_only(self) -> None:
        path = self._write("flip.mosaic", flip=True, flop=False)
        positions = _tile_yx_from_mosaic(path)
        self.assertAlmostEqual(positions["a.png"][0], -20.0, places=5)  # Y
        self.assertAlmostEqual(positions["a.png"][1], 10.0, places=5)   # X
        self.assertAlmostEqual(positions["b.png"][0], -40.0, places=5)
        self.assertAlmostEqual(positions["b.png"][1], 30.0, places=5)

    def test_flop_negates_x_only(self) -> None:
        path = self._write("flop.mosaic", flip=False, flop=True)
        positions = _tile_yx_from_mosaic(path)
        self.assertAlmostEqual(positions["a.png"][0], 20.0, places=5)
        self.assertAlmostEqual(positions["a.png"][1], -10.0, places=5)

    def test_utah_idoc_inverts_flip_flag(self) -> None:
        """idoc Write uses Flip=not Flip — section in FlipList → mosaic Flip=False."""
        section_in_flip_list = True
        mosaic_flip = not section_in_flip_list
        path = self._write("utah.mosaic", flip=mosaic_flip, flop=False)
        positions = _tile_yx_from_mosaic(path)
        # FlipList entry → Flip=False → Y stays positive.
        self.assertAlmostEqual(positions["a.png"][0], 20.0, places=5)
        self.assertAlmostEqual(positions["a.png"][1], 10.0, places=5)


if __name__ == "__main__":
    unittest.main()
