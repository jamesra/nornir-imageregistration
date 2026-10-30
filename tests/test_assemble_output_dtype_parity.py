"""Parallel assemble must use the same output dtype as serial/threaded (#112).

``TilesToImage`` / ``TilesToImageThreaded`` took ``tiles[0].Image.dtype``.
``TilesToImageParallel`` used ``default_image_dtype()`` (float16), so a mosaic of
in-memory float32 tiles requested a different canvas dtype. Multiprocess parallel
cannot ship in-memory tile arrays, so the parallel case is asserted at buffer
allocation time.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration
import nornir_imageregistration.assemble_tiles as assemble_tiles
from nornir_imageregistration.mosaic_tileset import MosaicTileset
from nornir_imageregistration.tile import Tile
from nornir_imageregistration.transforms.rigid import Rigid


def _float32_tileset(n_tiles: int = 2, shape=(24, 24)) -> MosaicTileset:
    tileset = MosaicTileset(image_to_source_space_scale=1.0)
    stride = float(shape[1]) * 0.9
    for i in range(n_tiles):
        img = np.full(shape, fill_value=0.25 * (i + 1), dtype=np.float32)
        tileset[i] = Tile(
            Rigid(target_offset=(0.0, i * stride)),
            img,
            image_to_source_space_scale=1.0,
            ID=i)
    return tileset


class TestAssembleOutputDtypeParity(unittest.TestCase):

    def setUp(self):
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy)
        self.assertNotEqual(
            np.dtype(nornir_imageregistration.default_image_dtype()),
            np.dtype(np.float32))

    def test_helper_uses_first_tile_dtype(self):
        tileset = _float32_tileset()
        self.assertEqual(np.dtype(np.float32), np.dtype(assemble_tiles._assemble_output_dtype(tileset)))

    def test_serial_and_threaded_preserve_tile_dtype(self):
        serial, _ = assemble_tiles.TilesToImage(_float32_tileset())
        threaded, _ = assemble_tiles.TilesToImageThreaded(_float32_tileset())
        self.assertEqual(np.float32, serial.dtype)
        self.assertEqual(serial.dtype, threaded.dtype)

    def test_parallel_allocates_canvas_with_tile_dtype(self):
        seen: list[np.dtype] = []

        def capture(height, width, dtype):
            seen.append(np.dtype(dtype))
            raise RuntimeError('stop after dtype selection')

        with mock.patch.object(
                assemble_tiles, '__CreateOutputBufferForArea', side_effect=capture):
            with self.assertRaisesRegex(RuntimeError, 'stop after dtype'):
                assemble_tiles.TilesToImageParallel(_float32_tileset())
        self.assertEqual([np.dtype(np.float32)], seen)


if __name__ == '__main__':
    unittest.main()
