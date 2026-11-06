"""A pyramid tile reads the finer level from the local copy without staging or statting the share."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from nornir_imageregistration import tileset_functions

_NAMES = ('tl.png', 'tr.png', 'bl.png', 'br.png')


def _write(path: str) -> None:
  Image.fromarray(np.full((16, 16), 90, dtype=np.uint8)).save(path)


class TestLocalInputs(unittest.TestCase):

  def _build(self, root: str, local: tuple[str, ...], on_share: tuple[str, ...]):
    share, cache, out_temp, dest = (os.path.join(root, d) for d in ('share', 'cache', 'out_temp', 'dest'))
    for d in (share, cache, out_temp, dest):
      os.makedirs(d)
    for name in on_share:
      _write(os.path.join(share, name))
    for name in local:
      _write(os.path.join(cache, name))
    sources = [os.path.join(share, n) for n in _NAMES]
    tileset_functions.CreateOneTilesetTileWithPillowOverNetwork(
      (16, 16), *sources, OutputFileFullPath=os.path.join(dest, 'out.png'), temp_input_dir=cache,
      output_level_temp_dir=out_temp, source_tile_names=frozenset(on_share))
    return cache, dest

  def test_local_inputs_are_used_not_restaged_and_are_consumed(self) -> None:
    with tempfile.TemporaryDirectory() as root, \
        mock.patch.object(tileset_functions, '_copy_source_tile_local', side_effect=AssertionError('staged')):
      cache, dest = self._build(root, local=_NAMES, on_share=_NAMES)
      self.assertTrue(os.path.isfile(os.path.join(dest, 'out.png')))
      self.assertEqual(os.listdir(cache) if os.path.isdir(cache) else [], [])

  def test_an_input_missing_locally_is_read_from_the_share_and_the_share_is_left_alone(self) -> None:
    with tempfile.TemporaryDirectory() as root:
      cache, dest = self._build(root, local=_NAMES[:2], on_share=_NAMES)
      self.assertTrue(os.path.isfile(os.path.join(dest, 'out.png')))
      self.assertTrue(all(os.path.isfile(os.path.join(root, 'share', n)) for n in _NAMES))


if __name__ == '__main__':
  unittest.main()
