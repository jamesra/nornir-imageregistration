"""Regressions for #194 / #195 arrange_mosaic helpers."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import nornir_imageregistration as nir
from nornir_imageregistration import arrange_mosaic


class TestTileOffsetRemoteExcessScalar(unittest.TestCase):
    def test_caller_excess_scalar_is_forwarded(self) -> None:
        """#195: do not overwrite the excess_scalar argument with a literal."""
        rect = nir.Rectangle.CreateFromBounds((0.0, 0.0, 32.0, 32.0))
        seen: list[float] = []

        def _fake_overlap(image, overlapping_rect, excess_scalar=1.0, mask_extrema=True):
            seen.append(float(excess_scalar))
            h = w = 16
            img = np.linspace(0.1, 0.9, h * w, dtype=np.float32).reshape(h, w)
            mask = np.ones((h, w), dtype=bool)
            return arrange_mosaic._OverlapRegion(img, mask, (h, w), np.zeros(2))

        tile_offset = getattr(arrange_mosaic, '__tile_offset_remote')
        found = nir.masked_correlation.MaskedOffsetResult(
            nir.AlignmentRecord(peak=(0.0, 0.0), weight=1.0), 0.5, 256)
        with patch.object(arrange_mosaic, '_overlap_region_with_validity', side_effect=_fake_overlap), \
                patch.object(nir, 'LoadImage', return_value=np.ones((64, 64), dtype=np.float32)), \
                patch.object(nir.masked_correlation, 'masked_find_offset', return_value=found):
            tile_offset(
                'a.png', 'b.png', rect, rect, OffsetAdjustment=(0.0, 0.0),
                excess_scalar=1.25)

        self.assertEqual(seen, [1.25, 1.25])


class TestAlignmentScoreRemoteFinally(unittest.TestCase):
    def test_load_failure_does_not_raise_nameerror_in_finally(self) -> None:
        """#194: finally del must not mask the original exception."""
        rect = nir.Rectangle.CreateFromBounds((0.0, 0.0, 8.0, 8.0))
        alignment_score = getattr(arrange_mosaic, '__AlignmentScoreRemote')

        with patch.object(nir, 'ImageParamToImageArray', side_effect=OSError('missing')):
            with self.assertRaises(OSError) as ctx:
                alignment_score('a.png', 'b.png', rect, rect)
        self.assertNotIsInstance(ctx.exception, NameError)
        self.assertIn('missing', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
