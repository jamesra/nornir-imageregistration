"""Regression for #180: single count_nonzero of find_peak overlap mask."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import nornir_imageregistration.phasecorrelation as pc


class TestFindPeakMaskCount(unittest.TestCase):
    def test_overlap_mask_counted_once(self) -> None:
        """#180: reuse the empty-mask count for the mean denominator."""
        image = np.zeros((24, 24), dtype=np.float64)
        image[8, 8] = 1.0
        mask = np.ones((24, 24), dtype=bool)
        calls = {'n': 0}
        real = np.count_nonzero

        def _counting(a, *args, **kwargs):
            calls['n'] += 1
            return real(a, *args, **kwargs)

        with patch.object(np, 'count_nonzero', side_effect=_counting):
            pc.find_peak(image, overlap_mask=mask, cutoff=0.0)

        self.assertEqual(calls['n'], 1)


if __name__ == '__main__':
    unittest.main()
