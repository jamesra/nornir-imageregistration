import numpy as np
from numpy.typing import NDArray
import unittest

import nornir_imageregistration
import nornir_imageregistration.spatial.converters as converters


class TestConverters(unittest.TestCase):

    def test_ArcAngle(self):
        origin = np.array([0, 0])
        A = np.array([1, 0])
        B = np.array([0, 1])
        angle = abs(converters.ArcAngle(origin, A, B)[0])
        self.assertAlmostEqual(angle, np.pi / 2)

        self.CheckArcAngle(origin, A, B, np.pi / 2)
        self.CheckAnglesSumToPi(origin, A, B)

    def test_repro_case(self):
        """There was a bug that occurred because ArcAngle was changing input arrays,
           This was the data at the time but there is nothing special about it"""
        self.CheckAnglesSumToPi(A=(1012.16, 148.212),
                                B=(167.083, 89.787979),
                                C=(-122.412, -18.266))

    def CheckArcAngle(self,
                      origin: NDArray[np.floating],
                      A: NDArray[np.floating],
                      B: NDArray[np.floating],
                      expected_angle: float):
        origin = nornir_imageregistration.EnsurePointsAre2DNumpyArray(origin)
        A = nornir_imageregistration.EnsurePointsAre2DNumpyArray(A)
        B = nornir_imageregistration.EnsurePointsAre2DNumpyArray(B)
        angle = abs(converters.ArcAngle(origin, A, B)[0])
        self.assertAlmostEqual(angle, expected_angle)

    def CheckAnglesSumToPi(self,
                           A: NDArray[np.floating],
                           B: NDArray[np.floating],
                           C: NDArray[np.floating]):
        """The angles of a triangle should always sum to pi radians"""

        A = nornir_imageregistration.EnsurePointsAre2DNumpyArray(A)
        B = nornir_imageregistration.EnsurePointsAre2DNumpyArray(B)
        C = nornir_imageregistration.EnsurePointsAre2DNumpyArray(C)

        angleABC = abs(converters.ArcAngle(A, B, C)[0])
        angleBAC = abs(converters.ArcAngle(B, A, C)[0])
        angleCAB = abs(converters.ArcAngle(C, A, B)[0])

        total = angleABC + angleBAC + angleCAB
        self.assertAlmostEqual(total, np.pi, places=5)
