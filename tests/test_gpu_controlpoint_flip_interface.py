"""GPU control-point bases must advertise ``ITransformFlip`` (#116).

``ControlPointBase`` inherits ``ITransformFlip``; ``ControlPointBase_GPUComponent``
implemented ``Flip`` but omitted the interface, so ``isinstance(..., ITransformFlip)``
(e.g. Pyre's transform controller) silently skipped Flip for every GPU control-point
transform.
"""

from __future__ import annotations

import unittest

from nornir_imageregistration.transforms.base import ITransformFlip
from nornir_imageregistration.transforms.controlpointbase import (
    ControlPointBase,
    ControlPointBase_GPUComponent,
)
from nornir_imageregistration.transforms.gridtransform import (
    GridTransform,
    GridTransform_GPUComponent,
)
from nornir_imageregistration.transforms.landmark import Landmark_CPU, Landmark_GPU


class TestGpuControlPointFlipInterface(unittest.TestCase):

    def test_gpu_base_is_transform_flip(self):
        self.assertTrue(issubclass(ControlPointBase, ITransformFlip))
        self.assertTrue(issubclass(ControlPointBase_GPUComponent, ITransformFlip))

    def test_concrete_cpu_and_gpu_twins_are_transform_flip(self):
        self.assertTrue(issubclass(GridTransform, ITransformFlip))
        self.assertTrue(issubclass(GridTransform_GPUComponent, ITransformFlip))
        self.assertTrue(issubclass(Landmark_CPU, ITransformFlip))
        self.assertTrue(issubclass(Landmark_GPU, ITransformFlip))


if __name__ == '__main__':
    unittest.main()
