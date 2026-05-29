"""
Created on Mar 12, 2013

@author: u0490822
"""
import os
import unittest
from typing import Any

import numpy as np

import nornir_imageregistration
import setup_imagetest


def _cp_for_get_array_module():
    try:
        import cupy as cp
    except ModuleNotFoundError:
        import nornir_imageregistration.cupy_thunk as cp
    except ImportError:
        import nornir_imageregistration.cupy_thunk as cp
    return cp


class ImageBlurBase(setup_imagetest.ImageTestBase):

    def setUp(self):
        super(ImageBlurBase, self).setUp()

        self.TEMImagePath16bpp = os.path.join(self.TestInputPath, "PlatformRaw", "IDOC", "RC2_Micro", "17")
        self.TEMImagePath8bpp = os.path.join(self.TestInputPath, "Images", "Alignment")
        self.LMImagePath = os.path.join(self.TestInputPath, "PlatformRaw", "PMG", "6259", "6259_9778_WDF_40xOil_05_G")

    def _tile_image_path(self) -> str:
        return os.path.join(self.LMImagePath, "Tile000012.bmp")

    def _run_smartblur(self, *, backend: str) -> tuple[Any, Any]:
        image_path = self._tile_image_path()
        config = nornir_imageregistration.blur.SmartBlurConfig(
            kernel_size=7, sigma=5, low_threshold=0.15, high_threshold=0.3
        )
        img = nornir_imageregistration.LoadImage(image_path, dtype=np.float32, backend=backend)  # type: ignore[arg-type]
        blurred_image = nornir_imageregistration.blur.smart_blur(img, config)
        nornir_imageregistration.ShowGrayscale(
            [img, blurred_image], title=os.path.basename(image_path), PassFail=True
        )
        return img, blurred_image


class test_smart_blur(ImageBlurBase):

    def test_smartblur_numpy(self):
        cp = _cp_for_get_array_module()
        prev = nornir_imageregistration.GetActiveComputationLib()
        try:
            nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
            img, blurred = self._run_smartblur(backend="numpy")
            self.assertIs(cp.get_array_module(img), np)
            self.assertIs(cp.get_array_module(blurred), np)
        finally:
            nornir_imageregistration.SetActiveComputationLib(prev)

    @unittest.skipIf(not nornir_imageregistration.HasCupy(), "CuPy not available")
    def test_smartblur_cupy(self):
        import cupy as cp

        prev = nornir_imageregistration.GetActiveComputationLib()
        try:
            nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
            img, blurred = self._run_smartblur(backend="cupy")
            self.assertIs(cp.get_array_module(img), cp)
            self.assertIs(cp.get_array_module(blurred), cp)
        finally:
            nornir_imageregistration.SetActiveComputationLib(prev)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
