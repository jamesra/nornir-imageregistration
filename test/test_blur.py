"""
Created on Mar 12, 2013

@author: u0490822
"""
import os
import unittest
import numpy as np

from PIL import Image

import nornir_shared.plot as plot
import nornir_imageregistration
import setup_imagetest


class ImageBlurBase(setup_imagetest.ImageTestBase):

    def setUp(self):
        super(ImageBlurBase, self).setUp()

        self.TEMImagePath16bpp = os.path.join(self.TestInputPath, "PlatformRaw", "IDoc", "RC2_Micro", "17")
        self.TEMImagePath8bpp = os.path.join(self.TestInputPath, "Images", "Alignment")
        self.LMImagePath = os.path.join(self.TestInputPath, "PlatformRaw", "PMG", "6259", "6259_9778_WDF_40xOil_05_G")


class test_smart_blur(ImageBlurBase):

    def test_smartblur(self):
        image_paths = []
        # image_paths.append(os.path.join(self.TEMImagePath16bpp, '10000.tif'))
        # image_paths.append(os.path.join(self.TEMImagePath16bpp, '10016.tif'))
        image_paths.append(os.path.join(self.LMImagePath, 'Tile000012.bmp'))

        config = nornir_imageregistration.blur.SmartBlurConfig(kernel_size=7, sigma=5, low_threshold=.15,
                                                               high_threshold=.3)

        for image_path in image_paths:
            img = nornir_imageregistration.LoadImage(image_path, dtype=np.float32)
            blurred_image = nornir_imageregistration.blur.smart_blur(img, config)
            nornir_imageregistration.ShowGrayscale([img, blurred_image], title=os.path.basename(image_path),
                                                   PassFail=True)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
