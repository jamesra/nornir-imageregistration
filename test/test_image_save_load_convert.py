'''
Created on Feb 20, 2019

@author: u0490822
'''

import logging
import os
import unittest
import numpy

import nornir_shared.images
import nornir_imageregistration
import nornir_imageregistration.stos_brute as stos_brute

from . import setup_imagetest

class ImageProperties(object):
    '''Helper class describing properties of images we expect in tests'''
    @property
    def bpp(self):
        return self._bpp
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def ext(self):
        return self._ext
    
    def __init__(self, bpp, dtype, ext):
        self._bpp = int(bpp)
        self._dtype = dtype
        self._ext = ext
        
    def GenFilename(self, InputImageFullPath):
        '''
        Generates a filename for an image by appending the expected image properties to the name
        :param str output_filename: A filename with no extension.
        '''
        input_basename = os.path.basename(InputImageFullPath)
        (output_filename, _) = os.path.splitext(input_basename) 
        return '{0}_as_{1}-bit_{2}.{3}'.format(output_filename, self.bpp, self.dtype().dtype.char, self.ext)


class TestImageSaveLoadConvert(setup_imagetest.ImageTestBase):

    
    def RunConvertImageTest(self, input_image_fullpath):
        '''Tests converting an image using the min/max/gamma parameters.  Displays to user
        to ensure valid results'''
        original_image = nornir_imageregistration.ImageParamToImageArray(input_image_fullpath)
                
        Bpp = nornir_shared.images.GetImageBpp(input_image_fullpath) 
        hist = nornir_imageregistration.Histogram(input_image_fullpath)
        hist = nornir_shared.histogram.Histogram.TryRemoveMaxValueOutlier(hist)
        hist = nornir_shared.histogram.Histogram.TryRemoveMinValueOutlier(hist)
        iMinCutoff = hist.MinNonEmptyBin()
        iMaxCutoff = hist.MaxNonEmptyBin()
        
        min_val = hist.BinValue(iMinCutoff)
        max_val = hist.BinValue(iMaxCutoff, 1.0)
         
        leveled_image = nornir_imageregistration.core._ConvertSingleImage(input_image_fullpath, MinMax=(min_val, max_val), Bpp=Bpp)
        
#         self.assertTrue(nornir_imageregistration.ShowGrayscale([original_image,leveled_image],
#                                                                PassFail=True,
#                                                                title="The original image"))
        
        inverted_leveled_image = nornir_imageregistration.core._ConvertSingleImage(input_image_fullpath, MinMax=(min_val, max_val), Bpp=Bpp, Invert=True)
        gamma_lowered_image = nornir_imageregistration.core._ConvertSingleImage(input_image_fullpath, MinMax=(min_val, max_val), Bpp=Bpp, Gamma=0.7)
        gamma_raised_image = nornir_imageregistration.core._ConvertSingleImage(input_image_fullpath, MinMax=(min_val, max_val), Bpp=Bpp, Gamma=1.3)
        
        self.assertTrue(nornir_imageregistration.ShowGrayscale([[original_image, leveled_image, inverted_leveled_image],
                                                                [gamma_lowered_image, gamma_raised_image]], 
                                                               PassFail=True,
                                                               title="First Row: The original image, the leveled image, the inverted image\nSecond Row: Lower gamma, Raised Gamma"))
        
        self.RunSaveLoadImageTest_BppOnly(leveled_image, os.path.join(self.TestOutputPath, 'leveled.png'), Bpp)
        self.RunSaveLoadImageTest_BppOnly(inverted_leveled_image, os.path.join(self.TestOutputPath, 'inverted_leveled.png'), Bpp)
        self.RunSaveLoadImageTest_BppOnly(gamma_lowered_image, os.path.join(self.TestOutputPath, 'gamma_raised.png'), Bpp)
        self.RunSaveLoadImageTest_BppOnly(gamma_raised_image, os.path.join(self.TestOutputPath, 'gamma_lowered.png'), Bpp)
        
        return leveled_image
    
    def RunSaveLoadImageTest(self, input_image_fullpath, expected_input_properties, expected_output_properties):
        self.assertTrue(isinstance(expected_input_properties, ImageProperties))
        self.assertTrue(isinstance(expected_output_properties, ImageProperties))
         
        self.assertTrue(os.path.exists(input_image_fullpath), "Missing test input: {0}".format(input_image_fullpath))
        input_image = nornir_imageregistration.ImageParamToImageArray(input_image_fullpath)
         
        output_path = os.path.join(self.TestOutputPath, expected_output_properties.GenFilename(input_image_fullpath))
                
        wrong_input_bpp_error_msg = "Expected {0}-bit image".format(expected_input_properties.bpp)
        wrong_output_bpp_error_msg = "Expected {0}-bit image".format(expected_output_properties.bpp)
        
        bpp = nornir_imageregistration.ImageBpp(input_image)
        self.assertEqual(bpp, expected_input_properties.bpp, wrong_input_bpp_error_msg)
        
        nornir_imageregistration.SaveImage(output_path, input_image, bpp=expected_output_properties.bpp)
        reloaded_image = nornir_imageregistration.ImageParamToImageArray(output_path)
        MagickBpp = nornir_shared.images.GetImageBpp(output_path)
        self.assertEqual(MagickBpp, expected_output_properties.bpp, wrong_output_bpp_error_msg)
        
        reloaded_bpp = nornir_imageregistration.ImageBpp(reloaded_image)
        self.assertEqual(reloaded_bpp, expected_output_properties.bpp, wrong_output_bpp_error_msg)
        
    def RunSaveLoadImageTest_BppOnly(self, input_image, output_fullpath, expected_bpp):
                 
        wrong_input_bpp_error_msg = "Expected {0}-bit image".format(expected_bpp)
        wrong_output_bpp_error_msg = "Expected {0}-bit image".format(expected_bpp)
        
        bpp = nornir_imageregistration.ImageBpp(input_image)
        self.assertEqual(bpp, expected_bpp, wrong_input_bpp_error_msg)
        
        nornir_imageregistration.SaveImage(output_fullpath, input_image, bpp=expected_bpp)
        reloaded_image = nornir_imageregistration.ImageParamToImageArray(output_fullpath)
        MagickBpp = nornir_shared.images.GetImageBpp(output_fullpath)
        self.assertEqual(MagickBpp, expected_bpp, wrong_output_bpp_error_msg)
        
        reloaded_bpp = nornir_imageregistration.ImageBpp(reloaded_image)
        self.assertEqual(reloaded_bpp, expected_bpp, wrong_output_bpp_error_msg)

    def test_8Bit_ConvertImage(self):
        self.FixedImagePath = os.path.join(self.ImportedDataPath, "BrightfieldShading.png")
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")
        self.RunConvertImageTest(self.FixedImagePath)
    
    def test_16Bit_ConvertImage(self):
        filename = '10001_RPC2_590.tif'
        self.FixedImagePath = os.path.join(self.ImportedDataPath,'16-bit', filename)
        basename = os.path.basename(filename)
        output_path = os.path.join(self.TestOutputPath,  basename + '_converted.tif')
                
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input: {0}".format(self.FixedImagePath))
        output = self.RunConvertImageTest(self.FixedImagePath)
        
        nornir_imageregistration.SaveImage(output_path, output)
        
    def test_16Bit_to_8Bit_ConvertImage(self):
        filename = '10001_RPC2_590.tif'
        self.FixedImagePath = os.path.join(self.ImportedDataPath,'16-bit', filename)
        basename = os.path.basename(filename)
        output_path = os.path.join(self.TestOutputPath,  basename + '_converted.tif')
                
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input: {0}".format(self.FixedImagePath))
        output = self.RunConvertImageTest(self.FixedImagePath)
        
        nornir_imageregistration.SaveImage(output_path, output)
        
    def test_16Bit_SaveImage(self):
        input_image_fullpath = os.path.join(self.ImportedDataPath, '16-bit', "10000.tif")
        self.RunSaveLoadImageTest(input_image_fullpath, 
                                  expected_input_properties=ImageProperties(16, numpy.int16, '.tif'), 
                                  expected_output_properties=ImageProperties(16, numpy.uint16, '.png'))
        
        self.RunSaveLoadImageTest(input_image_fullpath, 
                                  expected_input_properties=ImageProperties(16, numpy.uint16, '.tif'), 
                                  expected_output_properties=ImageProperties(16, numpy.uint16, '.tif'))
        
    def test_16Bit_to_8Bit_SaveImage(self):
        input_image_fullpath = os.path.join(self.ImportedDataPath, '16-bit', "10000.tif")
        self.RunSaveLoadImageTest(input_image_fullpath, 
                                  expected_input_properties=ImageProperties(16, numpy.uint16, '.tif'), 
                                  expected_output_properties=ImageProperties(8, numpy.uint8, '.png'))
        
        self.RunSaveLoadImageTest(input_image_fullpath, 
                                  expected_input_properties=ImageProperties(16, numpy.uint16, '.tif'), 
                                  expected_output_properties=ImageProperties(8, numpy.uint8, '.tif'))
        
    def test_8Bit_SaveImage(self):
        input_image_fullpath = os.path.join(self.ImportedDataPath, "BrightfieldShading.png")
        self.RunSaveLoadImageTest(input_image_fullpath, 
                                  expected_input_properties=ImageProperties(8,numpy.uint16, '.png'), 
                                  expected_output_properties=ImageProperties(8,numpy.uint16, '.png'))
        
        self.RunSaveLoadImageTest(input_image_fullpath, 
                                  expected_input_properties=ImageProperties(8,numpy.uint16, '.png'), 
                                  expected_output_properties=ImageProperties(8,numpy.uint16, '.tif'))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()