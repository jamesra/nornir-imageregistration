'''
Created on Mar 12, 2013

@author: u0490822
'''
import os
import unittest

from PIL import Image

import nornir_pools as pools
import nornir_shared.images
import nornir_shared.plot as plot
import setup_imagetest
from nornir_imageregistration import im_histogram_parser
from nornir_imageregistration import image_stats


class ImageStatsBase(setup_imagetest.ImageTestBase):

    def setUp(self):
        super(ImageStatsBase, self).setUp()

        self.ImagePath16bpp = os.path.join(self.TestInputPath, "PlatformRaw", "IDoc", "RC2_Micro", "17")
        self.ImagePath8bpp = os.path.join(self.TestInputPath, "Images", "Alignment")


class testHistogram(ImageStatsBase):

    def HistogramFromFileImageMagick(self, File):
        '''Create a histogram for a file, put FilePrefix in front of any files written'''
        Bpp = nornir_shared.images.GetImageBpp(File)
        # maxVal = (1 << Bpp) - 1; 
        maxVal = None
        # Scale = 0.125
        Scale = 1

        numBins = None

        if Bpp == 8:
            numBins = 256
        else:
            numBins = 2048

        self.assertTrue(os.path.exists(File), "Input image missing " + File)

        task = image_stats.__HistogramFileImageMagick__(File, ProcPool=pools.GetGlobalProcessPool(), Bpp=Bpp,
                                                        Scale=Scale)
        taskOutput = task.wait_return()

        self.assertIsNotNone(taskOutput, "No output from __HistogramFileImageMagick__ on image")
        taskOutput = taskOutput.splitlines()

        self.assertGreater(len(taskOutput), 0, "No output from histograming image")
        histogram = im_histogram_parser.Parse(taskOutput, maxVal=maxVal, numBins=numBins)

        self.assertEqual(histogram.NumBins, numBins)

        return histogram

    def HistogramFromFileSciPy(self, File):
        '''Create a histogram for a file, put FilePrefix in front of any files written'''
        Bpp = nornir_shared.images.GetImageBpp(File)
        Scale = 0.125
        numBins = 2048
        if (Bpp == 8):
            numBins = 256
        else:
            numBins = 2048

        self.assertTrue(os.path.exists(File), "Input image missing " + File)

        Im = None
        MinVal = None
        MaxVal = None
        with Image.open(File, mode='r') as img:
            img_I = img.convert("I")
            (MinVal, MaxVal) = img.getextrema()

        histogram = image_stats.__HistogramFileSciPy__(File, Bpp=Bpp, numBins=numBins, MinVal=MinVal, MaxVal=MaxVal)

        self.assertIsNotNone(histogram, "No output from __HistogramFileSciPy__ on image")

        return histogram

    def SaveHistogram(self, histogram, FilePrefix):

        HistogramDataFullPath = os.path.join(self.VolumeDir, FilePrefix + '_Histogram.xml')
        histogram.Save(HistogramDataFullPath)
        self.assertTrue(os.path.exists(HistogramDataFullPath))

        HistogramImageFullPath = os.path.join(self.VolumeDir, FilePrefix + '_Histogram.png')

        plot.Histogram(HistogramDataFullPath, HistogramImageFullPath)

        self.assertTrue(os.path.exists(HistogramImageFullPath))

    def testHistogramParse16bpp(self):
        tileAFullPath = os.path.join(self.ImagePath16bpp, '10000.tif')
        tileBFullPath = os.path.join(self.ImagePath16bpp, '10016.tif')

        histA = self.HistogramFromFileImageMagick(tileAFullPath)
        self.assertIsNotNone(histA)
        self.assertEqual(histA.MinValue, 290)
        self.assertEqual(histA.MaxValue, 8853)
        self.SaveHistogram(histA, 'A')

        #    Pillow and matplotlib do not like reading our 16-bit tif files
        #         histA_Scipy = self.HistogramFromFileSciPy(tileAFullPath)
        #         self.assertIsNotNone(histA_Scipy)
        #         self.assertEqual(histA_Scipy.MinValue, 348)
        #         self.assertEqual(histA_Scipy.MaxValue, 7934)
        #         self.SaveHistogram(histA_Scipy, 'A')

        histB = self.HistogramFromFileImageMagick(tileBFullPath)
        self.assertIsNotNone(histB)
        self.assertEqual(histB.MinValue, 317)
        self.assertEqual(histB.MaxValue, 7912)
        # Can we add them together?
        self.SaveHistogram(histB, 'B')

        #    Pillow and matplotlib do not like reading our 16-bit tif files
        #         histB_Scipy = self.HistogramFromFileSciPy(tileBFullPath);
        #         self.assertIsNotNone(histB_Scipy)
        #         self.assertEqual(histB_Scipy.MinValue, 404)
        #         self.assertEqual(histB_Scipy.MaxValue, 7384)
        #         # Can we add them together?
        #         self.SaveHistogram(histB_Scipy, 'B')

        HistogramComposite = image_stats.Histogram([tileAFullPath, tileBFullPath], Scale=1, Bpp=16)
        self.assertEqual(HistogramComposite.MinValue, min(histA.MinValue, histB.MinValue))
        self.assertEqual(HistogramComposite.MaxValue, max(histA.MaxValue, histB.MaxValue))

        # We know that histA has the lower value, so our first bin value should match
        self.assertEqual(HistogramComposite.Bins[0], histA.Bins[0])

        self.SaveHistogram(HistogramComposite, 'Composite')

    def testHistogramParse8bpp(self):
        '''TODO 12/17/2014 This test for 8bpp histogramming needs review.  The Scipy, Photoshop, and ImageMagick results do not agree.
           These are being checkin in because the tests did not previously exist and the failure should be noted'''

        tileAFullPath = os.path.join(self.ImagePath8bpp, '401.png')
        tileBFullPath = os.path.join(self.ImagePath8bpp, '402.png')

        histA_Scipy = self.HistogramFromFileSciPy(tileAFullPath)
        self.assertIsNotNone(histA_Scipy)
        self.assertEqual(histA_Scipy.MinValue, 0)
        self.assertEqual(histA_Scipy.MaxValue, 255)

        histA = self.HistogramFromFileImageMagick(tileAFullPath)
        self.assertIsNotNone(histA)
        self.assertEqual(histA.MinValue, 0)
        self.assertEqual(histA.MaxValue, 255)
        self.SaveHistogram(histA, 'A')

        for i, binVal in enumerate(histA.Bins):
            assert binVal == histA_Scipy.Bins[i], "Histogram A Bin {0} has mismatched values {1} vs {2}".format(i,
                                                                                                                binVal,
                                                                                                                histA_Scipy.Bins[
                                                                                                                    i])

        self.assertEqual(histA.MinValue, histA_Scipy.MinValue)
        self.assertEqual(histA.MaxValue, histA_Scipy.MaxValue)

        histB_Scipy = self.HistogramFromFileSciPy(tileBFullPath)
        self.assertIsNotNone(histB_Scipy)
        self.assertEqual(histB_Scipy.MinValue, 0)
        self.assertEqual(histB_Scipy.MaxValue, 255)

        # Can we add them together? 

        histB = self.HistogramFromFileImageMagick(tileBFullPath)
        self.assertIsNotNone(histB)
        self.assertEqual(histA.MinValue, 0)
        self.assertEqual(histA.MaxValue, 255)
        # Can we add them together?
        self.SaveHistogram(histB, 'B')

        for i, binVal in enumerate(histB.Bins):
            assert binVal == histB_Scipy.Bins[i], "Histogram B Bin {0} has mismatched values {1} vs {2}".format(i,
                                                                                                                binVal,
                                                                                                                histB_Scipy.Bins[
                                                                                                                    i])

        self.assertEqual(histB.MinValue, histB_Scipy.MinValue)
        self.assertEqual(histB.MaxValue, histB_Scipy.MaxValue)

        HistogramComposite = image_stats.Histogram([tileAFullPath, tileBFullPath], Scale=0.125, Bpp=16)
        self.assertEqual(HistogramComposite.MinValue, min(histA.MinValue, histB.MinValue))
        self.assertEqual(HistogramComposite.MaxValue, max(histA.MaxValue, histB.MaxValue))

        TestComposite = histA
        TestComposite.AddHistogram(histB_Scipy)

        for i, binVal in enumerate(TestComposite.Bins):
            assert binVal == HistogramComposite.Bins[i], "Composite Bin {0} has mismatched values {1} vs {2}".format(i,
                                                                                                                     binVal,
                                                                                                                     HistogramComposite.Bins[
                                                                                                                         i])

        # We know that histA has the lower value, so our first bin value should match
        self.assertEqual(HistogramComposite.Bins[0], histA.Bins[0])

        self.SaveHistogram(HistogramComposite, 'Composite')

    def testHistogram16bpp(self):
        tileAFullPath = os.path.join(self.ImagePath16bpp, '10000.tif')
        tileBFullPath = os.path.join(self.ImagePath16bpp, '10016.tif')

        InputImagePath16bpp = os.path.join(self.TestInputPath, "Images", "RPC2", "0794", "10001.tif")

        histA_Scipy = self.HistogramFromFileSciPy(InputImagePath16bpp)
        self.assertIsNotNone(histA_Scipy)

        histA = self.HistogramFromFileImageMagick(InputImagePath16bpp)
        self.assertIsNotNone(histA)

        self.assertEqual(len(histA.Bins), len(histA_Scipy.Bins))
        self.assertEqual(histA.MinValue, histA_Scipy.MinValue)
        self.assertEqual(histA.MaxValue, histA_Scipy.MaxValue)

        for i, binVal in enumerate(histA.Bins):
            assert binVal == histA_Scipy.Bins[i], "Histogram A Bin {0} has mismatched values {1} vs {2}".format(i,
                                                                                                                binVal,
                                                                                                                histA_Scipy.Bins[
                                                                                                                    i])

        plot.Histogram(histA_Scipy, os.path.join(self.TestOutputPath, "RPC2_0794_Scipy.png"), 1, 99,
                       Title="Scipy Histogram for RPC2 794 001.tif")
        plot.Histogram(histA, os.path.join(self.TestOutputPath, "RPC2_0794_ImageMagick.png"), 1, 99,
                       Title="ImageMagick Histogram for RPC2 794 001.tif")

    def testPrune(self):
        '''Create a histogram for a file, put FilePrefix in front of any files written'''
        File = os.path.join(self.ImagePath8bpp, '401.png')

        self.assertTrue(os.path.exists(File), "Input image missing " + File)

        score = image_stats.__PruneFileSciPy__(File, overlap=0.15)

        self.assertGreater(score, 0, msg="Non-zero score expected for prune score")
        return


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
