'''
Created on Mar 12, 2013

@author: u0490822
'''
import logging
import os
import shutil
import unittest

from nornir_imageregistration import im_histogram_parser
from nornir_imageregistration import image_stats
import nornir_imageregistration

import nornir_imageregistration.core as core
import nornir_pools as pools
import nornir_shared.histogram as histogram
import nornir_shared.plot as plot
import nornir_shared.images 

from . import setup_imagetest


class ImageStatsBase(setup_imagetest.ImageTestBase):

    def setUp(self):

        super(ImageStatsBase, self).setUp()

        self.ImagePath16bpp = os.path.join(self.TestInputPath, "PlatformRaw", "IDoc", "RC2_Micro", "17");
        self.ImagePath8bpp = os.path.join(self.TestInputPath, "Images", "Alignment");


class testHistogram(ImageStatsBase):

    def HistogramFromFileImageMagick(self, File):
        '''Create a histogram for a file, put FilePrefix in front of any files written'''
        Bpp = nornir_shared.images.GetImageBpp(File)
        # maxVal = (1 << Bpp) - 1; 
        maxVal = None;
        Scale = 0.125
        numBins = 2048

        self.assertTrue(os.path.exists(File), "Input image missing " + File)

        task = image_stats.__HistogramFileImageMagick__(File, ProcPool=pools.GetGlobalProcessPool(), Bpp=Bpp, Scale=Scale)
        taskOutput = task.wait_return();

        self.assertIsNotNone(taskOutput, "No output from __HistogramFileImageMagick__ on image")
        taskOutput = taskOutput.splitlines();

        self.assertGreater(len(taskOutput), 0, "No output from histograming image")
        histogram = im_histogram_parser.Parse(taskOutput, maxVal=maxVal, numBins=numBins)

        self.assertEqual(histogram.NumBins, numBins)

        return histogram
    
    def HistogramFromFileSciPy(self, File):
        '''Create a histogram for a file, put FilePrefix in front of any files written'''
        Bpp = nornir_shared.images.GetImageBpp(File)
        # maxVal = (1 << Bpp) - 1;
        maxVal = None;
        Scale = 0.125
        numBins = 2048
        if(Bpp == 8):
            numBins = 256

        self.assertTrue(os.path.exists(File), "Input image missing " + File)

        histogram = image_stats.__HistogramFileSciPy__(File, Bpp=Bpp, numBins=numBins)
         
        self.assertIsNotNone(histogram, "No output from __HistogramFileSciPy__ on image") 
  
        return histogram
    
    def SaveHistogram(self, histogram, FilePrefix):

        HistogramDataFullPath = os.path.join(self.VolumeDir, FilePrefix + '_Histogram.xml');
        histogram.Save(HistogramDataFullPath);
        self.assertTrue(os.path.exists(HistogramDataFullPath));

        HistogramImageFullPath = os.path.join(self.VolumeDir, FilePrefix + '_Histogram.png');

        plot.Histogram(HistogramDataFullPath, HistogramImageFullPath);

        self.assertTrue(os.path.exists(HistogramImageFullPath));

    def testHistogramParse16bpp(self):
        tileAFullPath = os.path.join(self.ImagePath16bpp, '10000.tif');
        tileBFullPath = os.path.join(self.ImagePath16bpp, '10016.tif');

        histA = self.HistogramFromFileImageMagick(tileAFullPath)
        self.assertIsNotNone(histA)
        self.assertEqual(histA.MinValue, 348)
        self.assertEqual(histA.MaxValue, 7934)
        self.SaveHistogram(histA, 'A')
        
#    Pillow and matplotlib do not like reading our 16-bit tif files
#         histA_Scipy = self.HistogramFromFileSciPy(tileAFullPath)
#         self.assertIsNotNone(histA_Scipy)
#         self.assertEqual(histA_Scipy.MinValue, 348)
#         self.assertEqual(histA_Scipy.MaxValue, 7934)
#         self.SaveHistogram(histA_Scipy, 'A')

        histB = self.HistogramFromFileImageMagick(tileBFullPath);
        self.assertIsNotNone(histB)
        self.assertEqual(histB.MinValue, 404)
        self.assertEqual(histB.MaxValue, 7384)
        # Can we add them together?
        self.SaveHistogram(histB, 'B')
        
        #    Pillow and matplotlib do not like reading our 16-bit tif files
#         histB_Scipy = self.HistogramFromFileSciPy(tileBFullPath);
#         self.assertIsNotNone(histB_Scipy)
#         self.assertEqual(histB_Scipy.MinValue, 404)
#         self.assertEqual(histB_Scipy.MaxValue, 7384)
#         # Can we add them together?
#         self.SaveHistogram(histB_Scipy, 'B')

        HistogramComposite = image_stats.Histogram([tileAFullPath, tileBFullPath], Scale=0.125, Bpp=16);
        self.assertEqual(HistogramComposite.MinValue, min(histA.MinValue, histB.MinValue))
        self.assertEqual(HistogramComposite.MaxValue, max(histA.MaxValue, histB.MaxValue))

        # We know that histA has the lower value, so our first bin value should match
        self.assertEqual(HistogramComposite.Bins[0], histA.Bins[0])

        self.SaveHistogram(HistogramComposite, 'Composite');
        
    def testHistogramParse8bpp(self):
        '''TODO 12/17/2014 This test for 8bpp histogramming needs review.  The Scipy, Photoshop, and ImageMagick results do not agree.
           These are being checkin in because the tests did not previously exist and the failure should be noted'''
        
        tileAFullPath = os.path.join(self.ImagePath8bpp, '401.png');
        tileBFullPath = os.path.join(self.ImagePath8bpp, '402.png');
 
        histA_Scipy = self.HistogramFromFileSciPy(tileAFullPath)
        self.assertIsNotNone(histA_Scipy)
        self.assertEqual(histA_Scipy.MinValue, 0)
        self.assertEqual(histA_Scipy.MaxValue, 1) 
        
        histA = self.HistogramFromFileImageMagick(tileAFullPath)
        self.assertIsNotNone(histA)
        self.assertEqual(histA.MinValue, 0)
        self.assertEqual(histA.MaxValue, 247)
        self.SaveHistogram(histA, 'A')
        
        self.assertEqual(histA.MinValue, histA_Scipy.MinValue)
        self.assertEqual(histA.MaxValue, histA_Scipy.MaxValue)
        
        histB_Scipy = self.HistogramFromFileSciPy(tileBFullPath);
        self.assertIsNotNone(histB_Scipy)
        self.assertEqual(histB_Scipy.MinValue, 0)
        self.assertEqual(histB_Scipy.MaxValue, 1)
         
        # Can we add them together? 

        histB = self.HistogramFromFileImageMagick(tileBFullPath);
        self.assertIsNotNone(histB)
        self.assertEqual(histA.MinValue, 0)
        self.assertEqual(histA.MaxValue, 247)
        # Can we add them together?
        self.SaveHistogram(histB, 'B')
        
        self.assertEqual(histB.MinValue, histB_Scipy.MinValue)
        self.assertEqual(histB.MaxValue, histB_Scipy.MaxValue)
          
        HistogramComposite = image_stats.Histogram([tileAFullPath, tileBFullPath], Scale=0.125, Bpp=16);
        self.assertEqual(HistogramComposite.MinValue, min(histA.MinValue, histB.MinValue))
        self.assertEqual(HistogramComposite.MaxValue, max(histA.MaxValue, histB.MaxValue))

        # We know that histA has the lower value, so our first bin value should match
        self.assertEqual(HistogramComposite.Bins[0], histA.Bins[0])

        self.SaveHistogram(HistogramComposite, 'Composite');
    
    def testPrune(self):
        '''Create a histogram for a file, put FilePrefix in front of any files written'''
        File = os.path.join(self.ImagePath8bpp, '401.png');
        
        self.assertTrue(os.path.exists(File), "Input image missing " + File)

        (filename, score) = image_stats.__PruneFileSciPy__(File, overlap=0.15)
        
        self.assertGreater(score, 0, msg="Non-zero score expected for prune score")
        self.assertEqual(filename, File, "output filename should match input filename")
        return


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
