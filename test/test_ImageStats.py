'''
Created on Mar 12, 2013

@author: u0490822
'''
import unittest
import nornir_imageregistration
import os
import shutil
import logging
import setup_imagetest
import nornir_pools as pools
import nornir_shared.plot as plot
from nornir_imageregistration import im_histogram_parser
from nornir_imageregistration import image_stats

class ImageStatsBase(setup_imagetest.ImageTestBase):

    def setUp(self):

        super(ImageStatsBase, self).setUp()

        self.ImagePath = os.path.join(self.TestInputPath, "PlatformRaw", "IDoc", "RC2_Micro", "17");


class testHistogram(ImageStatsBase):

    def HistogramFromFile(self, File):
        '''Create a histogram for a file, put FilePrefix in front of any files written'''
        Bpp = 16
        # maxVal = (1 << Bpp) - 1;
        maxVal = None;
        Scale = 0.125
        numBins = 2048

        self.assertTrue(os.path.exists(File), "Input image missing " + File)

        task = image_stats.__HistogramFileImageMagick__(File, ProcPool=pools.GetGlobalProcessPool(), Bpp=Bpp, Scale=Scale)
        taskOutput = task.wait_return();

        self.assertIsNotNone(taskOutput, "No output from histograming image")
        taskOutput = taskOutput.splitlines();

        self.assertGreater(len(taskOutput), 0, "No output from histograming image")
        histogram = im_histogram_parser.Parse(taskOutput, maxVal=maxVal, numBins=numBins)

        self.assertEqual(histogram.NumBins, numBins)

        return histogram

    def SaveHistogram(self, histogram, FilePrefix):

        HistogramDataFullPath = os.path.join(self.VolumeDir, FilePrefix + '_Histogram.xml');
        histogram.Save(HistogramDataFullPath);
        self.assertTrue(os.path.exists(HistogramDataFullPath));

        HistogramImageFullPath = os.path.join(self.VolumeDir, FilePrefix + '_Histogram.png');

        plot.Histogram(HistogramDataFullPath, HistogramImageFullPath);

        self.assertTrue(os.path.exists(HistogramImageFullPath));

    def testHistogramParse(self):
        tileAFullPath = os.path.join(self.ImagePath, '10000.tif');
        tileBFullPath = os.path.join(self.ImagePath, '10016.tif');

        histA = self.HistogramFromFile(tileAFullPath)
        self.assertIsNotNone(histA)
        self.assertEqual(histA.MinValue, 348)
        self.assertEqual(histA.MaxValue, 7934)
        self.SaveHistogram(histA, 'A')

        histB = self.HistogramFromFile(tileBFullPath);
        self.assertIsNotNone(histB)
        self.assertEqual(histB.MinValue, 404)
        self.assertEqual(histB.MaxValue, 7384)
        # Can we add them together?
        self.SaveHistogram(histB, 'B')

        HistogramComposite = image_stats.Histogram([tileAFullPath, tileBFullPath], Scale=0.125, Bpp=16);
        self.assertEqual(HistogramComposite.MinValue, min(histA.MinValue, histB.MinValue))
        self.assertEqual(HistogramComposite.MaxValue, max(histA.MaxValue, histB.MaxValue))

        # We know that histA has the lower value, so our first bin value should match
        self.assertEqual(HistogramComposite.Bins[0], histA.Bins[0])

        self.SaveHistogram(HistogramComposite, 'Composite');


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()