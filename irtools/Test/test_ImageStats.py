'''
Created on Mar 12, 2013

@author: u0490822
'''
import unittest
import irtools
import os
import shutil
import logging
import setup_imagetest
import pools
import utils.misc
import PlotHistogram
from irtools import im_histogram_parser
from irtools import image_stats

class ImageStatsBase(setup_imagetest.ImageTestBase):

    def setUp(self):

        super(ImageStatsBase, self).setUp()

        self.TestDataSource = os.path.join(os.getcwd(), "Test", "Data", "PlatformRaw", "IDoc", "17");


class testHistogram(ImageStatsBase):

    def HistogramFromFile(self, File):
        '''Create a histogram for a file, put FilePrefix in front of any files written'''
        Bpp = 16
        # maxVal = (1 << Bpp) - 1;
        maxVal = None;
        Scale = 0.125
        numBins = 2048

        task = image_stats.__HistogramFileImageMagick__(File, ProcPool=pools.GetGlobalProcessPool(), Bpp=Bpp, Scale=Scale)
        taskOutput = task.wait_return();
        taskOutput = taskOutput.splitlines();
        histogram = im_histogram_parser.Parse(taskOutput, maxVal = maxVal, numBins = numBins)

        self.assertEqual(histogram.NumBins, numBins)

        return histogram


    def SaveHistogram(self, histogram, FilePrefix):

        HistogramDataFullPath = os.path.join(self.VolumeDir, FilePrefix + '_Histogram.xml');
        histogram.Save(HistogramDataFullPath);
        self.assertTrue(os.path.exists(HistogramDataFullPath));

        HistogramImageFullPath = os.path.join(self.VolumeDir, FilePrefix + '_Histogram.png');

        PlotHistogram.PlotHistogram(HistogramDataFullPath, HistogramImageFullPath);

        self.assertTrue(os.path.exists(HistogramImageFullPath));


    def testHistogramParse(self):
        tileAFullPath = os.path.join(self.TestDataSource, '10000.tif');
        tileBFullPath = os.path.join(self.TestDataSource, '10016.tif');

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

        HistogramComposite = image_stats.Histogram([tileAFullPath, tileBFullPath], Scale = 0.125, Bpp = 16);
        self.assertEqual(HistogramComposite.MinValue, min(histA.MinValue, histB.MinValue))
        self.assertEqual(HistogramComposite.MaxValue, max(histA.MaxValue, histB.MaxValue))

        # We know that histA has the lower value, so our first bin value should match
        self.assertEqual(HistogramComposite.Bins[0], histA.Bins[0])

        self.SaveHistogram(HistogramComposite, 'Composite');


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()