'''
Created on Mar 21, 2013

@author: u0490822
'''
import unittest
import os
import nornir_imageregistration.core as core
import logging
import setup_imagetest
import nornir_shared.images as images
import nornir_imageregistration.stos_brute as stos_brute
from nornir_imageregistration import alignment_record
import nornir_imageregistration
import nornir_imageregistration.files
import nornir_imageregistration.transforms

def CheckAlignmentRecord(test, arecord, angle, X, Y, adelta=None, sdelta=None):
        '''Verifies that an alignment record is more or less equal to expected values'''

        angle = float(angle)
        X = float(X)
        Y = float(Y)

        if adelta is None:
            adelta = 1.0
        if sdelta is None:
            sdelta = 1.5

        test.assertIsNotNone(arecord)
        test.assertAlmostEqual(arecord.angle, angle, msg="Wrong angle found: %s" % str(arecord) , delta=adelta)
        test.assertAlmostEqual(arecord.peak[1], X, msg="Wrong X offset: %s" % str(arecord), delta=sdelta)
        test.assertAlmostEqual(arecord.peak[0], Y, msg="Wrong Y offset: %s" % str(arecord), delta=sdelta)

class TestStos(setup_imagetest.ImageTestBase):

    def testStosWrite(self):
        InputDir = 'C:\\Buildscript\\Test\\images\\'
        OutputDir = 'C:\\Temp\\'

        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        peak = (-4.4, 22.41)
        # peak = (0,0)
        # imWarpedSize = core.GetImageSize(WarpedImagePath)
        # imFixedSize = core.GetImageSize(FixedImagePath)
        # peak = (peak[0] - ((imWarpedSize[0] - imFixedSize[0])/2), peak[1] - ((imWarpedSize[1] - imFixedSize[1])/2))

        rec = alignment_record.AlignmentRecord(peak, 1, 229.2)
        self.assertIsNotNone(rec)

        stosObj = rec.ToStos(FixedImagePath, WarpedImagePath, PixelSpacing=32)
        self.assertIsNotNone(stosObj)

        stosObj.Save(os.path.join(self.VolumeDir, '17-18_brute.stos'))

        print str(rec)


class TestStosBrute(setup_imagetest.ImageTestBase):



    def testStosBrute(self):

        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        # In photoshop the correct transform is X: -4  Y: 22 Angle: 132

        # Check both clustered and non-clustered output
        AlignmentRecord = stos_brute.SliceToSliceBruteForce(FixedImagePath,
                               WarpedImagePath, SingleThread=True, AngleSearchRange=range(-140, -130))

        self.Logger.info("Best alignment: " + str(AlignmentRecord))
        CheckAlignmentRecord(self, AlignmentRecord, angle=-132.0, X=-4, Y=22)

        # Check both clustered and non-clustered output
        AlignmentRecord = stos_brute.SliceToSliceBruteForce(FixedImagePath,
                               WarpedImagePath, SingleThread=False, Cluster=True)

        self.Logger.info("Best alignment: " + str(AlignmentRecord))
        CheckAlignmentRecord(self, AlignmentRecord, angle=-132.0, X=-4, Y=22)

        AlignmentRecord = stos_brute.SliceToSliceBruteForce(FixedImagePath,
                               WarpedImagePath)

        self.Logger.info("Best alignment: " + str(AlignmentRecord))
        CheckAlignmentRecord(self, AlignmentRecord, angle=-132.0, X=-4, Y=22)

        # OK, try to save the stos file and reload it.  Make sure the transforms match
        savedstosObj = AlignmentRecord.ToStos(FixedImagePath, WarpedImagePath, PixelSpacing=1)
        self.assertIsNotNone(savedstosObj)

        FixedSize = core.GetImageSize(FixedImagePath)
        WarpedSize = core.GetImageSize(WarpedImagePath)

        alignmentTransform = AlignmentRecord.ToTransform(FixedSize, WarpedSize)

        stosfilepath = os.path.join(self.VolumeDir, '17-18_brute.stos')

        savedstosObj.Save(stosfilepath)

        loadedStosObj = nornir_imageregistration.files.stosfile.StosFile.Load(stosfilepath)
        self.assertIsNotNone(loadedStosObj)

        loadedTransform = nornir_imageregistration.transforms.factory.LoadTransform(loadedStosObj.Transform)
        self.assertIsNotNone(loadedTransform)


    def testStosBruteWithMask(self):
        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        WarpedImageMaskPath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImageMaskPath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        AlignmentRecord = stos_brute.SliceToSliceBruteForce(FixedImagePath,
                               WarpedImagePath,
                               FixedImageMaskPath,
                               WarpedImageMaskPath)

        self.Logger.info("Best alignment: " + str(AlignmentRecord))
        CheckAlignmentRecord(self, AlignmentRecord, angle=-132.0, X=-4, Y=22)

class TestStosBruteToSameImage(setup_imagetest.ImageTestBase):

#    def testSameSimpleImage(self):
#        '''Make sure the same image aligns to itself with peak (0,0) and angle 0'''
#        FixedImagePath = os.path.join(self.ImportedDataPath, "fixed.png")
#        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")
#        FixedImageMaskPath = os.path.join(self.ImportedDataPath, "fixedmask.png")
#        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")
#
#        AlignmentRecord = stos_brute.SliceToSliceBruteForce(FixedImagePath, FixedImagePath)
#
#        CheckAlignmentRecord(self, AlignmentRecord, angle = 0.0, X = 0, Y = 0)
#
#
#    def testSameSimpleImageWithMask(self):
#        '''Make sure the same image aligns to itself with peak (0,0) and angle 0'''
#        FixedImagePath = os.path.join(self.ImportedDataPath, "fixed.png")
#        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")
#        FixedImageMaskPath = os.path.join(self.ImportedDataPath, "fixedmask.png")
#        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")
#
#        AlignmentRecord = stos_brute.SliceToSliceBruteForce(FixedImagePath,
#                       FixedImagePath,
#                       FixedImageMaskPath,
#                       FixedImageMaskPath)
#        CheckAlignmentRecord(self, AlignmentRecord, angle = 0.0, X = 0, Y = 0)

    def testSameTEMImage(self):
        '''Make sure the same image aligns to itself with peak (0,0) and angle 0'''
        FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")
        FixedImageMaskPath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        AlignmentRecord = stos_brute.SliceToSliceBruteForce(FixedImagePath,
                               FixedImagePath,
                               FixedImageMaskPath,
                               FixedImageMaskPath)
        CheckAlignmentRecord(self, AlignmentRecord, angle=0.0, X=0, Y=0, adelta=1.5)

if __name__ == "__main__":
    # import syssys.argv = ['', 'Test.testName']
    unittest.main()