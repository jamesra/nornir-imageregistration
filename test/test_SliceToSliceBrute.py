'''
Created on Mar 21, 2013

@author: u0490822
'''
import logging
import os
import unittest

from nornir_imageregistration import alignment_record
import nornir_imageregistration
import nornir_imageregistration.files
import nornir_imageregistration.transforms

import nornir_imageregistration.core as core
import nornir_imageregistration.scripts.nornir_rotate_translate
import nornir_imageregistration.stos_brute as stos_brute
import nornir_shared.images as images

from . import setup_imagetest


def CheckAlignmentRecord(test, arecord, angle, X, Y, flipud=False, adelta=None, sdelta=None):
        '''Verifies that an alignment record is more or less equal to expected values'''

        angle = float(angle)
        X = float(X)
        Y = float(Y)

        if adelta is None:
            adelta = 1.0
        if sdelta is None:
            sdelta = 2.0

        test.assertIsNotNone(arecord)
        test.assertEqual(arecord.flippedud, flipud, "Flip Up/Down mismatch: %s" % str(arecord))
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

        print(str(rec))


class TestStosBrute(setup_imagetest.ImageTestBase):



    def testStosBrute(self):
        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.RunBasicBruteAlignment(FixedImagePath, WarpedImagePath, FlipUD=False)

    def testStosBruteWithFlip(self):
        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8_flippedud.png")
        FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.RunBasicBruteAlignment(FixedImagePath, WarpedImagePath, FlipUD=True)

    def RunBasicBruteAlignment(self, FixedImagePath, WarpedImagePath, FlipUD):

        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        # In photoshop the correct transform is X: -4  Y: 22 Angle: 132

        # Check both clustered and non-clustered output
        AlignmentRecord = stos_brute.SliceToSliceBruteForce(FixedImagePath,
                               WarpedImagePath, SingleThread=True, AngleSearchRange=list(range(-140, -130)))

        self.Logger.info("Best alignment: " + str(AlignmentRecord))
        CheckAlignmentRecord(self, AlignmentRecord, angle=-132.0, X=-4, Y=22, flipud=FlipUD)

        # Check both clustered and non-clustered output
        AlignmentRecord = stos_brute.SliceToSliceBruteForce(FixedImagePath,
                               WarpedImagePath, SingleThread=False, Cluster=True)

        self.Logger.info("Best alignment: " + str(AlignmentRecord))
        CheckAlignmentRecord(self, AlignmentRecord, angle=-132.0, X=-4, Y=22, flipud=FlipUD)

        AlignmentRecord = stos_brute.SliceToSliceBruteForce(FixedImagePath,
                               WarpedImagePath)

        self.Logger.info("Best alignment: " + str(AlignmentRecord))
        CheckAlignmentRecord(self, AlignmentRecord, angle=-132.0, X=-4, Y=22, flipud=FlipUD)

        # OK, try to save the stos file and reload it.  Make sure the transforms match
        savedstosObj = AlignmentRecord.ToStos(FixedImagePath, WarpedImagePath, PixelSpacing=1)
        self.assertIsNotNone(savedstosObj)

        FixedSize = core.GetImageSize(FixedImagePath)
        WarpedSize = core.GetImageSize(WarpedImagePath)

        alignmentTransform = AlignmentRecord.ToTransform(FixedSize, WarpedSize)

        if FlipUD:
            stosfilepath = os.path.join(self.VolumeDir, '17-18_brute_flipped.stos')
        else:
            stosfilepath = os.path.join(self.VolumeDir, '17-18_brute.stos')

        savedstosObj.Save(stosfilepath)

        loadedStosObj = nornir_imageregistration.files.stosfile.StosFile.Load(stosfilepath)
        self.assertIsNotNone(loadedStosObj)

        self.assertFalse(loadedStosObj.HasMasks, "Stos file saved without masks should return false in HasMasks check")
        self.assertIsNone(loadedStosObj.ControlMaskName, "Mask in .stos does not match mask used in alignment\n")
        self.assertIsNone(loadedStosObj.MappedMaskName, "Mask in .stos does not match mask used in alignment\n")

        loadedTransform = nornir_imageregistration.transforms.factory.LoadTransform(loadedStosObj.Transform)
        self.assertIsNotNone(loadedTransform)


    def testStosBruteWithMask(self):
        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        controlMaskName = "mini_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png"
        warpedMaskName = "0017_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png"

        WarpedImageMaskPath = os.path.join(self.ImportedDataPath, warpedMaskName)
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImageMaskPath = os.path.join(self.ImportedDataPath, controlMaskName)
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        AlignmentRecord = stos_brute.SliceToSliceBruteForce(FixedImagePath,
                               WarpedImagePath,
                               FixedImageMaskPath,
                               WarpedImageMaskPath)

        self.Logger.info("Best alignment: " + str(AlignmentRecord))
        CheckAlignmentRecord(self, AlignmentRecord, angle=-132.0, X=-4, Y=22)

        savedstosObj = AlignmentRecord.ToStos(FixedImagePath, WarpedImagePath, FixedImageMaskPath, WarpedImageMaskPath, PixelSpacing=1)
        self.assertIsNotNone(savedstosObj)

        stosfilepath = os.path.join(self.VolumeDir, '17-18_brute_WithMask.stos')
        savedstosObj.Save(stosfilepath)

        loadedStosObj = nornir_imageregistration.files.stosfile.StosFile.Load(stosfilepath)
        self.assertIsNotNone(loadedStosObj)

        self.assertTrue(loadedStosObj.HasMasks, ".stos file is expected to have masks")

        self.assertEqual(loadedStosObj.ControlMaskName, controlMaskName, "Mask in .stos does not match mask used in alignment\n")
        self.assertEqual(loadedStosObj.MappedMaskName, warpedMaskName, "Mask in .stos does not match mask used in alignment\n")

    def testStosBruteExecute(self):

        WarpedImagePath = os.path.join(self.ImportedDataPath, "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImagePath = os.path.join(self.ImportedDataPath, "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        controlMaskName = "mini_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png"
        warpedMaskName = "0017_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png"

        WarpedImageMaskPath = os.path.join(self.ImportedDataPath, warpedMaskName)
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImageMaskPath = os.path.join(self.ImportedDataPath, controlMaskName)
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        stosfilepath = os.path.join(self.VolumeDir, '17-18_brute_WithMask.stos')


        nornir_imageregistration.scripts.nornir_rotate_translate.Execute(ExecArgs=['-f', FixedImagePath,
                                                                             '-w', WarpedImagePath,
                                                                             '-fm', FixedImageMaskPath,
                                                                             '-wm', WarpedImageMaskPath,
                                                                             '-o', stosfilepath])

        self.assertTrue(os.path.exists(stosfilepath), "Stos brute script should create output")



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
