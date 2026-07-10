'''
Created on Mar 21, 2013

@author: u0490822
'''
import os
import unittest
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Check if cupy is available, and if it is not import thunks that refer to scipy/numpy
try:
    import cupy as cp
    import cupyx

    init_context = cp.zeros((64, 64))
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx

import scipy
from nornir_shared.tasktimer import TaskTimerContext

import nornir_imageregistration
from nornir_imageregistration.headless import inspect_png_output, is_headless
from nornir_imageregistration import AlignmentRecord, alignment_record
import nornir_imageregistration.core as core
import nornir_imageregistration.files
import nornir_imageregistration.scripts.nornir_rotate_translate
import nornir_imageregistration.stos_brute as stos_brute
import nornir_imageregistration.transforms
from nornir_imageregistration.settings import SliceToSliceMethod
from nornir_shared.tasktimer import TaskTimer
from mathfuncs.angles import are_angle_radians_equal, are_angle_degrees_equal, assert_angles_equal, \
    assert_angles_equal_degrees

# from . import setup_imagetest
import setup_imagetest


def CreateRotatedAndOffsetImage(image: np.ndarray | str,
                                mask: np.ndarray | str,
                                angle: float,
                                offset: tuple[int, int]) -> nornir_imageregistration.ImagePermutationHelper:
    input_image_data = nornir_imageregistration.ImagePermutationHelper(image, mask)

    if angle != 0:
        rotated_image = scipy.ndimage.rotate(input_image_data.Image.astype(np.float32), -angle, reshape=False)
        rotated_mask = scipy.ndimage.rotate(input_image_data.Mask, -angle, reshape=False)
    else:
        rotated_image = input_image_data.Image
        rotated_mask = input_image_data.Mask

    rotated_translated_image = nornir_imageregistration.CropImage(rotated_image,
                                                                  Xo=-offset[1],
                                                                  Yo=-offset[0],
                                                                  Width=input_image_data.Image.shape[1],
                                                                  Height=input_image_data.Image.shape[0],
                                                                  image_stats=input_image_data.Stats)

    rotated_translated_mask = nornir_imageregistration.CropImage(rotated_mask,
                                                                 Xo=-offset[1],
                                                                 Yo=-offset[0],
                                                                 Width=input_image_data.Image.shape[1],
                                                                 Height=input_image_data.Image.shape[0],
                                                                 image_stats=input_image_data.Stats)

    return nornir_imageregistration.ImagePermutationHelper(rotated_translated_image,
                                                           rotated_translated_mask)


def CheckAlignmentRecord(test: unittest.TestCase, arecord: alignment_record.AlignmentRecord, angle: float, X: float,
                         Y: float, flipud: bool = False, adelta: float | None = None, sdelta: float | None = None):
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
    test.assertAlmostEqual(arecord.angle, angle, msg="Wrong angle found: %s" % str(arecord), delta=adelta)
    test.assertAlmostEqual(arecord.peak[1], X, msg="Wrong X offset: %s" % str(arecord), delta=sdelta)
    test.assertAlmostEqual(arecord.peak[0], Y, msg="Wrong Y offset: %s" % str(arecord), delta=sdelta)


class TestStos(setup_imagetest.ImageTestBase):

    def testStosWrite(self):
        InputDir = 'C:\\Buildscriptd\\Test\\images\\'
        OutputDir = 'C:\\Temp\\'

        WarpedImagePath = os.path.join(self.ImportedDataPath,
                                       "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        FixedImagePath = os.path.join(self.ImportedDataPath,
                                      "mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
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

    def setUp(self):
        super(TestStosBrute, self).setUp()
        self.WarpedImagePath = self.GetImagePath("0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.FixedImagePath = self.GetImagePath("mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.WarpedImagePathFlipped = self.GetImagePath(
            "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8_FlippedUD.png")

    def testStosBrute_SingleThread(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.RunBasicBruteAlignment(self.FixedImagePath, self.WarpedImagePath, SingleThread=True, FlipUD=False)

    def testStosBrute_MultiThread(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.RunBasicBruteAlignment(self.FixedImagePath, self.WarpedImagePath, SingleThread=False, FlipUD=False)

    def testStosBrute_Cluster(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.RunBasicBruteAlignment(self.FixedImagePath, self.WarpedImagePath, SingleThread=False, Cluster=True,
                                    FlipUD=False)

    def testStosBrute_GPU(self):
        if not nornir_imageregistration.HasCupy():
            return
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        self.RunBasicBruteAlignment(self.FixedImagePath, self.WarpedImagePath, SingleThread=True, FlipUD=False)

    def testStosBruteWithFlip_SingleThread(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.RunBasicBruteAlignment(self.FixedImagePath, self.WarpedImagePathFlipped, SingleThread=True, FlipUD=True)

    def testStosBruteWithFlip_MultiThread(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.RunBasicBruteAlignment(self.FixedImagePath, self.WarpedImagePathFlipped, SingleThread=False, FlipUD=True)

    def testStosBruteWithFlip_Cluster(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.RunBasicBruteAlignment(self.FixedImagePath, self.WarpedImagePathFlipped, SingleThread=False, Cluster=True,
                                    FlipUD=True)

    def testStosBruteWithFlip_GPU(self):
        if not nornir_imageregistration.HasCupy():
            return

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        self.RunBasicBruteAlignment(self.FixedImagePath, self.WarpedImagePathFlipped, SingleThread=True, FlipUD=True)

    def RunBasicBruteAlignment(self, FixedImagePath: str,
                               WarpedImagePath: str,
                               FlipUD: bool = False,
                               SingleThread: bool = False,
                               Cluster: bool = False):

        use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")

        SingleThread = True if nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy else SingleThread

        MinOverlap = 0.5

        timer = TaskTimer()

        # In photoshop the correct transform is X: -4  Y: 22 Angle: 132

        timer.Start(f"\nSliceToSliceBrute No Mask - Cluster={Cluster} - SingleThread={SingleThread} - GPU={use_cp}")
        # Check both clustered and non-clustered output
        AlignmentRecord = stos_brute.SliceToSliceRigidRegistration(target_image=FixedImagePath,
                                                                   source_image=WarpedImagePath,
                                                                   SingleThread=SingleThread,
                                                                   AngleSearchRange=None,
                                                                   # AngleSearchRange=list(range(130, 140)),#AngleSearchRange=None, #
                                                                   TestFlip=FlipUD,
                                                                   MinOverlap=MinOverlap,
                                                                   Cluster=Cluster,
                                                                   method=nornir_imageregistration.settings.SliceToSliceMethod.BruteForce)

        self.Logger.info("Best alignment: " + str(AlignmentRecord))
        timer.End(f"\nSliceToSliceBrute No Mask - Cluster={Cluster} - SingleThread={SingleThread} - GPU={use_cp}")

        CheckAlignmentRecord(self, AlignmentRecord, angle=132.0, X=-4, Y=22, flipud=FlipUD)

        # OK, try to save the stos file and reload it.  Make sure the transforms match
        savedstosObj = AlignmentRecord.ToStos(FixedImagePath, WarpedImagePath, PixelSpacing=1)
        self.assertIsNotNone(savedstosObj)

        FixedSize = core.GetImageSize(FixedImagePath)
        WarpedSize = core.GetImageSize(WarpedImagePath)

        alignmentTransform = AlignmentRecord.ToImageTransform(source_image_shape=FixedSize,
                                                              target_image_shape=WarpedSize)

        if FlipUD:
            stosfilepath = os.path.join(self.VolumeDir, '17-18_brute_flipped.stos')
        else:
            stosfilepath = os.path.join(self.VolumeDir, '17-18_brute.stos')

        if os.path.isfile(stosfilepath):
            os.remove(stosfilepath)
        savedstosObj.Save(stosfilepath)

        loadedStosObj = nornir_imageregistration.files.stosfile.StosFile.Load(stosfilepath)
        self.assertIsNotNone(loadedStosObj)

        self.assertFalse(loadedStosObj.HasMasks, "Stos file saved without masks should return false in HasMasks check")
        self.assertIsNone(loadedStosObj.ControlMaskName, "Mask in .stos does not match mask used in alignment\n")
        self.assertIsNone(loadedStosObj.MappedMaskName, "Mask in .stos does not match mask used in alignment\n")

        loadedTransform = nornir_imageregistration.transforms.factory.LoadTransform(loadedStosObj.Transform)
        self.assertIsNotNone(loadedTransform)


class TestStosBruteWithMask(setup_imagetest.ImageTestBase):

    def setUp(self):
        super(TestStosBruteWithMask, self).setUp()
        self.WarpedImagePath = self.GetImagePath("0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.FixedImagePath = self.GetImagePath("mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.WarpedImagePathFlipped = self.GetImagePath(
            "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8_FlippedUD.png")
        self.WarpedImageMaskPath = self.GetImagePath("0017_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.FixedImageMaskPath = self.GetImagePath("mini_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png")

    def testStosBruteWithMask_MultiThread(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        AlignmentRecord = self.RunBasicBruteAlignmentWithMask(self.FixedImagePath, self.WarpedImagePath,
                                                              self.FixedImageMaskPath, self.WarpedImageMaskPath,
                                                              SingleThread=False, FlipUD=False)
        CheckAlignmentRecord(self, AlignmentRecord, angle=132.0, X=-4, Y=22)
        savedstosObj = AlignmentRecord.ToStos(self.FixedImagePath, self.WarpedImagePath,
                                              self.FixedImageMaskPath, self.WarpedImageMaskPath,
                                              PixelSpacing=1)
        self.CheckStosObj(savedstosObj, '17-18_brute_WithMask.stos', self.FixedImageMaskPath, self.WarpedImageMaskPath)

    def testStosBruteWithMask_GPU(self):
        if not nornir_imageregistration.HasCupy():
            return

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        AlignmentRecord = self.RunBasicBruteAlignmentWithMask(self.FixedImagePath, self.WarpedImagePath,
                                                              self.FixedImageMaskPath, self.WarpedImageMaskPath,
                                                              SingleThread=True, FlipUD=False)
        CheckAlignmentRecord(self, AlignmentRecord, angle=132.0, X=-4, Y=22)
        savedstosObj = AlignmentRecord.ToStos(self.FixedImagePath, self.WarpedImagePath,
                                              self.FixedImageMaskPath, self.WarpedImageMaskPath,
                                              PixelSpacing=1)
        self.CheckStosObj(savedstosObj, '17-18_brute_WithMask_GPU.stos', self.FixedImageMaskPath,
                          self.WarpedImageMaskPath)

    def RunBasicBruteAlignmentWithMask(self,
                                       FixedImagePath: str,
                                       WarpedImagePath: str,
                                       FixedImageMaskPath: str,
                                       WarpedImageMaskPath: str,
                                       AngleSearchRange: list[float] | None = None,
                                       WarpedImageScaleFactors=None,
                                       FlipUD: bool = False,
                                       SingleThread: bool = False,
                                       Cluster: bool = False,
                                       method: SliceToSliceMethod = SliceToSliceMethod.BruteForce) -> nornir_imageregistration.AlignmentRecord:
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(WarpedImageMaskPath), "Missing test input")
        self.assertTrue(os.path.exists(FixedImageMaskPath), "Missing test input")

        use_cp = nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy
        timer = TaskTimer()
        timer.Start(f"\nSliceToSliceBrute WithMask - Cluster={Cluster} - SingleThread={SingleThread} - GPU={use_cp}")

        AlignmentRecord = stos_brute.SliceToSliceRigidRegistration(target_image=FixedImagePath,
                                                                   source_image=WarpedImagePath,
                                                                   target_mask=FixedImageMaskPath,
                                                                   source_mask=WarpedImageMaskPath,
                                                                   LargestDimension=1024,
                                                                   AngleSearchRange=AngleSearchRange,
                                                                   WarpedImageScaleFactors=WarpedImageScaleFactors,
                                                                   SingleThread=SingleThread,
                                                                   TestFlip=FlipUD,
                                                                   Cluster=Cluster,
                                                                   method=method)

        self.Logger.info("Best alignment: " + str(AlignmentRecord))
        timer.End(f"\nSliceToSliceBrute WithMask - Cluster={Cluster} - SingleThread={SingleThread} - GPU={use_cp}")

        return AlignmentRecord

    def CheckStosObj(self,
                     stosObj: nornir_imageregistration.StosFile,
                     stosfilename: str,
                     FixedImageMaskPath: str,
                     WarpedImageMaskPath: str):

        self.assertIsNotNone(stosObj)

        stosfilepath = os.path.join(self.VolumeDir, stosfilename)
        if os.path.isfile(stosfilepath):
            os.remove(stosfilepath)
        stosObj.Save(stosfilepath)

        loadedStosObj = nornir_imageregistration.files.stosfile.StosFile.Load(stosfilepath)
        self.assertIsNotNone(loadedStosObj)

        self.assertTrue(loadedStosObj.HasMasks, ".stos file is expected to have masks")

        controlMaskName = os.path.basename(FixedImageMaskPath)
        warpedMaskName = os.path.basename(WarpedImageMaskPath)

        self.assertEqual(loadedStosObj.ControlMaskName, controlMaskName,
                         "Mask in .stos does not match mask used in alignment\n")
        self.assertEqual(loadedStosObj.MappedMaskName, warpedMaskName,
                         "Mask in .stos does not match mask used in alignment\n")

    def testStosBruteScaleMismatchWithMask(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.runStosBruteScaleMismatchWithMask()

    def testStosBruteScaleMismatchWithMask_GPU(self):
        if not nornir_imageregistration.HasCupy():
            return
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        self.runStosBruteScaleMismatchWithMask()

    def testStosBruteScaleMismatchWithMask_LogPolar(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        ImageRootPath = os.path.join(self.ImportedDataPath, "Alignment", "CaptureResolutionMismatch")
        Downsample = '032'
        Filter = 'Leveled'
        TEM1Resolution = 2.176
        TEM2Resolution = 2.143
        WarpedImagePath = os.path.join(ImageRootPath, "502", Filter, "Images", str(Downsample),
                                       "0502_TEM_{0}.png".format(Filter))
        FixedImagePath = os.path.join(ImageRootPath, "503", Filter, "Images", str(Downsample),
                                      "0503_TEM_{0}.png".format(Filter))
        WarpedImageMaskPath = os.path.join(ImageRootPath, "502", "Mask", "Images", str(Downsample),
                                           "0502_TEM_Mask.png")
        FixedImageMaskPath = os.path.join(ImageRootPath, "503", "Mask", "Images", str(Downsample),
                                          "0503_TEM_Mask.png")
        WarpedImageScalar = TEM2Resolution / TEM1Resolution
        AlignmentRecord = self.RunBasicBruteAlignmentWithMask(
            FixedImagePath, WarpedImagePath, FixedImageMaskPath, WarpedImageMaskPath,
            WarpedImageScaleFactors=WarpedImageScalar,
            FlipUD=False,
            AngleSearchRange=range(160, 200, 1),
            method=SliceToSliceMethod.LogPolar)
        self.assertAlmostEqual(AlignmentRecord.scale, WarpedImageScalar, delta=0.05)

    def runStosBruteScaleMismatchWithMask(self):
        ImageRootPath = os.path.join(self.ImportedDataPath, "Alignment", "CaptureResolutionMismatch")
        Downsample = '032'
        Filter = 'Leveled'
        TEM1Resolution = 2.176  # nm/pixel, section 503, Fixed
        TEM2Resolution = 2.143  # nm/pixel, section 502, Warped

        # Approximate correct answer
        # X: -165
        # Y: +90
        # Angle: 176

        # We are registering 502 onto 503, so TEM2 is warped and TEM1 is fixed

        WarpedImagePath = os.path.join(ImageRootPath, "502", Filter, "Images", str(Downsample),
                                       "0502_TEM_{0}.png".format(Filter))
        FixedImagePath = os.path.join(ImageRootPath, "503", Filter, "Images", str(Downsample),
                                      "0503_TEM_{0}.png".format(Filter))

        WarpedImageMaskPath = os.path.join(ImageRootPath, "502", "Mask", "Images", str(Downsample),
                                           "0502_TEM_Mask.png")
        FixedImageMaskPath = os.path.join(ImageRootPath, "503", "Mask", "Images", str(Downsample),
                                          "0503_TEM_Mask.png")

        WarpedImageScalar = TEM2Resolution / TEM1Resolution
        # WarpedImageScalar = 0.91 #TEM2Resolution / TEM1Resolution

        AlignmentRecord = self.RunBasicBruteAlignmentWithMask(FixedImagePath,
                                                              WarpedImagePath,
                                                              FixedImageMaskPath,
                                                              WarpedImageMaskPath,
                                                              WarpedImageScaleFactors=WarpedImageScalar,
                                                              FlipUD=False,
                                                              AngleSearchRange=range(160, 200, 1))

        self.Logger.info("Best alignment: " + str(AlignmentRecord))

        savedstosObj = AlignmentRecord.ToStos(FixedImagePath, WarpedImagePath, FixedImageMaskPath, WarpedImageMaskPath,
                                              PixelSpacing=1)
        self.CheckStosObj(savedstosObj, '502-503_brute_WithMask_scalemismatch_GPU.stos', FixedImageMaskPath,
                          WarpedImageMaskPath)

    def runStosBruteExecuteWithMask(self):
        self.assertTrue(os.path.exists(self.WarpedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(self.WarpedImageMaskPath), "Missing test input")
        self.assertTrue(os.path.exists(self.FixedImageMaskPath), "Missing test input")

        stosfilepath = os.path.join(self.VolumeDir, '17-18_brute_WithMask.stos')

        nornir_imageregistration.scripts.nornir_rotate_translate.Execute(ExecArgs=['-f', self.FixedImagePath,
                                                                                   '-w', self.WarpedImagePath,
                                                                                   '-fm', self.FixedImageMaskPath,
                                                                                   '-wm', self.WarpedImageMaskPath,
                                                                                   '-o', stosfilepath])

        self.assertTrue(os.path.exists(stosfilepath), "Stos brute script should create output")

    def testStosBruteExecuteWithMask(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.runStosBruteExecuteWithMask()

    def testStosBruteExecuteWithMask_GPU(self):
        if not nornir_imageregistration.HasCupy():
            return

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        self.runStosBruteExecuteWithMask()


class TestStosBruteToSameImage(setup_imagetest.ImageTestBase):

    def setUp(self):
        super(TestStosBruteToSameImage, self).setUp()
        self.FixedImagePath = self.GetImagePath("mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.FixedImageMaskPath = self.GetImagePath("mini_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png")

    #    def testSameSimpleImage(self):
    #        '''Make sure the same image aligns to itself with peak (0,0) and angle 0'''
    #        FixedImagePath = os.path.join(self.ImportedDataPath, "fixed.png")
    #        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")
    #        FixedImageMaskPath = os.path.join(self.ImportedDataPath, "fixedmask.png")
    #        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")
    #
    #        AlignmentRecord = stos_brute.SliceToSliceRigidRegistration(FixedImagePath, FixedImagePath)
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
    #        AlignmentRecord = stos_brute.SliceToSliceRigidRegistration(FixedImagePath,
    #                       FixedImagePath,
    #                       FixedImageMaskPath,
    #                       FixedImageMaskPath)
    #        CheckAlignmentRecord(self, AlignmentRecord, angle = 0.0, X = 0, Y = 0)

    def testSameTEMImageFast_SingleThread(self):
        '''Make sure the same image aligns to itself with peak (0,0) and angle 0'''

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(self.FixedImageMaskPath), "Missing test input")

        self.RunBasicBruteAlignmentToSameImage(self.FixedImagePath,
                                               self.FixedImagePath,
                                               self.FixedImageMaskPath,
                                               self.FixedImageMaskPath,
                                               AngleSearchRange=[-2, 0, 2],
                                               SingleThread=True)

    def testSameTEMImageFast_MultiThread(self):
        '''Make sure the same image aligns to itself with peak (0,0) and angle 0'''
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(self.FixedImageMaskPath), "Missing test input")

        self.RunBasicBruteAlignmentToSameImage(self.FixedImagePath,
                                               self.FixedImagePath,
                                               self.FixedImageMaskPath,
                                               self.FixedImageMaskPath,
                                               AngleSearchRange=[-2, 0, 2],
                                               SingleThread=False)

    def testSameTEMImageFast_GPU(self):
        '''Make sure the same image aligns to itself with peak (0,0) and angle 0'''
        if not nornir_imageregistration.HasCupy():
            return

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(self.FixedImageMaskPath), "Missing test input")

        self.RunBasicBruteAlignmentToSameImage(self.FixedImagePath,
                                               self.FixedImagePath,
                                               self.FixedImageMaskPath,
                                               self.FixedImageMaskPath,
                                               AngleSearchRange=[-2, 0, 2],
                                               SingleThread=True)

    # def testSameTEMImage_SingleThread(self):
    #     '''Make sure the same image aligns to itself with peak (0,0) and angle 0'''
    #     self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")
    #     self.assertTrue(os.path.exists(self.FixedImageMaskPath), "Missing test input")
    #
    #     self.RunBasicBruteAlignmentToSameImage(self.FixedImagePath,
    #                                            self.FixedImagePath,
    #                                            self.FixedImageMaskPath,
    #                                            self.FixedImageMaskPath,
    #                                            SingleThread=True)

    def testSameTEMImage_MultiThread(self):
        '''Make sure the same image aligns to itself with peak (0,0) and angle 0'''

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(self.FixedImageMaskPath), "Missing test input")

        self.RunBasicBruteAlignmentToSameImage(self.FixedImagePath,
                                               self.FixedImagePath,
                                               self.FixedImageMaskPath,
                                               self.FixedImageMaskPath,
                                               SingleThread=False)

    def testSameTEMImage_GPU(self):
        '''Make sure the same image aligns to itself with peak (0,0) and angle 0'''
        if not nornir_imageregistration.HasCupy():
            return

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        self.assertTrue(os.path.exists(self.FixedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(self.FixedImageMaskPath), "Missing test input")

        self.RunBasicBruteAlignmentToSameImage(self.FixedImagePath,
                                               self.FixedImagePath,
                                               self.FixedImageMaskPath,
                                               self.FixedImageMaskPath,
                                               SingleThread=True)

    def RunBasicBruteAlignmentToSameImage(self,
                                          FixedImagePath: str,
                                          WarpedImagePath: str,
                                          FixedImageMaskPath: str,
                                          WarpedImageMaskPath: str,
                                          angle: float = 0.0,  # Angle to rotate the target image by
                                          offset: tuple[int, int] = (0, 0),  # Offset to apply to the target image
                                          AngleSearchRange: list[float] | None = None,
                                          FlipUD: bool = False,
                                          SingleThread: bool = False,
                                          Cluster: bool = False):
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(FixedImageMaskPath), "Missing test input")
        self.assertTrue(os.path.exists(WarpedImageMaskPath), "Missing test input")

        source_image_data = nornir_imageregistration.ImagePermutationHelper(FixedImagePath,
                                                                            FixedImageMaskPath)

        target_image_data = CreateRotatedAndOffsetImage(WarpedImagePath, WarpedImageMaskPath, angle,
                                                        offset)

        AlignmentRecord = stos_brute.SliceToSliceRigidRegistration(target_image=FixedImagePath,
                                                                   source_image=WarpedImagePath,
                                                                   target_mask=FixedImageMaskPath,
                                                                   source_mask=WarpedImageMaskPath,
                                                                   AngleSearchRange=AngleSearchRange,
                                                                   SingleThread=SingleThread,
                                                                   method=nornir_imageregistration.settings.SliceToSliceMethod.BruteForce)
        print(AlignmentRecord)
        CheckAlignmentRecord(self, AlignmentRecord, angle=angle, X=offset[1], Y=offset[0], adelta=1.5)

    def RunBruteAlignmentToSameImageWithRotateTranslate(self,
                                                        FixedImagePath: str,
                                                        WarpedImagePath: str,
                                                        FixedImageMaskPath: str,
                                                        WarpedImageMaskPath: str,
                                                        angle: float = 0.0,  # Angle to rotate the target image by
                                                        offset: tuple[int, int] = (0, 0),
                                                        # Offset to apply to the target image
                                                        AngleSearchRange: list[float] | None = None,
                                                        FlipUD: bool = False,
                                                        SingleThread: bool = False,
                                                        Cluster: bool = False):
        self.assertTrue(os.path.exists(FixedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(WarpedImagePath), "Missing test input")
        self.assertTrue(os.path.exists(FixedImageMaskPath), "Missing test input")
        self.assertTrue(os.path.exists(WarpedImageMaskPath), "Missing test input")

        source_image_data = nornir_imageregistration.ImagePermutationHelper(FixedImagePath,
                                                                            FixedImageMaskPath)

        target_image_data = CreateRotatedAndOffsetImage(WarpedImagePath, WarpedImageMaskPath, angle,
                                                        offset)

        settings = nornir_imageregistration.settings.StosBruteSettings(
            method=SliceToSliceMethod.BruteForce,
            min_overlap=0.5,
            try_flipped=True,
            angles=AngleSearchRange)

        AlignmentRecord = stos_brute.SliceToSliceRigidRegistrationWithPreprocessedImages(
            source_image_data=source_image_data,
            target_image_data=target_image_data,
            settings=settings,
            SingleThread=SingleThread,
            Cluster=Cluster)
        print(AlignmentRecord)
        CheckAlignmentRecord(self, AlignmentRecord, angle=angle, X=offset[1], Y=offset[0], adelta=1.5)

    def testTranslateOnly(self):
        self.RunBruteAlignmentToSameImageWithRotateTranslate(self.FixedImagePath,
                                                             self.FixedImagePath,
                                                             self.FixedImageMaskPath,
                                                             self.FixedImageMaskPath,
                                                             angle=0,
                                                             offset=(0, 64),
                                                             AngleSearchRange=[0])


class TestLogPolarStosWithMask(setup_imagetest.ImageTestBase):

    def setUp(self):
        super(TestLogPolarStosWithMask, self).setUp()
        self.WarpedImagePath = self.GetImagePath("0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.FixedImagePath = self.GetImagePath("mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.WarpedImagePathFlipped = self.GetImagePath(
            "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8_FlippedUD.png")
        self.WarpedImageMaskPath = self.GetImagePath("0017_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png")
        self.FixedImageMaskPath = self.GetImagePath("mini_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png")

    def test_simple(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

        target_image_data = nornir_imageregistration.ImagePermutationHelper(self.FixedImagePath,
                                                                            self.FixedImageMaskPath)
        source_image_data = nornir_imageregistration.ImagePermutationHelper(self.WarpedImagePath,
                                                                            self.WarpedImageMaskPath)

        results = stos_brute._find_angle_and_scale_with_logpolar(source_image=source_image_data.ImageWithMaskAsNoise,
                                                                 target_image=target_image_data.ImageWithMaskAsNoise,
                                                                 source_stats=source_image_data.Stats,
                                                                 target_stats=target_image_data.Stats)

    def test_known_rotation_offset(self, angle=132, source_to_target_offset=(34, 100)):
        """
        Using the same image as source and target, rotate the target through a full circle and plot the
        measured angle and shift at each angle.  The resulting plot should be a line with a slope of 1
        :param angle:
        :param source_to_target_offset:
        :return:
        """
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

        source_image_data = nornir_imageregistration.ImagePermutationHelper(self.WarpedImagePath,
                                                                            self.WarpedImageMaskPath)

        target_image_data = CreateRotatedAndOffsetImage(self.WarpedImagePath, self.WarpedImageMaskPath, angle,
                                                        source_to_target_offset)

        # Check that we can get the correct angle and scale calling logpolar directly
        results = stos_brute._find_angle_and_scale_with_logpolar(source_image=source_image_data.ImageWithMaskAsNoise,
                                                                 target_image=target_image_data.ImageWithMaskAsNoise,
                                                                 source_stats=source_image_data.Stats,
                                                                 target_stats=source_image_data.Stats)

        assert_angles_equal_degrees(self, results.angle, angle, tolerance=2.0, msg="Angle mismatch")
        self.assertAlmostEqual(results.scale, 1.0, delta=0.1, msg="Scale mismatch")

        # Check that we can get the correct translation vector by calling the full alignment routine
        settings = nornir_imageregistration.settings.StosBruteSettings(
            min_overlap=0.5,
            method=SliceToSliceMethod.LogPolar,
            try_flipped=False
        )

        rigid_results = stos_brute.SliceToSliceRigidRegistrationWithPreprocessedImages(
            source_image_data=source_image_data,
            target_image_data=target_image_data,
            settings=settings)
        self.assertAlmostEqual(rigid_results.peak[0], source_to_target_offset[0], delta=2.0)
        self.assertAlmostEqual(rigid_results.peak[1], source_to_target_offset[1], delta=2.0)
        assert_angles_equal_degrees(self, rigid_results.angle, angle, tolerance=2.0)

    def test_known_offset(self):
        self.test_known_rotation_offset(angle=0, source_to_target_offset=(0, 64))

    def test_identity(self):
        self.test_known_rotation_offset(angle=0, source_to_target_offset=(0, 0))

    def test_known_rotation_plot(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

        with TaskTimerContext() as timer:
            angles, y, weight, diff = self.get_angle_plot()

        print(f"{timer}")
        self.plot_angles(angles, y, weight, diff)

    def get_angle_plot(self) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        Using the same image as source and target, rotate the target through a full circle and plot the
        measured angle and shift at each angle.  The resulting plot should be a line with a slope of 1
        :param angle:
        :param source_to_target_offset:
        :return:
        """
        angles = list(range(-180, 180, 30))
        source_to_target_offset = (0, 0)

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

        source_image_data = nornir_imageregistration.ImagePermutationHelper(self.WarpedImagePath,
                                                                            self.WarpedImageMaskPath)
        y = []
        weight = []
        diff = []
        timer = TaskTimer()
        for angle in angles:
            target_image_data = CreateRotatedAndOffsetImage(self.WarpedImagePath, self.WarpedImageMaskPath, angle,
                                                            source_to_target_offset)
            print(f"angle: {angle}")
            timer.Start(f"find angle")
            result = stos_brute._find_angle_and_scale_with_logpolar(source_image=source_image_data.ImageWithMaskAsNoise,
                                                                    target_image=target_image_data.ImageWithMaskAsNoise,
                                                                    source_stats=source_image_data.Stats,
                                                                    target_stats=target_image_data.Stats)
            timer.End(f"find angle")

            y.append(result.angle)
            diff.append(result.angle - angle)
            weight.append(result.weight)

        print(f"{timer}")
        return angles, y, weight, diff

    def plot_angles(self, angles, y, weight, diff):
        plt.plot(angles, y)
        plt.plot(angles, weight)
        plt.plot(angles, diff)
        plt.legend(['angle', 'weight', 'diff'])
        plt.xlabel('actual angle')
        plt.ylabel('measured angle')
        plt.gca().set_aspect('equal')

        if is_headless():
            path = os.path.join(self.TestOutputPath, 'known_rotation_plot_angles.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            inspect_png_output(path)
        else:
            plt.show()


class TestHybridAdaptiveFallback(unittest.TestCase):
    """Unit and property tests for continuous ambiguous fallback helpers."""

    def test_smooth01_monotonic(self):
        values = [0.5, 1.0, 1.25, 1.5, 2.0]
        mapped = [stos_brute._smooth01(v, 1.0, 1.5) for v in values]
        self.assertEqual(mapped[0], 0.0)
        self.assertEqual(mapped[-1], 1.0)
        for left, right in zip(mapped, mapped[1:]):
            self.assertLessEqual(left, right)

    def test_logpolar_confidence_weakest_signal(self):
        weak = stos_brute.LogPolarDiagnostics(
            angle_peak_ratio=1.5,
            translation_peak_ratio=1.0,
            strength_delta_ratio=0.25,
            degrees_per_pixel=0.5,
            peak_strength=10.0,
        )
        self.assertAlmostEqual(stos_brute._logpolar_confidence(weak), 0.0)

        strong = stos_brute.LogPolarDiagnostics(
            angle_peak_ratio=1.5,
            translation_peak_ratio=1.3,
            strength_delta_ratio=0.25,
            degrees_per_pixel=0.5,
            peak_strength=10.0,
        )
        self.assertAlmostEqual(stos_brute._logpolar_confidence(strong), 1.0)

    def test_fallback_search_geometry_monotonic(self):
        dpp = 0.35
        prev_hw = float('inf')
        for confidence in (0.0, 0.25, 0.5, 0.75, 1.0):
            hw, coarse_step, _, _ = stos_brute._fallback_search_geometry(confidence, dpp)
            self.assertLessEqual(hw, prev_hw)
            prev_hw = hw
            self.assertGreaterEqual(coarse_step, 0.2)

    def test_median_ambiguous_diagnostics_near_fixed_c(self):
        """Typical ambiguous pair (ratios just below thresholds) should land near ±20°."""
        diagnostics = stos_brute.LogPolarDiagnostics(
            angle_peak_ratio=1.19,
            translation_peak_ratio=1.10,
            strength_delta_ratio=0.10,
            degrees_per_pixel=360.0 / 512.0,
            peak_strength=5.0,
        )
        confidence = stos_brute._logpolar_confidence(diagnostics)
        hw, _, _, _ = stos_brute._fallback_search_geometry(confidence, diagnostics.degrees_per_pixel)
        self.assertGreaterEqual(hw, 15.0)
        self.assertLessEqual(hw, 30.0)
        angles = stos_brute._adaptive_fallback_angle_range(12.0, diagnostics)
        self.assertGreaterEqual(len(angles), 25)
        self.assertLessEqual(len(angles), 65)

    def test_brute_fallback_needs_widen(self):
        diagnostics = stos_brute.LogPolarDiagnostics(
            angle_peak_ratio=1.1,
            translation_peak_ratio=1.05,
            strength_delta_ratio=0.08,
            degrees_per_pixel=0.5,
            peak_strength=10.0,
        )
        logpolar = stos_brute.AngleScaleResult(
            angle=0.0, scale=1.0, weight=10.0, translation=(0.0, 0.0),
            ambiguous=True, diagnostics=diagnostics,
        )
        good = AlignmentRecord((0, 0), 9.0, 0.0)
        bad = AlignmentRecord((0, 0), 8.0, 0.0)
        self.assertFalse(stos_brute._brute_fallback_needs_widen(good, logpolar))
        self.assertTrue(stos_brute._brute_fallback_needs_widen(bad, logpolar))

    def test_adaptive_fallback_angle_count_decreases_with_confidence(self):
        import hypothesis
        import hypothesis.strategies as st

        @hypothesis.given(
            angle_ratio=st.floats(1.0, 1.5),
            trans_ratio=st.floats(1.0, 1.3),
            delta_ratio=st.floats(0.0, 0.25),
        )
        @hypothesis.settings(max_examples=50, deadline=None)
        def angle_count_monotonic(angle_ratio, trans_ratio, delta_ratio):
            low_conf = stos_brute.LogPolarDiagnostics(
                angle_peak_ratio=angle_ratio,
                translation_peak_ratio=trans_ratio,
                strength_delta_ratio=delta_ratio,
                degrees_per_pixel=360.0 / 512.0,
                peak_strength=1.0,
            )
            high_conf = stos_brute.LogPolarDiagnostics(
                angle_peak_ratio=max(angle_ratio, 1.45),
                translation_peak_ratio=max(trans_ratio, 1.25),
                strength_delta_ratio=max(delta_ratio, 0.22),
                degrees_per_pixel=360.0 / 512.0,
                peak_strength=1.0,
            )
            low_c = stos_brute._logpolar_confidence(low_conf)
            high_c = stos_brute._logpolar_confidence(high_conf)
            hypothesis.assume(high_c > low_c + 0.05)
            low_count = len(stos_brute._adaptive_fallback_angle_range(0.0, low_conf))
            high_count = len(stos_brute._adaptive_fallback_angle_range(0.0, high_conf))
            hypothesis.assume(low_count > 5 and high_count > 5)
            assert low_count >= high_count

        angle_count_monotonic()


if __name__ == "__main__":
    # import syssys.argv = ['', 'Test.testName']
    unittest.main()
