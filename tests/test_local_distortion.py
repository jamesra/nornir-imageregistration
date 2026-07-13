"""
Created on Sep 26, 2018

@author: u0490822
"""
import os
import os.path
import tempfile
import unittest
import re
from typing import Any, cast

import numpy as np

import picklehelper

try:
    import cupy as cp
    import cupyx

    init_context = cp.zeros((64, 64))  # Attempt to initialize CUDA context if we get this far
    init_context = init_context.mean()
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx
except Exception:
    # CuPy is installed but CUDA user-mode is unusable (e.g. missing libnvrtc without GPU mount).
    import nornir_imageregistration.cupy_thunk as cp
    import nornir_imageregistration.cupyx_thunk as cupyx

import nornir_imageregistration
from nornir_imageregistration import local_distortion_correction
from nornir_imageregistration.alignment_record import EnhancedAlignmentRecord
import nornir_imageregistration.assemble
from nornir_imageregistration.local_distortion_correction import AlignRecordsToControlPoints, RefineStosFile, \
    _RefineGridPointsForTwoImages
import nornir_imageregistration.scripts.nornir_stos_grid_refinement
import nornir_pools
import setup_imagetest


def _fake_refine_tile_alignment_remote(*args, **kwargs):
    refine_dtype = np.dtype([('SourceAY', 'f4'),
                             ('SourceAX', 'f4'),
                             ('SourceBY', 'f4'),
                             ('SourceBX', 'f4'),
                             ('BaseTargetY', 'f4'),
                             ('BaseTargetX', 'f4'),
                             ('TargetY', 'f4'),
                             ('TargetX', 'f4'),
                             ('DisplacementY', 'f4'),
                             ('DisplacementX', 'f4'),
                             ('Weight', 'f4'),
                             ('Angle', 'f4')])
    point_pairs = np.zeros((2, 2), dtype=refine_dtype)
    rows = [10.0, 30.0]
    cols = [10.0, 30.0]
    for i, y in enumerate(rows):
        for j, x in enumerate(cols):
            point_pairs[i, j] = (y, x, y, x + 8.0, y, x + 4.0, y + 2.0, x + 4.0, 2.0, 0.0, 1.0, 0.0)
    return point_pairs, np.asarray((2.0, 0.0, 1.0), dtype=np.float32)


def _fake_refine_tile_overlap_batch_remote(anchor_tile, overlap_batch, image_scale, subregion_shape):
    """Return one stub refinement result per overlap in the batch."""
    single = _fake_refine_tile_alignment_remote(
        anchor_tile,
        overlap_batch[0].B if overlap_batch else anchor_tile,
        overlap_batch[0].scaled_overlapping_source_rect_A if overlap_batch else None,
        overlap_batch[0].scaled_overlapping_source_rect_B if overlap_batch else None,
        overlap_batch[0].scaled_offset if overlap_batch else np.zeros(2),
        image_scale,
        subregion_shape)
    return [single] * len(overlap_batch)


def _serial_refinement_pool(_target_space_scale: float):
    return nornir_pools.GetGlobalSerialPool()


# class TestLocalDistortion(setup_imagetest.TransformTestBase):
# 
#     @property
#     def TestName(self):
#         return "GridRefinement"
#     
#     def setUp(self):
#         super(TestLocalDistortion, self).setUp()
#         return 
# 
# 
#     def tearDown(self):
#         super(TestLocalDistortion, self).tearDown()
#         return
# 
# 
#     def testRefineMosaic(self):
#         '''
#         This is a test for the refine mosaic feature which is not fully implemented
#         '''
#         tilesDir = self.GetTileFullPath()
#         mosaicFile = self.GetMosaicFile("Translated")
#         mosaic = nornir_imageregistration.mosaic.Mosaic.LoadFromMosaicFile(mosaicFile)
#         mosaic.RefineMosaic(tilesDir, usecluster=False)
#         pass


class TestSliceToSliceRefinement(setup_imagetest.TransformTestBase, picklehelper.PickleHelper):

    def __init__(self, methodName='runTest'):
        self._TestName = 'SliceToSliceRefinement'

        # if methodName.startswith('test'):
        # self._TestName = methodName[len('test'):]
        # else:
        # self._TestName = methodName

        setup_imagetest.TransformTestBase.__init__(self, methodName=methodName)

    @property
    def TestName(self):
        return self._TestName

    def setUp(self):
        super(TestSliceToSliceRefinement, self).setUp()
        return

    def tearDown(self):
        super(TestSliceToSliceRefinement, self).tearDown()
        return

    @property
    def ImageDir(self):
        return os.path.join(self.ImportedDataPath, '..\\..\\Images\\')

    # def testStosRefinementRC1_258(self): 
    # #self.TestName = "StosRefinementRC1_258"
    # stosFile = self.GetStosFile("0258-0257_grid_16.stos")
    # self.RunStosRefinement(stosFile, ImageDir=self.TestInputDataPath, SaveImages=False, SavePlots=True)

    #     def testStosRefinementRC2_162(self):
    #         SaveImages = False
    #         SavePlots = True
    #         self.TestName = "StosRefinementRC2_162"
    #         stosFile = self.GetStosFile("0164-0162_brute_32")
    #         self.RunStosRefinement(stosFile, self.ImageDir, SaveImages=False, SavePlots=True)
    #
    # def testStosRefinementRC2_617(self):
    #     # self.TestName = "StosRefinementRC2_617"
    #     stosFilePath = self.GetStosFilePath("StosRefinementRC2_617", "0617-0618_brute_32_pyre")
    #     # self.RunStosRefinement(stosFilePath, ImageDir=os.path.dirname(stosFilePath), SaveImages=False, SavePlots=True)
    #     RefineStosFile(InputStos=stosFilePath,
    #                    OutputStosPath=os.path.join(self.TestOutputPath, 'Final.stos'),
    #                    num_iterations=5,
    #                    cell_size=(256, 256),
    #                    grid_spacing=(128, 128),
    #                    angles_to_search=[-5, 0, 5],
    #                    max_travel_for_finalization=1,
    #                    max_travel_for_finalization_improvement=256,
    #                    min_alignment_overlap=0.5,
    #                    SaveImages=False,
    #                    SavePlots=True)

    #     def testStosRefinementRC2_1034(self):
    #         #self.TestName = "StosRefinementRC2_1034"
    #         stosFilePath = self.GetStosFilePath("StosRefinementRC2_1034","1034-1032_ctrl-TEM_Leveled_map-TEM_Leveled_original.stos")
    #         self.RunStosRefinement(stosFilePath, ImageDir=os.path.dirname(stosFilePath), SaveImages=False, SavePlots=True)
    # #         RefineStosFile(InputStos=stosFile,
    #                        OutputStosPath=os.path.join(self.TestOutputPath, 'Final.stos'),
    #                        num_iterations=10,
    #                        cell_size=(128,128),
    #                        grid_spacing=(128,128),
    #                        angles_to_search=[-2.5, 0, 2.5],
    #                        min_travel_for_finalization=0.5,
    #                        min_alignment_overlap=0.5,
    #                        SaveImages=True,
    #                        SavePlots=True)

    #    def testStosRefinementRC2_1034_Mini(self):
    # self.TestName = "StosRefinementRC2_1034_Mini"
    #        stosFilePath = self.GetStosFilePath("StosRefinementRC2_1034_Mini", "1032-1034_brute_32_pyre_crude_across.stos")
    #        self.RunStosRefinement(stosFilePath, ImageDir=os.path.dirname(stosFilePath), SaveImages=False, SavePlots=True)
    #         RefineStosFile(InputStos=stosFile,
    #                        OutputStosPath=os.path.join(self.TestOutputPath, 'Final.stos'),
    #                        num_iterations=10,
    #                        cell_size=(128,128),
    #                        grid_spacing=(128,128),
    #                        angles_to_search=[-2.5, 0, 2.5],
    #                        min_travel_for_finalization=0.5,
    #                        min_alignment_overlap=0.5,
    #                        SaveImages=True,
    #                        SavePlots=True)
    #        return

    # def test_StosRefinementCPED_3_2(self):
    #     # self.TestName = "StosRefinementRC2_1034_Mini"
    #     try:
    #         os.remove(self.CachePath)
    #     except Exception as e:
    #         print(f"Exception cleaning cache directory: {self.CachePath}\n{e}")
    #         pass
    #
    #     stosFilePath = self.GetStosFilePath("StosRefinementRPC3_14_13", "14-13_ctrl-TEM_Leveled_map-TEM_Leveled.stos")
    #     # self.RunStosRefinement(stosFilePath, ImageDir=None, SaveImages=False, SavePlots=True)
    #     RefineStosFile(InputStos=stosFilePath,
    #                    OutputStosPath=os.path.join(self.TestOutputPath, 'Final.stos'),
    #                    num_iterations=10,
    #                    cell_size=(256, 256),
    #                    grid_spacing=(128, 128),
    #                    angles_to_search=[0], 
    #                    min_alignment_overlap=0.25,
    #                    num_iterations=5, 
    #                    SaveImages=False,
    #                    SavePlots=True)
    #     return

    # def testStosRefinementRPC3_13_14_Incorrect_Brute_Alignment(self):
    #     """
    #     This is an incorrectly aligned brute output.  The goal is to have the alignment exit without going off the rails or producing horrible output.
    #     """
    #
    #     # Do not stress about this test until you verify the input transform was
    #     # not affected by the grid transform saving bug and that it is a valid
    #     # starting point
    #
    #     # self.TestName = "StosRefinementRC2_617"
    #     stosFilePath = self.GetStosFilePath("StosRefinementRPC3_14_13_DS32_From_Brute",
    #                                         "14-13_ctrl-TEM_Leveled_map-TEM_Leveled.stos")
    #     # self.RunStosRefinement(stosFilePath, ImageDir=os.path.dirname(stosFilePath), SaveImages=False, SavePlots=True)
    #     RefineStosFile(InputStos=stosFilePath,
    #                    OutputStosPath=os.path.join(self.TestOutputPath, 'Final.stos'),
    #                    num_iterations=5,
    #                    cell_size=(256, 256),
    #                    grid_spacing=(128, 128),
    #                    angles_to_search=[0],
    #                    max_travel_for_finalization=None,
    #                    max_travel_for_finalization_improvement=None,
    #                    min_alignment_overlap=0.5,
    #                    min_unmasked_area=0.49,
    #                    SaveImages=False,
    #                    SavePlots=False)

    # def testStosRefinementRC2_27_26_Grid32_to_Grid16(self):
    #     """
    #     This is an incorrectly aligned brute output.  The goal is to have the alignment exit without going off the rails or producing horrible output.
    #     """
    #
    #     # Do not stress about this test until you verify the input transform was
    #     # not affected by the grid transform saving bug and that it is a valid
    #     # starting point
    #
    #     # self.TestName = "StosRefinementRC2_617"
    #     # stosFilePath = self.GetStosFilePath("StosRefinementRPC3_14_13_DS32_From_Brute",
    #     #                                     "14-13_ctrl-TEM_Leveled_map-TEM_Leveled.stos")
    #     # self.RunStosRefinement(stosFilePath, ImageDir=os.path.dirname(stosFilePath), SaveImages=False, SavePlots=True)
    #     stosFilePath = os.path.join("D:", "Data", "RC2", "TEM", "Grid32", "Automatic",
    #                                 "26-27_ctrl-TEM_Blob_map-TEM_Blob.stos")
    #     RefineStosFile(InputStos=stosFilePath,
    #                    OutputStosPath=os.path.join(self.TestOutputPath, 'Final.stos'),
    #                    num_iterations=5,
    #                    cell_size=(256, 256),
    #                    grid_spacing=(128, 128),
    #                    angles_to_search=[0],
    #                    max_travel_for_finalization=None,
    #                    max_travel_for_finalization_improvement=None,
    #                    min_alignment_overlap=0.5,
    #                    min_unmasked_area=0.49,
    #                    SaveImages=True,
    #                    SavePlots=True)

    def testStosRefinementRPC2_1156_1155_Grid16_to_Grid8(self):
        """
        Optional developer fixture: incorrectly aligned brute output that must
        refine without going off the rails.

        Skipped unless the machine-local RPC2 path exists. Prefer
        ``testStosRefinementIDoc690_691`` for CI / portable coverage.
        """

        # Do not stress about this test until you verify the input transform was
        # not affected by the grid transform saving bug and that it is a valid
        # starting point
        stosFilePath = os.path.join("D:", "Data", "RPC2", "TEM", "Grid8", "Automatic",
                                    "1156-1155_ctrl-TEM_Leveled_map-TEM_Leveled.stos")
        if not os.path.isfile(stosFilePath):
            self.skipTest(f"Machine-local RPC2 STOS fixture not present: {stosFilePath}")
        RefineStosFile(InputStos=stosFilePath,
                       OutputStosPath=os.path.join(self.TestOutputPath, 'Final.stos'),
                       num_iterations=5,
                       cell_size=(256, 256),
                       grid_spacing=(128, 128),
                       angles_to_search=[0],
                       max_travel_for_finalization=None,
                       max_travel_for_finalization_improvement=None,
                       min_alignment_overlap=0.5,
                       min_unmasked_area=0.49,
                       SaveImages=True,
                       SavePlots=True)

    def testStosRefinementIDoc690_691(self):
        """RefineStosFile on the bundled idoc 690/691 brute fixture must emit a .stos."""
        stosFilePath = os.path.join(
            os.path.dirname(__file__),
            "fixtures",
            "idoc_690_691",
            "StosBrute16",
            "690-691_ctrl-TEM_Leveled_map-TEM_Leveled.stos",
        )
        if not os.path.isfile(stosFilePath):
            self.skipTest(f"Bundled STOS fixture missing: {stosFilePath}")

        output_path = os.path.join(self.TestOutputPath, "Final_idoc_690_691.stos")
        RefineStosFile(InputStos=stosFilePath,
                       OutputStosPath=output_path,
                       num_iterations=2,
                       cell_size=(64, 64),
                       grid_spacing=(64, 64),
                       angles_to_search=[0],
                       max_travel_for_finalization=None,
                       max_travel_for_finalization_improvement=None,
                       min_alignment_overlap=0.5,
                       min_unmasked_area=0.49,
                       SaveImages=False,
                       SavePlots=False)
        self.assertTrue(os.path.isfile(output_path), "RefineStosFile did not write output .stos")
        refined = nornir_imageregistration.files.StosFile.Load(output_path)
        self.assertIsNotNone(refined.Transform)
        self.assertGreater(len(refined.Transform.strip()), 0)

    # def testStosRefinementRPC3_449_450(self):
    #     """
    #     This is a simple test case where the rigid translation is accurate and the images are barely rotated relative to each other
    #     """
    #
    #     stosFilePath = self.GetStosFilePath("testStosRefinementRPC3_449_450_From_Brute", "449-450_ctrl-TEM_Leveled_map-TEM_Leveled.stos")
    #     self.RunStosRefinement(stosFilePath, ImageDir=os.path.dirname(stosFilePath), SaveImages=False, SavePlots=True)
    #     RefineStosFile(InputStos=stosFilePath,
    #                    OutputStosPath=os.path.join(self.TestOutputPath, 'Final.stos'),
    #                    num_iterations=5,
    #                    cell_size=(256, 256),
    #                    grid_spacing=(128, 128),
    #                    angles_to_search=[0],
    #                    max_travel_for_finalization=None,
    #                    max_travel_for_finalization_improvement=None,
    #                    min_alignment_overlap=0.5,
    #                    min_unmasked_area=0.49,
    #                    SaveImages=False,
    #                    SavePlots=True)

    # def testStosRefinementRPC3_13_14(self):
    #     # self.TestName = "StosRefinementRC2_617"
    #     stosFilePath = self.GetStosFilePath("StosRefinementRPC3_14_13", "14-13_ctrl-TEM_Leveled_map-TEM_Leveled.stos")
    #     # self.RunStosRefinement(stosFilePath, ImageDir=os.path.dirname(stosFilePath), SaveImages=False, SavePlots=True)
    #     RefineStosFile(InputStos=stosFilePath,
    #                    OutputStosPath=os.path.join(self.TestOutputPath, 'Final.stos'),
    #                    num_iterations=5,
    #                    cell_size=(256, 256),
    #                    grid_spacing=(128, 128),
    #                    angles_to_search=[0],
    #                    max_travel_for_finalization=None,
    #                    max_travel_for_finalization_improvement=None,
    #                    min_alignment_overlap=0.5,
    #                    min_unmasked_area=0.49,
    #                    SaveImages=False,
    #                    SavePlots=True)

    def RunStosRefinement(self, stosFilePath: str, ImageDir: str | None = None, SaveImages: bool = False,
                          SavePlots: bool = True):
        """
        This is a test for the refine mosaic feature which is not fully implemented
        """
        use_cache = False

        # stosFile = self.GetStosFile("0164-0162_brute_32")
        # stosFile = self.GetStosFile("0617-0618_brute_64")
        stosObj = nornir_imageregistration.files.StosFile.Load(stosFilePath)
        # stosObj.Downsample = 64.0
        # stosObj.Scale(2.0)
        # stosObj.Save(os.path.join(self.TestOutputPath, "0617-0618_brute_32.stos"))

        fixedImage = stosObj.ControlImageFullPath
        warpedImage = stosObj.MappedImageFullPath

        target_image_fullpath = stosObj.ControlImageFullPath if ImageDir is None else os.path.join(ImageDir,
                                                                                                   stosObj.ControlImageFullPath)
        source_image_fullpath = stosObj.MappedImageFullPath if ImageDir is None else os.path.join(ImageDir,
                                                                                                  stosObj.MappedImageFullPath)

        target_mask_fullpath = None
        if stosObj.ControlMaskName is not None:
            target_mask_fullpath = os.path.join(
                ImageDir if ImageDir is not None else os.path.dirname(stosObj.ControlImageFullPath),
                stosObj.ControlMaskName)

        source_mask_fullpath = None
        if stosObj.MappedMaskName is not None:
            source_mask_fullpath = os.path.join(
                ImageDir if ImageDir is not None else os.path.dirname(stosObj.MappedImageFullPath),
                stosObj.MappedMaskName)

        stosTransform = nornir_imageregistration.transforms.factory.LoadTransform(stosObj.Transform, 1)

        # unrefined_image_path = os.path.join(self.TestOutputPath, 'unrefined_transform.png')

        #        if not os.path.exists(unrefined_image_path):
        #            unrefined_warped_image = nornir_imageregistration.assemble.TransformStos(stosTransform,
        #                                                                                     fixedImage=fixedImage,
        #                                                                                     warpedImage=warpedImage)
        #            nornir_imageregistration.SaveImage(unrefined_image_path, unrefined_warped_image, bpp=8)
        #        else:
        #            unrefined_warped_image = nornir_imageregistration.LoadImage(unrefined_image_path)

        i = 1

        finalized_points = {}

        # min_percentile_included = 5.0
        min_alignment_overlap = 0.5
        min_unmasked_area = 0.49

        final_pass = False

        # CutoffPercentilePerIteration = 10.0
        target_image_data = nornir_imageregistration.ImagePermutationHelper(img=target_image_fullpath,
                                                                            mask=target_mask_fullpath,
                                                                            extrema_mask_size_cuttoff=None,
                                                                            dtype=nornir_imageregistration.default_image_dtype())

        source_image_data = nornir_imageregistration.ImagePermutationHelper(img=source_image_fullpath,
                                                                            mask=source_mask_fullpath,
                                                                            extrema_mask_size_cuttoff=None,
                                                                            dtype=nornir_imageregistration.default_image_dtype())

        with nornir_imageregistration.settings.GridRefinement.CreateWithPreprocessedImages(
                target_img_data=target_image_data,
                source_img_data=source_image_data,
                num_iterations=10,
                cell_size=128, grid_spacing=96,
                angles_to_search=None, final_pass_angles=[0],
                max_travel_for_finalization=None,
                min_alignment_overlap=0.5,
                min_unmasked_area=0.49) as settings:

            FirstPassWeightScoreCutoff = None
            FirstPassCompositeScoreCutoff = None
            FirstPassFinalizeValue = None  # The score required to finalize a control point on the first pass.
            # The first score is recorded to prevent the best scores from being finalized and then later
            # groups of poor scores looking falsely good because the correct registrations are all finalized
            first_pass_weight_distance_composite_scores = None
            transform_inclusion_percentile = 33.3  # - (CutoffPercentilePerIteration * i)
            finalize_percentile = 66.6
            finalize_range = 33.3

            while i <= settings.num_iterations:

                cachedFileName = '_{6}_{5}_pass{0}_alignment_Cell_{2}x{1}_Grid_{4}x{3}'.format(
                    os.path.basename(stosFilePath), i,
                    settings.cell_size[0], settings.cell_size[1],
                    settings.grid_spacing[0], settings.grid_spacing[1],
                    self.TestName)

                alignment_points = self.ReadOrCreateVariable(cachedFileName) if use_cache else None

                if alignment_points is None:
                    alignment_points = _RefineGridPointsForTwoImages(stosTransform,
                                                                     finalized=finalized_points,
                                                                     settings=settings)
                    self.SaveVariable(alignment_points, cachedFileName)
                else:
                    duplicate_check = {a.ID: a for a in alignment_points}
                    for key in duplicate_check.keys():
                        if key in finalized_points:
                            raise ValueError("Cached alignment has a duplicate point.  Delete the cache and try again")

                print(f"Pass {i} aligned {len(alignment_points)} points")

                # if i == 1:
                #   cell_size = cell_size / 2.0

                updated_and_finalized_alignment_points = alignment_points + list(finalized_points.values())
                updated_and_finalized_weights_distance = local_distortion_correction._alignment_records_to_composite_scores(
                    updated_and_finalized_alignment_points)

                transform_inclusion_percentile_this_pass = 33.3  # - (CutoffPercentilePerIteration * i)
                transform_inclusion_percentile_this_pass = np.clip(transform_inclusion_percentile_this_pass, 10.0,
                                                                   100.0)

                (updatedTransform, included_alignment_records,
                 weight_distance_composite_scores) = local_distortion_correction._PeakListToTransform(
                    alignment_points,
                    AlignRecordsToControlPoints(finalized_points.values()),
                    percentile=transform_inclusion_percentile_this_pass,
                    cutoff=FirstPassCompositeScoreCutoff)

                if FirstPassCompositeScoreCutoff is None:
                    FirstPassCompositeScoreCutoff = np.percentile(weight_distance_composite_scores[:, 2],
                                                                  100.0 - transform_inclusion_percentile_this_pass)
                    FirstPassWeightScoreCutoff = np.percentile(weight_distance_composite_scores[:, 0],
                                                               transform_inclusion_percentile_this_pass)

                finalize_percentile = 66.6
                finalize_percentile = np.clip(finalize_percentile, 10.0, 100.0)

                FinalizeCutoffThisPass = None
                if FirstPassFinalizeValue is not None:
                    # cutoff_range = np.abs(FirstPassFinalizeValue - FirstPassWeightScoreCutoff)
                    # fraction = i / (num_iterations - 1)
                    # FinalizeCutoffThisPass = FirstPassFinalizeValue - (cutoff_range * fraction)
                    FinalizeCutoffThisPass = np.percentile(first_pass_weight_distance_composite_scores[:, 0],
                                                           finalize_percentile)
                    print(f'Finalize cutoff this pass: {FinalizeCutoffThisPass}')

                if SavePlots:
                    histogram_filename = os.path.join(self.TestOutputPath, f'weight_histogram_pass{i}.png')
                    nornir_imageregistration.views.PlotWeightHistogram(alignment_points, histogram_filename,
                                                                       transform_cutoff=transform_inclusion_percentile_this_pass / 100.0,
                                                                       finalize_cutoff=finalize_percentile / 100,
                                                                       line_pos_list=None if FirstPassFinalizeValue is None else [
                                                                           FirstPassFinalizeValue],
                                                                       title=f"Histogram of Weights, pass #{i}")
                    vector_field_filename = os.path.join(self.TestOutputPath, f'Vector_field_pass{i}.png')
                    nornir_imageregistration.views.PlotPeakList(alignment_points, list(finalized_points.values()),
                                                                vector_field_filename,
                                                                ylim=(0, settings.target_image.shape[1]),
                                                                xlim=(0, settings.target_image.shape[0]),
                                                                attrib='weight')

                new_finalized_points = local_distortion_correction.CalculateFinalizedAlignmentPointsMask(
                    alignment_points,
                    percentile=finalize_percentile,
                    max_travel_distance=settings.max_travel_for_finalization,
                    weight_cutoff=FinalizeCutoffThisPass)

                if FirstPassFinalizeValue is None:
                    FirstPassFinalizeValue = np.percentile(weight_distance_composite_scores[:, 0], finalize_percentile)

                if first_pass_weight_distance_composite_scores is None:
                    first_pass_weight_distance_composite_scores = weight_distance_composite_scores

                print(f"Finalizing {len(new_finalized_points)} points this pass.")

                new_finalized_alignments_list = list(
                    filter(lambda index_item: new_finalized_points[index_item[0]], enumerate(alignment_points)))
                new_finalized_alignments_dict = {fp[1].ID: fp[1] for fp in new_finalized_alignments_list}

                (improved_finalized_dict, improved_alignments) = local_distortion_correction.TryToImproveAlignments(
                    updatedTransform,
                    new_finalized_alignments_dict,
                    settings)

                print(f"Improved {len(improved_alignments)} finalized alignments")

                new_finalization_count = len(improved_finalized_dict)
                finalized_points = {**finalized_points, **improved_finalized_dict}

                # new_finalizations = 0
                # for (ir, record) in enumerate(alignment_points):
                #     if not new_finalized_points[ir]:
                #         continue
                #
                #     key = tuple(record.SourcePoint)
                #     if key in finalized_points:
                #         continue
                #
                #     # See if we can improve the final alignment
                #     refined_align_record = nornir_imageregistration.stos_brute.SliceToSliceRigidRegistration(record.TargetROI,
                #                                                                                       record.SourceROI,
                #                                                                                       AngleSearchRange=settings.final_pass_angles,
                #                                                                                       MinOverlap=min_alignment_overlap,
                #                                                                                       SingleThread=True,
                #                                                                                       Cluster=False,
                #                                                                                       TestFlip=False)
                #
                #     if refined_align_record.weight > record.weight and np.linalg.norm(refined_align_record.peak) < settings.max_travel_for_finalization:
                #         oldPSDDelta = record.PSDDelta
                #         record = nornir_imageregistration.alignment_record.EnhancedAlignmentRecord(ID=record.ID,
                #                                                                      TargetPoint=record.TargetPoint,
                #                                                                      SourcePoint=record.SourcePoint,
                #                                                                      peak=refined_align_record.peak,
                #                                                                      weight=refined_align_record.weight,
                #                                                                      angle=refined_align_record.angle,
                #                                                                      flipped_ud=refined_align_record.flippedud)
                #         record.PSDDelta = oldPSDDelta
                #
                #     # Create a record that is unmoving
                #     finalized_points[key] = EnhancedAlignmentRecord(record.ID,
                #                                                      TargetPoint=record.AdjustedTargetPoint,
                #                                                      SourcePoint=record.SourcePoint,
                #                                                      peak=np.asarray((0, 0), dtype=np.float32),
                #                                                      weight=record.weight, angle=0,
                #                                                      flipped_ud=record.flippedud)
                #
                #     finalized_points[key].PSDDelta = record.PSDDelta
                #
                #     new_finalizations += 1

                print(
                    f"Pass {i} has locked {new_finalization_count} new points, {len(finalized_points)} of {len(updated_and_finalized_alignment_points)} are locked")

                # Update the transform with the adjusted points
                if len(improved_alignments) > 0:
                    combined_records_this_pass = {a.ID: a for a in included_alignment_records}
                    for item in finalized_points.items():
                        combined_records_this_pass[item[0]] = item[1]

                    updatedTransform = nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(
                        AlignRecordsToControlPoints(combined_records_this_pass.values()))

                stosObj.Transform = updatedTransform
                stosObj.Save(os.path.join(self.TestOutputPath, "UpdatedTransform_pass{0}.stos".format(i)))

                if SaveImages:
                    warpedToFixedImage = nornir_imageregistration.assemble.TransformStos(updatedTransform,
                                                                                         fixedImage=settings.target_image,
                                                                                         warpedImage=settings.source_image)

                    Delta = warpedToFixedImage - settings.target_image
                    ComparisonImage = np.abs(Delta)
                    ComparisonImage = ComparisonImage / ComparisonImage.max()

                    # nornir_imageregistration.SaveImage(os.path.join(self.TestOutputPath, 'delta_pass{0}.png'.format(i)), ComparisonImage, bpp=8)
                    # nornir_imageregistration.SaveImage(os.path.join(self.TestOutputPath, 'image_pass{0}.png'.format(i)), warpedToFixedImage, bpp=8)

                    pool = nornir_pools.GetGlobalThreadPool()
                    pool.add_task(f'delta_pass{i}.png', nornir_imageregistration.SaveImage,
                                  os.path.join(self.TestOutputPath, f'delta_pass{i}.png'), np.copy(ComparisonImage),
                                  bpp=8)
                    pool.add_task(f'image_pass{i}.png', nornir_imageregistration.SaveImage,
                                  os.path.join(self.TestOutputPath, f'image_pass{i}.png'), np.copy(warpedToFixedImage),
                                  bpp=8)

                # nornir_imageregistration.core.ShowGrayscale([target_image, unrefined_warped_image, warpedToFixedImage, ComparisonImage])

                i += 1

                # Build a final transform using only finalized points
                # stosTransform = updatedTransform
                # stosTransform = local_distortion_correction._PeakListToTransform(list(finalized_points.values()), percentile=percentile)
                stosTransform = updatedTransform

                if final_pass:
                    break

                if i == settings.num_iterations:
                    final_pass = True
                    # angles_to_search = settings.final_pass_angles

                # If we've locked 10% of the points and have not locked any new ones we are done
                if len(finalized_points) > len(
                        updated_and_finalized_alignment_points) * 0.1 and new_finalization_count == 0:
                    final_pass = True
                    # angles_to_search = settings.final_pass_angles

                # If we've locked 90% of the points we are done
                if len(finalized_points) > len(updated_and_finalized_alignment_points) * 0.9:
                    final_pass = True
                    # angles_to_search = settings.final_pass_angles

            # Make one more pass to see if we can improve finalized points
            finalTransform = nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(
                AlignRecordsToControlPoints(finalized_points.values()))
            stosObj.Transform = finalTransform
            stosObj.Save(os.path.join(self.TestOutputPath, "Final_Mesh_Transform.stos"))

            (nudged_finalized_points, nudged_point_keys) = local_distortion_correction.TryToImproveAlignments(
                finalTransform, finalized_points, settings)
            print(f'Final tuning of points adjusted {len(improved_alignments)} of {len(finalized_points)} points')

            finalTransform = nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(
                AlignRecordsToControlPoints(nudged_finalized_points.values()))
            stosObj.Transform = finalTransform
            stosObj.Save(os.path.join(self.TestOutputPath, "Final_Mesh_Transform_Improved.stos"))

            # Convert the transform to a grid transform and persist to disk
            # finalTransform = nornir_imageregistration.transforms.meshwithrbffallback.MeshWithRBFFallback(AlignRecordsToControlPoints(finalized_points.values()))

            stosObj.Transform = nornir_imageregistration.transforms.ConvertTransformToGridTransform(finalTransform,
                                                                                                    source_image_shape=settings.source_image.shape,
                                                                                                    cell_size=settings.cell_size,
                                                                                                    grid_spacing=settings.grid_spacing)
            stosObj.Save(os.path.join(self.TestOutputPath, "Final_Transform.stos"))
            return

    # def testGridRefineScript(self):
    #     stosFile = self.GetStosFilePath("StosRefinementRC2_617", "0617-0618_brute_32_pyre")
    #     args = ['-input', stosFile,
    #             '-output', os.path.join(self.TestOutputPath, 'scriptTestResult.stos'),
    #             '-min_overlap', '0.5',
    #             '-grid_spacing', '128,128',
    #             '-it', '1',
    #             '-c', '256,256',
    #             '-angles', '0',
    #             '-travel_cutoff', '0.5']
    #
    #     nornir_imageregistration.scripts.nornir_stos_grid_refinement.Execute(args)
    #    return

    def _rotate_points(self, points, rotcenter, rangle):

        t = nornir_imageregistration.transforms.Rigid(target_offset=(0, 0), source_rotation_center=rotcenter,
                                                      angle=rangle)
        return t.Transform(points)

    def testTransformReductionToRigidTransform(self):
        """
        Takes the control points of a transform and converts each point to a rigid transform that approximates the offset and angle centered at that point
        """

        # A set of control points offset by (10,10)
        InitialTargetPoints = np.asarray([[0, 0],
                                          [10, 0],
                                          [0, 10],
                                          [10, 10]], dtype=np.float64)

        angle = 30.0
        rangle = (angle / 180.0) * np.pi

        CalculatedSourcePoints = self._rotate_points(InitialTargetPoints, rotcenter=(0, 0), rangle=-rangle)

        xp = cp.get_array_module(CalculatedSourcePoints)
        # Optional offset to add as an additional test (must match points' backend under CuPy)
        CalculatedSourcePoints = CalculatedSourcePoints + xp.asarray((-1, 4), dtype=xp.float64)

        controlPoints = xp.hstack(
            (xp.asarray(InitialTargetPoints, dtype=xp.float64), CalculatedSourcePoints))
        reference_transform = nornir_imageregistration.transforms.MeshWithRBFFallback(controlPoints)

        ValidationTestPoints = reference_transform.InverseTransform(InitialTargetPoints)
        np.testing.assert_allclose(
            nornir_imageregistration.EnsureNumpyArray(ValidationTestPoints),
            nornir_imageregistration.EnsureNumpyArray(CalculatedSourcePoints))

        # OK, check that the rigid transforms returned for the InitialTargetPoints perfectly match our reference_transform
        local_rigid_transforms = local_distortion_correction.ApproximateRigidTransformByTargetPoints(
            reference_transform, InitialTargetPoints)

        for i, t in enumerate(local_rigid_transforms):
            test_source_points = t.InverseTransform(InitialTargetPoints)
            np.testing.assert_allclose(
                nornir_imageregistration.EnsureNumpyArray(test_source_points),
                nornir_imageregistration.EnsureNumpyArray(CalculatedSourcePoints), atol=.006,
                err_msg="Inverse Transform Iteration {0}".format(i))

            test_target_points = t.Transform(CalculatedSourcePoints)
            np.testing.assert_allclose(
                nornir_imageregistration.EnsureNumpyArray(test_target_points),
                InitialTargetPoints, atol=.005,
                err_msg="Transform Iteration {0}".format(i))

        return

    def testAlignmentRecordsToTransforms(self):
        """
        Converts a set of alignment records into a transform
        """

        # A set of control points offset by (10,10)
        InitialTransformPoints = [[0, 0, 10, 10],
                                  [10, 0, 20, 10],
                                  [0, 10, 10, 20],
                                  [10, 10, 20, 20]]
        T = nornir_imageregistration.transforms.MeshWithRBFFallback(InitialTransformPoints)

        transformed = T.InverseTransform(T.TargetPoints)
        self.assertTrue(np.allclose(T.SourcePoints, transformed),
                        "Transform could not map the initial input to the test correctly")

        transform_testPoints = [[-1, -2]]
        expectedOutOfBounds = np.asarray([[9, 8]], dtype=np.float64)
        transformed_out_of_bounds = T.InverseTransform(transform_testPoints)
        self.assertTrue(np.allclose(transformed_out_of_bounds, expectedOutOfBounds),
                        "Transform could not map the initial input to the test correctly")

        inverse_transformed_out_of_bounds = T.Transform(expectedOutOfBounds)
        self.assertTrue(np.allclose(transform_testPoints, inverse_transformed_out_of_bounds),
                        "Transform could not map the initial input to the test correctly")

        peak_shift = (1, 0)
        # Create fake alignment results requiring us to shift the transform up one
        a = EnhancedAlignmentRecord((0, 0), (0, 0), (10, 10), peak_shift, 1.5)
        b = EnhancedAlignmentRecord((1, 0), (10, 0), (20, 10), peak_shift, 2.5)
        c = EnhancedAlignmentRecord((0, 1), (0, 10), (10, 20), peak_shift, 3)
        d = EnhancedAlignmentRecord((1, 1), (10, 10), (20, 20), peak_shift, 2)

        records = [a, b, c, d]

        # #Transforming the adjusted fixed point with the old transform generates an incorrect result
        for r in records:
            r.CalculatedWarpedPoint = T.InverseTransform(r.AdjustedTargetPoint)
        ########

        # If this is failing check that at least three records make it past the filter criteria
        (transform, included_alignment_records, calculated_cutoff) = local_distortion_correction._PeakListToTransform(
            records, nornir_imageregistration.WeightMethod.Composite)

        test1 = np.asarray(((0, 0), (5, 5), (10, 10)))
        expected1 = np.asarray(((10, 10), (15, 15), (20, 20)))
        expected1 -= np.array(peak_shift)
        actual1 = transform.InverseTransform(test1)

        self.assertTrue(np.allclose(expected1, actual1))

        records2 = []
        ####Begin 2nd pass.  Pretend we need to shift every over one
        for iRow in range(0, transform.SourcePoints.shape[0]):
            r = EnhancedAlignmentRecord(records[iRow].ID,
                                        transform.TargetPoints[iRow, :],
                                        transform.SourcePoints[iRow, :],
                                        (0, -1),
                                        5.0)
            records2.append(r)

        (
            transform2, included_alignment_records,
            calculated_cutoff_2) = local_distortion_correction._PeakListToTransform(
            records2, nornir_imageregistration.WeightMethod.Composite)
        test2 = np.asarray(((0, 0), (5, 5), (10, 10)))
        expected2 = np.asarray(((9, 11), (14, 16), (19, 21)))
        actual2 = transform2.InverseTransform(test2)

        self.assertTrue(np.allclose(expected2, actual2))

        pass

    def test_mesh_to_grid_transform(self):

        InitialTransformPoints = [[10, 10, 0, 0],
                                  [20, 10, 10, 0],
                                  [10, 20, 0, 10],
                                  [20, 20, 10, 10]]
        offset = (10, 10)

        mesh_t = nornir_imageregistration.transforms.MeshWithRBFFallback(InitialTransformPoints)

        grid_t = nornir_imageregistration.transforms.ConvertTransformToGridTransform(mesh_t,
                                                                                     source_image_shape=(
                                                                                         10, 10),
                                                                                     cell_size=1,
                                                                                     grid_dims=(10, 5))

        test_points = np.asarray(((0, 0), (5, 5), (10, 10)))
        expected_points = np.asarray(((10, 10), (15, 15), (20, 20)))

        mesh_transformed_points = mesh_t.Transform(test_points)
        grid_transformed_points = grid_t.Transform(test_points)

        self.assertTrue(np.array_equal(mesh_transformed_points, expected_points))
        self.assertTrue(np.array_equal(mesh_transformed_points, grid_transformed_points))

        grid_t_grid_spacing = nornir_imageregistration.transforms.ConvertTransformToGridTransform(
            mesh_t,
            source_image_shape=(10, 10),
            cell_size=1,
            grid_spacing=(1, 2))

        grid_spacing_transformed_points = grid_t_grid_spacing.Transform(test_points)
        self.assertTrue(np.array_equal(grid_spacing_transformed_points, grid_transformed_points))


class TestMosaicGridRefinementApi(unittest.TestCase):

    def test_refinement_base_target_uses_inverse_warp_anchor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_shape = (64, 64)
            image_a = np.zeros(image_shape, dtype=np.float32)
            image_b = np.zeros(image_shape, dtype=np.float32)
            image_a[16:48, 16:48] = 1.0
            image_b[16:48, 20:52] = 1.0

            image_a_path = os.path.join(temp_dir, "1.png")
            image_b_path = os.path.join(temp_dir, "2.png")
            nornir_imageregistration.SaveImage(image_a_path, image_a, bpp=8)
            nornir_imageregistration.SaveImage(image_b_path, image_b, bpp=8)

            transforms = {
                "1.png": nornir_imageregistration.transforms.RigidTranslation((0, 0)),
                "2.png": nornir_imageregistration.transforms.RigidTranslation((0, 40)),
            }
            tiles = nornir_imageregistration.mosaic_tileset.Create(
                list(transforms.values()),
                [image_a_path, image_b_path],
                image_to_source_space_scale=1.0)
            tile_a, tile_b = list(tiles.values())
            overlap = nornir_imageregistration.tile_overlap.TileOverlap(
                tile_a, tile_b, image_to_source_space_scale=1.0)
            subregion_shape = np.array([32, 32], dtype=np.int64)
            geometry = local_distortion_correction._compute_padded_overlap_geometry(
                overlap.scaled_overlapping_source_rect_A,
                overlap.scaled_overlapping_source_rect_B,
                overlap.overlapping_target_rect,  # type: ignore[arg-type]
                subregion_shape,
                1.0)
            subregion_offset = np.array([32.0, 32.0])
            _, _, _, base_target = local_distortion_correction._refinement_cell_geometry(
                subregion_offset,
                geometry.target_region_rect,
                1.0,
                tile_a,
                tile_b)
            expected = local_distortion_correction._target_center_for_refinement_cell(
                subregion_offset,
                geometry.target_region_rect,
                1.0)
            self.assertTrue(np.allclose(base_target, expected, atol=1e-3))

    def test_inverse_warp_cell_mapping_differs_from_linear_for_nonlinear_grid(self):
        """Non-linear transforms must inverse-warp the target cell center for source anchors."""
        rigid = nornir_imageregistration.transforms.RigidTranslation((100.0, 200.0))
        grid = nornir_imageregistration.transforms.ConvertTransformToGridTransform(
            rigid,
            source_image_shape=np.asarray([128, 128], dtype=np.int64),
            cell_size=(16, 16),
            grid_dims=(9, 9))
        bent_targets = np.asarray(grid.TargetPoints, dtype=np.float64).copy()
        bent_targets[:, 0] += np.linspace(-8.0, 8.0, bent_targets.shape[0])
        for index in range(bent_targets.shape[0]):
            grid.UpdateTargetPointsByIndex(index, bent_targets[index])

        with tempfile.TemporaryDirectory() as temp_dir:
            image = np.zeros((128, 128), dtype=np.float32)
            image[32:96, 32:96] = 1.0
            image_path = os.path.join(temp_dir, "tile.png")
            nornir_imageregistration.SaveImage(image_path, image, bpp=8)
            tile_a = nornir_imageregistration.mosaic_tileset.Create(
                [grid],
                [image_path],
                image_to_source_space_scale=1.0)[0]
            tile_b = tile_a
            subregion_shape = np.array([32, 32], dtype=np.int64)
            overlapping_target_rect = nornir_imageregistration.Rectangle.CreateFromPointAndArea(
                (100.0, 200.0), (64.0, 64.0))
            geometry = local_distortion_correction._compute_padded_overlap_geometry(
                nornir_imageregistration.Rectangle.CreateFromPointAndArea((16.0, 16.0), (64.0, 64.0)),
                nornir_imageregistration.Rectangle.CreateFromPointAndArea((16.0, 16.0), (64.0, 64.0)),
                overlapping_target_rect,
                subregion_shape,
                1.0)
            subregion_offset = np.array([16.0, 16.0])
            _, inverse_source_a, _, _ = local_distortion_correction._refinement_cell_geometry(
                subregion_offset,
                geometry.target_region_rect,
                1.0,
                tile_a,
                tile_b)
            linear_source_a, _ = local_distortion_correction._full_source_points_for_refinement_cell(
                subregion_offset,
                geometry.padded_scaled_source_rect_a,
                geometry.padded_scaled_source_rect_b,
                1.0)
            self.assertFalse(np.allclose(inverse_source_a, linear_source_a, atol=1e-3))

    def test_refinement_base_target_uses_transform_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_shape = (64, 64)
            image_a = np.zeros(image_shape, dtype=np.float32)
            image_b = np.zeros(image_shape, dtype=np.float32)
            image_a[16:48, 16:48] = 1.0
            image_b[16:48, 20:52] = 1.0

            image_a_path = os.path.join(temp_dir, "1.png")
            image_b_path = os.path.join(temp_dir, "2.png")
            nornir_imageregistration.SaveImage(image_a_path, image_a, bpp=8)
            nornir_imageregistration.SaveImage(image_b_path, image_b, bpp=8)

            transforms = {
                "1.png": nornir_imageregistration.transforms.RigidTranslation((0, 0)),
                "2.png": nornir_imageregistration.transforms.RigidTranslation((0, 40)),
            }
            tiles = nornir_imageregistration.mosaic_tileset.Create(
                list(transforms.values()),
                [image_a_path, image_b_path],
                image_to_source_space_scale=1.0)
            tile_a, tile_b = list(tiles.values())
            overlap = nornir_imageregistration.tile_overlap.TileOverlap(
                tile_a, tile_b, image_to_source_space_scale=1.0)
            subregion_offset = np.array([32.0, 32.0])
            full_a, full_b = local_distortion_correction._full_source_points_for_refinement_cell(
                subregion_offset,
                overlap.scaled_overlapping_source_rect_A,
                overlap.scaled_overlapping_source_rect_B,
                1.0)
            base_target = local_distortion_correction._base_target_for_refinement_cell(
                tile_a, tile_b, full_a, full_b)
            expected = tile_a.Transform.Transform(np.asarray([full_a], dtype=np.float64))[0]
            self.assertTrue(np.allclose(base_target, expected, atol=1e-3))

    def test_split_displacements_balances_offsets(self):
        refine_dtype = np.dtype([('SourceAY', 'f4'),
                                 ('SourceAX', 'f4'),
                                 ('SourceBY', 'f4'),
                                 ('SourceBX', 'f4'),
                                 ('BaseTargetY', 'f4'),
                                 ('BaseTargetX', 'f4'),
                                 ('TargetY', 'f4'),
                                 ('TargetX', 'f4'),
                                 ('DisplacementY', 'f4'),
                                 ('DisplacementX', 'f4'),
                                 ('Weight', 'f4'),
                                 ('Angle', 'f4')])

        point_pairs = np.zeros((1, 1), dtype=refine_dtype)
        point_pairs[0, 0] = (20, 10, 21, 11, 100, 200, 102, 198, 2, -2, 0.8, 0)

        a_updates, b_updates = local_distortion_correction.SplitDisplacements(None, None, point_pairs)
        self.assertEqual(1, a_updates.shape[0])
        self.assertEqual(1, b_updates.shape[0])

        self.assertTrue(np.allclose([101, 199], [a_updates['TargetY'][0], a_updates['TargetX'][0]]))
        self.assertTrue(np.allclose([99, 201], [b_updates['TargetY'][0], b_updates['TargetX'][0]]))
        self.assertEqual(a_updates['Weight'][0], b_updates['Weight'][0])

    def test_refine_grid_mosaic_returns_grid_transforms(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_shape = (64, 64)
            image_a = np.zeros(image_shape, dtype=np.float32)
            image_b = np.zeros(image_shape, dtype=np.float32)
            image_a[16:48, 16:48] = 1.0
            image_b[16:48, 20:52] = 1.0

            image_a_path = os.path.join(temp_dir, "1.png")
            image_b_path = os.path.join(temp_dir, "2.png")
            nornir_imageregistration.SaveImage(image_a_path, image_a, bpp=8)
            nornir_imageregistration.SaveImage(image_b_path, image_b, bpp=8)

            transforms = {
                "1.png": nornir_imageregistration.transforms.RigidTranslation((0, 0)),
                "2.png": nornir_imageregistration.transforms.RigidTranslation((0, 8)),
            }
            mosaic = nornir_imageregistration.mosaic.Mosaic(transforms)

            refine_fn_name = "__RefineTileOverlapBatchRemote"
            original_fn = getattr(local_distortion_correction, refine_fn_name)
            original_pool_fn = local_distortion_correction._refinement_pool_for_overlap_tasks

            setattr(local_distortion_correction, refine_fn_name, _fake_refine_tile_overlap_batch_remote)
            local_distortion_correction._refinement_pool_for_overlap_tasks = _serial_refinement_pool
            try:
                refined_mosaic, diagnostics = nornir_imageregistration.RefineGridMosaic(
                    mosaic,
                    temp_dir,
                    iterations=1,
                    cell_size=(32, 32),
                    displacement_threshold=0.0,
                    return_diagnostics=True)
            finally:
                setattr(local_distortion_correction, refine_fn_name, original_fn)
                local_distortion_correction._refinement_pool_for_overlap_tasks = original_pool_fn

            first = refined_mosaic.ImageToTransform["1.png"]
            second = refined_mosaic.ImageToTransform["2.png"]
            self.assertTrue(isinstance(first, nornir_imageregistration.transforms.IGridTransform))
            self.assertTrue(isinstance(second, nornir_imageregistration.transforms.IGridTransform))
            self.assertEqual(diagnostics.iterations_completed, 1)

            mfile = refined_mosaic.ToMosaicFile()
            self.assertIn("GridTransform_double_2_2", mfile.ImageToTransformString["1.png"])
            self.assertIn("GridTransform_double_2_2", mfile.ImageToTransformString["2.png"])

    def test_refine_grid_mosaic_uses_full_resolution_source_bounds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            full_shape = (80, 80)
            downsample = 4
            ds_shape = (full_shape[0] // downsample, full_shape[1] // downsample)

            image_a = np.zeros(ds_shape, dtype=np.float32)
            image_b = np.zeros(ds_shape, dtype=np.float32)
            image_a[4:16, 4:16] = 1.0
            image_b[4:16, 6:18] = 1.0

            image_a_path = os.path.join(temp_dir, "1.png")
            image_b_path = os.path.join(temp_dir, "2.png")
            nornir_imageregistration.SaveImage(image_a_path, image_a, bpp=8)
            nornir_imageregistration.SaveImage(image_b_path, image_b, bpp=8)

            transforms = {
                "1.png": nornir_imageregistration.transforms.RigidTranslation((0, 0)),
                "2.png": nornir_imageregistration.transforms.RigidTranslation((0, 40)),
            }
            mosaic = nornir_imageregistration.mosaic.Mosaic(transforms)

            refine_fn_name = "__RefineTileOverlapBatchRemote"
            original_fn = getattr(local_distortion_correction, refine_fn_name)
            original_pool_fn = local_distortion_correction._refinement_pool_for_overlap_tasks

            setattr(local_distortion_correction, refine_fn_name, _fake_refine_tile_overlap_batch_remote)
            local_distortion_correction._refinement_pool_for_overlap_tasks = _serial_refinement_pool
            try:
                refined_mosaic = cast(
                    nornir_imageregistration.mosaic.Mosaic,
                    nornir_imageregistration.RefineGridMosaic(
                        cast(Any, mosaic),
                        temp_dir,
                        iterations=1,
                        cell_size=(16, 16),
                        imageScale=1.0 / downsample,
                        displacement_threshold=0.0))
            finally:
                setattr(local_distortion_correction, refine_fn_name, original_fn)
                local_distortion_correction._refinement_pool_for_overlap_tasks = original_pool_fn

            mfile = refined_mosaic.ToMosaicFile()
            transform_str = mfile.ImageToTransformString["1.png"]
            self.assertIn("GridTransform_double_2_2", transform_str)

            fp_match = re.search(
                r"\bfp\s+\d+\s+\d+\s+\d+\s+\d+\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)",
                transform_str)
            self.assertIsNotNone(fp_match, "Expected fixed-parameter bounds in transform string")
            if fp_match is None:
                self.fail("Expected fixed-parameter bounds in transform string")
            _, _, max_x_str, max_y_str = fp_match.groups()
            max_x = float(max_x_str)
            max_y = float(max_y_str)

            # fp bounds should reflect full-resolution source geometry, not downsampled tile size.
            self.assertGreaterEqual(max_x, full_shape[1] - 2)
            self.assertGreaterEqual(max_y, full_shape[0] - 2)

    def test_grid_sparse_update_preserves_unmeasured_cells(self):
        """Measured cells move; unmeasured grid corners stay fixed."""
        rigid = nornir_imageregistration.transforms.RigidTranslation((100.0, 200.0))
        grid = nornir_imageregistration.transforms.ConvertTransformToGridTransform(
            rigid,
            source_image_shape=np.asarray([64, 64], dtype=np.int64),
            cell_size=(16, 16),
            grid_dims=(5, 5))

        before = np.asarray(grid.TargetPoints, dtype=np.float64)
        corner_index = 0
        corner_before = before[corner_index].copy()

        center_source = np.asarray(grid.SourcePoints[grid.NumControlPoints // 2], dtype=np.float64)
        point_pairs = np.asarray(
            [[110.0, 210.0, center_source[0], center_source[1]]],
            dtype=np.float64)

        updated = local_distortion_correction._apply_point_pair_updates_to_grid_transform(grid, point_pairs)
        self.assertEqual(1, updated)

        after = np.asarray(grid.TargetPoints, dtype=np.float64)
        np.testing.assert_allclose(after[corner_index], corner_before, atol=1e-4)
        self.assertFalse(np.allclose(after[corner_index], point_pairs[0, 0:2]))

    def test_grid_multi_update_averages_same_cell(self):
        """Two measurements mapping to the same grid cell produce a weighted-average delta."""
        rigid = nornir_imageregistration.transforms.RigidTranslation((0.0, 0.0))
        grid = nornir_imageregistration.transforms.ConvertTransformToGridTransform(
            rigid,
            source_image_shape=np.asarray([64, 64], dtype=np.int64),
            cell_size=(16, 16),
            grid_dims=(5, 5))

        source_point = np.asarray(grid.SourcePoints[0], dtype=np.float64)
        point_pairs = np.asarray([
            [5.0, 5.0, source_point[0], source_point[1]],
            [7.0, 7.0, source_point[0], source_point[1]],
        ], dtype=np.float64)

        local_distortion_correction._apply_point_pair_updates_to_grid_transform(grid, point_pairs)
        expected = np.asarray([6.0, 6.0], dtype=np.float64)
        np.testing.assert_allclose(
            nornir_imageregistration.EnsureNumpyArray(grid.TargetPoints[0]),
            expected,
            atol=1e-4)

    def test_grid_update_with_less_than_three_points(self):
        """Single-point updates must still adjust the grid (legacy sparse-cell behavior)."""
        rigid = nornir_imageregistration.transforms.RigidTranslation((0.0, 0.0))
        grid = nornir_imageregistration.transforms.ConvertTransformToGridTransform(
            rigid,
            source_image_shape=np.asarray([64, 64], dtype=np.int64),
            cell_size=(32, 32),
            grid_dims=(3, 3))

        before = np.asarray(grid.TargetPoints[4], dtype=np.float64)
        source_point = np.asarray(grid.SourcePoints[4], dtype=np.float64)
        point_pairs = np.asarray([[4.0, 6.0, source_point[0], source_point[1]]], dtype=np.float64)

        updated = local_distortion_correction._apply_point_pair_updates_to_grid_transform(grid, point_pairs)
        self.assertEqual(1, updated)
        after_center = nornir_imageregistration.EnsureNumpyArray(grid.TargetPoints[4])
        self.assertFalse(np.allclose(after_center, before, atol=1e-4))

    def test_mesh_rebuild_resamples_full_output_grid(self):
        """Merged overlap pairs must rebuild a mesh and repopulate every output grid node."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_shape = (64, 64)
            image_path = os.path.join(temp_dir, "tile.png")
            nornir_imageregistration.SaveImage(
                image_path,
                np.zeros(image_shape, dtype=np.float32),
                bpp=8)

            rigid = nornir_imageregistration.transforms.RigidTranslation((10.0, 20.0))
            source_shape = np.asarray(image_shape, dtype=np.int64)
            grid = nornir_imageregistration.transforms.ConvertTransformToGridTransform(
                rigid,
                source_image_shape=source_shape,
                cell_size=(16, 16),
                grid_dims=(5, 5))

            sources = np.asarray(grid.SourcePoints, dtype=np.float64)
            targets = np.asarray(grid.Transform(sources), dtype=np.float64)
            targets[0] += np.asarray([2.0, 3.0], dtype=np.float64)
            targets[4] += np.asarray([-1.0, 1.5], dtype=np.float64)
            targets[20] += np.asarray([0.5, -0.5], dtype=np.float64)
            point_pairs = np.hstack([targets[[0, 4, 20]], sources[[0, 4, 20]]])

            tile = nornir_imageregistration.Tile(
                grid,
                image_path,
                image_to_source_space_scale=1.0,
                ID=0)
            updated = local_distortion_correction._update_tile_transform_from_merged_overlap_pairs(
                tile,
                point_pairs,
                resolved_cell_size=(16, 16),
                resolved_mesh_shape=(5, 5))
            self.assertEqual(3, updated)
            self.assertTrue(isinstance(tile.Transform, nornir_imageregistration.transforms.IGridTransform))
            self.assertEqual((5, 5), tile.Transform.grid_dims)

    def test_refine_grid_mosaic_initializes_grid_before_iteration(self):
        """RefineGridMosaic must emit grid transforms even when overlap refinement is stubbed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_shape = (64, 64)
            image_a = np.zeros(image_shape, dtype=np.float32)
            image_b = np.zeros(image_shape, dtype=np.float32)
            image_a[16:48, 16:48] = 1.0
            image_b[16:48, 20:52] = 1.0

            image_a_path = os.path.join(temp_dir, "1.png")
            image_b_path = os.path.join(temp_dir, "2.png")
            nornir_imageregistration.SaveImage(image_a_path, image_a, bpp=8)
            nornir_imageregistration.SaveImage(image_b_path, image_b, bpp=8)

            transforms = {
                "1.png": nornir_imageregistration.transforms.RigidTranslation((0, 0)),
                "2.png": nornir_imageregistration.transforms.RigidTranslation((0, 8)),
            }
            mosaic = nornir_imageregistration.mosaic.Mosaic(transforms)

            refine_fn_name = "__RefineTileOverlapBatchRemote"
            original_fn = getattr(local_distortion_correction, refine_fn_name)
            original_pool_fn = local_distortion_correction._refinement_pool_for_overlap_tasks

            def _empty_overlap_refinement_batch(_anchor, overlap_batch, *_args, **_kwargs):
                refine_dtype = np.dtype([('SourceAY', 'f4'),
                                         ('SourceAX', 'f4'),
                                         ('SourceBY', 'f4'),
                                         ('SourceBX', 'f4'),
                                         ('BaseTargetY', 'f4'),
                                         ('BaseTargetX', 'f4'),
                                         ('TargetY', 'f4'),
                                         ('TargetX', 'f4'),
                                         ('DisplacementY', 'f4'),
                                         ('DisplacementX', 'f4'),
                                         ('Weight', 'f4'),
                                         ('Angle', 'f4')])
                empty = np.zeros((1, 1), dtype=refine_dtype)
                empty_result = (empty, np.asarray((0.0, 0.0, 0.0), dtype=np.float32))
                return [empty_result] * len(overlap_batch)

            setattr(local_distortion_correction, refine_fn_name, _empty_overlap_refinement_batch)
            local_distortion_correction._refinement_pool_for_overlap_tasks = _serial_refinement_pool
            try:
                refined_mosaic = nornir_imageregistration.RefineGridMosaic(
                    mosaic,
                    temp_dir,
                    iterations=1,
                    cell_size=(32, 32),
                    mesh_shape=(3, 3),
                    displacement_threshold=0.0)
            finally:
                setattr(local_distortion_correction, refine_fn_name, original_fn)
                local_distortion_correction._refinement_pool_for_overlap_tasks = original_pool_fn

            for transform in refined_mosaic.ImageToTransform.values():
                self.assertTrue(isinstance(transform, nornir_imageregistration.transforms.IGridTransform))
                self.assertEqual((3, 3), transform.grid_dims)

    def test_refinement_pool_serial_when_cupy(self):
        original_using_cupy = nornir_imageregistration.UsingCupy
        original_mt_pool = nornir_pools.GetGlobalMultithreadingPool
        original_serial_pool = nornir_pools.GetGlobalSerialPool
        try:
            nornir_imageregistration.UsingCupy = lambda: True  # type: ignore[method-assign]
            nornir_pools.GetGlobalMultithreadingPool = lambda: object()  # type: ignore[assignment]
            nornir_pools.GetGlobalSerialPool = lambda: "serial-pool"  # type: ignore[assignment]
            self.assertIs(
                local_distortion_correction._refinement_pool_for_overlap_tasks(0.25),
                "serial-pool")
        finally:
            nornir_imageregistration.UsingCupy = original_using_cupy  # type: ignore[method-assign]
            nornir_pools.GetGlobalMultithreadingPool = original_mt_pool
            nornir_pools.GetGlobalSerialPool = original_serial_pool

    def test_refinement_pool_serial_at_full_resolution(self):
        original_using_cupy = nornir_imageregistration.UsingCupy
        original_mt_pool = nornir_pools.GetGlobalMultithreadingPool
        original_serial_pool = nornir_pools.GetGlobalSerialPool
        try:
            nornir_imageregistration.UsingCupy = lambda: False  # type: ignore[method-assign]
            nornir_pools.GetGlobalMultithreadingPool = lambda: object()  # type: ignore[assignment]
            nornir_pools.GetGlobalSerialPool = lambda: "serial-pool"  # type: ignore[assignment]
            self.assertIs(
                local_distortion_correction._refinement_pool_for_overlap_tasks(1.0),
                "serial-pool")
        finally:
            nornir_imageregistration.UsingCupy = original_using_cupy  # type: ignore[method-assign]
            nornir_pools.GetGlobalMultithreadingPool = original_mt_pool
            nornir_pools.GetGlobalSerialPool = original_serial_pool

    def test_compute_padded_overlap_geometry_uses_overlap_not_full_tile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_shape = (256, 256)
            image_a = np.zeros(image_shape, dtype=np.float32)
            image_b = np.zeros(image_shape, dtype=np.float32)
            image_a[64:192, 64:192] = 1.0
            image_b[64:192, 96:224] = 1.0
            image_a_path = os.path.join(temp_dir, "1.png")
            image_b_path = os.path.join(temp_dir, "2.png")
            nornir_imageregistration.SaveImage(image_a_path, image_a, bpp=8)
            nornir_imageregistration.SaveImage(image_b_path, image_b, bpp=8)

            tiles = nornir_imageregistration.mosaic_tileset.Create(
                [nornir_imageregistration.transforms.RigidTranslation((0, 0)),
                 nornir_imageregistration.transforms.RigidTranslation((0, 32))],
                [image_a_path, image_b_path],
                image_to_source_space_scale=1.0)
            tile_a, tile_b = list(tiles.values())
            overlap = nornir_imageregistration.tile_overlap.TileOverlap(
                tile_a, tile_b, image_to_source_space_scale=1.0)
            subregion_shape = np.array([32, 32], dtype=np.int64)

            geometry = local_distortion_correction._compute_padded_overlap_geometry(
                overlap.scaled_overlapping_source_rect_A,
                overlap.scaled_overlapping_source_rect_B,
                overlap.overlapping_target_rect,  # type: ignore[arg-type]
                subregion_shape,
                1.0)

            self.assertLess(
                geometry.padded_scaled_source_rect_a.Area,
                float(tile_a.ImageSize[0] * tile_a.ImageSize[1]))
            self.assertLess(geometry.target_region_rect.Width, tile_a.ImageSize[1] * 2)

    def test_warp_overlap_for_grid_refine_spills_large_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image = np.random.rand(128, 128).astype(np.float32)
            image_path = os.path.join(temp_dir, "tile.png")
            nornir_imageregistration.SaveImage(image_path, image, bpp=8)
            tile = nornir_imageregistration.mosaic_tileset.Create(
                [nornir_imageregistration.transforms.RigidTranslation((0, 0))],
                [image_path],
                image_to_source_space_scale=1.0)[0]

            source_rect = nornir_imageregistration.Rectangle.CreateFromPointAndArea((16, 16), (96, 96))
            target_rect = nornir_imageregistration.Rectangle.CreateFromPointAndArea((16, 16), (96, 96))

            original_create = (
                nornir_imageregistration.transformed_image_data_temp_files.TransformedImageDataViaTempFile.Create)
            create_kwargs = []

            def _recording_create(*args, **kwargs):
                create_kwargs.append(kwargs)
                return original_create(*args, **kwargs)

            nornir_imageregistration.transformed_image_data_temp_files.TransformedImageDataViaTempFile.Create = (
                _recording_create)
            try:
                result = local_distortion_correction._warp_overlap_for_grid_refine(
                    tile,
                    source_rect,
                    target_rect,
                    target_space_scale=1.0,
                    single_threaded_invoke=False)
            finally:
                nornir_imageregistration.transformed_image_data_temp_files.TransformedImageDataViaTempFile.Create = (
                    original_create)

            self.assertEqual(1, len(create_kwargs))
            self.assertFalse(create_kwargs[0]['SingleThreadedInvoke'])
            self.assertFalse(isinstance(
                result,
                nornir_imageregistration.transformed_image_data.TransformedImageDataError))

    def test_warp_overlap_matches_transform_tile(self):
        """Overlap crop warp must match the legacy TransformTile overlap ROI path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_shape = (128, 128)
            image_a = np.random.rand(*image_shape).astype(np.float32)
            image_b = np.random.rand(*image_shape).astype(np.float32)
            image_a_path = os.path.join(temp_dir, "a.png")
            image_b_path = os.path.join(temp_dir, "b.png")
            nornir_imageregistration.SaveImage(image_a_path, image_a, bpp=8)
            nornir_imageregistration.SaveImage(image_b_path, image_b, bpp=8)

            tiles = nornir_imageregistration.mosaic_tileset.Create(
                [nornir_imageregistration.transforms.RigidTranslation((0, 0)),
                 nornir_imageregistration.transforms.RigidTranslation((0, 64))],
                [image_a_path, image_b_path],
                image_to_source_space_scale=1.0)
            tile_a, tile_b = list(tiles.values())
            overlap = nornir_imageregistration.tile_overlap.TileOverlap(
                tile_a, tile_b, image_to_source_space_scale=1.0)
            subregion_shape = np.array([32, 32], dtype=np.int64)
            grid_dim = np.asarray(
                nornir_imageregistration.TileGridShape(
                    overlap.scaled_overlapping_source_rect_A.Size, subregion_shape),
                dtype=np.int64)
            target_region = local_distortion_correction._legacy_overlap_target_region(
                tile_a, tile_b, grid_dim, subregion_shape, 1.0)
            padded_a, padded_b = local_distortion_correction._padded_scaled_overlap_source_rects(
                overlap.scaled_overlapping_source_rect_A,
                overlap.scaled_overlapping_source_rect_B,
                grid_dim,
                subregion_shape)

            for tile, padded_rect in ((tile_a, padded_a), (tile_b, padded_b)):
                custom = local_distortion_correction._warp_overlap_for_grid_refine(
                    tile, padded_rect, target_region, 1.0, False)
                legacy = nornir_imageregistration.assemble_tiles.TransformTile(
                    tile,
                    TargetRegion=target_region,
                    target_space_scale=1.0,
                    SingleThreadedInvoke=False)
                self.assertFalse(isinstance(
                    custom,
                    nornir_imageregistration.transformed_image_data.TransformedImageDataError))
                self.assertFalse(isinstance(
                    legacy,
                    nornir_imageregistration.transformed_image_data.TransformedImageDataError))
                self.assertEqual(custom.image.shape, legacy.image.shape)
                np.testing.assert_allclose(custom.image, legacy.image, rtol=0, atol=1e-6)

    def test_refine_overlap_pair_keeps_target_coordinates_bounded(self):
        """Refinement must not produce runaway target-space displacements on rigid mosaics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_shape = (256, 256)
            image_a = np.zeros(image_shape, dtype=np.float32)
            image_b = np.zeros(image_shape, dtype=np.float32)
            image_a[64:192, 64:192] = np.random.rand(128, 128).astype(np.float32)
            image_b[64:192, 96:224] = image_a[64:192, 96:224]
            image_a_path = os.path.join(temp_dir, "a.png")
            image_b_path = os.path.join(temp_dir, "b.png")
            nornir_imageregistration.SaveImage(image_a_path, image_a, bpp=8)
            nornir_imageregistration.SaveImage(image_b_path, image_b, bpp=8)

            tiles = nornir_imageregistration.mosaic_tileset.Create(
                [nornir_imageregistration.transforms.RigidTranslation((0, 0)),
                 nornir_imageregistration.transforms.RigidTranslation((0, 32))],
                [image_a_path, image_b_path],
                image_to_source_space_scale=1.0)
            tile_a, tile_b = list(tiles.values())
            overlap = nornir_imageregistration.tile_overlap.TileOverlap(
                tile_a, tile_b, image_to_source_space_scale=1.0)
            subregion_shape = np.array([32, 32], dtype=np.int64)
            local_distortion_correction._initialize_tile_grid_transforms(
                [tile_a, tile_b],
                resolved_cell_size=(32, 32),
                resolved_mesh_shape=(8, 8))

            point_pairs, _ = local_distortion_correction._refine_single_tile_overlap_pair(
                tile_a,
                tile_b,
                overlap.scaled_overlapping_source_rect_A,
                overlap.scaled_overlapping_source_rect_B,
                overlap.overlapping_target_rect,  # type: ignore[arg-type]
                1.0,
                subregion_shape)

            flattened = point_pairs.reshape(-1)
            overlap_target = overlap.overlapping_target_rect  # type: ignore[union-attr]
            margin = float(max(overlap_target.Width, overlap_target.Height))
            valid = flattened[flattened['Weight'] > 0]
            if valid.size > 0:
                self.assertLess(np.max(np.abs(valid['DisplacementY'])), margin)
                self.assertLess(np.max(np.abs(valid['DisplacementX'])), margin)
                self.assertLess(np.max(valid['TargetY']), overlap_target.MaxY + margin)
                self.assertLess(np.max(valid['TargetX']), overlap_target.MaxX + margin)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
