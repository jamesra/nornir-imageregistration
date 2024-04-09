'''
Created on Sep 26, 2018

@author: u0490822
'''
import os
import os.path
import unittest

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

import nornir_imageregistration
from nornir_imageregistration import local_distortion_correction
from nornir_imageregistration.alignment_record import EnhancedAlignmentRecord
import nornir_imageregistration.assemble
from nornir_imageregistration.local_distortion_correction import AlignRecordsToControlPoints, RefineStosFile, \
    _RefineGridPointsForTwoImages
import nornir_imageregistration.scripts.nornir_stos_grid_refinement
import nornir_pools
import setup_imagetest


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

    def testStosRefinementRC2_27_26_Grid32_to_Grid16(self):
        """
        This is an incorrectly aligned brute output.  The goal is to have the alignment exit without going off the rails or producing horrible output.
        """

        # Do not stress about this test until you verify the input transform was
        # not affected by the grid transform saving bug and that it is a valid
        # starting point

        # self.TestName = "StosRefinementRC2_617"
        # stosFilePath = self.GetStosFilePath("StosRefinementRPC3_14_13_DS32_From_Brute",
        #                                     "14-13_ctrl-TEM_Leveled_map-TEM_Leveled.stos")
        # self.RunStosRefinement(stosFilePath, ImageDir=os.path.dirname(stosFilePath), SaveImages=False, SavePlots=True)
        stosFilePath = os.path.join("D:", "Data", "RC2", "TEM", "Grid32", "Automatic",
                                    "26-27_ctrl-TEM_Blob_map-TEM_Blob.stos")
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

    # def testStosRefinementRPC2_1156_1155_Grid16_to_Grid8(self):
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
    #     stosFilePath = os.path.join("D:", "Data", "RPC2", "TEM", "Grid8", "Automatic",
    #                                 "1156-1155_ctrl-TEM_Leveled_map-TEM_Leveled.stos")
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
        '''
        This is a test for the refine mosaic feature which is not fully implemented
        '''
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
                #     refined_align_record = nornir_imageregistration.stos_brute.SliceToSliceBruteForce(record.TargetROI,
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
        '''
        Takes the control points of a transform and converts each point to a rigid transform that approximates the offset and angle centered at that point
        '''

        # A set of control points offset by (10,10)
        InitialTargetPoints = np.asarray([[0, 0],
                                          [10, 0],
                                          [0, 10],
                                          [10, 10]], dtype=np.float64)

        angle = 30.0
        rangle = (angle / 180.0) * np.pi

        CalculatedSourcePoints = self._rotate_points(InitialTargetPoints, rotcenter=(0, 0), rangle=-rangle)

        # Optional offset to add as an additional test
        CalculatedSourcePoints += np.array((-1, 4))

        controlPoints = np.hstack((InitialTargetPoints, CalculatedSourcePoints))
        reference_transform = nornir_imageregistration.transforms.MeshWithRBFFallback(controlPoints)

        ValidationTestPoints = reference_transform.InverseTransform(InitialTargetPoints)
        np.testing.assert_allclose(ValidationTestPoints, CalculatedSourcePoints)

        # OK, check that the rigid transforms returned for the InitialTargetPoints perfectly match our reference_transform
        local_rigid_transforms = local_distortion_correction.ApproximateRigidTransformByTargetPoints(
            reference_transform, InitialTargetPoints)

        for i, t in enumerate(local_rigid_transforms):
            test_source_points = t.InverseTransform(InitialTargetPoints)
            np.testing.assert_allclose(test_source_points, CalculatedSourcePoints, atol=.006,
                                       err_msg="Inverse Transform Iteration {0}".format(i))

            test_target_points = t.Transform(CalculatedSourcePoints)
            np.testing.assert_allclose(test_target_points, InitialTargetPoints, atol=.005,
                                       err_msg="Transform Iteration {0}".format(i))

        return

    def testAlignmentRecordsToTransforms(self):
        '''
        Converts a set of alignment records into a transform
        '''

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
            records)

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
            records2)
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


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
