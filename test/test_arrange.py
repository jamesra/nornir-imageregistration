'''
Created on Dec 13, 2013

@author: James Anderson
'''

import glob
import os
import unittest

import matplotlib
import matplotlib.pyplot
from nornir_imageregistration.alignment_record import AlignmentRecord
from nornir_imageregistration.files.mosaicfile import MosaicFile
import nornir_imageregistration.layout
from nornir_imageregistration.mosaic import Mosaic
import nornir_imageregistration.mosaic
from scipy import stats
from scipy.misc import imsave

import nornir_imageregistration.arrange_mosaic as arrange
import nornir_imageregistration.assemble_tiles as at
import nornir_imageregistration.core as core
import nornir_imageregistration.tileset as tileset 
import nornir_imageregistration.transforms.factory as tfactory
import nornir_pools
import nornir_shared.plot
import nornir_shared.histogram
from nornir_shared.tasktimer import TaskTimer
import numpy as np

from nornir_imageregistration.views import plot_tile_overlap

from . import setup_imagetest


# from pylab import *
def _GetWorstOffsetPair(layout_obj):
    
    offsets = nornir_imageregistration.layout.OffsetsSortedByWeight(layout_obj)
    TileA_ID = offsets[-1, 0]
    TileB_ID = offsets[-1, 1]
    return (TileA_ID, TileB_ID)

def _GetBestOffsetPair(layout_obj):
    
    offsets = nornir_imageregistration.layout.OffsetsSortedByWeight(layout_obj)
    TileA_ID = offsets[0, 0]
    TileB_ID = offsets[0, 1]
    return (TileA_ID, TileB_ID)


class TestBasicTileAlignment(setup_imagetest.TransformTestBase):

    def test_Alignments(self):

        Downsample = 1.0
        DownsampleString = "%03d" % Downsample 
        self.TilesPath = os.path.join(self.ImportedDataPath, "PMG1", "Leveled", "TilePyramid", DownsampleString)

        Tile1Filename = "Tile000001.png"
        Tile2Filename = "Tile000002.png"
        Tile5Filename = "Tile000005.png"
        Tile6Filename = "Tile000006.png"
        Tile7Filename = "Tile000007.png"
        Tile9Filename = "Tile000009.png"


        self.RunAlignment(Tile7Filename, Tile9Filename, (908 / Downsample, 0))
        self.RunAlignment(Tile5Filename, Tile6Filename, (2 / Downsample, 1260 / Downsample))
        self.RunAlignment(Tile1Filename, Tile2Filename, (4 / Downsample, 1260 / Downsample))


    def test_MismatchSizeAlignments(self):

        self.TilesPath = os.path.join(self.TestInputPath, "Images", "Alignment")

        Tile1Filename = "402.png"
        Tile2Filename = "401_Subset.png"

        # self.RunAlignment(Tile1Filename, Tile2Filename, (-529, -93))


    def RunAlignment(self, TileAFilename, TileBFilename, ExpectedOffset):
        '''ExpectedOffset is (Y,X)'''

        imFixed = core.LoadImage(os.path.join(self.TilesPath, TileAFilename))
        imMoving = core.LoadImage(os.path.join(self.TilesPath, TileBFilename))

        imFixedPadded = core.PadImageForPhaseCorrelation(imFixed)
        imMovingPadded = core.PadImageForPhaseCorrelation(imMoving)

        alignrecord = core.FindOffset(imFixedPadded, imMovingPadded)
        
        print(str(alignrecord))

        # self.assertAlmostEqual(alignrecord.peak[1], ExpectedOffset[1], delta=2, msg="X dimension incorrect: " + str(alignrecord.peak) + " != " + str(ExpectedOffset))
        # self.assertAlmostEqual(alignrecord.peak[0], ExpectedOffset[0], delta=2, msg="Y dimension incorrect: " + str(alignrecord.peak) + " != " + str(ExpectedOffset))


class TestMosaicArrange(setup_imagetest.TransformTestBase, setup_imagetest.PickleHelper):

    @property
    def Dataset(self):
        return "PMG1"
    
    @property
    def MosaicFiles(self, testName=None):
        if testName is None:
            testName = self.Dataset

        return glob.glob(os.path.join(self.ImportedDataPath, testName, "Stage.mosaic"))


    def RigidTransformForTile(self, tile, offset=None):
        if offset is None:
            offset = np.zeros((2))
            
        r =  tfactory.CreateRigidTransform(target_image_shape=tile.OriginalImageSize,
                                             source_image_shape=tile.OriginalImageSize,
                                             rangle=0,
                                             warped_offset=offset)
        
#         m =  tfactory.CreateRigidTransform(target_image_shape=tile.OriginalImageSize,
#                                              source_image_shape=tile.OriginalImageSize,
#                                              rangle=0,
#                                              warped_offset=offset)
#         
#         rf = r.FixedBoundingBox
#         mf = m.FixedBoundingBox
#         
#         #"Transforms should have identical bounds regardless of type")
#         np.testing.assert_allclose(rf.Corners,mf.Corners, atol=1e-5)        
        return r

    def ShowTilesWithOffset(self, layout_obj, tiles_list, TileA_ID, TileB_ID, filename, openwindow):
        
        if not (layout_obj.Contains(TileA_ID) and layout_obj.Contains(TileB_ID)):
            print("Tile offset for {0},{1} cannot be shown, Tiles missing in layout".format(TileA_ID, TileB_ID))
            return
        
        if not layout_obj.ContainsOffset((TileA_ID, TileB_ID)):
            print("Tile offset for {0},{1} cannot be shown, offset not specified".format(TileA_ID, TileB_ID))
            return
        
        NodeA = layout_obj.nodes[TileA_ID]
        NodeB_Offset = NodeA.GetOffset(TileB_ID)
        
        tileA = tiles_list[TileA_ID]
        tileB = tiles_list[TileB_ID]
        
        # transformA = self.RigidTransformForTile(tileA, AlignmentRecord((0, -624 * 2.0), 0, 0))
        transformA = self.RigidTransformForTile(tileA)
        transformB = self.RigidTransformForTile(tileB, NodeB_Offset)

        ImageToTransform = {}
        ImageToTransform[tileA.ImagePath] = transformA
        ImageToTransform[tileB.ImagePath] = transformB

        mosaic = Mosaic(ImageToTransform)
        mosaic.TranslateToZeroOrigin()
        
        info_str = "%d -> %d\noffset: (%gx, %gy)\nweight: %g" % (TileA_ID, TileB_ID, NodeB_Offset[1], NodeB_Offset[0], NodeA.GetWeight(TileB_ID))

        self._ShowMosaic(mosaic, usecluster=False, title=info_str, mosaic_path=os.path.join(self.TestOutputPath, filename + "_%d-%d.png" % (TileA_ID, TileB_ID)), openwindow=openwindow)
        
    
    def PlotLayoutWeightHistogram(self, layout, ImageFilename=None, title=None, openwindow=True):
        
        matplotlib.pyplot.clf()
        offsets = nornir_imageregistration.layout.OffsetsSortedByWeight(layout)
        weight_scores = offsets[:, 4]
        
        numBins = len(weight_scores) / 4.0
        if numBins < 10:
            numBins = 10
        
        if not title is None:
            matplotlib.pyplot.title(title)
            
        matplotlib.pyplot.hist(weight_scores, int(numBins))
        
        if(ImageFilename is not None):
            # plt.show()
            OutputImageFullPath = os.path.join(self.TestOutputPath, ImageFilename + '.png')
            matplotlib.pyplot.savefig(OutputImageFullPath)
        
        if openwindow:
            matplotlib.pyplot.show()
            
        matplotlib.pyplot.clf()

    def _ShowMosaic(self, mosaic, mosaic_path=None, openwindow=True, usecluster=True, title=None, target_space_scale=None, source_space_scale=None):

        
        (assembledImage, mask) = mosaic.AssembleImage(tilesPath=None, usecluster=usecluster,
                                                      target_space_scale=target_space_scale,
                                                      source_space_scale=source_space_scale)
        
        if not mosaic_path is None: 
            pool = nornir_pools.GetGlobalThreadPool()
            pool.add_task("Save %s" % mosaic_path, core.SaveImage, mosaic_path, assembledImage)
            #core.SaveImage(mosaic_path, assembledImage)
        
        if openwindow:
            if title is None:
                title = "A mosaic with no tiles out of place"

            self.assertTrue(nornir_imageregistration.ShowGrayscale(assembledImage, title=title, PassFail=True))


    def __CheckNoOffsetsToSelf(self, layout):

        for i, node in layout.nodes.items():
            self.assertFalse(i in node.ConnectedIDs, "Tiles should not be registered to themselves, node {0}".format(i))
            
    def __CheckAllOffsetsPresent(self, layout, expected_offsets):

        for offset in expected_offsets:
            self.assertTrue(layout.ContainsOffset(offset), "{0} should have an offset in the layout but does not".format(offset))
            
    def __CheckNoExtraOffsets(self, layout, expected_offsets):

        offset_keys = frozenset([o.ID for o in expected_offsets])
        for i, node in layout.nodes.items():
            for other in node.ConnectedIDs:
                key = (node.ID, other)
                if key[0] > key[1]:
                    key = (other, node.ID)
                    
                self.assertTrue(key in offset_keys, "{0} is an unexpected offset".format(key))


    def __RemoveExtraImages(self, mosaic):

        '''Remove all but the first two images'''
        keys = list(mosaic.ImageToTransform.keys())
        keys.sort()

        for i, k in enumerate(keys):
            if i >= 2:
                del mosaic.ImageToTransform[k]

 
    
    def CalculateOffsetsForTiles(self, tile_offsets_to_update, all_tile_offsets, excess_scalar, imageScale, existing_layout):
     
        translate_layout = arrange._FindTileOffsets(tile_offsets_to_update, excess_scalar, imageScale, existing_layout)
        
        self.__CheckNoOffsetsToSelf(translate_layout)
        self.__CheckAllOffsetsPresent(translate_layout, all_tile_offsets)
        self.__CheckNoExtraOffsets(translate_layout, all_tile_offsets)
        
        return translate_layout

    def ArrangeMosaicDirect(self, mosaicFilePath, 
                            TilePyramidDir=None,
                            downsample=None,
                            openwindow=False, 
                            max_relax_iterations=None,
                            max_relax_tension_cutoff=None,
                            inter_tile_distance_scale = None,
                            feature_score_threshold = 0.5):

        if downsample is None:
            downsample = 1
            
        #pool = nornir_pools.GetGlobalThreadPool()
            
        downsamplePath = '%03d' % downsample
        
        minWeight = 0
        maxWeight = 1.0
        
        mosaicBaseName = os.path.basename(mosaicFilePath)
        (mosaicBaseName, ext) = os.path.splitext(mosaicBaseName)     

        scale = 1.0 / float(downsample)
        
        TilesDir = None
        if TilePyramidDir is None:
            TilesDir = os.path.join(self.ImportedDataPath, self.Dataset, 'Leveled', 'TilePyramid', downsamplePath)
        else:
            TilesDir = os.path.join(TilePyramidDir, downsamplePath)

        mosaic = Mosaic.LoadFromMosaicFile(mosaicFilePath)
        mosaic.EnsureTransformsHaveMappedBoundingBoxes(image_scale=scale, image_path=TilesDir)
        mosaic.TranslateToZeroOrigin()
          
#        mosaic.TranslateToZeroOrigin()

        # self.__RemoveExtraImages(mosaic)

        # assembleScale = tiles.MostCommonScalar(mosaic.ImageToTransform.values(), mosaic.TileFullPaths(TilesDir))

        # expectedScale = 1.0 / float(downsamplePath)

        #  self.assertEqual(assembleScale, expectedScale, "Scale for assemble does not match the expected scale")

        timer = TaskTimer()
        timer.Start("ArrangeTiles " + TilesDir)

        tilesPathList = sorted(mosaic.CreateTilesPathList(TilesDir))
        transforms = list(mosaic._TransformsSortedByKey())
        
        imageScale = self.ReadOrCreateVariable(self.id() + "_imageScale_%03d" % downsample, tileset.MostCommonScalar, transforms=transforms, imagepaths=tilesPathList)
        first_pass_excess_scalar = 3 #This needs to be 3 to ensure we can detect any offset, otherwise quadrant of the peak is ambiguous
        excess_scalar = first_pass_excess_scalar
        
        initial_tiles = nornir_imageregistration.tile.CreateTiles( transforms=transforms, imagepaths=tilesPathList)
        
        
        min_overlap = 0.075
        
        if inter_tile_distance_scale is None:
            inter_tile_distance_scale = 1.0
        
        #tile_overlap_feature_scores = arrange.ScoreTileOverlaps(tiles=initial_tiles, imageScale=imageScale)
        last_pass_overlaps = None
        translated_layout = None
        
        stage_reported_overlaps = None
        
        for iPass in range(0,5):
            (distinct_overlaps, new_overlaps, updated_overlaps, removed_overlap_IDs, non_overlapping_IDs) = arrange.GenerateTileOverlaps(tiles=initial_tiles,
                                                             existing_overlaps=last_pass_overlaps,
                                                             offset_epsilon=1.0,
                                                             image_scale=imageScale,
                                                             min_overlap=min_overlap,
                                                             inter_tile_distance_scale=inter_tile_distance_scale)
            
            if stage_reported_overlaps is None:
                stage_reported_overlaps = {to.ID: to.offset for to in new_overlaps}
            
            
            #After the first pass trust the mosaic layout
            inter_tile_distance_scale = 1.0
            
            new_or_updated_overlaps = list(new_overlaps)
            new_or_updated_overlaps.extend(updated_overlaps)
            
            arrange.ScoreTileOverlaps(distinct_overlaps)
            
            overlap_colors = ['green'] * len(new_overlaps)
            overlap_colors.extend(['blue'] * len(updated_overlaps))
             
            if translated_layout is not None:
                for ID in removed_overlap_IDs:
                    translated_layout.RemoveOverlap(ID)
                for ID in non_overlapping_IDs:
                    translated_layout.RemoveOverlap(ID)    
                
            
    #         tile_overlap_feature_scores = self.ReadOrCreateVariable(self.id() + "tile_overlap_feature_scores_%03d_%03g" % (downsample, first_pass_excess_scalar),
    #                                                       arrange.GenerateScoredTileOverlaps,
    #                                                       tiles=initial_tiles,
    #                                                       imageScale=imageScale)
            
            Scores = []
            for overlap in distinct_overlaps: 
                Scores.extend(overlap.feature_scores)
        
            Scores.sort()
            h = nornir_shared.histogram.Histogram.Init(minVal=np.min(Scores), maxVal=np.max(Scores), numBins=int(np.sqrt(len(Scores)) * 10))
            h.Add(Scores)
            
            #nornir_shared.plot.Histogram(h)
            nornir_shared.plot.Histogram(h, ImageFilename=os.path.join(self.TestOutputPath, "{0:d}pass_FeatureScoreHistogram.png".format(iPass)), Title="Tile overlap feature scores")
#            nornir_shared.plot.Histogram(h, ImageFilename=os.path.join(mosaicBaseName + "_{0d}pass_FeatureScoreHistogram.png", Title="Tile overlap feature scores")
            
            #pool.add_task('Plot prune histogram', nornir_shared.plot.Histogram,h, ImageFilename=mosaicBaseName + "_PrunePlotHistogram.png", Title="Tile overlap feature scores")
            #pool.add_task('Plot prune histogram', nornir_shared.plot.Histogram,h, ImageFilename=mosaicBaseName + "_PrunePlotHistogram.png", Title="Tile overlap feature scores")
            
            #self.assertAlmostEqual(imageScale, 1.0 / downsample, "Calculated image scale should match downsample value passed to test")
        
            #Create a list of offsets requiring updates
            filtered_overlaps_needing_offsets = []
            for overlap in new_or_updated_overlaps:
                if overlap.feature_scores[0] >= feature_score_threshold and overlap.feature_scores[1] >= feature_score_threshold:
                    filtered_overlaps_needing_offsets.append(overlap)
                else:
                    if translated_layout is not None:
                        translated_layout.RemoveOverlap(overlap)
                    
            #Create a list of every offset that should be found in the layout for debugging
            filtered_distinct_offsets = []
            for overlap in distinct_overlaps:
                if overlap.feature_scores[0] >= feature_score_threshold and overlap.feature_scores[1] >= feature_score_threshold:
                    filtered_distinct_offsets.append(overlap)
                    
            #Find the overlaps that are locked
            new_or_updated_dict = {o.ID for o in new_or_updated_overlaps}
            locked_overlaps = []
            for d in distinct_overlaps:
                if not d.ID in new_or_updated_dict:
                    locked_overlaps.append(d)
                    if translated_layout.ContainsOffset(d.ID):
                        overlap_colors.extend(['gold'])
                    else:
                        overlap_colors.extend(['red'])
            
            self.SaveVariable((new_overlaps, updated_overlaps, locked_overlaps, overlap_colors),
                              os.path.join(self.TestOutputPath, "pass_{0}_tile_overlaps.pickle".format(iPass)))
                    
            nornir_imageregistration.views.plot_tile_overlaps(new_overlaps + updated_overlaps + locked_overlaps,
                                                              colors=overlap_colors,
                                                              OutputFilename=os.path.join(self.TestOutputPath, "pass_{0}_tile_overlaps.svg".format(iPass)))
             
            translated_layout = self.ReadOrCreateVariable(self.id() + "_{0:d}pass_tile_layout_{1:03d}_{2:03g}".format(iPass, downsample, first_pass_excess_scalar),
                                                          self.CalculateOffsetsForTiles,
                                                          tile_offsets_to_update=filtered_overlaps_needing_offsets,
                                                          all_tile_offsets=filtered_distinct_offsets,
                                                          excess_scalar=excess_scalar, 
                                                          imageScale=imageScale,
                                                          existing_layout=translated_layout)
            excess_scalar = 3.0
            # Each tile should contain a dictionary with the known offsets.  Show the overlapping images using the calculated offsets
    
            (tileA_ID, tileB_ID) = _GetWorstOffsetPair(translated_layout)
            self.ShowTilesWithOffset(translated_layout, initial_tiles, tileA_ID, tileB_ID, "_{0:d}pass_Worst1stPass".format(iPass), openwindow=openwindow)
            
            (tileA_ID, tileB_ID) = _GetBestOffsetPair(translated_layout)
            self.ShowTilesWithOffset(translated_layout, initial_tiles, tileA_ID, tileB_ID, "_{0:d}pass_Best1stPass".format(iPass), openwindow=openwindow)
            
            self.ShowTilesWithOffset(translated_layout, initial_tiles, 0, 1, "{0:d}_pass".format(iPass), openwindow=openwindow)
            self.ShowTilesWithOffset(translated_layout, initial_tiles, 0, 60, "{0:d}_pass".format(iPass), openwindow=openwindow)
            self.ShowTilesWithOffset(translated_layout, initial_tiles, 1, 61, "{0:d}_pass".format(iPass), openwindow=openwindow)
            self.ShowTilesWithOffset(translated_layout, initial_tiles, 60, 61, "{0:d}_pass".format(iPass), openwindow=openwindow)
            #self.ShowTilesWithOffset(translated_layout, initial_tiles, 53, 56, "{0:d}_pass".format(iPass), openwindow=openwindow)
            
            # mosaic.ArrangeTilesWithTranslate(TilesDir, usecluster=parallel)
            scaled_translated_layout = translated_layout.copy()
            
            nornir_imageregistration.layout.ScaleOffsetWeightsByPopulationRank(scaled_translated_layout,
                                                                               min_allowed_weight=minWeight,
                                                                               max_allowed_weight=maxWeight)
            
#            nornir_imageregistration.layout.NormalizeOffsetWeights(scaled_translated_layout)
        
            
            
            self.PlotLayoutWeightHistogram(scaled_translated_layout, mosaicBaseName + "_{0:d}pass_weight_histogram".format(iPass), openwindow=False)
            translated_final_layouts = nornir_imageregistration.layout.BuildLayoutWithHighestWeightsFirst(scaled_translated_layout)
            translated_final_layout = nornir_imageregistration.layout.MergeDisconnectedLayoutsWithOffsets(translated_final_layouts, stage_reported_overlaps)
            
            translated_mosaic = self.CreateSaveShowMosaic(mosaicBaseName + "_{0:d}pass_Weighted".format(iPass),
                                                          translated_final_layout,
                                                          initial_tiles,
                                                          openwindow=openwindow,
                                                          target_space_scale=1/4.0,
                                                          source_space_scale=None)
            
#             relaxed_layout = self._Relax_Layout(scaled_translated_layout,
#                                                 max_iter=max_relax_iterations,
#                                                 max_tension_cutoff=max_relax_tension_cutoff,
#                                                 dirname_postfix="_pass{0:d}".format(iPass))
            
            relaxed_layout = nornir_imageregistration.layout.RelaxLayout(translated_final_layout,
                                            max_iter=max_relax_iterations,
                                            max_tension_cutoff=max_relax_tension_cutoff,
                                            plotting_output_path=os.path.join(self.TestOutputPath, "relax_pass_{0:d}".format(iPass)),
                                            plotting_interval=10)

                  
            relaxed_layout.UpdateTileTransforms(initial_tiles)
            
            relaxed_mosaic = self.CreateSaveShowMosaic(mosaicBaseName + "_{0:d}pass_relaxed".format(iPass),
                                                       relaxed_layout,
                                                       initial_tiles,
                                                       openwindow=openwindow,
                                                       target_space_scale=1/4.0,
                                                       source_space_scale=None)
            
            last_pass_overlaps = distinct_overlaps
            
            #Copy the relaxed layout positions back into the translated layout
            for ID,node in relaxed_layout.nodes.items():
                tnode = translated_layout.nodes[ID]
                tnode.Position = node.Position
                    
            nornir_pools.WaitOnAllPools()
             
            
        #self.CreateSaveShowMosaic(mosaicBaseName + "_{0:d}pass_relaxed".format(iPass), relaxed_layout, initial_tiles, openwindow)
            
        #             
        original_score = mosaic.QualityScore(TilesDir)
        translated_score = translated_mosaic.QualityScore(TilesDir)
        relaxed_score = relaxed_mosaic.QualityScore(TilesDir)
        #translated_refined_score = translated_refined_mosaic.QualityScore(TilesDir)
        #translated_refined_relaxed_score = translated_refined_relaxed_mosaic.QualityScore(TilesDir) 
        
        print("Original Quality Score: %g" % (original_score))
        #print("Translated Quality Score: %g" % (translated_score))
        #print("Relaxed Quality Score: %g" % (relaxed_score))
        #print("Translated refined Quality Score: %g" % (translated_refined_score))
        #print("Translated refined relaxed Quality Score: %g" % (translated_refined_relaxed_score))
        
        # self.assertLess(translated_score, original_score, "Translated worse than original")
        # self.assertLess(relaxed_score, translated_score, "Translated worse than original")
        
        
    def CreateMosaic(self, name, layout_obj, tiles, *args, **kwargs):
        OutputDir = os.path.join(self.TestOutputPath, name + '.mosaic')
        # OutputMosaicDir = os.path.join(self.TestOutputPath, name + '.png')
        
        created_mosaic = layout_obj.ToMosaic(tiles)
        created_mosaic.SaveToMosaicFile(OutputDir)
        return created_mosaic
        
        
    def CreateSaveShowMosaic(self, name, layout_obj, tiles, openwindow=False,
                                                          target_space_scale=None,
                                                          source_space_scale=None):
        
        created_mosaic = self.CreateMosaic(name, layout_obj, tiles)
        OutputMosaicDir = os.path.join(self.TestOutputPath, name + '.png')
        
#         if not openwindow: 
#             pool = nornir_pools.GetGlobalThreadPool()
#             pool.add_task(OutputMosaicDir, self._ShowMosaic, created_mosaic, OutputMosaicDir, openwindow=openwindow)
#         else:
        self._ShowMosaic(created_mosaic, OutputMosaicDir, openwindow=openwindow, 
                                                          target_space_scale=target_space_scale,
                                                          source_space_scale=source_space_scale)
        
        return created_mosaic
          
    
    def _Relax_Layout(self, layout_obj, max_tension_cutoff=None, max_iter=None, dirname_postfix=None):
                
        if max_tension_cutoff is None:
            max_tension_cutoff = 1.0
            
        if max_iter is None:
            max_iter = 100
            
        if dirname_postfix is None:
            dirname_postfix = ""
            
        max_tension = layout_obj.MaxWeightedNetTensionMagnitude[1]
        min_plotting_tension = max_tension_cutoff * 20
        plotting_max_tension = max(min_plotting_tension, max_tension)
        i = 0
        
        pool = nornir_pools.GetGlobalMultithreadingPool()
        
        MovieImageDir = os.path.join(self.TestOutputPath, "relax_movie" + dirname_postfix)
        if not os.path.exists(MovieImageDir):
            os.makedirs(MovieImageDir)
            
        while max_tension > max_tension_cutoff and i < max_iter:
            print("%d %g" % (i, max_tension))
            node_movement = nornir_imageregistration.layout.Layout.RelaxNodes(layout_obj)
            max_tension = layout_obj.MaxWeightedNetTensionMagnitude[1]
            plotting_max_tension = max_tension
            plotting_max_tension = max(min_plotting_tension, max_tension)
            # node_distance = setup_imagetest.array_distance(node_movement[:,1:3])             
            # max_distance = np.max(node_distance,0)
            
            
            filename = os.path.join(MovieImageDir, "%d.png" % i)
            
            #pool.add_task("Plot step #%d" % (i), nornir_imageregistration.views.plot_layout, 
#                           layout_obj=layout_obj,
#                           OutputFilename=filename,
#                           max_tension=plotting_max_tension)
            
            if i % 10 == 0:
#                 nornir_imageregistration.views.plot_layout( 
#                            layout_obj=layout_obj,
#                            OutputFilename=filename,
#                            max_tension=plotting_max_tension)
             
                pool.add_task("Plot step #%d" % (i), nornir_imageregistration.views.plot_layout, 
                              layout_obj=layout_obj.copy(),
                              OutputFilename=filename,
                              max_tension=plotting_max_tension)

#             nornir_shared.plot.VectorField(Points=layout_obj.GetPositions(),
#                                            Offsets=layout_obj.WeightedNetTensionVectors(),
#                                            weights=nornir_imageregistration.array_distance(layout_obj.WeightedNetTensionVectors()) / layout_obj.MaxWeightedNetTensionMagnitude,
#                                            OutputFilename=filename)
            i += 1
#             
        return layout_obj

    def ArrangeMosaic(self, mosaicFilePath,
                      TilePyramidDir=None,
                      downsample=None, 
                      openwindow=False,
                      max_relax_iterations=None,
                      max_relax_tension_cutoff=None,
                      inter_tile_distance_scale = None,
                      min_translate_iterations=None):
 
        if downsample is None:
            downsample = 1
            
        if min_translate_iterations is None:
            min_translate_iterations = 5
            
        if inter_tile_distance_scale is None:
            inter_tile_distance_scale = 0.5
            
        downsamplePath = '%03d' % downsample

        mosaic = Mosaic.LoadFromMosaicFile(mosaicFilePath)
        mosaicBaseName = os.path.basename(mosaicFilePath)

        (mosaicBaseName, ext) = os.path.splitext(mosaicBaseName)

        TilesDir = None
        if TilePyramidDir is None:
            TilesDir = os.path.join(self.ImportedDataPath, self.Dataset, 'Leveled', 'TilePyramid', downsamplePath)
        else:
            TilesDir = os.path.join(TilePyramidDir, downsamplePath)

        mosaic.TranslateToZeroOrigin()
        
        original_score = mosaic.QualityScore(TilesDir)

        # self.__RemoveExtraImages(mosaic)

        assembleScale = tileset.MostCommonScalar(mosaic.ImageToTransform.values(), mosaic.TileFullPaths(TilesDir))

        expectedScale = 1.0 / float(downsamplePath)

        self.assertEqual(assembleScale, expectedScale, "Scale for assemble does not match the expected scale")

        timer = TaskTimer()

        timer.Start("ArrangeTiles " + TilesDir)

        translated_mosaic = mosaic.ArrangeTilesWithTranslate(TilesDir, excess_scalar=1.5,
                                                             min_translate_iterations=min_translate_iterations,
                                                             max_relax_iterations=max_relax_iterations,
                                                             max_relax_tension_cutoff=max_relax_tension_cutoff,
                                                             inter_tile_distance_scale=inter_tile_distance_scale,
                                                             min_overlap=0.05)

        timer.End("ArrangeTiles " + TilesDir, True)
        
        translated_score = translated_mosaic.QualityScore(TilesDir)
        
        print("Original Quality Score: %g" % (original_score))
        print("Translate Quality Score: %g" % (translated_score))
        
        OutputDir = os.path.join(self.TestOutputPath, mosaicBaseName + '.mosaic')
        OutputMosaicDir = os.path.join(self.TestOutputPath, mosaicBaseName + '.png')

        if openwindow:
            self._ShowMosaic(translated_mosaic, OutputMosaicDir)
         
        
#     def test_RC2_0197_Mosaic(self):
#         
#         self.ArrangeMosaicDirect(mosaicFilePath="D:\\RC2\\TEM\\0197\\TEM\\stage.mosaic", TilePyramidDir="D:\\RC2\\TEM\\0197\\TEM\\Leveled\\TilePyramid", downsample=4, openwindow=False)
# 
#         print("All done")
#         
#     def test_RC2_0001_Mosaic(self):
#         
#         self.ArrangeMosaicDirect(mosaicFilePath="D:\\RC2\\TEM\\0001\\TEM\\stage.mosaic", TilePyramidDir="D:\\RC2\\TEM\\0001\\TEM\\Leveled\\TilePyramid", downsample=4, openwindow=False)
# 
#         print("All done")
#         
#     def test_RC2_0380_Mosaic(self):
#         
#         self.ArrangeMosaicDirect(mosaicFilePath="D:\\RC2\\TEM\\0380\\TEM\\Prune_Thr10.0.mosaic", TilePyramidDir="D:\\RC2\\TEM\\0380\\TEM\\Leveled\\TilePyramid", downsample=4, openwindow=False)
# 
#         print("All done")
# #         
#     def test_RC2_0192_Mosaic(self):
#               
#         self.ArrangeMosaicDirect(mosaicFilePath="C:\\Data\\RC2\\TEM\\0192\\TEM\\Prune_Thr10.0.mosaic", TilePyramidDir="C:\\Data\\RC2\\TEM\\0192\\TEM\\Leveled\\TilePyramid", downsample=4, max_relax_iterations=150, openwindow=False)
#       
#         print("All done")

#     def test_RC1_0060_Mosaic(self):
#                   
#         self.ArrangeMosaicDirect(mosaicFilePath="C:\\Data\\RC1\\TEM\\0060\\TEM\\Stage.mosaic",
#                                  TilePyramidDir="C:\\Data\\RC1\\TEM\\0060\\TEM\\Leveled\\TilePyramid",
#                                  downsample=4,
#                                  max_relax_iterations=500,
#                                  openwindow=False,
#                                  max_relax_tension_cutoff=0.1)
#          
#         print("All done")
           
#     def test_RC2_0192_Smaller_Mosaic(self):
#                
#         self.ArrangeMosaicDirect(mosaicFilePath="C:\\Data\\RC2\\TEM\\0192\\TEM\\Stage_cropped.mosaic",
#                                  TilePyramidDir="C:\\Data\\RC2\\TEM\\0192\\TEM\\Leveled\\TilePyramid",
#                                  downsample=4,
#                                  max_relax_iterations=150,
#                                  openwindow=False)
#        
#         print("All done")

#     def test_RPC2_1013_Mosaic(self):
#                  
#         self.ArrangeMosaicDirect(mosaicFilePath="C:\\Data\\RPC2\\1013\\TEM\\Stage.mosaic",
#                                  TilePyramidDir="C:\\Data\\RPC2\\1013\\TEM\\Leveled\\TilePyramid",
#                                  downsample=4,
#                                  max_relax_iterations=500,
#                                  openwindow=False)
#          
#         print("All done")
        
    
    def test_TEM2_Sahler_13208_Mosaic(self):
                 
        self.ArrangeMosaicDirect(mosaicFilePath="C:\\Data\\TEM2_Sahler\\13208\\Stage_limited.mosaic",
                                 TilePyramidDir="C:\\Data\\TEM2_Sahler\\13208\\Raw8\\TilePyramid",
                                 downsample=4,
                                 max_relax_iterations=500,
                                 openwindow=False)
         
        print("All done")

#     def test_RC2_1034_Mosaic(self):
#         self.ArrangeMosaicDirect(mosaicFilePath="C:\\Data\\RC2\\TEM\\1034\\TEM\\Stage.mosaic",
#                                  TilePyramidDir="C:\\Data\\RC2\\TEM\\1034\\TEM\\Leveled\\TilePyramid",
#                                  downsample=4,
#                                  max_relax_iterations=500,
#                                  openwindow=False,
#                                  max_relax_tension_cutoff=0.1)
#                                  #inter_tile_distance_scale=0.5)
#          
#         print("All done")

#     def test_PMG_0006_E_Mosaic(self):
#         self.ArrangeMosaicDirect(mosaicFilePath="C:\\Data\\PMG\\0006\\E\\Stage.mosaic",
#                                  TilePyramidDir="C:\\Data\\PMG\\0006\\E\\Leveled\\TilePyramid",
#                                  downsample=2,
#                                  max_relax_iterations=500,
#                                  openwindow=False,
#                                  max_relax_tension_cutoff=0.1)
#                                  #inter_tile_distance_scale=0.5)
#           
#         print("All done")

#     def test_EM2_0007(self):
#         self.ArrangeMosaicDirect(mosaicFilePath="C:\\Data\\EM2\\TEM\\0007\\TEM\\Stage.mosaic",
#                                  TilePyramidDir="C:\\Data\\EM2\\TEM\\0007\\TEM\\Raw8\\TilePyramid",
#                                  downsample=4,
#                                  max_relax_iterations=500,
#                                  openwindow=False,
#                                  max_relax_tension_cutoff=0.1,
#                                  feature_score_threshold=1.0)
#                                  #inter_tile_distance_scale=0.5)
            
#        print("All done")
        
#     def test_DM4_0476_Mosaic(self):
#         self.ArrangeMosaicDirect(mosaicFilePath="C:\\Data\\DM4\\0476\\SEM\\Stage.mosaic",
#                                  TilePyramidDir="C:\\Data\\DM4\\0476\\SEM\\Leveled\\TilePyramid",
#                                  downsample=4,
#                                  max_relax_iterations=500,
#                                  openwindow=False,
#                                  max_relax_tension_cutoff=0.1)
#                                  #inter_tile_distance_scale=0.5)
#         
#        print("All done")
        
#     def test_Neitz_110__Mosaic(self):
#               
#         self.ArrangeMosaicDirect(mosaicFilePath="C:\\src\\git\\nornir-testdata\\Transforms\\mosaics\\Neitz\\Stage.mosaic",
#                                  TilePyramidDir="C:\\src\\git\\nornir-testdata\\Transforms\\mosaics\\Neitz\\Leveled\\TilePyramid",
#                                  downsample=8,
#                                  max_relax_iterations=150,
#                                  openwindow=True)
#       
#         print("All done")
        
#     def test_RC2_0626_Mosaic(self):
#         
#         self.ArrangeMosaicDirect(mosaicFilePath="D:\\RC2\\TEM\\0626\\TEM\\Prune_Thr10.0.mosaic", TilePyramidDir="D:\\RC2\\TEM\\0626\\TEM\\Leveled\\TilePyramid", downsample=4, max_relax_iterations=150, openwindow=False)
# 
#         print("All done")

#     def test_ArrangeMosaic(self):
#         
#         for m in self.MosaicFiles:
#             self.ArrangeMosaic(m, TilePyramidDir=None, downsample=1, max_relax_iterations=150, max_relax_tension_cutoff=1.0)
# 
#         print("All done")
 
#     def test_ArrangeMosaicDirect(self):
#  
#         for m in self.MosaicFiles:
#             self.ArrangeMosaicDirect(m, TilePyramidDir=None, downsample=1, openwindow=False)
#  
#         print("All done")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']

    import nornir_shared.misc
    nornir_shared.misc.RunWithProfiler("unittest.main()")
