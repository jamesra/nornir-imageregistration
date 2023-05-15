'''
Created on Dec 13, 2013

@author: James Anderson
'''

import glob
import os
import unittest

import matplotlib
import matplotlib.pyplot
import nornir_imageregistration
from nornir_imageregistration import AlignmentRecord, MosaicFile, MosaicTileset, Mosaic
# from nornir_imageregistration.files.mosaicfile import MosaicFile
import nornir_imageregistration.layout
# from nornir_imageregistration.mosaic import Mosaic
# from nornir_imageregistration.mosaic_tileset import MosaicTileset
import nornir_imageregistration.mosaic
from scipy import stats 

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

import setup_imagetest
import mosaic_tileset


# from pylab import *
def _GetWorstOffsetPair(layout_obj):
    
    offsets = nornir_imageregistration.layout.OffsetsSortedByWeight(layout_obj)
    TileA_ID = int(offsets[-1, 0])
    TileB_ID = int(offsets[-1, 1])
    Weight = offsets[-1, -1]
    return (TileA_ID, TileB_ID, Weight)


def _GetBestOffsetPair(layout_obj):
    
    offsets = nornir_imageregistration.layout.OffsetsSortedByWeight(layout_obj)
    TileA_ID = int(offsets[0, 0])
    TileB_ID = int(offsets[0, 1])
    Weight = offsets[0, -1]
    return (TileA_ID, TileB_ID, Weight)

 
def RigidTransformForTile(tile, offset=None):
    if offset is None:
        offset = np.zeros((2))
        
    r = tfactory.CreateRigidTransform(target_image_shape=tile.FullResolutionImageSize,
                                         source_image_shape=tile.FullResolutionImageSize,
                                         rangle=0,
                                         warped_offset=offset)
    
#         m =  tfactory.CreateRigidTransform(target_image_shape=tile.FullResolutionImageSize,
#                                              source_image_shape=tile.FullResolutionImageSize,
#                                              rangle=0,
#                                              warped_offset=offset)
#         
#         rf = r.FixedBoundingBox
#         mf = m.FixedBoundingBox
#         
#         #"Transforms should have identical bounds regardless of type")
#         np.testing.assert_allclose(rf.Corners,mf.Corners, atol=1e-5)        
    return r


def ShowTilesWithOffset(test, layout_obj, tiles_list, TileA_ID:int, TileB_ID:int, output_dir:str, filename:str, image_to_source_space_scale:float, openwindow:bool, weight:float=None):
    
    if not (layout_obj.Contains(TileA_ID) and layout_obj.Contains(TileB_ID)):
        print("Tile offset for {0},{1} cannot be shown, Tiles missing in layout".format(TileA_ID, TileB_ID))
        return
    
    if not layout_obj.ContainsOffset((TileA_ID, TileB_ID)):
        print("Tile offset for {0},{1} cannot be shown, offset not specified".format(TileA_ID, TileB_ID))
        return
    
    NodeA = layout_obj.nodes[TileA_ID]
    NodeB_Offset = NodeA.GetOffset(TileB_ID)
    
    if weight is None:
        weight = NodeA.GetWeight(TileB_ID)
    
    tileA = tiles_list[TileA_ID]
    tileB = tiles_list[TileB_ID]
    
    # transformA = RigidTransformForTile(tileA, AlignmentRecord((0, -624 * 2.0), 0, 0))
    transformA = RigidTransformForTile(tileA)
    transformB = RigidTransformForTile(tileB, NodeB_Offset)

    mosaic_set = nornir_imageregistration.mosaic_tileset.MosaicTileset(image_to_source_space_scale=image_to_source_space_scale)
    newTileA = nornir_imageregistration.tile.Tile(transform=transformA,
                                                    imagepath=tileA.ImagePath,
                                                    image_to_source_space_scale=image_to_source_space_scale,
                                                    ID=tileA.ID)
    newTileB = nornir_imageregistration.tile.Tile(transform=transformB,
                                                    imagepath=tileB.ImagePath,
                                                    image_to_source_space_scale=image_to_source_space_scale,
                                                    ID=tileB.ID)
    mosaic_set[newTileA.ImagePath] = newTileA
    mosaic_set[newTileB.ImagePath] = newTileB
    
    mosaic_set.TranslateToZeroOrigin()
    
    # ImageToTransform = {}
    # ImageToTransform[tileA.ImagePath] = transformA
    # ImageToTransform[tileB.ImagePath] = transformB

    # mosaic = Mosaic(ImageToTransform)
    # mosaic.TranslateToZeroOrigin()
    
    info_str = "%d -> %d\noffset: (%gx, %gy)\nweight: %g" % (TileA_ID, TileB_ID, NodeB_Offset[1], NodeB_Offset[0], NodeA.GetWeight(TileB_ID))

    ShowMosaicSet(test, mosaic_set, usecluster=False, title=info_str,
                     path=os.path.join(output_dir, f"{filename}_{TileA_ID}-{TileB_ID}_Weight_{float(weight):.5f}.png"),
                     image_to_source_space_scale=image_to_source_space_scale, openwindow=openwindow)
 
 
def ShowMosaicSet(test, mosaicTileset, path=None, openwindow=True, usecluster=True, title:str=None, target_space_scale:float=None, image_to_source_space_scale:float=None):
    '''
    :param mosaicTileset: The tileset we are assembling an image for
    :param unittest test: The test we are running and should use for assertions
    :param path: Where to save the output image, will not be saved if None is passed
    :param openwindow: Set to true to display the assembled image
    '''
    (assembledImage, mask) = mosaicTileset.AssembleImage(usecluster=usecluster,
                                                  target_space_scale=target_space_scale)
    
    if path is not None:
        pool = nornir_pools.GetGlobalThreadPool()
        pool.add_task("Save %s" % path, core.SaveImage, path, assembledImage, bpp=8)
        # core.SaveImage(mosaic_path, assembledImage)
    
    if openwindow:
        if title is None:
            title = "A mos = aic with no tiles out of place"

        test.assertTrue(nornir_imageregistration.ShowGrayscale(assembledImage, title=title, PassFail=True))


def CreateSaveShowMosaic(test, name, layout_obj, tileset, output_dir,
                            openwindow=False,
                            target_space_scale=None):
         
    updated_tileset = layout_obj.ToMosaicTileset(tileset)
    OutputMosaicPath = os.path.join(output_dir, name + '.mosaic')
    updated_tileset.SaveMosaic(OutputMosaicPath)
    
    OutputMosaicDir = os.path.join(output_dir, name + '.png')
    
#         if not openwindow: 
#             pool = nornir_pools.GetGlobalThreadPool()
#             pool.add_task(OutputMosaicDir, self._ShowMosaic, created_mosaic, OutputMosaicDir, openwindow=openwindow)
#         else:
    ShowMosaicSet(test, updated_tileset, OutputMosaicDir, openwindow=openwindow,
                                                      target_space_scale=target_space_scale,
                                                      image_to_source_space_scale=tileset.image_to_source_space_scale)
    
    return updated_tileset.ToMosaic()


class TestMosaicArrange(setup_imagetest.TransformTestBase, setup_imagetest.PickleHelper):
    '''
    Test layouts of entire mosaics
    See test_TileToTileAlignment to explore individual tile to tile registration when debugging
    '''

    @property
    def TestName(self):
        return self.__name__
    
    @property
    def Dataset(self):
        return "PMG1"
    
    @property
    def MosaicFiles(self, testName=None):
        if testName is None:
            testName = self.Dataset

        return glob.glob(os.path.join(self.ImportedDataPath, testName, "Stage.mosaic"))

    def GetStandardTranslateSettings(self, known_offsets=None):
        return nornir_imageregistration.settings.TranslateSettings(min_overlap=0.02,
                                                            max_translate_iterations=5,
                                                            max_relax_iterations=500,
                                                            max_relax_tension_cutoff=0.5,
                                                            known_offsets=known_offsets)
    
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

    def _ShowMosaic(self, mosaic, mosaic_path=None, openwindow=True, usecluster=True, title:str=None, target_space_scale:float=None, image_to_source_space_scale:float=None):
        
        mosaicTileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaic,
                                                                                 image_folder=None,
                                                                                 image_to_source_space_scale=image_to_source_space_scale)
        
        return ShowMosaicSet(self, mosaicTileset, path=mosaic_path,
                              openwindow=openwindow, usecluster=usecluster,
                              title=title, target_space_scale=target_space_scale,
                              image_to_source_space_scale=image_to_source_space_scale)
    
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
    
    def CalculateOffsetsForTiles(self, tile_offsets_to_update, all_tile_offsets, excess_scalar, image_to_source_space_scale, existing_layout, use_feature_score:bool):
     
        translate_layout = arrange._FindTileOffsets(tile_offsets_to_update, excess_scalar, image_to_source_space_scale, existing_layout, use_feature_score=use_feature_score)
        
        self.__CheckNoOffsetsToSelf(translate_layout)
        self.__CheckAllOffsetsPresent(translate_layout, all_tile_offsets)
        self.__CheckNoExtraOffsets(translate_layout, all_tile_offsets)
        
        return translate_layout

    def ArrangeMosaicDirect(self, config:nornir_imageregistration.settings.TranslateSettings, mosaicFilePath,
                            TilePyramidDir=None,
                            downsample=None,
                            openwindow=False):

        if downsample is None:
            downsample = 1
            
        # pool = nornir_pools.GetGlobalThreadPool()
            
        downsamplePath = '%03d' % downsample
          
        mosaicBaseName = os.path.basename(mosaicFilePath)
        (mosaicBaseName, ext) = os.path.splitext(mosaicBaseName)
 
        image_to_source_space_scale = downsample

        debug_output_downsample = None
        if downsample < 4:
            debug_output_downsample = 4.0
        else:
            debug_output_downsample = downsample
        
        TilesDir = None
        if TilePyramidDir is None:
            TilesDir = os.path.join(self.ImportedDataPath, self.Dataset, 'Leveled', 'TilePyramid', downsamplePath)
        else:
            TilesDir = os.path.join(TilePyramidDir, downsamplePath)

        mosaic = Mosaic.LoadFromMosaicFile(mosaicFilePath)
         
        timer = TaskTimer()
        timer.Start("ArrangeTiles " + TilesDir)

        mosaic_tileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaic,
                                                                                  image_folder=TilesDir,
                                                                                  image_to_source_space_scale=float(downsample))
        
        mosaic_tileset.TranslateToZeroOrigin()
        
        self.SaveVariable(mosaic_tileset,
                          os.path.join(self.TestOutputPath, "mosaic_tileset.pickle"))
        
        # tilesPathList = sorted(mosaic.CreateTilesPathList(TilesDir))
        # transforms = list(mosaic._TransformsSortedByKey())
        
        # image_to_source_space_scale = self.ReadOrCreateVariable(self.id() + "_imageScale_%03d" % downsample, tileset.MostCommonScalar, transforms=transforms, imagepaths=tilesPathList)
        # config.first_pass_excess_scalar = 3  # This needs to be 3 to ensure we can detect any offset, otherwise quadrant of the peak is ambiguous
        excess_scalar = config.first_pass_excess_scalar
        
        # initial_tiles = nornir_imageregistration.mosaic_tileset.CreateTiles( transforms=transforms, imagepaths=tilesPathList, source_space_scale=scale)
        
        min_overlap = 0.02
        
        # tile_overlap_feature_scores = arrange.ScoreTileOverlaps(tiles=initial_tiles, image_to_source_space_scale=image_to_source_space_scale)
        last_pass_overlaps = None
        translated_layout = None
        
        stage_reported_overlaps = None
        
        for iPass in range(0, config.max_translate_iterations):
            (distinct_overlaps, new_overlaps, updated_overlaps, removed_overlap_IDs, non_overlapping_IDs) = arrange.GenerateTileOverlaps(mosaic_tileset,
                                                             existing_overlaps=last_pass_overlaps,
                                                             offset_epsilon=1.0,
                                                             min_overlap=min_overlap,
                                                             inter_tile_distance_scale=config.inter_tile_distance_scale,
                                                             exclude_diagonal_overlaps=config.exclude_diagonal_overlaps)
            
            if stage_reported_overlaps is None:
                stage_reported_overlaps = {to.ID: to.offset for to in new_overlaps}
            
            new_or_updated_overlaps = list(new_overlaps)
            new_or_updated_overlaps.extend(updated_overlaps)
            # If there is nothing to update we are done
            if len(new_or_updated_overlaps) == 0:
                break
            
            overlap_colors = ['green'] * len(new_overlaps)
            overlap_colors.extend(['blue'] * len(updated_overlaps))
            
            if translated_layout is not None:
                for ID in removed_overlap_IDs:
                    translated_layout.RemoveOverlap(ID)
                for ID in non_overlapping_IDs:
                    self.assertTrue(translated_layout.nodes[ID].ConnectedIDs.shape[0] == 0, "Non-overlapping node should not have overlaps")
#                    translated_layout.RemoveNode(ID)    
            
            filtered_overlaps_needing_offsets = []
            if False == config.feature_score_calculations_required:
                filtered_overlaps_needing_offsets = new_or_updated_overlaps
            else:
                arrange.ScoreTileOverlaps(distinct_overlaps)
                arrange.NormalizeOverlapFeatureScores(distinct_overlaps)
               
                Scores = []
                for overlap in distinct_overlaps: 
                    Scores.extend(overlap.feature_scores)
            
                Scores.sort()
                h = nornir_shared.histogram.Histogram.Init(minVal=np.min(Scores), maxVal=np.max(Scores), numBins=int(np.sqrt(len(Scores)) * 10))
                h.Add(Scores)
                
                # nornir_shared.plot.Histogram(h)
                nornir_shared.plot.Histogram(h, ImageFilename=os.path.join(self.TestOutputPath, "{0:d}pass_FeatureScoreHistogram.png".format(iPass)), Title="Tile overlap feature scores")
    #            nornir_shared.plot.Histogram(h, ImageFilename=os.path.join(mosaicBaseName + "_{0d}pass_FeatureScoreHistogram.png", Title="Tile overlap feature scores")

             
            
            
            # pool.add_task('Plot prune histogram', nornir_shared.plot.Histogram,h, ImageFilename=mosaicBaseName + "_PrunePlotHistogram.png", Title="Tile overlap feature scores")
            # pool.add_task('Plot prune histogram', nornir_shared.plot.Histogram,h, ImageFilename=mosaicBaseName + "_PrunePlotHistogram.png", Title="Tile overlap feature scores")
            
            # self.assertAlmostEqual(image_to_source_space_scale, 1.0 / downsample, "Calculated image scale should match downsample value passed to test")
        
                # Create a list of offsets requiring updates
                for overlap in new_or_updated_overlaps:
                    if overlap.feature_scores[0] >= config.feature_score_threshold and overlap.feature_scores[1] >= config.feature_score_threshold:
                        filtered_overlaps_needing_offsets.append(overlap)
                    else:
                        if translated_layout is not None:
                            translated_layout.RemoveOverlap(overlap)
              
                # Create a list of every offset that should be found in the layout for debugging
            if config.feature_score_calculations_required:
                filtered_distinct_offsets = []
                for overlap in distinct_overlaps:
                    if overlap.feature_scores[0] >= config.feature_score_threshold and overlap.feature_scores[1] >= config.feature_score_threshold:
                        filtered_distinct_offsets.append(overlap)
            else:
                filtered_distinct_offsets = distinct_overlaps
                    
            # Find the overlaps that are locked
            new_or_updated_dict = {o.ID for o in new_or_updated_overlaps}
            locked_overlaps = []
            for d in distinct_overlaps:
                if not d.ID in new_or_updated_dict:
                    locked_overlaps.append(d)
                    if translated_layout.ContainsOffset(d.ID):
                        overlap_colors.extend(['gold'])
                    else:
                        overlap_colors.extend(['red'])
            
            self.SaveVariable((distinct_overlaps, new_overlaps, updated_overlaps, removed_overlap_IDs, locked_overlaps, overlap_colors),
                              os.path.join(self.TestOutputPath, "pass_{0}_tile_overlaps.pickle".format(iPass)))

            nornir_imageregistration.views.plot_tile_overlaps(new_overlaps + updated_overlaps + locked_overlaps,
                                                              colors=overlap_colors,
                                                              OutputFilename=os.path.join(self.TestOutputPath, "pass_{0}_tile_overlaps.svg".format(iPass)))
             
            translated_layout = self.ReadOrCreateVariable(self.id() + "_{0:d}pass_tile_layout_{1:03d}_{2:03g}".format(iPass, downsample, config.first_pass_excess_scalar),
                                                          self.CalculateOffsetsForTiles,
                                                          tile_offsets_to_update=filtered_overlaps_needing_offsets,
                                                          all_tile_offsets=filtered_distinct_offsets,
                                                          excess_scalar=excess_scalar,
                                                          image_to_source_space_scale=image_to_source_space_scale,
                                                          existing_layout=translated_layout,
                                                          use_feature_score=config.use_feature_score)
            excess_scalar = 3.0
            # Each tile should contain a dictionary with the known offsets.  Show the overlapping images using the calculated offsets
    
            (tileA_ID, tileB_ID, worst_weight) = _GetWorstOffsetPair(translated_layout)
            ShowTilesWithOffset(self, translated_layout, mosaic_tileset,
                                     tileA_ID, tileB_ID,
                                     self.TestOutputPath,
                                     "_{0:d}pass_Worst1stPass".format(iPass),
                                     image_to_source_space_scale=downsample,
                                     openwindow=openwindow,
                                     weight=worst_weight)
            
            (tileA_ID, tileB_ID, best_weight) = _GetBestOffsetPair(translated_layout)
            ShowTilesWithOffset(self, translated_layout, mosaic_tileset,
                                     tileA_ID, tileB_ID,
                                     self.TestOutputPath,
                                     "_{0:d}pass_Best1stPass".format(iPass),
                                     image_to_source_space_scale=downsample,
                                     openwindow=openwindow,
                                     weight=best_weight)
            
            # ShowTilesWithOffset(translated_layout, initial_tiles, 0, 1, "{0:d}_pass".format(iPass), openwindow=openwindow)
            # ShowTilesWithOffset(translated_layout, initial_tiles, 0, 60, "{0:d}_pass".format(iPass), openwindow=openwindow)
            # ShowTilesWithOffset(translated_layout, initial_tiles, 1, 61, "{0:d}_pass".format(iPass), openwindow=openwindow)
            # ShowTilesWithOffset(translated_layout, initial_tiles, 60, 61, "{0:d}_pass".format(iPass), openwindow=openwindow)
            # ShowTilesWithOffset(translated_layout, initial_tiles, 53, 56, "{0:d}_pass".format(iPass), openwindow=openwindow)
            
            # mosaic.ArrangeTilesWithTranslate(TilesDir, usecluster=parallel)
            scaled_translated_layout = translated_layout.copy() 
            # nornir_imageregistration.layout.SetUniformOffsetWeights(scaled_translated_layout)
            nornir_imageregistration.layout.NormalizeOffsetWeights(scaled_translated_layout,
                                                                            min_allowed_weight=config.min_offset_weight,
                                                                            max_allowed_weight=config.max_offset_weight)
            # nornir_imageregistration.layout.ScaleOffsetWeightsByPopulationRank(scaled_translated_layout,
                                                                               # min_allowed_weight=config.min_offset_weight,
                                                                               # max_allowed_weight=config.max_offset_weight)   
            
#            nornir_imageregistration.layout.NormalizeOffsetWeights(scaled_translated_layout)
            
            self.PlotLayoutWeightHistogram(scaled_translated_layout, mosaicBaseName + "_{0:d}pass_weight_histogram".format(iPass), openwindow=False)
            translated_final_layouts = nornir_imageregistration.layout.BuildLayoutWithHighestWeightsFirst(scaled_translated_layout)
            translated_final_layout = nornir_imageregistration.layout.MergeDisconnectedLayoutsWithOffsets(translated_final_layouts, stage_reported_overlaps)
            translated_final_layout.TranslateToZeroOrigin()
            
            # translated_mosaic = CreateSaveShowMosaic(self, mosaicBaseName + "_{0:d}pass_Weighted".format(iPass),
            #                                               translated_final_layout,
            #                                               tileset=mosaic_tileset,
            #                                               output_dir=self.TestOutputPath,
            #                                               openwindow=openwindow,
            #                                               target_space_scale=1.0 / debug_output_downsample)
            
#             relaxed_layout = self.Relax_Layout(scaled_translated_layout,
#                                                 max_iter=config.max_relax_iterations,
#                                                 max_tension_cutoff=config.max_relax_tension_cutoff,

#                                                 dirname_postfix="_pass{0:d}".format(iPass))
    
            relaxed_layouts = []
            for layout in translated_final_layouts:
                relaxed_layout = nornir_imageregistration.layout.RelaxLayout(layout,
                                                max_iter=config.max_relax_iterations,
                                                max_tension_cutoff=config.max_relax_tension_cutoff,
                                                plotting_output_path=os.path.join(self.TestOutputPath, "relax_pass_{0:d}".format(iPass)),
                                                plotting_interval=10)
                relaxed_layouts.append(relaxed_layout)
                
            relaxed_layout = nornir_imageregistration.layout.MergeDisconnectedLayoutsWithOffsets(relaxed_layouts, stage_reported_overlaps)
            
            relaxed_layout.TranslateToZeroOrigin()

            relaxed_layout.UpdateTileTransforms(mosaic_tileset)
            
            relaxed_mosaic = CreateSaveShowMosaic(self, mosaicBaseName + "_{0:d}pass_relaxed".format(iPass),
                                                       relaxed_layout,
                                                       tileset=mosaic_tileset,
                                                       output_dir=self.TestOutputPath,
                                                       openwindow=openwindow,
                                                       target_space_scale=1.0 / debug_output_downsample)
            
            last_pass_overlaps = distinct_overlaps
            
            # Copy the relaxed layout positions back into the translated layout
            for ID, node in relaxed_layout.nodes.items():
                tnode = translated_layout.nodes[ID]
                tnode.Position = node.Position
                    
            nornir_pools.WaitOnAllPools()
            
        #  CreateSaveShowMosaic(mosaicBaseName + "_{0:d}pass_relaxed".format(iPass), relaxed_layout, initial_tiles, openwindow)
            
        #             
        original_score = mosaic.QualityScore(TilesDir, downsample)
        # translated_score = translated_mosaic.QualityScore(TilesDir, downsample)
        relaxed_score = relaxed_mosaic.QualityScore(TilesDir, downsample)
        # translated_refined_score = translated_refined_mosaic.QualityScore(TilesDir)
        # translated_refined_relaxed_score = translated_refined_relaxed_mosaic.QualityScore(TilesDir) 
        
        print("Original Quality Score: %g" % (original_score))
        # print("Translated Quality Score: %g" % (translated_score))
        print("Relaxed Quality Score: %g" % (relaxed_score))
        # print("Translated refined Quality Score: %g" % (translated_refined_score))
        # print("Translated refined relaxed Quality Score: %g" % (translated_refined_relaxed_score))
        
        # self.assertLess(translated_score, original_score, "Translated worse than original")
        # self.assertLess(relaxed_score, translated_score, "Translated worse than original")
        
    def CreateMosaic(self, name, layout_obj, tiles, *args, **kwargs):
        OutputDir = os.path.join(self.TestOutputPath, name + '.mosaic')
    #    # OutputMosaicDir = os.path.join(self.TestOutputPath, name + '.png')
    #
    #     created_mosaic = layout_obj.ToMosaic(tiles)
    #     created_mosaic.SaveToMosaicFile(OutputDir)
    #     return created_mosaic

    def ArrangeMosaic(self,
                      config:nornir_imageregistration.settings.TranslateSettings,
                      mosaicFilePath,
                      TilePyramidDir=None,
                      downsample=None,
                      openwindow=False):
 
        if downsample is None:
            downsample = 1
              
        downsamplePath = '%03d' % downsample

        mosaic = Mosaic.LoadFromMosaicFile(mosaicFilePath)
        mosaicBaseName = os.path.basename(mosaicFilePath)
        
        (mosaicBaseName, ext) = os.path.splitext(mosaicBaseName)

        TilesDir = None
        if TilePyramidDir is None:
            TilesDir = os.path.join(self.ImportedDataPath, self.Dataset, 'Leveled', 'TilePyramid', downsamplePath)
        else:
            TilesDir = os.path.join(TilePyramidDir, downsamplePath)
            
        mosaictileset = mosaic_tileset.CreateFromMosaic(mosaic,
                                                        image_folder=TilesDir,
                                                        image_to_source_space_scale=downsample)
        mosaictileset.TranslateToZeroOrigin()
        
        print("Calculating quality score")
        original_score = mosaic.QualityScore(TilesDir, downsample)
        print(f"Original mosaic quality score: {original_score}")

        # self.__RemoveExtraImages(mosaic)

        # assembleScale = tileset.MostCommonScalar(mosaic.ImageToTransform.values(), mosaic.TileFullPaths(TilesDir))

        # expectedScale = 1.0 / float(downsamplePath)

        # self.assertEqual(assembleScale, expectedScale, "Scale for assemble does not match the expected scale")

        timer = TaskTimer()

        timer.Start("ArrangeTiles " + TilesDir)

        translated_mosaic = mosaictileset.ArrangeTilesWithTranslate(config=config)

        timer.End("ArrangeTiles " + TilesDir, True)
        
        OutputDir = os.path.join(self.TestOutputPath, mosaicBaseName + '_translated.mosaic')
        OutputMosaicDir = os.path.join(self.TestOutputPath, mosaicBaseName + '.png')
        
        translated_mosaic.SaveMosaic(OutputDir)
        
        translated_score = translated_mosaic.QualityScore()
        
        print("Original Quality Score: %g" % (original_score))
        print("Translate Quality Score: %g" % (translated_score))
         
        if openwindow:
            self._ShowMosaic(translated_mosaic, OutputMosaicDir)
         
#     def test_RC2_0001_Mosaic(self):

#    def test_RC2_0197_Mosaic(self):         
#        self.ArrangeMosaicDirect(mosaicFilePath="D:\\RC2\\TEM\\0197\\TEM\\stage.mosaic", TilePyramidDir="D:\\RC2\\TEM\\0197\\TEM\\Leveled\\TilePyramid", downsample=4, openwindow=False)
#        print("All done")
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
#    def test_RC2_0192_Mosaic(self):
#               
#        self.ArrangeMosaicDirect(mosaicFilePath="C:\\Data\\RC2\\TEM\\0192\\TEM\\Prune_Thr10.0.mosaic", TilePyramidDir="C:\\Data\\RC2\\TEM\\0192\\TEM\\Leveled\\TilePyramid", downsample=4, max_relax_iterations=150, openwindow=False)
#       
#        print("All done")

    # def test_RC1_0060_Mosaic(self):
    #
    #     self.ArrangeMosaicDirect(mosaicFilePath="C:\\Data\\RC1\\TEM\\0060\\TEM\\Stage.mosaic",
    #                              TilePyramidDir="C:\\Data\\RC1\\TEM\\0060\\TEM\\Leveled\\TilePyramid",
    #                              downsample=4, 
    #                              openwindow=False,
    #                              config=config)
    #
    #     print("All done")
    
    # def test_Redmond_13611_Tiny_Mosaic(self):
    #
    #     self.ArrangeMosaicDirect(mosaicFilePath=r"C:\src\git\nornir-testdata\Images\Alignment\Redmond\Stage_two_tiles.mosaic",
    #                              TilePyramidDir=r"C:\src\git\nornir-testdata\Images\Alignment\Redmond",
    #                              downsample=4, 
    #                              openwindow=True,
    #                              config=config)
    #
    #     print("All done")
    
    # def test_Redmond_13611_Mosaic(self):
    #
    #     self.ArrangeMosaicDirect(mosaicFilePath=r"D:\\Data\\Redmond\\13611\\TEM\\Prune_Thr10.0.mosaic",
    #                              TilePyramidDir=r"D:\\Data\\Redmond\\13611\\TEM\\Leveled\\TilePyramid\\",
    #                              downsample=4,  
    #                              openwindow=False,
    #                              config=config)
    #
    #     print("All done")
           
    # def test_RC2_0192_Smaller_Mosaic(self):
    #
    #     config = self.GetStandardTranslateSettings()
    #     #config.feature_score_threshold = None
    #     #config.use_feature_score = None 
    #     self.ArrangeMosaicDirect(mosaicFilePath="D:\\Data\\RC2\\TEM\\0192\\TEM\\Stage_cropped.mosaic",
    #                              TilePyramidDir="D:\\Data\\RC2\\TEM\\0192\\TEM\\Leveled\\TilePyramid",
    #                              downsample=4,
    #                              openwindow=False,
    #                              config=config)
    #
    #     print("All done")

    # def test_RPC2_1013_Mosaic(self):
    #
    #     self.ArrangeMosaicDirect(mosaicFilePath="D:\\Data\\RPC2\\1013\\TEM\\Stage.mosaic",
    #                              TilePyramidDir="D:\\Data\\RPC2\\1013\\TEM\\Leveled\\TilePyramid",
    #                              downsample=4, 
    #                              openwindow=False,
    #                              config=config)
    #
    #     print("All done")
    
    # def test_RPC2_0989_Mosaic(self):
    #
    #     config = self.GetStandardTranslateSettings()
    #     self.ArrangeMosaicDirect(mosaicFilePath="D:\\Data\\RPC2\\0989\\TEM\\Stage.mosaic",
    #                              TilePyramidDir="D:\\Data\\RPC2\\0989\\TEM\\Leveled\\TilePyramid",
    #                              downsample=4,  
    #                              openwindow=False,
    #                              config=config)
    #
    #     print("All done")
        
    def test_RPC2_0989_Mosaic(self):
    
        config = self.GetStandardTranslateSettings()
        self.ArrangeMosaic(mosaicFilePath="D:\\Data\\RPC2\\0989\\TEM\\Stage.mosaic",
                           TilePyramidDir="D:\\Data\\RPC2\\0989\\TEM\\Leveled\\TilePyramid",
                           downsample=4,  
                           openwindow=False,
                           config=config)
    
        print("All done")
    
    # def test_RC3_0001_Mosaic(self):
    #
    #     config = self.GetStandardTranslateSettings() 
    #     self.ArrangeMosaicDirect(mosaicFilePath="D:\\Data\\RC3\\TEM\\0001\\TEM\\Stage.mosaic",
    #                              TilePyramidDir="D:\\Data\\RC3\\TEM\\0001\\TEM\\Leveled\\TilePyramid",
    #                              downsample=4, 
    #                              openwindow=False,
    #                              config=config)
    # #
    #     print("All done")
    #
    # def test_RC3_0001_Mosaic_Production(self):
    #
    #     self.ArrangeMosaic(mosaicFilePath="D:\\Data\\RC3\\TEM\\0001\\TEM\\Stage.mosaic",
    #                              TilePyramidDir="D:\\Data\\RC3\\TEM\\0001\\TEM\\Leveled\\TilePyramid",
    #                              downsample=4, 
    #                              openwindow=False,
    #                              config=config)
    #
    #     print("All done")
         
    # def test_RC3_0203_Mosaic(self): 
    #     config = self.GetStandardTranslateSettings() 
    #     self.ArrangeMosaicDirect(mosaicFilePath="D:\\Data\\RC3\\TEM\\0203\\TEM\\Stage.mosaic",
    #                              TilePyramidDir="D:\\Data\\RC3\\TEM\\0203\\TEM\\Leveled\\TilePyramid",
    #                              downsample=4, 
    #                              openwindow=False,
    #                              config=config)
    #
    #     print("All done")
        
    # def test_RC3_1619_Mosaic(self):
    #     config = self.GetStandardTranslateSettings()
    #     self.ArrangeMosaicDirect(mosaicFilePath="D:\\Data\\RC3\\TEM\\1619\\TEM\\Stage.mosaic",
    #                              TilePyramidDir="D:\\Data\\RC3\\TEM\\1619\\TEM\\Leveled\\TilePyramid",
    #                              downsample=4,
    #                              openwindow=False,
    #                              config=config)
    #
    #     print("All done")
    
    # def test_RC3_1492_Mosaic(self):
    #     config = self.GetStandardTranslateSettings() 
    #     self.ArrangeMosaicDirect(mosaicFilePath="D:\\Data\\RC3\\TEM\\1492\\TEM\\Stage.mosaic",
    #                              TilePyramidDir="D:\\Data\\RC3\\TEM\\1492\\TEM\\Leveled\\TilePyramid",
    #                              downsample=4,
    #                              openwindow=False,
    #                              config=config)
    #
    #     print("All done")
        
    # def test_RC3_1492_Mosaic_Production(self):
    #
    #     config = self.GetStandardTranslateSettings() 
    #     self.ArrangeMosaic(mosaicFilePath="D:\\Data\\RC3\\TEM\\1492\\TEM\\Stage.mosaic",
    #                              TilePyramidDir="D:\\Data\\RC3\\TEM\\1492\\TEM\\Leveled\\TilePyramid",
    #                              downsample=4, 
    #                              openwindow=False,
    #                              config=config)
    #
    #     print("All done")
    #
    # def test_RC3_1154_Mosaic(self):
    #
    #     config = self.GetStandardTranslateSettings()
    #     config.feature_score_threshold = None
    #     config.use_feature_score = None 
    #     self.ArrangeMosaic(mosaicFilePath=r"D:\Data\RC3\TEM\1154\TEM\Prune_Thr10.0.mosaic",
    #                              TilePyramidDir=r"D:\Data\RC3\TEM\1154\TEM\Leveled\TilePyramid",
    #                              downsample=4,
    #                              openwindow=False,
    #                              config=config)
    #
    #     print("All done")
    
    # def test_RC3_1154_Mosaic_Direct(self):
    #
    #     config = self.GetStandardTranslateSettings()
    #     self.ArrangeMosaicDirect(mosaicFilePath=r"D:\Data\RC3\TEM\1154\TEM\Prune_Thr10.0.mosaic",
    #                              TilePyramidDir=r"D:\Data\RC3\TEM\1154\TEM\Leveled\TilePyramid",
    #                              downsample=4, 
    #                              openwindow=False,
    #                              config=config)
    #
    #     print("All done")
        
    # def test_Neitz_Mosaic(self):
    #
    #     config = self.GetStandardTranslateSettings()
    #     self.ArrangeMosaicDirect(mosaicFilePath="D:\\Data\\cped_sm\\SEM\\0855\\SEM\\Stage.mosaic",
    #                              TilePyramidDir="D:\\Data\\cped_sm\\SEM\\0855\\SEM\\Leveled\\TilePyramid",
    #                              downsample=4, 
    #                              openwindow=False,
    #                              config=config)
    #
    #     print("All done")
#         
#     
#     def test_TEM2_Sahler_13208_Mosaic(self):
#         config = self.GetStandardTranslateSettings()                
#         self.ArrangeMosaicDirect(mosaicFilePath="C:\\Data\\TEM2_Sahler\\13208\\Stage_limited.mosaic",
#                                  TilePyramidDir="C:\\Data\\TEM2_Sahler\\13208\\Raw8\\TilePyramid",
#                                  downsample=4, 
#                                  openwindow=False,
#                                  config=config)
#          
#         print("All done")

    # def test_RC2_1034_Mosaic(self):
    #     """
    #     This section has a massive tear and many folds.  It is a great test case for manual offsets
    #     """
    #     config = self.GetStandardTranslateSettings()
    #     self.ArrangeMosaicDirect(mosaicFilePath="D:\\Data\\RC2\\TEM\\1034\\TEM\\Stage.mosaic",
    #                             TilePyramidDir="D:\\Data\\RC2\\TEM\\1034\\TEM\\Leveled\\TilePyramid",
    #                             downsample=4,
    #                             openwindow=False,
    #                             config=config)
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
