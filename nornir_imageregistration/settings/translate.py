'''
Created on Aug 4, 2022

@author: u0490822
''' 


class TranslateSettings(object):
    '''
    Settings for translating a mosaic
    '''


    def __init__(self, 
                 min_overlap:float=None,
                 max_relax_iterations:int=None,
                 max_relax_tension_cutoff:float=None,
                 feature_score_threshold:float=None,
                 offset_acceptance_threshold:float=None,
                 min_translate_iterations:int=None,
                 max_translate_iterations:int=None,
                 inter_tile_distance_scale:float=None,
                 first_pass_inter_tile_distance_scale:float=None,
                 first_pass_excess_scalar:float=None,
                 min_offset_weight:float=None,
                 max_offset_weight:float=None,
                 excess_scalar:float=None,
                 use_feature_score:bool=False,
                 exclude_diagonal_overlaps:bool=True,
                 known_offsets:list=None): 
                 #**kwargs): #Catcher for serialized deprecated arguments
        '''
        :param float min_overlap: The percentage of area that two tileset must overlap before being considered by the layout model
        :param int max_relax_iterations: Maximum number of iterations in the relax stage
        :param int max_relax_tension_cutoff: Stop relaxation stage if the maximum tension vector is below this value
        :param int feature_score_threshold: The minimum average power spectral density per pixel measurement required to believe there is enough texture in overlapping regions for registration algorithms
        :param int offset_acceptance_threshold: The distance the expected offset between tiles has to change before we recalculate feature scores and registration.  If the difference in offset magnitude from predicted to measured falls below this value the previous registration will be used if it exists.
        :param int min_translate_iterations: The min number of iterations of the alignment+relaxation cycle that will be run even if we hit a cutoff value first
        :param int min_translate_iterations: The max number of iterations of the alignment+relaxation cycle that will be run unless we hit a cutoff value first 
        :param int inter_tile_distance_scale: A scalar from 0 to 1.  1 indicates to trust the overlap reported by the transforms.  0 indicates to test the entire tile for overlaps.  Use this value to increase the area searched for correlations if the stage input is not reliable.
        :param int first_pass_inter_tile_distance_scale: 
        :param int first_pass_excess_scalar: excess_scalar value to use on only the first pass.  If specified a larger value can be useful to find larger offsets and then a smaller excess_scalar can be used on later passes to search smaller regions for speed.
        :param float min_offset_weight: The minimum weight we will allow an offset measurement between two tiles to have.  Set equal to max_offset_weight to treat all tiles equally.
        :param float max_offset_weight: The maximum weight we will allow an offset measurement between two tiles to have.  Set equal to min_offset_weight to treat all tiles equally.
        :param int excess_scalar:  How much additional area should we pad the overlapping regions with.  Increase this value if you want larger offsets to be found.
        :param bool use_feature_score:  Multiply the alignment quality metric for the registration by the min power spectral density of the overlapping regions.  The idea is to down-weight featureless regions.  However the latest peak weight currently downweights itself if there were many strong peaks to choose from.
        :param bool exclude_diagonal_overlaps:  Do not include overlaps between rectangles whose center is outside the min/max range of the partner rectangle along both axes.  Reduces calculations by eliminating calculations for small overlapping corner regions.
        :param list known_offsets: Optional list of known tile offsets to prevent seams from opening up with questionable input with areas of low overlap.
        '''
        
        self.min_overlap = 0.02 if min_overlap is None else min_overlap
        self.max_relax_iterations = 150 if max_relax_iterations is None else max_relax_iterations
        self.max_relax_tension_cutoff = 1.0 if max_relax_tension_cutoff is None else max_relax_tension_cutoff
        self.feature_score_threshold = 0.035 if feature_score_threshold is None else feature_score_threshold
        self.offset_acceptance_threshold = 1.0 if offset_acceptance_threshold is None else offset_acceptance_threshold
        self.min_translate_iterations = 1 if min_translate_iterations is None else min_translate_iterations
        self.max_translate_iterations = self.min_translate_iterations * 4 if max_translate_iterations is None else max_translate_iterations
        self.inter_tile_distance_scale = 1.0 if inter_tile_distance_scale is None else inter_tile_distance_scale
        self.first_pass_inter_tile_distance_scale = self.inter_tile_distance_scale / 2 if first_pass_inter_tile_distance_scale is None else first_pass_inter_tile_distance_scale
        self.first_pass_excess_scalar = 3.0 if first_pass_excess_scalar is None else first_pass_excess_scalar
        self.min_offset_weight = 0   if min_offset_weight is None else min_offset_weight
        self.max_offset_weight = 1.0 if max_offset_weight is None else max_offset_weight
        self.excess_scalar = 3.0 if excess_scalar is None else excess_scalar
        self.use_feature_score = False if use_feature_score is None else use_feature_score
        self.exclude_diagonal_overlaps = True if exclude_diagonal_overlaps is None else exclude_diagonal_overlaps
        self.known_offsets = [] if known_offsets is None else known_offsets
        
        if self.min_translate_iterations > self.max_translate_iterations:
            raise ValueError("min_translate_iterations > max_translate_iterations")
        
        if self.min_offset_weight > self.max_offset_weight:
            raise ValueError("min_offset_weight > max_offset_weight")