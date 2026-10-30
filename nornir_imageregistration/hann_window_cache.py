import nornir_imageregistration.image_filter_cache as image_filter_cache
import skimage.filters

HannWindowCache = image_filter_cache.CreateWindowFilterCache("hann")
