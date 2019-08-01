'''
Created on Apr 23, 2019

@author: u0490822
'''

from PIL import Image
import numpy as np


def _try_read_bpp_from_pillow_mode(im):
    '''
    Pillow allows some modes to add the number of bits by adding a semicolon to the mode and a number. ex: 'I;16'
    :return: The number of bits if specified, otherwise None
    '''
    mode = None
    if isinstance(im, str):
        mode = im
    else:
        mode = im.mode
        
    bits = None
    try:
        parts = mode.split(';')
        if len(parts) > 1:
            bits = int(parts[1])
    except:
        pass
    
    return bits

def _try_estimate_dtype_from_extrema(im):
    '''
    Pillow allows some modes to add the number of bits by adding a semicolon to the mode and a number. ex: 'I;16'
    :return: The number of bits if specified, otherwise None
    '''
    mode = None
    if isinstance(im, str):
        mode = im
    else:
        mode = im.mode
        
    if not (mode[0] == 'I' or mode[0] == 'F'):
        raise ValueError('Image mode must be I or F')
    
    (min_val, max_val) = im.getextrema()
    if mode[0] == 'I':
        if max_val <= 255:
            assert(min_val >= 0)
            return np.uint8
        elif max_val < (1 << 15):
            return np.int16
        elif max_val < (1 << 16):
            assert(min_val >= 0)
            return np.uint16
        elif max_val < (1 << 31):
            return np.int32
        elif max_val < (1 << 32):
            assert(min_val >= 0)
            return np.uint32
        else:
            return np.int64
        
    elif mode[0] == 'F':
        return np.float32 #This is a floating point image already, just trust that float32 is plenty
    
    raise ValueError("Unexpected image or mode passed")


def dtype_for_pillow_image(im):
    mode = im.mode
         
    if mode == '1':
        return np.bool
    if mode[0] == 'L':
        return np.uint8
    if mode[0] == 'P':
        return np.uint8
    if mode == 'RGB':
        return np.uint8 #Numpy should create a 3rd axis for other channels
    if mode == 'RGBA':
        return np.uint8 #Numpy should create a 3rd axis for other channels
    if mode == 'CMYK':
        return np.uint8 #Numpy should create a 3rd axis for other channels
    if mode == 'YCbCr':
        return np.uint8 #Numpy should create a 3rd axis for other channels
    if mode == 'LAB':
        return np.uint8 #Numpy should create a 3rd axis for other channels
    if mode == 'HSV':
        return np.uint8 #Numpy should create a 3rd axis for other channels
    if mode[0] == 'I':
        
        bits = _try_read_bpp_from_pillow_mode(im)
        if bits is not None:
            if bits == 1:
                return np.bool
            elif bits <= 8:
                return np.uint8
            elif bits <= 16:
                return np.uint16
            else:
                return np.uint32 #According to Pillow docs the 32-bit integers are signed
        else:
            return _try_estimate_dtype_from_extrema(im)
        
    if mode[0] == 'F': 
        bits = _try_read_bpp_from_pillow_mode(im)
        if bits <= 8:
            return np.float16
        elif bits <= 16:
            return np.float16
        else:
            return np.float32 #According to Pillow docs the 32-bit integers are signed
    
    raise ValueError("Unexpected pillow image mode: {0}".format(mode))

if __name__ == '__main__':
    pass