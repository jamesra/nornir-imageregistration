'''
Created on Sep 10, 2019

@author: u0490822

These functions generate tileset image pyramid levels.  The network implementation
of these functions copies the images locally and writes the output locally before
moving it to the final output directory.  This saves trips over the network as
we build the pyramid, which tends to be slow for sometimes hundreds of thousands 
of small files.  This also helps the image I/O, which at this time is implemented
by pillow as lots of small I/O requests against the image file. 
'''

import numpy
from PIL import Image
#Disable decompression bomb protection since we are dealing with huge images on purpose
Image.MAX_IMAGE_PIXELS = None
import threading
import tempfile
import os
import shutil
import nornir_pools
#import nornir_shared.prettyoutput as prettyoutput

def ClearTempDirectories(level_paths):
    '''Deletes temporary directories used to generate levels'''
    
    if level_paths is None:
        return 
    
    if len(level_paths) == 0:
        return 
    
    temp_dir = tempfile.gettempdir()
    
    pool = nornir_pools.GetGlobalThreadPool()
    for level_path in level_paths:
        LevelDir = os.path.join(temp_dir, os.path.basename(level_path))
        pool.add_task("Remove temp directory {0}".format(LevelDir), shutil.rmtree, LevelDir, ignore_errors=True)
        
    pool.wait_completion()
        
def GetTempPathForTile(fullpath):
    '''
    Given a tileset image, return the temporary filename for the tile
    '''
    LevelDir = os.path.basename(os.path.dirname(fullpath))
    return os.path.join(tempfile.gettempdir(), LevelDir)

def GetTempDirForLevelDir(fullpath):
    '''
    Given a tileset level, return the temporary level directory
    '''
    return os.path.join(tempfile.gettempdir(), os.path.basename(fullpath))

def CreateOneTilesetTileWithPillowOverNetwork(TileDims, TopLeft, TopRight, BottomLeft, BottomRight, OutputFileFullPath ):
    '''Copy files to a local temp directory before access to improve IO over the network since Pillow tends to issue lots 
       of small IO calls instead of reading the entire file. 
       The temporary files are not removed so the next tileset level can utilize the local data.
       Use ClearTempDirectories to clean up the temporary data'''
    
    temp_dir = tempfile.gettempdir()
    
    TopLeftBase = os.path.basename(TopLeft)
    TopRightBase = os.path.basename(TopRight)
    BottomLeftBase = os.path.basename(BottomLeft)
    BottomRightBase = os.path.basename(BottomRight)
    
    LevelDir = os.path.basename(os.path.dirname(TopLeft))
    
    temp_input_dir = os.path.join(tempfile.gettempdir(), LevelDir)
    
    os.makedirs(temp_input_dir, exist_ok=True)
    
    temp_TopLeft = os.path.join(temp_input_dir, TopLeftBase)
    temp_TopRight = os.path.join(temp_input_dir, TopRightBase)
    temp_BottomLeft = os.path.join(temp_input_dir, BottomLeftBase)
    temp_BottomRight = os.path.join(temp_input_dir, BottomRightBase)
    
    try:
        if not os.path.exists(temp_TopLeft):
            shutil.copyfile(TopLeft, temp_TopLeft)
    except:
#        prettyoutput.Log("Missing input file {0}".format(TopLeft)) 
        pass
    
    try:
        if not os.path.exists(temp_TopRight):
            shutil.copyfile(TopRight, temp_TopRight)
    except: 
        #prettyoutput.Log("Missing input file {0}".format(TopRight)) 
        pass
    
    try:
        if not os.path.exists(temp_BottomLeft):
            shutil.copyfile(BottomLeft, temp_BottomLeft)
    except: 
        #prettyoutput.Log("Missing input file {0}".format(BottomLeft))
        pass
    
    try:
        if not os.path.exists(temp_BottomRight):
            shutil.copyfile(BottomRight, temp_BottomRight)
    except: 
        #prettyoutput.Log("Missing input file {0}".format(BottomRight))
        pass
    
    outputbase = os.path.basename(OutputFileFullPath)
    output_level_dir = os.path.basename(os.path.dirname(OutputFileFullPath))
    
    temp_output_dir = os.path.join(temp_dir, output_level_dir)
    temp_output = os.path.join(temp_output_dir, outputbase) 
                
    os.makedirs(temp_output_dir, exist_ok=True)
    
    CreateOneTilesetTileWithPillow(TileDims, temp_TopLeft, temp_TopRight, temp_BottomLeft, temp_BottomRight, temp_output)
    
    #Copy the file, but leave the temp in case we genereate the next level
    shutil.copyfile(temp_output, OutputFileFullPath)
    
    #Remove the input because this function is used to generate levels, and once we generate the next level we don't need the source level
    try:
        os.remove(temp_TopLeft)
    except IOError:
        pass
    
    try:
        os.remove(temp_TopRight)
    except IOError:
        pass
    
    try:
        os.remove(temp_BottomLeft)
    except IOError:
        pass
    
    try:
        os.remove(temp_BottomRight)
    except IOError:
        pass
    
#     try: We don't need to remove output if it was moved instead of copied
#         os.remove(temp_output)
#     except IOError:
#         pass

    #Run a thread for the move so this worker can perform other tasks
#    thread = threading.Thread(None, shutil.move, args=[temp_output, OutputFileFullPath])
#    thread.daemon = False
#    thread.run()       
    

def CreateOneTilesetTileWithPillow(TileDims, TopLeft, TopRight, BottomLeft, BottomRight, OutputFileFullPath ):
    '''Create a single tile by merging four tiles from a higher resolution and downsampling
    :param tuple TileDims: (Height, Width) of tiles'''
    
    TileSize = numpy.asarray((TileDims[1], TileDims[0]), dtype=numpy.int64) #Pillow uses the opposite ordering of axis
    DoubleTileSize = TileSize * 2 #Double the size 
    
    imComposite = None
            
    try:
        with Image.open(TopLeft) as imTopLeft:
            if imComposite is None:
                imComposite = Image.new(imTopLeft.mode, size=(DoubleTileSize[0], DoubleTileSize[1]), color=0) 
            imComposite.paste(imTopLeft, box=(0,0))
    except IOError as e:
#        prettyoutput.Log("Missing input file {0}".format(TopLeft)) 
        pass
    
    try:
        with Image.open(TopRight) as imTopRight:
            if imComposite is None:
                imComposite = Image.new(imTopRight.mode, size=(DoubleTileSize[0], DoubleTileSize[1]), color=0)
            imComposite.paste(imTopRight, box=(TileSize[0],0))
    except IOError as e:
#        prettyoutput.Log("Missing input file {0}".format(TopRight)) 
        pass
    
    try:
        with Image.open(BottomLeft) as imBottomLeft:
            if imComposite is None:
                imComposite = Image.new(imBottomLeft.mode, size=(DoubleTileSize[0], DoubleTileSize[1]), color=0)
            imComposite.paste(imBottomLeft, box=(0,TileSize[1]))
    except IOError as e:
#        prettyoutput.Log("Missing input file {0}".format(BottomLeft)) 
        pass
    
    try:
        with Image.open(BottomRight) as imBottomRight:
            if imComposite is None:
                imComposite = Image.new(imBottomRight.mode, size=(DoubleTileSize[0], DoubleTileSize[1]), color=0)
            imComposite.paste(imBottomRight, box=(TileSize[0],TileSize[1]))
    except IOError as e:
#        prettyoutput.Log("Missing input file {0}".format(BottomRight)) 
        pass
    
    if imComposite is not None:    
        with imComposite.resize(imTopLeft.size, resample=Image.LANCZOS) as imFinal:
            try:  
                imFinal.save(OutputFileFullPath, optimize=True) 
            except FileExistsError:
                pass
            
        del imComposite
    
    return 
    

if __name__ == '__main__':
    pass