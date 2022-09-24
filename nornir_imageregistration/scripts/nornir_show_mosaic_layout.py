'''
Created on May 21, 2013

@author: u0490822
'''

import argparse
import logging
import os
import sys
  
import nornir_shared.misc
import nornir_imageregistration


def __CreateArgParser(ExecArgs=None):

    # conflict_handler = 'resolve' replaces old arguments with new if both use the same option flag
    parser = argparse.ArgumentParser(description="Produce a registered image for the moving image in a .stos file")

    parser.add_argument('-input', '-i',
                        action='store',
                        required=True,
                        type=str,
                        help='Input .mosaic file path',
                        dest='inputpath')

    parser.add_argument('-output', '-o',
                        action='store',
                        required=True,
                        type=str,
                        help='Output image file path',
                        dest='outputpath')

    parser.add_argument('-tilepath', '-t',
                        action='store',
                        required=True,
                        type=str,
                        help='Path to directory containing tiles listed in mosaic.  Either an absolute path or a path relative to the .mosaic file',
                        dest='tilepath'
                        )
    
    parser.add_argument('-minoverlap', '-min',
                        action='store',
                        required=False,
                        type=float,
                        default=0.1,
                        help='Minimum overlap from 0 to 1 we expect tiles to have before being considered overlapping.',
                        dest='min_overlap'
                        )
    
    parser.add_argument('-d', '-downsample',
                        action='store',
                        required=False,
                        type=float,
                        default=None,
                        help='Downsample level of image folder being passed.  If not specified the program will attempt to convert the folder name to to a number',
                        dest='downsample'
                        )
    
    parser.add_argument('-s', '-addscores',
                        action='store_true',
                        required=False, 
                        default=False,
                        help='If true, add feature scores to the output file',
                        dest='addscores'
                        )
    

    return parser


def ParseArgs(ExecArgs=None):
    if ExecArgs is None:
        ExecArgs = sys.argv

    parser = __CreateArgParser()

    return parser.parse_known_args(args=ExecArgs)


def OnUseError(message):
    parser = __CreateArgParser()
    parser.print_usage()

    log = logging.getLogger('nornir-assemble')
    log.error(message)

    sys.exit()


def ValidateArgs(Args):
    if not os.path.exists(Args.inputpath):
        OnUseError("Input mosaic file not found: " + Args.inputpath)
        
    if Args.tilepath is None:
        Args.tilepath = os.path.dirname(Args.inputpath)
        
    if not '.' in Args.outputpath:
        Args.outputpath = Args.outputpath + '.svg'

    output_file = GenerateAbsOrMosaicRelativePath(Args.outputpath, Args.inputpath)
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    tile_path = GenerateAbsOrMosaicRelativePath(Args.tilepath, Args.inputpath)
    if not os.path.exists(os.path.dirname(tile_path)):
        OnUseError("Tile path not found: " + tile_path)
            
    
        
def GenerateAbsOrMosaicRelativePath(arg, mosaic_path):
    
    output_path = arg
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(mosaic_path), arg)
        
    return output_path

def Execute(ExecArgs=None):
    if ExecArgs is None:
        ExecArgs = sys.argv[1:]
        
    (Args, extra) = ParseArgs(ExecArgs)

    ValidateArgs(Args)
 
    mosaic = nornir_imageregistration.Mosaic.LoadFromMosaicFile(Args.inputpath)
    
    TilesDir = GenerateAbsOrMosaicRelativePath(Args.tilepath, Args.inputpath)
    output_file = GenerateAbsOrMosaicRelativePath(Args.outputpath, Args.inputpath)
    
    downsample = Args.downsample
    
    if downsample is None:
        try:
            folder_name = os.path.basename(TilesDir)
            downsample = float(folder_name) 
        except ValueError:
            print("downsample not specified and could not be determined from filename.  Please pass the downsample parameter.")
            return
             
    initial_tiles = nornir_imageregistration.mosaic_tileset.CreateFromMosaic( mosaic, image_folder=TilesDir, image_to_source_space_scale=downsample)
        
    (distinct_overlaps, new_overlaps, updated_overlaps, removed_overlap_IDs, non_overlapping_IDs) = nornir_imageregistration.arrange_mosaic.GenerateTileOverlaps(initial_tiles,
                                                             existing_overlaps=None,
                                                             offset_epsilon=1.0, 
                                                             min_overlap=Args.min_overlap,
                                                             inter_tile_distance_scale=1,
                                                             exclude_diagonal_overlaps=True)
    
    if Args.addscores:
        nornir_imageregistration.arrange_mosaic.ScoreTileOverlaps(new_overlaps)

    
    nornir_imageregistration.views.plot_tile_overlaps(new_overlaps,
                                                      colors=None,
                                                      OutputFilename=output_file)
    
    if os.path.exists(output_file):
        print("Wrote: " + output_file)
    else:
        print("Output file is missing, unknown error: " + output_file)
         

if __name__ == '__main__':
    (args, extra) = ParseArgs()
    nornir_shared.misc.SetupLogging(OutputPath=os.path.join(os.path.dirname(os.path.dirname(args.inputpath)), "Logs"))

    Execute()
