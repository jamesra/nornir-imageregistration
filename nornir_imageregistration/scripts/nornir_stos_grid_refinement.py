'''
Created on Oct 29, 2018

@author: u0490822
''' 

import argparse
import logging
import os
import sys

import nornir_imageregistration
import nornir_imageregistration.local_distortion_correction

import nornir_shared.misc
from nornir_shared.argparse_helpers import NumberPair, NumberList

from nornir_imageregistration.files.stos_override_args import StosOverrideArgs


def __CreateArgParser(ExecArgs=None):



    # conflict_handler = 'resolve' replaces old arguments with new if both use the same option flag
    parser = argparse.ArgumentParser(description="Produce a registered image for the moving image in a .stos file")

    parser.add_argument('-input', '-i',
                        action='store',
                        required=True,
                        type=str,
                        help='Input .stos file path',
                        dest='inputpath')

    parser.add_argument('-output', '-o',
                        action='store',
                        required=True,
                        type=str,
                        help='Output .stos file path',
                        dest='outputpath')
    
    StosOverrideArgs.ExtendParser(parser, RequireInputImages=False)

    parser.add_argument('-min_overlap', '-mino',
                        action='store',
                        required=False,
                        type=float,
                        default=0.5,
                        help='images are known to overlap by at least this percentage',
                        dest='min_alignment_overlap'
                        )
    
    parser.add_argument('-cell_size', '-c',
                        action='store',
                        required=False,
                        type=NumberPair,
                        default=(128,128),
                        help='Dimensions of cells (subsets of the images) used for registration.  (The first iteration uses double-sized cells.)',
                        dest='cell_size'
                        )
    
    parser.add_argument('-grid_spacing', '-gs',
                        action='store',
                        required=False,
                        type=NumberPair,
                        default=(256,256),
                        help='Distances between centers of cells used for registration.',
                        dest='grid_spacing'
                        )
    
    parser.add_argument('-iterations', '-it',
                        action='store',
                        required=False,
                        type=int,
                        default=10,
                        help='Maximum number of iterations',
                        dest='num_iterations'
                        )
    
    parser.add_argument('-angles', '-a',
                        action='store',
                        required=False,
                        type=NumberList,
                        default=None,
                        help='Rotate each cell by each of the specified degrees and choose the best alignment, slower but may be more accurate.  Default is to not rotate.',
                        dest='angles_to_search'
                        )
    
    parser.add_argument('-travel_cutoff', '-t',
                        action='store',
                        required=False,
                        type=float,
                        default=0.5,
                        help='If the registration for a cell translates by less than travel_cutoff the cell is "finalized" and is not checked for future iterations.',
                        dest='min_travel_for_finalization'
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

    log = logging.getLogger('nornir-stos-grid-refinement')
    log.error(message)

    sys.exit()


def Execute(ExecArgs=None):
    
    if ExecArgs is None:
        ExecArgs = sys.argv[1:]
        
    (Args, extra) = ParseArgs(ExecArgs)

    stosArgs = StosOverrideArgs(Args)
    inputStos = stosArgs.MergeStosAndArgs(Args)
    
    if not os.path.exists(os.path.dirname(Args.outputpath)):
        os.makedirs(os.path.dirname(Args.outputpath))

    nornir_imageregistration.local_distortion_correction.RefineStosFile(inputStos, 
                   Args.outputpath, 
                   num_iterations=Args.num_iterations,
                   cell_size=Args.cell_size,
                   grid_spacing=Args.grid_spacing,
                   angles_to_search=Args.angles_to_search,
                   min_travel_for_finalization=Args.min_travel_for_finalization,
                   min_alignment_overlap=Args.min_alignment_overlap)
    
    
    # self.assertTrue(os.path.exists(stosArgs.stosOutput), "No output stos file created")

    if os.path.exists(stosArgs.stosOutput):
        print("Wrote: " + stosArgs.stosOutput)
    else:
        print("Outputfile is missing, unknown error: " + stosArgs.stosOutput)


if __name__ == '__main__':

    (args, extra) = ParseArgs()

    nornir_shared.misc.SetupLogging(os.path.join(os.path.dirname(args.outputpath), "Logs"))

    Execute()
