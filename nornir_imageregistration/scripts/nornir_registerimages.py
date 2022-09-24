'''
Created on May 21, 2016

@author: u0490822
'''

import argparse
import logging
import os
import sys

import numpy

import nornir_imageregistration.core as core
import nornir_imageregistration.spatial as spatial
import nornir_shared.misc


def __CreateArgParser(ExecArgs=None):



    # conflict_handler = 'resolve' replaces old arguments with new if both use the same option flag
    parser = argparse.ArgumentParser(description="Maps the control space of the warped transform to the control space of the fixed transform and saves the resulting transform as a new .stos file.")

#     parser.add_argument('-output', '-o',
#                         action='store',
#                         required=True,
#                         type=str,
#                         help='Output transform file path',
#                         dest='outputpath')

    parser.add_argument('-fixed', '-f',
                        action='store',
                        required=True,
                        type=str,
                        default=None,
                        help='Fixed transform path',
                        dest='fixedpath'
                        )

    parser.add_argument('-warped', '-w',
                        action='store',
                        required=True,
                        type=str,
                        default=None,
                        help='Warped transform path, ',
                        dest='warpedpath'
                        )
    
    parser.add_argument('-log', '-l',
                        action='store',
                        required=False,
                        type=str,
                        default=None,
                        help="Output path for log files",
                        dest="logpath")
    
    return parser


def ParseArgs(ExecArgs=None):
    if ExecArgs is None:
        ExecArgs = sys.argv

    parser = __CreateArgParser()

    return parser.parse_known_args(args=ExecArgs)


def OnUseError(message):
    parser = __CreateArgParser()
    parser.print_usage()

    log = logging.getLogger('RegisterImages')
    log.error(message)

    sys.exit()

def ValidateArgs(Args):
    if not os.path.exists(Args.fixedpath):
        OnUseError("Fixed image file not found: " + Args.fixedpath)

    if not os.path.exists(Args.warpedpath):
        OnUseError("Warped image file not found: " + Args.warpedpath)        

  
def Execute(ExecArgs=None):
    if ExecArgs is None:
        ExecArgs = sys.argv[1:]
        
    (Args, extra) = ParseArgs(ExecArgs)

    ValidateArgs(Args)
    
    fixed_image = core.LoadImage(Args.fixedpath)
    warped_image = core.LoadImage(Args.warpedpath)
    
    # align_record = core.FindOffset(fixed_image, warped_image)
    
    # print("Overall alignment: %s" % str(align_record))
    
    control_point_coord = numpy.asarray((777, 765))
    fixed_image_area = numpy.asarray((128, 128))
    fixed_control_points_bbox = spatial.Rectangle.CreateFromCenterPointAndArea(control_point_coord, fixed_image_area)
    
    warped_control_points_bbox = spatial.Rectangle.CreateFromCenterPointAndArea(control_point_coord, fixed_image_area)
    
    # Pull a subtile from the images.
    cropped_fixed_image = core.CropImageRect(fixed_image, fixed_control_points_bbox, cval=0)
    cropped_warped_image = core.CropImageRect(warped_image, warped_control_points_bbox, cval=0)
    
    # cropped_fixed_padded_image = core.PadImageForPhaseCorrelation(cropped_fixed_image, 0, NewWidth=warped_control_points_bbox.Width, NewHeight=warped_control_points_bbox.Height)
    
    control_point_align_record = core.FindOffset(cropped_fixed_image, cropped_warped_image, MinOverlap=0.5)
    
    print("Control point alignment: %s" % str(control_point_align_record))
    
    
    core.ShowGrayscale((fixed_image, warped_image, cropped_fixed_image, cropped_warped_image) , title="Control point regions")
 

if __name__ == '__main__':

    (args, extra) = ParseArgs()

    if not args.logpath is None:
        nornir_shared.misc.SetupLogging(OutputPath=os.path.join(os.path.dirname(args.logpath), "Logs"))

    Execute()

    pass
