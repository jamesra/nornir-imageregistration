'''
Created on May 21, 2013

@author: u0490822
'''

import argparse
import logging
import os
import sys

from nornir_imageregistration.files.stos_override_args import StosOverrideArgs
import nornir_imageregistration.stos_brute as sb
import nornir_shared.misc


def __CreateArgParser(ExecArgs=None):
    # conflict_handler = 'resolve' replaces old arguments with new if both use the same option flag
    parser = argparse.ArgumentParser(description="Produce a registered image for the moving image in a .stos file")

    parser.add_argument('-input', '-i',
                        action='store',
                        required=False,
                        type=str,
                        default=None,
                        help='Input .stos file path',
                        dest='inputpath')

    parser.add_argument('-output', '-o',
                        action='store',
                        required=True,
                        type=str,
                        help='Output .stos file path',
                        dest='outputpath')

    StosOverrideArgs.ExtendParser(parser, RequireInputImages=True)

    parser.add_argument('-min_overlap', '-mino',
                        action='store',
                        required=False,
                        type=float,
                        default=0.5,
                        help='images are known to overlap by at least this percentage',
                        dest='min_overlap'
                        )

    parser.add_argument('-checkflip', '-flip',
                        action='store_true',
                        required=False,
                        help='If true, a vertically flipped version of the warped image will also be searched for the best alignment',
                        dest='testflip'
                        )

    parser.add_argument('-cuda', '-c',
                        action='store_true',
                        required=False,
                        # type=bool,
                        # default=False,
                        help='Use GPU for calculations if available',
                        dest='use_cp'
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

    log = logging.getLogger('nornir-stos-brute')
    log.error(message)

    sys.exit()


def Execute(ExecArgs=None):
    if ExecArgs is None:
        ExecArgs = sys.argv[1:]

    (Args, extra) = ParseArgs(ExecArgs)

    stosArgs = StosOverrideArgs(Args)

    if not os.path.exists(os.path.dirname(Args.outputpath)):
        os.makedirs(os.path.dirname(Args.outputpath))

    alignRecord = sb.SliceToSliceBruteForce(stosArgs.ControlImage,
                                            stosArgs.WarpedImage,
                                            stosArgs.ControlMask,
                                            stosArgs.WarpedMask,
                                            MinOverlap=Args.min_overlap,
                                            TestFlip=Args.testflip)

    if not (stosArgs.ControlMask is None or stosArgs.WarpedMask is None):
        stos = alignRecord.ToStos(stosArgs.ControlImage,
                                  stosArgs.WarpedImage,
                                  stosArgs.ControlMask,
                                  stosArgs.WarpedMask,
                                  PixelSpacing=1)

        stos.Save(Args.outputpath)
    else:
        stos = alignRecord.ToStos(ImagePath=stosArgs.ControlImage,
                                  WarpedImagePath=stosArgs.WarpedImage,
                                  PixelSpacing=1)

        stos.Save(Args.outputpath, AddMasks=False)

    # self.assertTrue(os.path.exists(stosArgs.stosOutput), "No output stos file created")

    if os.path.exists(stosArgs.stosOutput):
        print("Wrote: " + stosArgs.stosOutput)
    else:
        print("Outputfile is missing, unknown error: " + stosArgs.stosOutput)


if __name__ == '__main__':
    (args, extra) = ParseArgs()

    nornir_shared.misc.SetupLogging(OutputPath=os.path.join(os.path.dirname(args.outputpath), "Logs"))

    Execute()
