'''
Created on May 21, 2013

@author: u0490822
'''

import argparse
import os
import nornir_imageregistration.stos_brute as sb
import nornir_shared.misc
import logging
import sys

def __CreateArgParser(ExecArgs=None):



    # conflict_handler = 'resolve' replaces old arguments with new if both use the same option flag
    parser = argparse.ArgumentParser(description="Produce a registered image for the moving image in a .stos file")

    parser.add_argument('-input', '-i',
                        action='store',
                        required=False,
                        type=str,
                        help='Input .stos file path',
                        dest='inputpath')

    parser.add_argument('-output', '-o',
                        action='store',
                        required=True,
                        type=str,
                        help='Output .stos file path',
                        dest='outputpath')

    parser.add_argument('-scale', '-s',
                        action='store',
                        required=False,
                        type=float,
                        default=1.0,
                        help='The input images are a different size than the transform, scale the transform by the specified factor',
                        dest='scalar'
                        )

    parser.add_argument('-fixedimage', '-f',
                        action='store',
                        required=True,
                        type=str,
                        default=None,
                        help='Fixed image, overrides .stos file fixed image',
                        dest='fixedimagepath'
                        )

    parser.add_argument('-warpedimage', '-w',
                        action='store',
                        required=True,
                        type=str,
                        default=None,
                        help='warped image, overrides .stos file warped image',
                        dest='warpedimagepath'
                        )

    parser.add_argument('-fixedmask', '-fm',
                        action='store',
                        required=False,
                        type=str,
                        default=None,
                        help='Fixed mask, overrides .stos file fixed mask',
                        dest='fixedmaskpath'
                        )

    parser.add_argument('-warpedmask', '-wm',
                        action='store',
                        required=False,
                        type=str,
                        default=None,
                        help='warped mask, overrides .stos file warped mask',
                        dest='warpedmaskpath'
                        )

    parser.add_argument('-minoverlap', '-mino',
                        action='store',
                        required=False,
                        type=float,
                        default=None,
                        help='images are known to overlap by at least this percentage',
                        dest='minoverlap'
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

class StosBruteArgs(object):

    @property
    def ControlImage(self):
        return self.stos.ControlImageFullPath

    @property
    def WarpedImage(self):
        return self.stos.WarpedImageFullPath

    @property
    def ControlMask(self):
        return self.stos.ControlMaskFullPath

    @property
    def WarpedMask(self):
        return self.stos.MappedMaskFullPath


    def __init__(self, Args):
        self.stosPath = Args.inputPath
        self.stosOutput = Args.outputpath

        self.fixedImage = Args.fixedimagepath
        self.warpedImage = Args.warpedimagepath
        self.fixedMask = Args.fixedmaskpath
        self.warpedMask = Args.warpedmaskpath

        self.stos = self.MergeStosAndArgs(Args)


    def MergeStosAndArgs(self, Args):

        stos = StosFile()
        if not Args.inputPath is None:
            if os.path.exists(Args.inputPath):
                stos = StosFile.Load(Args.inputpath)

                if Args.scalar != 1.0:
                    stos.scale(Args.scalar)

        if not Args.fixedImage is None:
            stos.ControlImageFullPath = Args.fixedImage

        if not self.warpedImage is None:
            stos.MappedImageFullPath = Args.warpedImage

        if not self.fixedmaskpath is None:
            stos.ControlMaskFullPath = self.fixedmaskpath

        if not self.warpedmaskpath is None:
            self.MappedMaskFullPath = stos.warpedmaskpath

        if not os.path.exists(os.path.dirname(Args.outputpath)):
            os.makedirs(os.path.dirname(Args.outputpath))

        return stos


def Execute(ExecArgs=None):

    (Args, extra) = ParseArgs(ExecArgs)

    stosArgs = StosBruteArgs(Args)



    alignRecord = sb.SliceToSliceBruteForce(stosArgs.ControlImage, stosArgs.WarpedImage, stosArgs.ControlMask, stosArgs.WarpedMask, MinOverlap=stosArgs.minoverlap)

    if not (stosArgs.ControlMask is None or stosArgs.WarpedMask is None):
        stos = alignment.ToStos(stosArgs.ControlImage,
                                 stosArgs.WarpedImage,
                                 stosArgs.ControlMask,
                                 stosArgs.WarpedMask,
                                 PixelSpacing=1)

        stos.Save(stosNode.FullPath)
    else:
        stos = alignment.ToStos(ControlImageNode.FullPath,
                             MappedImageNode.FullPath,
                             PixelSpacing=1)

        stos.Save(stosArgs.stosOutput, AddMasks=False)

    # self.assertTrue(os.path.exists(stosArgs.stosOutput), "No output stos file created")

    if os.path.exists(stosArgs.stosOutput):
        print "Wrote: " + stosArgs.stosOutput
    else:
        print "Outputfile is missing, unknown error: " + stosArgs.stosOutput


if __name__ == '__main__':

    (args, extra) = ParseArgs()

    nornir_shared.misc.SetupLogging(os.path.join(os.path.dirname(args.outputpath), "Logs"))

    Execute()
