'''
Created on May 21, 2013

@author: u0490822
'''

import argparse
import logging
import os
import sys

import nornir_imageregistration
import nornir_imageregistration.assemble
import nornir_shared.misc


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

    parser.add_argument('-scale', '-s',
                        action='store',
                        required=False,
                        type=float,
                        default=1.0,
                        help='The input images are a different size than the transform, scale the transform by the specified factor',
                        dest='scalar'
                        )

    parser.add_argument('-tilepath', '-p',
                        action='store',
                        required=False,
                        type=str,
                        default=True,
                        help='Path to directory containing tiles listed in mosaic',
                        dest='tilepath'
                        )
    parser.add_argument('-cuda', '-c',
                        action='store_true',
                        required=False,
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

    log = logging.getLogger('nornir-assemble')
    log.error(message)

    sys.exit()


def ValidateArgs(Args):
    if not os.path.exists(Args.inputpath):
        OnUseError("Input mosaic file not found: " + Args.inputpath)

    if not os.path.exists(os.path.dirname(Args.outputpath)):
        os.makedirs(os.path.dirname(Args.outputpath), exist_ok=True)

    if not Args.tilepath is None:
        if not os.path.exists(Args.tilepath):
            OnUseError("Tile path not found: " + Args.tilepath)


def ReportFileWriteSuccessOrFailure(filepath):
    if os.path.exists(filepath):
        print("Wrote: " + filepath)
    else:
        print("{0} is missing, unknown error: ".format(filepath))


def Execute(ExecArgs=None):
    if ExecArgs is None:
        ExecArgs = sys.argv[1:]

    (Args, extra) = ParseArgs(ExecArgs)

    ValidateArgs(Args)

    mosaic = nornir_imageregistration.Mosaic.LoadFromMosaicFile(Args.inputpath)

    mosaicTileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaic, Args.tilepath,
                                                                             image_to_source_space_scale=1.0 / Args.scalar)
    mosaicTileset.TranslateToZeroOrigin()

    (mosaicImage, mosaicMask) = mosaicTileset.AssembleImage(usecluster=not Args.use_cp, use_cp=Args.use_cp,
                                                                target_space_scale=Args.scalar)

    output_dirname = os.path.dirname(Args.outputpath)
    output_filename = os.path.basename(Args.outputpath)
    (output_name, output_ext) = os.path.splitext(output_filename)

    if output_ext is None or len(output_ext) == 0:
        output_ext = '.png'

        if os.path.isdir(Args.outputpath):
            mosaic_basename = os.path.basename(Args.inputpath)
            (mosaic_name, _) = os.path.splitext(mosaic_basename)

            Args.outputpath = os.path.join(Args.outputpath, mosaic_name + output_ext)
        else:
            Args.outputpath += output_ext

    mask_output_fullpath = os.path.join(output_dirname, output_name + '_mask' + output_ext)

    nornir_imageregistration.core.SaveImage(Args.outputpath, mosaicImage, bpp=8)
    ReportFileWriteSuccessOrFailure(Args.outputpath)

    nornir_imageregistration.core.SaveImage(mask_output_fullpath, mosaicMask)
    ReportFileWriteSuccessOrFailure(mask_output_fullpath)


if __name__ == '__main__':
    (args, extra) = ParseArgs()

    nornir_shared.misc.SetupLogging(OutputPath=os.path.join(os.path.dirname(args.outputpath), "Logs"))

    Execute()
