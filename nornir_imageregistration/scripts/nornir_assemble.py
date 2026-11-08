'''
Created on May 21, 2013

@author: u0490822
'''

import argparse
import os
import sys

import nornir_imageregistration
import nornir_imageregistration.assemble
import nornir_shared.misc
from nornir_imageregistration.scripts._cli_paths import ensure_output_parent_dir, mosaic_default_tile_dir
from nornir_imageregistration.scripts._cli_use_error import exit_on_use_error


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
                        default=None,
                        help='Path to directory containing tiles listed in mosaic (default: the mosaic file\'s directory)',
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
    """Parse command-line arguments for the assemble script. Returns (namespace, unknown)."""
    if ExecArgs is None:
        ExecArgs = sys.argv

    parser = __CreateArgParser()

    return parser.parse_known_args(args=ExecArgs)


def OnUseError(message):
    """Print usage and exit with error (for invalid args)."""
    exit_on_use_error(__CreateArgParser(), 'nornir-assemble', message)


def ValidateArgs(Args):
    """Validate parsed args (paths exist, etc.); exits on failure. A missing tilepath becomes the mosaic's directory."""
    if not os.path.exists(Args.inputpath):
        OnUseError("Input mosaic file not found: " + Args.inputpath)

    ensure_output_parent_dir(Args.outputpath)

    if Args.tilepath is None:
        Args.tilepath = mosaic_default_tile_dir(Args.inputpath)

    if not os.path.exists(Args.tilepath):
        OnUseError("Tile path not found: " + Args.tilepath)


def ReportFileWriteSuccessOrFailure(filepath):
    if os.path.exists(filepath):
        print("Wrote: " + filepath)
    else:
        print("{0} is missing, unknown error: ".format(filepath))


def Execute(ExecArgs=None):
    """Run the assemble script with the given (or default) command-line args."""
    if ExecArgs is None:
        ExecArgs = sys.argv[1:]

    (Args, extra) = ParseArgs(ExecArgs)

    ValidateArgs(Args)

    if Args.use_cp:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
    else:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

    mosaic = nornir_imageregistration.Mosaic.LoadFromMosaicFile(Args.inputpath)
    
    mosaicTileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaic, Args.tilepath,
                                                                             image_to_source_space_scale = 1.0 / Args.scalar)
    mosaicTileset.TranslateToZeroOrigin()

    (mosaicImage, mosaicMask) = mosaicTileset.AssembleImage(target_space_scale=Args.scalar)

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
