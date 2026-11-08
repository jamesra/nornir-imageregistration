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
from nornir_imageregistration.scripts._cli_paths import mosaic_default_tile_dir
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
                        help='The input images are a different size than the transform, scale the transform by the specified inverse factor.  So if the images are downsampled by a factor of 4 pass 4 to this parameter.',
                        dest='scalar'
                        )

    parser.add_argument('-tilesize', '-t',
                        action='store',
                        required=False,
                        type=tuple,
                        default=(256, 256),
                        help="Tile size to generate, Width x Height",
                        dest="TileSize")

    parser.add_argument('-tilepath', '-p',
                        action='store',
                        required=False,
                        type=str,
                        default=None,
                        help='Path to directory containing tiles listed in mosaic (default: the mosaic file\'s directory)',
                        dest='tilepath')
    parser.add_argument('-cuda', '-c',
                        action='store_true',
                        required=False,
                        help='Use GPU for calculations if available',
                        dest='use_cp'
                        )

    return parser


def ParseArgs(ExecArgs=None):
    """Parse command-line arguments for the assemble_tiles script. Returns (namespace, unknown)."""
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

    os.makedirs(os.path.dirname(os.path.abspath(Args.outputpath)), exist_ok=True)

    if Args.tilepath is None:
        Args.tilepath = mosaic_default_tile_dir(Args.inputpath)

    if not os.path.exists(Args.tilepath):
        OnUseError("Tile path not found: " + Args.tilepath)


def Execute(ExecArgs=None):
    """Run the assemble_tiles script with the given (or default) command-line args."""
    if ExecArgs is None:
        ExecArgs = sys.argv[1:]

    (Args, extra) = ParseArgs(ExecArgs)

    ValidateArgs(Args)

    if Args.use_cp:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
    else:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

    mosaic = nornir_imageregistration.Mosaic.LoadFromMosaicFile(Args.inputpath)
    mosaicTileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaic=mosaic,
                                                                             image_folder=Args.tilepath,
                                                                             image_to_source_space_scale=1.0 / Args.scalar)
    mosaicTileset.TranslateToZeroOrigin()

    mosaicImage = mosaicTileset.AssembleImage(target_space_scale=Args.scalar)

    if not Args.outputpath.endswith('.png'):
        Args.outputpath += '.png'

    nornir_imageregistration.SaveImage(Args.outputpath, mosaicImage)
    if os.path.exists(Args.outputpath):
        print("Wrote: " + Args.outputpath)
    else:
        print("Outputfile is missing, unknown error: " + Args.outputpath)


if __name__ == '__main__':
    (args, extra) = ParseArgs()

    nornir_shared.misc.SetupLogging(OutputPath=os.path.join(os.path.dirname(args.outputpath), "Logs"))

    Execute()
