'''
Created on May 21, 2013

@author: u0490822
'''

import argparse
import os
import sys

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
                        help='Output mosaic file path',
                        dest='outputpath')

    parser.add_argument('-tilepath', '-s',
                        action='store',
                        required=False,
                        type=str,
                        default=None,
                        help='Path to directory containing tiles listed in mosaic (default: the mosaic file\'s directory)',
                        dest='tilepath'
                        )

    return parser


def ParseArgs(ExecArgs=None):
    """Parse command-line arguments for the translatemosaic script. Returns (namespace, unknown)."""
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

    if Args.tilepath is None:
        Args.tilepath = mosaic_default_tile_dir(Args.inputpath)

    os.makedirs(os.path.dirname(os.path.abspath(Args.outputpath)), exist_ok=True)

    if not os.path.exists(Args.tilepath):
        OnUseError("Tile path not found: " + Args.tilepath)

    if not '.' in Args.outputpath:
        Args.outputpath += '.mosaic'



def Execute(ExecArgs=None):
    """Run the translatemosaic script with the given (or default) command-line args."""
    if ExecArgs is None:
        ExecArgs = sys.argv[1:]

    (Args, extra) = ParseArgs(ExecArgs)

    ValidateArgs(Args)

    mosaic = nornir_imageregistration.Mosaic.LoadFromMosaicFile(Args.inputpath)

    # timer = TaskTimer()
    # timer.Start("ArrangeTiles " + Args.tilepath)
    tileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(
        mosaic, image_folder=str(Args.tilepath), image_to_source_space_scale=1.0
    )
    config = nornir_imageregistration.settings.TranslateSettings()
    translated_mosaic_tileset = tileset.ArrangeTilesWithTranslate(config=config)
    # timer.End("ArrangeTiles " + Args.tilepath, True)
    translated_mosaic_tileset.SaveMosaic(Args.outputpath)

    if os.path.exists(Args.outputpath):
        print("Wrote: " + Args.outputpath)
    else:
        print("Outputfile is missing, unknown error: " + Args.outputpath)


if __name__ == '__main__':
    (args, extra) = ParseArgs()
    nornir_shared.misc.SetupLogging(OutputPath=os.path.join(os.path.dirname(args.outputpath), "Logs"))

    Execute()
