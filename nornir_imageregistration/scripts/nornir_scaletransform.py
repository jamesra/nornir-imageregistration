'''
Created on May 21, 2013

@author: u0490822
'''

import argparse
import logging
import os
import sys

import nornir_imageregistration.files.stosfile as stosfile
import nornir_shared.misc


def __CreateArgParser(ExecArgs=None):
    # conflict_handler = 'resolve' replaces old arguments with new if both use the same option flag
    parser = argparse.ArgumentParser(
        description="Maps the control space of the warped transform to the control space of the fixed transform and saves the resulting transform as a new .stos file.")

    parser.add_argument('-output', '-o',
                        action='store',
                        required=True,
                        type=str,
                        help='Output transform file path',
                        dest='outputpath')

    parser.add_argument('-input', '-i',
                        action='store',
                        required=True,
                        type=str,
                        default=None,
                        help='Input transform file path',
                        dest='inputpath'
                        )

    parser.add_argument('-scale', '-s',
                        action='store',
                        required=True,
                        type=float,
                        default=None,
                        help='Scale value',
                        dest='scale'
                        )

    return parser


def ParseArgs(ExecArgs=None):
    """Parse command-line arguments for the scaletransform script. Returns (namespace, unknown)."""
    if ExecArgs is None:
        ExecArgs = sys.argv

    parser = __CreateArgParser()

    return parser.parse_known_args(args=ExecArgs)


def OnUseError(message):
    """Print usage and exit with error (for invalid args)."""
    parser = __CreateArgParser()
    parser.print_usage()

    log = logging.getLogger('AddTransforms')
    log.error(message)

    sys.exit()


def ValidateArgs(Args):
    """Validate parsed args (paths exist, etc.); exits on failure."""
    if not os.path.exists(Args.inputpath):
        OnUseError("Input stos file not found: " + Args.inputpath)

    if not os.path.exists(os.path.dirname(Args.outputpath)):
        os.makedirs(os.path.dirname(Args.outputpath))


def Execute(ExecArgs=None):
    """Run the scaletransform script with the given (or default) command-line args."""
    if ExecArgs is None:
        ExecArgs = sys.argv[1:]

    (Args, extra) = ParseArgs(ExecArgs)

    ValidateArgs(Args)

    stos = stosfile.StosFile.Load(Args.inputpath)
    stos.Scale(Args.scale)
    stos.Save(Args.outputpath)

    if os.path.exists(Args.outputpath):
        print("Wrote: " + Args.outputpath)
    else:
        print("Outputfile is missing, unknown error: " + Args.outputpath)


if __name__ == '__main__':
    (args, extra) = ParseArgs()

    nornir_shared.misc.SetupLogging(OutputPath=os.path.join(os.path.dirname(args.outputpath), "Logs"))

    Execute()

    pass
