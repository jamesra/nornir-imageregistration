'''
Created on May 21, 2013

@author: u0490822
'''

import argparse
import os
import nornir_imageregistration.io.stosfile as stosfile
import nornir_shared.misc
import logging
import sys

def __CreateArgParser(ExecArgs=None):



    # conflict_handler = 'resolve' replaces old arguments with new if both use the same option flag
    parser = argparse.ArgumentParser(description="Maps the control space of the warped transform to the control space of the fixed transform and saves the resulting transform as a new .stos file.")

    parser.add_argument('-output', '-o',
                        action='store',
                        required=True,
                        type=str,
                        help='Output transform file path',
                        dest='outputpath')

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

    return parser;

def ParseArgs(ExecArgs=None):
    if ExecArgs is None:
        ExecArgs = sys.argv;

    parser = __CreateArgParser();

    return parser.parse_known_args(args=ExecArgs)


def OnUseError(message):
    parser = __CreateArgParser()
    parser.print_usage()

    log = logging.getLogger('AddTransforms')
    log.error(message)

    sys.exit()

def ValidateArgs(Args):
    if not os.path.exists(Args.fixedpath):
        OnUseError("Fixed stos file not found: " + Args.fixedpath)

    if not os.path.exists(Args.warpedpath):
        OnUseError("Warped stos file not found: " + Args.warpedpath)

    if not os.path.exists(os.path.dirname(Args.outputpath)):
        os.makedirs(os.path.dirname(Args.outputpath))


def Execute(ExecArgs=None):

    (Args, extra) = ParseArgs(ExecArgs)

    ValidateArgs(Args)

    MToVStos = stosfile.AddStosTransforms(Args.warpedpath, Args.fixedpath)
    MToVStos.Save(Args.outputpath)

    if os.path.exists(Args.outputpath):
        print "Wrote: " + Args.outputpath
    else:
        print "Outputfile is missing, unknown error: " + Args.outputpath


if __name__ == '__main__':

    (args, extra) = ParseArgs();

    nornir_shared.misc.SetupLogging(os.path.join(os.path.dirname(args.outputpath), "Logs"))

    Execute();

    pass