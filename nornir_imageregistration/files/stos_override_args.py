'''
Created on Oct 29, 2018

@author: u0490822
'''

import os
import nornir_imageregistration

class StosOverrideArgs(object):
    '''
    This is a helper class that scripts can use to replace parameters of a .stos file
    with parameters from the command line.  It produces a new .stos file
    '''
    
    @classmethod 
    def ExtendParser(cls, parser, RequireInputImages):
        '''
        Adds the common command line parameters to the parser
        :param argparse.ArgumentParser parser: The command-line argument parser to extend
        '''
        
        parser.add_argument('-scale', '-s',
                        action='store',
                        required=False,
                        type=float,
                        default=1.0,
                        help='The input images are a different size than the desired transform, scale the transform by the specified factor',
                        dest='scalar'
                        )

        parser.add_argument('-fixedimage', '-f',
                            action='store',
                            required=RequireInputImages,
                            type=str,
                            default=None,
                            help='Fixed image, overrides .stos file fixed image',
                            dest='fixedimagepath'
                            )
    
        parser.add_argument('-warpedimage', '-w',
                            action='store',
                            required=RequireInputImages,
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

    @property
    def ControlImage(self):
        return self.stos.ControlImageFullPath

    @property
    def WarpedImage(self):
        return self.stos.MappedImageFullPath

    @property
    def ControlMask(self):
        return self.stos.ControlMaskFullPath

    @property
    def WarpedMask(self):
        return self.stos.MappedMaskFullPath


    def __init__(self, Args):
        self.stosPath = Args.inputpath
        self.stosOutput = Args.outputpath

        self.fixedImage = Args.fixedimagepath
        self.warpedImage = Args.warpedimagepath
        self.fixedMask = Args.fixedmaskpath
        self.warpedMask = Args.warpedmaskpath

        self.stos = self.MergeStosAndArgs(Args)


    def MergeStosAndArgs(self, Args):

        stos = nornir_imageregistration.files.StosFile()
        
        if not Args.inputpath is None:
            if os.path.exists(Args.inputpath):
                stos = nornir_imageregistration.files.StosFile.Load(Args.inputpath)

                if Args.scalar != 1.0:
                    stos.scale(Args.scalar)
            
            stosDir = os.path.dirname(Args.inputpath)
            stos.TryConvertRelativePathsToAbsolutePaths(stosDir)                

        if not Args.fixedimagepath is None:
            stos.ControlImageFullPath = self.fixedImage

        if not self.warpedImage is None:
            stos.MappedImageFullPath = self.warpedImage

        if not self.fixedMask is None:
            stos.ControlMaskFullPath = self.fixedMask

        if not self.warpedMask is None:
            stos.MappedMaskFullPath = self.warpedMask

        return stos
