from __future__ import annotations

import copy
import logging
import os
import typing

import nornir_imageregistration
import nornir_shared.checksum
import nornir_shared.files
import nornir_shared.prettyoutput as PrettyOutput


def __argumentToStos(Argument):
    stosObj = None
    if isinstance(Argument, str):
        stosObj = StosFile.Load(Argument)
    elif isinstance(Argument, StosFile):
        stosObj = Argument

    assert (stosObj is not None)

    return stosObj


class StosFile(object):
    """description of class"""

    @staticmethod
    def FileHasMasks(path: str) -> bool:
        stosObj = StosFile.Load(path)
        return stosObj.HasMasks

    @staticmethod
    def LoadChecksum(path: str):
        # assert(os.path.exists(path))
        stosObj = StosFile.Load(path)
        if stosObj is None:
            return None
        return stosObj.Checksum

    @property
    def Transform(self):
        return self._Transform

    @Transform.setter
    def Transform(self, val: str | nornir_imageregistration.transforms.ITransform | None):
        if val is None:
            self._Transform = None
            return

        if isinstance(val, nornir_imageregistration.transforms.ITransform):
            self._Transform = nornir_imageregistration.transforms.TransformToIRToolsString(val)
        elif isinstance(val, str):
            self._Transform = val
        else:
            raise TypeError("Transform must be a transform object or a ITK transform string")

        return

    @property
    def Downsample(self):
        return self._Downsample

    @Downsample.setter
    def Downsample(self, newDownsample: float | int):
        if self._Downsample is None:  # Don't scale if
            self._Downsample = newDownsample
        else:
            scalar = self._Downsample / newDownsample
            self.Scale(scalar)
            self._Downsample = newDownsample

    @property
    def ControlImageFullPath(self) -> str | None:
        return os.path.join(self.ControlImagePath, self.ControlImageName)

    @ControlImageFullPath.setter
    def ControlImageFullPath(self, val):

        if val is None:
            self.ControlImagePath = None
            self.ControlImageName = None
        else:
            d = os.path.dirname(val)
            f = os.path.basename(val)

            self.ControlImagePath = d.strip()
            self.ControlImageName = f.strip()

    @property
    def MappedImageFullPath(self) -> str | None:
        return os.path.join(self.MappedImagePath, self.MappedImageName)

    @MappedImageFullPath.setter
    def MappedImageFullPath(self, val):

        if val is None:
            self.MappedImagePath = None
            self.MappedImageName = None
        else:
            d = os.path.dirname(val)
            f = os.path.basename(val)
            self.MappedImagePath = d.strip()
            self.MappedImageName = f.strip()

    @property
    def ControlMaskFullPath(self) -> str | None:
        if self.ControlMaskPath is None or self.ControlMaskName is None:
            return None

        return os.path.join(self.ControlMaskPath, self.ControlMaskName)

    @ControlMaskFullPath.setter
    def ControlMaskFullPath(self, val: str | None):
        if val is None:
            self.ControlMaskPath = None
            self.ControlMaskName = None
            return

        d = os.path.dirname(val)
        f = os.path.basename(val)
        self.ControlMaskPath = d.strip()
        self.ControlMaskName = f.strip()

    @property
    def MappedMaskFullPath(self) -> str | None:
        if self.MappedMaskPath is None or self.MappedMaskName is None:
            return None

        return os.path.join(self.MappedMaskPath, self.MappedMaskName)

    @MappedMaskFullPath.setter
    def MappedMaskFullPath(self, val: str | None):
        if val is None:
            self.MappedMaskPath = None
            self.MappedMaskName = None
            return

        d = os.path.dirname(val)
        f = os.path.basename(val)

        self.MappedMaskPath = d.strip()
        self.MappedMaskName = f.strip()

    @property
    def Checksum(self) -> str:
        if self.Transform is None:
            return ""

        compressedString = StosFile.CompressedTransformString(self.Transform)
        return nornir_shared.checksum.DataChecksum(compressedString)

    @property
    def HasMasks(self) -> bool:
        return not (self.MappedMaskName is None or self.ControlMaskName is None)

    def ClearMasks(self):
        '''Remove masks from the file'''
        self.MappedMaskFullPath = None
        self.ControlMaskFullPath = None
        return

    #   NewImageNameTemplate = ("%(section)" + IrUtil.SectionFormat + "_%(channel)_%(type)_" + str(newspacing) + ".png\n")
    #   controlNewImageName = NewImageNameTemplate % {'section' : ControlSectionNumber}

    def __init__(self):
        self._Transform = None

        self.ControlImagePath = None
        self.MappedImagePath = None

        self.ControlImageName = None
        self.MappedImageName = None

        self.ControlMaskPath = None
        self.MappedMaskPath = None

        self.ControlMaskName = None
        self.MappedMaskName = None

        self.ControlSectionNumber = None
        self.MappedSectionNumber = None

        self.ControlChannel = None  # What channel was used to create the stos file?
        self.MappedChannel = None

        self.ControlMosaicFilter = None  # mosaic, blob, mask, etc...
        self.MappedMosaicFilter = None  # mosaic, blob, mask, etc...

        self._Downsample = None  # How much the images used for the stos file are downsampled

        self.UseMasksIfExist = True

        self.ControlImageDim = None
        self.MappedImageDim = None

        self.StosSource = None

        self.ImageToTransform = dict()
        return

    def __str__(self):
        return f'{self.ControlSectionNumber}<-{self.MappedSectionNumber} DS:{self._Downsample}'

    @classmethod
    def GetInfo(cls, filename: str):
        '''Returns details about a stos file we can learn from its filename
           returns  [mappedSection, controlSection, Channel, Filter, Source, Downsample]'''

        Logger = logging.getLogger(__name__ + str(cls.__class__))

        # Make sure extension is removed from filename
        [baseName, ext] = os.path.splitext(filename)
        baseName = os.path.basename(baseName)

        parts = baseName.split("_")
        try:
            sections = parts[0].split('-')
            mappedSection = int(sections[0])
            controlSection = int(sections[1])

        except:
            mappedSection = None
            controlSection = None
            Logger.info('Could not determine section numbers: ' + str(filename))
            # raise

        try:
            Channel = parts[-4]
        except:
            Channel = None
            Logger.info('Could not determine Channels: ' + str(filename))
            # raise

        try:
            Filter = parts[-3]
        except:
            Filter = None
            Logger.info('Could not determine Filter: ' + str(filename))
            # raise

        try:
            Source = parts[-2]
        except:
            Source = None
            Logger.info('Could not determine transform: ' + str(filename))
            # raise

        try:
            Downsample = int(parts[-1])
        except:
            Downsample = None
            Logger.info('Could not determine _Downsample: ' + str(filename))
            # raise

        return mappedSection, controlSection, Channel, Filter, Source, Downsample

    @staticmethod
    def Create(controlImageFullPath, mappedImageFullPath, Transform, controlMaskFullPath=None, mappedMaskFullPath=None):
        stosObj = StosFile()
        stosObj.ControlImageFullPath = controlImageFullPath
        stosObj.MappedImageFullPath = mappedImageFullPath
        stosObj.Transform = nornir_imageregistration.transforms.TransformToIRToolsString(Transform)

        if controlMaskFullPath is not None:
            stosObj.ControlMaskFullPath = controlMaskFullPath
            stosObj.MappedMaskFullPath = mappedMaskFullPath

        return stosObj

    @staticmethod
    def Load(filename: str) -> StosFile:

        obj = StosFile()

        try:
            [obj.MappedSectionNumber, obj.ControlSectionNumber, Channels, Filters, obj.StosSource,
             obj._Downsample] = StosFile.GetInfo(filename)
        except:
            pass

        lines = []

        try:
            with open(filename, 'r') as fMosaic:
                lines = fMosaic.readlines()
        except FileNotFoundError:
            PrettyOutput.LogErr("stos file not found: " + filename)
            raise
        except Exception as error:
            PrettyOutput.LogErr(f"Unexpected error {error} while opening stos file {filename}")
            raise

        if len(lines) < 7:
            PrettyOutput.LogErr("%s is not a valid stos file" % filename)
            raise ValueError("%s is not a valid stos file" % filename)

        obj.ControlImageFullPath = lines[0].strip()
        obj.MappedImageFullPath = lines[1].strip()

        ControlDims = lines[4].split()
        MappedDims = lines[5].split()

        obj.ControlImageDim = [float(x) for x in ControlDims]
        obj.MappedImageDim = [float(x) for x in MappedDims]

        obj.Transform = lines[6].strip()

        if len(lines) > 8:
            obj.ControlMaskFullPath = lines[8]
            obj.MappedMaskFullPath = lines[9]

        return obj

    @staticmethod
    def IsValid(filename) -> bool:
        '''#If stos-grid completely fails it uses the maximum float value for each data point.  This function loads the transform and ensures it is valid'''

        if not os.path.exists(filename):
            return False

        stos = StosFile.Load(filename)

        try:
            Transform = nornir_imageregistration.transforms.LoadTransform(stos.Transform, pixelSpacing=1)
        except:
            return False

        return True

    def Scale(self, scalar: float):
        '''Scale this stos transform by the requested amount'''

        # Adjust the mosaic and mask names if present
        self.ControlImageDim = [x * scalar for x in self.ControlImageDim]
        self.MappedImageDim = [x * scalar for x in self.MappedImageDim]

        # Adjust the grid points
        transformObj = nornir_imageregistration.transforms.LoadTransform(self.Transform, pixelSpacing=1)
        transformObj.Scale(scalar=scalar)

        #         if hasattr(transformObj, 'gridWidth'):
        #             # Save as a stos grid if we can
        #             self.Transform = nornir_imageregistration.transforms.TransformToIRToolsGridString(transformObj, transformObj.gridWidth, transformObj.gridHeight, bounds=self.MappedImageDim)
        #         else:
        #             self.Transform = nornir_imageregistration.transforms.TransformToIRToolsString(transformObj, bounds=self.MappedImageDim)
        self.Transform = nornir_imageregistration.transforms.TransformToIRToolsString(transformObj,
                                                                                      bounds=self.MappedImageDim)

        self._Downsample = self._Downsample * scalar

    def Save(self, filename: str, AddMasks: bool = True):
        # This function needs reworking to use different object variables'
        # assert(False)
        OutLines = list()

        # mosaic files to be warped
        OutLines.append(self.ControlImageFullPath)
        OutLines.append(self.MappedImageFullPath)

        # Write the header
        OutLines.append("0")
        OutLines.append("0")

        if os.path.exists(self.ControlImageFullPath):
            [ControlImageHeight, ControlImageWidth] = nornir_imageregistration.core.GetImageSize(
                self.ControlImageFullPath)
            self.ControlImageDim = [1.0, 1.0, int(ControlImageWidth), int(ControlImageHeight)]
        else:
            if len(self.ControlImageDim) == 2:
                self.ControlImageDim = [1.0, 1.0, int(self.ControlImageDim[0]), int(self.ControlImageDim[1])]

        if self.MappedImageDim is None:
            [MappedImageHeight, MappedImageWidth] = nornir_imageregistration.core.GetImageSize(self.MappedImageFullPath)
            self.MappedImageDim = [1.0, 1.0, MappedImageWidth, MappedImageHeight]
        else:
            if len(self.MappedImageDim) == 2:
                self.MappedImageDim = [1.0, 1.0, int(self.MappedImageDim[0]), int(self.MappedImageDim[1])]

        assert (self.ControlImageDim[2] >= 0)
        assert (self.ControlImageDim[3] >= 0)
        assert (self.MappedImageDim[2] >= 0)
        assert (self.MappedImageDim[3] >= 0)

        ControlDimStr = StosFile.__GetImageDimString(self.ControlImageDim)
        MappedDimStr = StosFile.__GetImageDimString(self.MappedImageDim)

        OutLines.append(ControlDimStr)
        OutLines.append(MappedDimStr)

        # OutLines.append(StosFile.CompressedTransformString(self.Transform))
        OutLines.append(self.Transform)

        if AddMasks and (not (self.ControlMaskName is None or self.MappedMaskName is None)):
            OutLines.append('two_user_supplied_masks:')
            OutLines.append(os.path.join(self.ControlMaskPath, self.ControlMaskName))
            OutLines.append(os.path.join(self.MappedMaskPath, self.MappedMaskName))

        for i, val in enumerate(OutLines):
            if not val[-1] == '\n':
                # print str(val) + '\n'
                OutLines[i] = val + '\n'

        OutFile = open(filename, "w")
        OutFile.writelines(OutLines)
        OutFile.close()

    @staticmethod
    def CompressedTransformString(transform: str) -> str:
        '''Given a list of parts builds a string where numbers are represented by the %g format
           This is no longer used when saving stos files because each transform needs a different level of precision.  However it is useful when computing checksums
        '''
        parts = None
        if isinstance(transform, str):
            parts = transform.split()
        else:
            parts = transform

        outputString = ""
        for part in parts:
            try:
                floatVal = float(part)
                outputString += "%g " % floatVal
            except:
                outputString += part + " "

        outputString.strip()
        outputString += "\n"
        return outputString

    @staticmethod
    def __GetImageDimsArray(ImageFullPath: str):
        '''Return a string compatible with the ITK .stos file image dimension entries'''

        [ImageHeight, ImageWidth] = nornir_imageregistration.core.GetImageSize(ImageFullPath)
        return [1.0, 1.0, ImageWidth, ImageHeight]

    @staticmethod
    def __GetImageDimString(ImageDimArray) -> str:
        ImageDimTemplate = "%(left)g %(bottom)g %(width)d %(height)d"
        DimStr = ImageDimTemplate % {'left': ImageDimArray[0],
                                     'bottom': ImageDimArray[1],
                                     'width': ImageDimArray[2] - (ImageDimArray[0] - 1),
                                     'height': ImageDimArray[3] - (ImageDimArray[1] - 1)}
        return DimStr

    def TryConvertRelativePathsToAbsolutePaths(self, stosDir: str):
        '''
        Converts any relative paths in the StosFile to an absolute path using the stosDir parameter.
        Existing absolute paths are left alone.  Relative paths are unchanged, just prepended with 
        the stosDir parameter 
        '''

        if stosDir is not None and len(stosDir) > 0:
            # Ensure any relative paths to images in the .stos file are relative to the position of the stos file
            if self.ControlImageFullPath is not None:
                if not os.path.isabs(self.ControlImageFullPath):
                    self.ControlImageFullPath = os.path.join(stosDir, self.ControlImageFullPath) 

            if self.MappedImageFullPath is not None:
                if not os.path.isabs(self.MappedImageFullPath):
                    self.MappedImageFullPath = os.path.join(stosDir, self.MappedImageFullPath) 

            if self.ControlMaskFullPath is not None:
                if not os.path.isabs(self.ControlMaskFullPath):
                    self.ControlMaskFullPath = os.path.join(stosDir, self.ControlMaskFullPath) 

            if self.MappedMaskFullPath is not None:
                if not os.path.isabs(self.MappedMaskFullPath):
                    self.MappedMaskFullPath = os.path.join(stosDir, self.MappedMaskFullPath)

    def BlendWithLinear(self, linear_factor: float):
        '''
        Blends a stos file using a control point transform with a rigid linear approximation of
        the same transform (rotation, translation, scaling) with the passed blending factor
        :param linear_factor:  0 to 1.0, amount of weight to assign points passed through linear transform
        :return:
        '''

        transformObj = nornir_imageregistration.transforms.LoadTransform(self.Transform, pixelSpacing=1.0)
        assert (transformObj is not None)

        if isinstance(transformObj, nornir_imageregistration.IControlPoints):
            blended_transform = nornir_imageregistration.transforms.utils.BlendWithLinear(transformObj, linear_factor)
            updated_transform = blended_transform.ToITKString()
            transform_changed = updated_transform != self.Transform
            self.Transform = updated_transform
            return transform_changed

        return False

    def ChangeStosGridPixelSpacing(self, oldspacing, newspacing, ControlImageFullPath,
                                   MappedImageFullPath,
                                   ControlMaskFullPath,
                                   MappedMaskFullPath,
                                   create_copy=True):
        '''
        :param oldspacing:
        :param newspacing:
        :param ControlImageFullPath:
        :param MappedImageFullPath:
        :param ControlMaskFullPath:
        :param MappedMaskFullPath:
        :param bool create_copy: True if a copy of the transform should be scaled, otherwise scales the transform we were called on
        '''
        if oldspacing == newspacing and \
                ControlImageFullPath == self.ControlImageFullPath and \
                MappedImageFullPath == self.MappedImageFullPath and \
                ControlMaskFullPath == self.ControlMaskFullPath and \
                MappedMaskFullPath == self.MappedMaskFullPath:
            if create_copy:
                return copy.deepcopy(self)
            else:
                return self

                # PrettyOutput.Log("ChangeStosGridPixelSpacing from " + str(oldspacing) + " to " + str(newspacing))
        scale = float(oldspacing) / float(newspacing)

        NewStosFile = StosFile()

        # NewStosFile.ControlImageDim = [x * scale for x in self.ControlImageDim]
        NewStosFile.ControlImageDim = copy.copy(self.ControlImageDim)
        NewStosFile.ControlImageDim[2] = self.ControlImageDim[2] * scale
        NewStosFile.ControlImageDim[3] = self.ControlImageDim[3] * scale
        # NewStosFile.MappedImageDim = [x * scale for x in self.MappedImageDim]

        # Update the filenames which are the first two lines of the file
        NewStosFile.MappedImageDim = copy.copy(self.MappedImageDim)
        NewStosFile.MappedImageDim[2] = self.MappedImageDim[2] * scale
        NewStosFile.MappedImageDim[3] = self.MappedImageDim[3] * scale

        NewStosFile.ControlImageFullPath = ControlImageFullPath
        NewStosFile.MappedImageFullPath = MappedImageFullPath
        NewStosFile.ControlMaskFullPath = ControlMaskFullPath
        NewStosFile.MappedMaskFullPath = MappedMaskFullPath

        if os.path.exists(ControlImageFullPath):
            NewStosFile.ControlImageDim = StosFile.__GetImageDimsArray(ControlImageFullPath)

        if os.path.exists(MappedImageFullPath):
            NewStosFile.MappedImageDim = StosFile.__GetImageDimsArray(MappedImageFullPath)

        # Adjust the transform points
        if scale == 1.0:
            NewStosFile.Transform = self.Transform
        else:
            transformObj = nornir_imageregistration.transforms.LoadTransform(self.Transform, pixelSpacing=1.0)
            assert (transformObj is not None)
            
            if isinstance(transformObj, nornir_imageregistration.transforms.ITransformScaling):
                transformObj.Scale(scalar=scale)
            elif transformObj.type == nornir_imageregistration.transforms.TransformType.RIGID:
                transformObj = nornir_imageregistration.transforms.ConvertRigidTransformToCenteredSimilarityTransform(transformObj)
                transformObj.Scale(scalar=scale)
            else:
                raise ValueError(f"Transform needs to be scaled but does not support ITransformScaling interface {transformObj}")

            NewStosFile._Downsample = newspacing

            #if hasattr(transformObj, 'gridWidth'):
                # Save as a stos grid if we can
            #    bounds = (NewStosFile.MappedImageDim[1], NewStosFile.MappedImageDim[0], NewStosFile.MappedImageDim[3],
            #              NewStosFile.MappedImageDim[2])
            #    NewStosFile.Transform = nornir_imageregistration.transforms.TransformToIRToolsString(transformObj,
            #                                                                                         bounds=bounds)
            #else:
            #NewStosFile.Transform = nornir_imageregistration.transforms.TransformToIRToolsString(
            #        transformObj)  # , bounds=NewStosFile.MappedImageDim)

            NewStosFile.Transform = transformObj.ToITKString()

        return NewStosFile

    def EqualizeStosGridPixelSpacing(self, control_spacing, mapped_spacing,
                                     MappedImageFullPath, MappedMaskFullPath,
                                     create_copy=True):
        '''
        Used to correct a mismatch between pixel spacings of the mapped and control images in a stos file.
        This was originally occuring when aligning light microscopy images to TEM images. 
        Nornir expects the spacings for Stos files to be equal.
        
        This function is implemented to keep the control spacing the same and adjust the mapped spacing to match.
        Stos files have no way to encode the spacing in the file itself unfortunately.
        '''

        if control_spacing == mapped_spacing:
            if create_copy:
                return copy.deepcopy(self)
            else:
                return self

        PrettyOutput.Log("ChangeStosGridPixelSpacing from {0:d} to {1:d}".format(mapped_spacing, control_spacing))

        control_spacing = float(control_spacing)
        mapped_spacing = float(mapped_spacing)

        mapped_space_scalar = mapped_spacing / control_spacing

        NewStosFile = StosFile()

        NewStosFile.ControlImageDim = copy.copy(self.ControlImageDim)
        # NewStosFile.MappedImageDim = [x * scale for x in self.MappedImageDim]

        # Update the filenames which are the first two lines of the file
        NewStosFile.MappedImageDim = copy.copy(self.MappedImageDim)
        NewStosFile.MappedImageDim[2] = self.MappedImageDim[2] * mapped_space_scalar
        NewStosFile.MappedImageDim[3] = self.MappedImageDim[3] * mapped_space_scalar

        NewStosFile.ControlImageFullPath = self.ControlImageFullPath
        NewStosFile.ControlMaskFullPath = self.ControlMaskFullPath
        NewStosFile.MappedImageFullPath = MappedImageFullPath
        NewStosFile.MappedMaskFullPath = MappedMaskFullPath

        if os.path.exists(MappedImageFullPath):
            NewStosFile.MappedImageDim = StosFile.__GetImageDimsArray(MappedImageFullPath)

        # Adjust the transform points 
        transformObj = nornir_imageregistration.transforms.LoadTransform(self.Transform, pixelSpacing=1.0)
        assert (transformObj is not None)
        transformObj.ScaleWarped(scalar=mapped_space_scalar)

        NewStosFile._Downsample = control_spacing

        if hasattr(transformObj, 'gridWidth'):
            # Save as a stos grid if we can
            bounds = (NewStosFile.MappedImageDim[1], NewStosFile.MappedImageDim[0], NewStosFile.MappedImageDim[3],
                      NewStosFile.MappedImageDim[2])
            NewStosFile.Transform = nornir_imageregistration.transforms.TransformToIRToolsString(transformObj,
                                                                                                 bounds=bounds)
        else:
            NewStosFile.Transform = nornir_imageregistration.transforms.TransformToIRToolsString(
                transformObj)  # , bounds=NewStosFile.MappedImageDim)

        return NewStosFile

    def StomOutputExists(self, StosPath: str, OutputPath: str, StosMapPath=None):

        '''If we haven't written the stos file itself, return false'''
        stosfullname = os.path.join(StosPath, self.FormattedStosFileName)
        if not os.path.exists(stosfullname):
            return False

        '''Checks whether valid stom output exists for this file.  Returns true if all output files are valid'''
        predictedMappedOutputName = self.OutputMappedImageName

        predictedMappedOutputFullname = os.path.join(OutputPath, predictedMappedOutputName)
        if not os.path.exists(predictedMappedOutputFullname):
            return False

        if nornir_shared.files.RemoveOutdatedFile(stosfullname, predictedMappedOutputFullname):
            return False

        predictedControlOutputName = self.OutputControlImageName
        predictedControlOutputFullname = os.path.join(OutputPath, predictedControlOutputName)
        if not os.path.exists(predictedControlOutputFullname):
            return False

        if nornir_shared.files.RemoveOutdatedFile(stosfullname, predictedControlOutputFullname):
            return False

        if StosMapPath is not None:
            if nornir_shared.files.RemoveOutdatedFile(StosMapPath, predictedControlOutputFullname):
                return False

        return True


def AddStosTransforms(A_To_B,
                      B_To_C,
                      EnrichTolerance: float | None) -> StosFile:
    '''
    :param EnrichTolerance:
    :param A_To_B: Commonly a single section transform, "4->3"
    :param B_To_C: Commonly the transform to the center of a volume, "3->1"
    '''
    A_To_B_Stos = __argumentToStos(A_To_B)
    B_To_C_Stos = __argumentToStos(B_To_C)

    # I'll need to make sure I remember to set the downsample factor when I warp the .mosaic files
    A_To_B_Transform = nornir_imageregistration.transforms.LoadTransform(A_To_B_Stos.Transform)
    B_To_C_Transform = nornir_imageregistration.transforms.LoadTransform(B_To_C_Stos.Transform)

    # OK, I should use a rotation/translation only transform to regularize the added transforms to knock down accumulated warps/errors

    A_To_C_Transform = B_To_C_Transform.AddTransform(A_To_B_Transform, EnrichTolerance, create_copy=False)

    A_To_C_Stos = copy.deepcopy(A_To_B_Stos)
    A_To_C_Stos.ControlSectionNumber = B_To_C_Stos.ControlSectionNumber
    A_To_C_Stos.ControlImageFullPath = B_To_C_Stos.ControlImageFullPath
    A_To_C_Stos.ControlMaskFullPath = B_To_C_Stos.ControlMaskFullPath

    A_To_C_Stos.Transform = nornir_imageregistration.transforms.TransformToIRToolsString(A_To_C_Transform)

    #     if hasattr(A_To_B_Transform, "gridWidth") and hasattr(A_To_B_Transform, "gridHeight"):
    #         A_To_C_Stos.Transform = nornir_imageregistration.transforms.TransformToIRToolsGridString(A_To_C_Transform, A_To_B_Transform.gridWidth, A_To_B_Transform.gridHeight)
    #     else:
    #         A_To_C_Stos.Transform = nornir_imageregistration.transforms.TransformToIRToolsString(A_To_C_Transform)

    A_To_C_Stos.ControlImageDim = B_To_C_Stos.ControlImageDim
    A_To_C_Stos.MappedImageDim = A_To_B_Stos.MappedImageDim

    return A_To_C_Stos
