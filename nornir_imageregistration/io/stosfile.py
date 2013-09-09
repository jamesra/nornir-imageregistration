import os
import sys
import nornir_shared.files
import nornir_shared.checksum
import nornir_shared.images
import nornir_shared.prettyoutput as PrettyOutput
import mosaicfile
import copy
import logging
from nornir_imageregistration.transforms import factory

StosNameTemplate = "%(mappedsection)04u-%(controlsection)04u_%(channels)s_%(mosaicfilters)s_%(stostype)s_%(downsample)u.stos"


def __argumentToStos(Argument):

    stosObj = None
    if isinstance(Argument, str):
        stosObj = StosFile.Load(Argument)
    elif isinstance(Argument, StosFile):
        stosObj = Argument

    assert(not stosObj is None)

    return stosObj

def AddStosTransforms(A_To_B, B_To_C):

    A_To_B_Stos = __argumentToStos(A_To_B)
    B_To_C_Stos = __argumentToStos(B_To_C)

    # I'll need to make sure I remember to set the downsample factor when I warp the .mosaic files
    A_To_B_Transform = factory.LoadTransform(A_To_B_Stos.Transform)
    B_To_C_Transform = factory.LoadTransform(B_To_C_Stos.Transform)

    A_To_C_Transform = B_To_C_Transform.AddTransform(A_To_B_Transform)

    A_To_C_Stos = copy.deepcopy(A_To_B_Stos)
    A_To_C_Stos.ControlImageFullPath = B_To_C_Stos.ControlImageFullPath
    A_To_C_Stos.Transform = factory.TransformToIRToolsGridString(A_To_C_Transform, A_To_B_Transform.gridWidth, A_To_B_Transform.gridHeight)
    A_To_C_Stos.ControlImageDim = B_To_C_Stos.ControlImageDim
    A_To_C_Stos.MappedImageDim = A_To_B_Stos.MappedImageDim

    return A_To_C_Stos

class StosFile:
    """description of class"""

    @classmethod
    def LoadChecksum(cls, path):
        assert(os.path.exists(path))
        stosObj = StosFile.Load(path)
        return stosObj.Checksum

    @property
    def Downsample(self):
        return self._Downsample

    @Downsample.setter
    def Downsample(self, newDownsample):
        if(self._Downsample is None):  # Don't scale if
            self._Downsample = newDownsample
        else:
            scalar = self._Downsample / newDownsample
            self.Scale(scalar)
            self._Downsample = newDownsample

    @property
    def ControlImageFullPath(self):
        return os.path.join(self.ControlImagePath, self.ControlImageName)

    @ControlImageFullPath.setter
    def ControlImageFullPath(self, val):

        d = os.path.dirname(val)
        f = os.path.basename(val)

        self.ControlImagePath = d.strip()
        self.ControlImageName = f.strip()

    @property
    def MappedImageFullPath(self):
        return os.path.join(self.MappedImagePath, self.MappedImageName)

    @MappedImageFullPath.setter
    def MappedImageFullPath(self, val):

        d = os.path.dirname(val)
        f = os.path.basename(val)
        self.MappedImagePath = d.strip()
        self.MappedImageName = f.strip()

    @property
    def ControlMaskFullPath(self):
        return os.path.join(self.ControlMaskPath, self.ControlMaskName)

    @ControlMaskFullPath.setter
    def ControlMaskFullPath(self, val):

        d = os.path.dirname(val)
        f = os.path.basename(val)
        self.ControlMaskPath = d.strip()
        self.ControlMaskName = f.strip()

    @property
    def MappedMaskFullPath(self):
        return os.path.join(self.MappedMaskPath, self.MappedMaskName)

    @MappedMaskFullPath.setter
    def MappedMaskFullPath(self, val):

        d = os.path.dirname(val)
        f = os.path.basename(val)

        self.MappedMaskPath = d.strip()
        self.MappedMaskName = f.strip()

    @property
    def Checksum(self):
        if self.Transform is None:
            return ""

        compressedString = StosFile.CompressedTransformString(self.Transform)
        return nornir_shared.checksum.DataChecksum(compressedString)

#   NewImageNameTemplate = ("%(section)" + IrUtil.SectionFormat + "_%(channel)_%(type)_" + str(newspacing) + ".png\n")
#   controlNewImageName = NewImageNameTemplate % {'section' : ControlSectionNumber}

    def __init__(self):

#        self.ControlImageName = property(self.__get__ControlImageName,
#                                         None,
#                                         None,
#                                         'Filename of Control Image')
#
#        self.MappedImageName = property(self.__get__MappedImageName,
#                                         None,
#                                         None,
#                                         'Filename of Mapped Image')
#
#        self.ControlMaskName = property(self.__get__ControlMaskName,
#                                         None,
#                                         None,
#                                         'Filename of Control Image Mask')
#
#        self.MappedMaskName = property(self.__get__MappedMaskName,
#                                         None,
#                                         None,
#                                         'Filename of Control Image Mask')

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

        self.Transform = None

        self.StosSource = None

        self.ImageToTransform = dict()
        return

    @classmethod
    def GetInfo(cls, filename):
        '''Returns details about a stos file we can learn from its name
           returns  [mappedSection, controlSection, Channel, Filter, Source, Downsample]'''

        Logger = logging.getLogger('stos')

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

        return [mappedSection, controlSection, Channel, Filter, Source, Downsample]

    @classmethod
    def Create(cls, controlImageFullPath, mappedImageFullPath, Transform, controlMaskFullPath=None, mappedMaskFullPath=None):
        stosObj = StosFile()
        stosObj.ControlImageFullPath = controlImageFullPath
        stosObj.MappedImageFullPath = mappedImageFullPath
        stosObj.Transform = factory.TransformToIRToolsString(Transform)

        if not controlMaskFullPath is None:
            stosObj.ControlMaskFullPath = controlMaskFullPath
            stosObj.MappedMaskFullPath = mappedMaskFullPath

        return stosObj

    @classmethod
    def Load(cls, filename):
        if not os.path.exists(filename):
            PrettyOutput.Log("Mosaic file not found: " + filename)
            return

        obj = StosFile()

        try:
            [obj.MappedSectionNumber, obj.ControlSectionNumber, Channels, Filters, obj.StosSource, obj._Downsample] = StosFile.GetInfo(filename)
        except:
            pass

        fMosaic = open(filename, 'r')
        lines = fMosaic.readlines()
        fMosaic.close()

        obj.ControlImagePath = os.path.dirname(lines[0].strip())
        obj.MappedImagePath = os.path.dirname(lines[1].strip())

        obj.ControlImageName = os.path.basename(lines[0].strip())
        obj.MappedImageName = os.path.basename(lines[1].strip())

        ControlDims = lines[4].split()
        MappedDims = lines[5].split()

        obj.ControlImageDim = [float(x) for x in ControlDims]
        obj.MappedImageDim = [float(x) for x in MappedDims]

        obj.Transform = lines[6].strip()

        if len(lines) > 8:
            obj.ControlMaskPath = lines[7]
            obj.MappedMaskPath = lines[8]

        return obj

    def Scale(self, scalar):
        '''Scale this stos transform by the requested amount'''

        # Adjust the mosaic and mask names if present
        self.ControlImageDim = [x * scalar for x in self.ControlImageDim]
        self.MappedImageDim = [x * scalar for x in self.MappedImageDim]

        # Adjust the grid points
        transformObj = factory.LoadTransform(self.Transform, pixelSpacing=1)
        transformObj.Scale(scalar=scalar)

        if hasattr(transformObj, 'gridWidth'):
            # Save as a stos grid if we can
            self.Transform = factory.TransformToIRToolsGridString(transformObj, transformObj.gridWidth, transformObj.gridHeight, bounds=self.MappedImageDim)
        else:
            self.Transform = factory.TransformToIRToolsString(transformObj, bounds=self.MappedImageDim)

        self._Downsample = self._Downsample * scalar

    def Save(self, filename, AddMasks=True):
        # This function needs reworking to use different object variables'
        # assert(False)
        OutLines = list()

        # mosaic files to be warped
        OutLines.append(self.ControlImageFullPath)
        OutLines.append(self.MappedImageFullPath)

        # Write the header
        OutLines.append("0")
        OutLines.append("0")

        if self.ControlImageDim is None:
            [ControlImageWidth, ControlImageHeight] = nornir_shared.images.GetImageSize(self.ControlImageFullPath)
            self.ControlImageDim = [1.0, 1.0, int(ControlImageWidth), int(ControlImageHeight)]

        if len(self.ControlImageDim) == 2:
            [ControlImageWidth, ControlImageHeight] = nornir_shared.images.GetImageSize(self.ControlImageFullPath)
            self.ControlImageDim = [self.ControlImageDim[0], self.ControlImageDim[1], int(ControlImageWidth), int(ControlImageHeight)]

        if self.MappedImageDim is None:
            [MappedImageWidth, MappedImageHeight] = nornir_shared.images.GetImageSize(self.MappedImageFullPath)
            self.MappedImageDim = [1.0, 1.0, (MappedImageWidth), (MappedImageHeight)]

        if len(self.MappedImageDim) == 2:
            [MappedImageWidth, MappedImageHeight] = nornir_shared.images.GetImageSize(self.MappedImageFullPath)
            self.MappedImageDim = [self.MappedImageDim[0], self.MappedImageDim[1], (MappedImageWidth), (MappedImageHeight)]

        ControlDimStr = StosFile.__GetImageDimString(self.ControlImageDim)
        MappedDimStr = StosFile.__GetImageDimString(self.MappedImageDim)

        OutLines.append(ControlDimStr)
        OutLines.append(MappedDimStr)

        OutLines.append(StosFile.CompressedTransformString(self.Transform))

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


    @classmethod
    def CompressedTransformString(cls, transform):
        '''Given a list of parts builds a string where numbers are represented by the %g format'''
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

    @classmethod
    def __GetImageDimsArray(cls, ImageFullPath):
        '''Return a string compatible with the ITK .stos file image dimension entries'''

        [ImageWidth, ImageHeight] = nornir_shared.images.GetImageSize(ImageFullPath)
        return [1.0, 1.0, ImageWidth, ImageHeight]

    @classmethod
    def __GetImageDimString(cls, ImageDimArray):
        ImageDimTemplate = "%(left)g %(bottom)g %(width)d %(height)d"
        DimStr = ImageDimTemplate % {'left' : ImageDimArray[0],
                                            'bottom' : ImageDimArray[1],
                                            'width' : ImageDimArray[2] - (ImageDimArray[0] - 1),
                                            'height' : ImageDimArray[3] - (ImageDimArray[1] - 1)}
        return DimStr


    def ChangeStosGridPixelSpacing(self, oldspacing, newspacing, ControlImageFullPath=None,
                                           MappedImageFullPath=None,
                                           ControlMaskFullPath=None,
                                           MappedMaskFullPath=None):
        PrettyOutput.Log("ChangeStosGridPixelSpacing from " + str(oldspacing) + " to " + str(newspacing))
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

        if(not ControlImageFullPath is None):
            NewStosFile.ControlImagePath = os.path.dirname(ControlImageFullPath)
            NewStosFile.ControlImageName = os.path.basename(ControlImageFullPath)
            NewStosFile.ControlImageDim = StosFile.__GetImageDimsArray(ControlImageFullPath)

        if not MappedImageFullPath is None:
            NewStosFile.MappedImagePath = os.path.dirname(MappedImageFullPath)
            NewStosFile.MappedImageName = os.path.basename(MappedImageFullPath)
            NewStosFile.MappedImageDim = StosFile.__GetImageDimsArray(MappedImageFullPath)

        if(not ControlMaskFullPath is None):
            NewStosFile.ControlMaskPath = os.path.dirname(ControlMaskFullPath)
            NewStosFile.ControlMaskName = os.path.basename(ControlMaskFullPath)

        if(not MappedMaskFullPath is None):
            NewStosFile.MappedMaskPath = os.path.dirname(MappedMaskFullPath)
            NewStosFile.MappedMaskName = os.path.basename(MappedMaskFullPath)

        # Adjust the grid points

        if scale == 1.0:
            NewStosFile.Transform = self.Transform
        else:
            transformObj = factory.LoadTransform(self.Transform, pixelSpacing=1.0)
            assert(not transformObj is None)
            transformObj.Scale(scalar=scale)

            NewStosFile._Downsample = newspacing

            if hasattr(transformObj, 'gridWidth'):
                # Save as a stos grid if we can
                NewStosFile.Transform = factory.TransformToIRToolsGridString(transformObj, transformObj.gridWidth, transformObj.gridHeight, bounds=NewStosFile.MappedImageDim)
            else:
                NewStosFile.Transform = factory.TransformToIRToolsString(transformObj)  # , bounds=NewStosFile.MappedImageDim)

        return NewStosFile


    def StomOutputExists(self, StosPath, OutputPath, StosMapPath=None):

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

        if(StosMapPath is not None):
            if nornir_shared.files.RemoveOutdatedFile(StosMapPath, predictedControlOutputFullname):
                return False


        return True
