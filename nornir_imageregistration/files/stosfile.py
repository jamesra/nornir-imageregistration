import copy
import logging
import os
import sys

import nornir_imageregistration
from nornir_imageregistration.transforms import factory

import nornir_shared.checksum
import nornir_shared.files
import nornir_shared.prettyoutput as PrettyOutput


def __argumentToStos(Argument):

    stosObj = None
    if isinstance(Argument, str):
        stosObj = StosFile.Load(Argument)
    elif isinstance(Argument, StosFile):
        stosObj = Argument

    assert(not stosObj is None)

    return stosObj

def AddStosTransforms(A_To_B, B_To_C, EnrichTolerance):

    A_To_B_Stos = __argumentToStos(A_To_B)
    B_To_C_Stos = __argumentToStos(B_To_C)

    # I'll need to make sure I remember to set the downsample factor when I warp the .mosaic files
    A_To_B_Transform = factory.LoadTransform(A_To_B_Stos.Transform)
    B_To_C_Transform = factory.LoadTransform(B_To_C_Stos.Transform)
    
    # OK, I should use a rotation/translation only transform to regularize the added transforms to knock down accumulated warps/errors
    
    
    A_To_C_Transform = B_To_C_Transform.AddTransform(A_To_B_Transform, EnrichTolerance, create_copy=False)

    A_To_C_Stos = copy.deepcopy(A_To_B_Stos)
    A_To_C_Stos.ControlImageFullPath = B_To_C_Stos.ControlImageFullPath
    A_To_C_Stos.ControlMaskFullPath = B_To_C_Stos.ControlMaskFullPath

    A_To_C_Stos.Transform = factory.TransformToIRToolsString(A_To_C_Transform)

#     if hasattr(A_To_B_Transform, "gridWidth") and hasattr(A_To_B_Transform, "gridHeight"):
#         A_To_C_Stos.Transform = factory.TransformToIRToolsGridString(A_To_C_Transform, A_To_B_Transform.gridWidth, A_To_B_Transform.gridHeight)
#     else:
#         A_To_C_Stos.Transform = factory.TransformToIRToolsString(A_To_C_Transform)

    A_To_C_Stos.ControlImageDim = B_To_C_Stos.ControlImageDim
    A_To_C_Stos.MappedImageDim = A_To_B_Stos.MappedImageDim 

    return A_To_C_Stos

class StosFile(object):
    """description of class"""
    
    @classmethod
    def FileHasMasks(cls, path):
        stosObj = StosFile.Load(path)
        return stosObj.HasMasks

    @classmethod
    def LoadChecksum(cls, path):
        assert(os.path.exists(path))
        stosObj = StosFile.Load(path)
        return stosObj.Checksum
    
    @property
    def Transform(self):
        return self._Transform
    
    @Transform.setter
    def Transform(self, val):
        if val is None:
            self._Transform = None
            return 
        
        if isinstance(val, nornir_imageregistration.transforms.base.Base):
            self._Transform = nornir_imageregistration.transforms.factory.TransformToIRToolsString(val)
        elif isinstance(val, str):
            self._Transform = val
        else:
            raise TypeError("Transform must be a transform object or a ITK transform string")
        
        return

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
        
        if val is None:
            self.ControlImagePath = None
            self.ControlImageName = None
        else:        
            d = os.path.dirname(val)
            f = os.path.basename(val)
    
            self.ControlImagePath = d.strip()
            self.ControlImageName = f.strip()

    @property
    def MappedImageFullPath(self):
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
    def ControlMaskFullPath(self):
        if self.ControlMaskPath is None or self.ControlMaskName is None:
            return None
        
        return os.path.join(self.ControlMaskPath, self.ControlMaskName)

    @ControlMaskFullPath.setter
    def ControlMaskFullPath(self, val):
        if val is None:
            self.ControlMaskPath = None
            self.ControlMaskName = None 
            return 
        
        d = os.path.dirname(val)
        f = os.path.basename(val)
        self.ControlMaskPath = d.strip()
        self.ControlMaskName = f.strip()

    @property
    def MappedMaskFullPath(self):
        if self.MappedMaskPath is None or self.MappedMaskName is None:
            return None
        
        return os.path.join(self.MappedMaskPath, self.MappedMaskName)

    @MappedMaskFullPath.setter
    def MappedMaskFullPath(self, val):
        if val is None:
            self.MappedMaskPath = None
            self.MappedMaskName = None 
            return 
        
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
    
    @property
    def HasMasks(self):
        return not (self.MappedMaskName is None or self.ControlMaskName is None)
    
    def ClearMasks(self):
        '''Remove masks from the file'''
        self.MappedMaskFullPath = None
        self.ControlMaskFullPath = None
        return 

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

        self.StosSource = None

        self.ImageToTransform = dict()
        return

    @classmethod
    def GetInfo(cls, filename):
        '''Returns details about a stos file we can learn from its name
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
            PrettyOutput.Log("Stos file not found: " + filename)
            return

        obj = StosFile()

        try:
            [obj.MappedSectionNumber, obj.ControlSectionNumber, Channels, Filters, obj.StosSource, obj._Downsample] = StosFile.GetInfo(filename)
        except:
            pass

        with open(filename, 'r') as fMosaic:
            lines = fMosaic.readlines()
            
        if len(lines) < 7:
            PrettyOutput.LogErr("%s is not a valid stos file" % (filename))
            raise ValueError("%s is not a valid stos file" % (filename))
            

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

    @classmethod
    def IsValid(cls, filename):
        '''#If stos-grid completely fails it uses the maximum float value for each data point.  This function loads the transform and ensures it is valid'''

        if not os.path.exists(filename):
            return False

        stos = StosFile.Load(filename)

        try:
            Transform = factory.LoadTransform(stos.Transform, pixelSpacing=1)
        except:
            return False

        return True


    def Scale(self, scalar):
        '''Scale this stos transform by the requested amount'''

        # Adjust the mosaic and mask names if present
        self.ControlImageDim = [x * scalar for x in self.ControlImageDim]
        self.MappedImageDim = [x * scalar for x in self.MappedImageDim]

        # Adjust the grid points
        transformObj = factory.LoadTransform(self.Transform, pixelSpacing=1)
        transformObj.Scale(scalar=scalar)

#         if hasattr(transformObj, 'gridWidth'):
#             # Save as a stos grid if we can
#             self.Transform = factory.TransformToIRToolsGridString(transformObj, transformObj.gridWidth, transformObj.gridHeight, bounds=self.MappedImageDim)
#         else:
#             self.Transform = factory.TransformToIRToolsString(transformObj, bounds=self.MappedImageDim)
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

       
        if os.path.exists(self.ControlImageFullPath):
            [ControlImageHeight, ControlImageWidth] = nornir_imageregistration.core.GetImageSize(self.ControlImageFullPath)
            self.ControlImageDim = [1.0, 1.0, int(ControlImageWidth), int(ControlImageHeight)]
        else:
            if len(self.ControlImageDim) == 2:
                self.ControlImageDim = [1.0, 1.0, int(self.ControlImageDim[0]), int(self.ControlImageDim[1])]

        if self.MappedImageDim is None:
            [MappedImageHeight, MappedImageWidth] = nornir_imageregistration.core.GetImageSize(self.MappedImageFullPath)
            self.MappedImageDim = [1.0, 1.0, (MappedImageWidth), (MappedImageHeight)]
        else:
            if len(self.MappedImageDim) == 2:
                self.MappedImageDim = [1.0, 1.0, int(self.MappedImageDim[0]), int(self.MappedImageDim[1])]
 
        assert(self.ControlImageDim[2] >= 0)
        assert(self.ControlImageDim[3] >= 0)
        assert(self.MappedImageDim[2] >= 0)
        assert(self.MappedImageDim[3] >= 0) 
        
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

        [ImageHeight, ImageWidth] = nornir_imageregistration.core.GetImageSize(ImageFullPath)
        return [1.0, 1.0, ImageWidth, ImageHeight]

    @classmethod
    def __GetImageDimString(cls, ImageDimArray):
        ImageDimTemplate = "%(left)g %(bottom)g %(width)d %(height)d"
        DimStr = ImageDimTemplate % {'left' : ImageDimArray[0],
                                            'bottom' : ImageDimArray[1],
                                            'width' : ImageDimArray[2] - (ImageDimArray[0] - 1),
                                            'height' : ImageDimArray[3] - (ImageDimArray[1] - 1)}
        return DimStr


    def ChangeStosGridPixelSpacing(self, oldspacing, newspacing, ControlImageFullPath,
                                           MappedImageFullPath,
                                           ControlMaskFullPath,
                                           MappedMaskFullPath,
                                           create_copy=True):
        '''
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
            transformObj = factory.LoadTransform(self.Transform, pixelSpacing=1.0)
            assert(not transformObj is None)
            transformObj.Scale(scalar=scale)

            NewStosFile._Downsample = newspacing

            if hasattr(transformObj, 'gridWidth'):
                # Save as a stos grid if we can
                bounds = (NewStosFile.MappedImageDim[1], NewStosFile.MappedImageDim[0], NewStosFile.MappedImageDim[3], NewStosFile.MappedImageDim[2])
                NewStosFile.Transform = factory.TransformToIRToolsString(transformObj, bounds=bounds)
            else:
                NewStosFile.Transform = factory.TransformToIRToolsString(transformObj)  # , bounds=NewStosFile.MappedImageDim)

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
        transformObj = factory.LoadTransform(self.Transform, pixelSpacing=1.0)
        assert(not transformObj is None)
        transformObj.ScaleWarped(scalar=mapped_space_scalar)

        NewStosFile._Downsample = control_spacing

        if hasattr(transformObj, 'gridWidth'):
            # Save as a stos grid if we can
            bounds = (NewStosFile.MappedImageDim[1], NewStosFile.MappedImageDim[0], NewStosFile.MappedImageDim[3], NewStosFile.MappedImageDim[2])
            NewStosFile.Transform = factory.TransformToIRToolsString(transformObj, bounds=bounds)
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
