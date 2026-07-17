from __future__ import annotations

import copy
import logging
import os

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import ITransform
from nornir_imageregistration.transforms.base import ITransformScaling, ITransformRelativeScaling
import nornir_shared.checksum
import nornir_shared.files
import nornir_shared.prettyoutput as PrettyOutput

_logger = logging.getLogger(__name__)


def _normalize_stos_path(path: str) -> str:
    """Normalize a path for on-disk STOS files using forward slashes."""
    return os.path.normpath(path).replace(os.sep, '/')


def _can_express_relative(full_path: str, stos_dir: str) -> bool:
    """Return True if full_path can be written relative to stos_dir."""
    if not full_path or not stos_dir:
        return False
    try:
        norm_full = os.path.normpath(full_path)
        norm_stos = os.path.normpath(stos_dir)
        os.path.commonpath([norm_full, norm_stos])
        os.path.relpath(norm_full, norm_stos)
        return True
    except ValueError:
        return False


def _path_for_stos_file(full_path: str, stos_dir: str) -> str:
    """Return a relative path when possible, otherwise a normalized absolute path."""
    if _can_express_relative(full_path, stos_dir):
        relative = os.path.relpath(os.path.normpath(full_path), os.path.normpath(stos_dir))
        return _normalize_stos_path(relative)
    absolute = _normalize_stos_path(os.path.abspath(full_path))
    _logger.warning(
        "STOS path cannot be expressed relative to %s; writing absolute: %s",
        stos_dir,
        absolute,
    )
    return absolute


def _path_from_stos_file(stored_path: str, stos_dir: str) -> str:
    """Resolve a stored STOS path (relative or absolute) to an absolute path."""
    stored_path = stored_path.strip()
    if os.path.isabs(stored_path):
        return os.path.normpath(stored_path)
    return os.path.normpath(os.path.join(stos_dir, stored_path))


def __argumentToStos(Argument):
    stosObj = None
    if isinstance(Argument, str):
        if not os.path.exists(Argument):
            raise FileNotFoundError(Argument)
        stosObj = StosFile.Load(Argument)
    elif isinstance(Argument, StosFile):
        stosObj = Argument

    assert (stosObj is not None)

    return stosObj


class StosFile(object):
    """Represents a slice-to-slice transform file: control/mapped image paths, transform string, and I/O."""

    ControlImagePath: str | None = None
    ControlImageName: str | None = None
    MappedImagePath: str | None = None
    MappedImageName: str | None = None
    _Transform: str | None = None
    _Downsample: float | int | None = None

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
            raise TypeError("transform must be a transform object or a ITK transform string")

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
    def ControlImageFullPath(self) -> str:
        if self.ControlImagePath is None:
            raise ValueError("ControlImagePath is not set")
        if self.ControlImageName is None:
            raise ValueError("ControlImageName is not set")

        return os.path.join(self.ControlImagePath, self.ControlImageName)

    @ControlImageFullPath.setter
    def ControlImageFullPath(self, val: str | None):

        if val is None:
            self.ControlImagePath = None
            self.ControlImageName = None
        else:
            d = os.path.dirname(val)
            f = os.path.basename(val)

            self.ControlImagePath = d.strip()
            self.ControlImageName = f.strip()

    @property
    def MappedImageFullPath(self) -> str:
        if self.MappedImagePath is None:
            raise ValueError("MappedImagePath is not set")
        if self.MappedImageName is None:
            raise ValueError("MappedImageName is not set")

        return os.path.join(self.MappedImagePath, self.MappedImageName)

    @MappedImageFullPath.setter
    def MappedImageFullPath(self, val: str | None):

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
        return nornir_shared.checksum.DataChecksum(compressedString)  # type: ignore[return-value]

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

        self.TargetSectionNumber = None
        self.SourceSectionNumber = None

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
        return f'{self.TargetSectionNumber}<-{self.SourceSectionNumber} DS:{self._Downsample}'

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
    def Create(target_image_fullpath: str,
               source_image_fullpath: str,
               transform: ITransform,
               target_mask_fullpath: str | None = None,
               source_mask_fullpath: str | None = None) -> StosFile:
        stosObj = StosFile()
        stosObj.ControlImageFullPath = target_image_fullpath
        stosObj.MappedImageFullPath = source_image_fullpath
        stosObj.Transform = nornir_imageregistration.transforms.TransformToIRToolsString(transform)

        if target_mask_fullpath is not None:
            stosObj.ControlMaskFullPath = target_mask_fullpath
            stosObj.MappedMaskFullPath = source_mask_fullpath

        return stosObj

    @staticmethod
    def Load(filename: str, resolve_paths: bool = True) -> StosFile:
        """Load a STOS file from disk, optionally resolving relative image paths."""
        obj = StosFile()

        try:
            [obj.SourceSectionNumber, obj.TargetSectionNumber, Channels, Filters, obj.StosSource,
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

        stos_dir = os.path.dirname(os.path.abspath(filename))

        def _resolve_stored_path(stored: str) -> str:
            stored = stored.strip()
            if resolve_paths:
                return _path_from_stos_file(stored, stos_dir)
            return stored

        obj.ControlImageFullPath = _resolve_stored_path(lines[0])
        obj.MappedImageFullPath = _resolve_stored_path(lines[1])

        ControlDims = lines[4].split()
        MappedDims = lines[5].split()

        obj.ControlImageDim = [float(x) for x in ControlDims]
        obj.MappedImageDim = [float(x) for x in MappedDims]

        obj.Transform = lines[6].strip()

        if len(lines) > 9 and lines[7].strip() == 'two_user_supplied_masks:':
            obj.ControlMaskFullPath = _resolve_stored_path(lines[8])
            obj.MappedMaskFullPath = _resolve_stored_path(lines[9])
        elif len(lines) > 8:
            obj.ControlMaskFullPath = _resolve_stored_path(lines[8])
            if len(lines) > 9:
                obj.MappedMaskFullPath = _resolve_stored_path(lines[9])

        return obj

    @staticmethod
    def IsValid(filename) -> bool:
        '''#If stos-grid completely fails it uses the maximum float value for each data point.  This function loads the transform and ensures it is valid'''

        try:
            stos = StosFile.Load(filename)
            Transform = nornir_imageregistration.transforms.LoadTransform(stos.Transform, pixelSpacing=1)  # type: ignore[arg-type]
        except FileNotFoundError:
            return False
        except:
            return False

        return True

    def Scale(self, scalar: float):
        '''Scale this stos transform by the requested amount'''

        # Adjust the mosaic and mask names if present
        self.ControlImageDim = [x * scalar for x in self.ControlImageDim]  # type: ignore[union-attr]
        self.MappedImageDim = [x * scalar for x in self.MappedImageDim]  # type: ignore[union-attr]

        # Adjust the grid points
        transformObj = nornir_imageregistration.transforms.LoadTransform(self.Transform, pixelSpacing=1)  # type: ignore[arg-type]
        if isinstance(transformObj, ITransformScaling):
            transformObj.Scale(scalar=scalar)

        #         if hasattr(transformObj, 'gridWidth'):
        #             # Save as a stos grid if we can
        #             self.transform = nornir_imageregistration.transforms.TransformToIRToolsGridString(transformObj, transformObj.gridWidth, transformObj.gridHeight, bounds=self.MappedImageDim)
        #         else:
        #             self.transform = nornir_imageregistration.transforms.TransformToIRToolsString(transformObj, bounds=self.MappedImageDim)
        self.Transform = nornir_imageregistration.transforms.TransformToIRToolsString(transformObj,
                                                                                      bounds=self.MappedImageDim)

        self._Downsample *= scalar  # type: ignore[operator]

    def Save(self, filename: str, AddMasks: bool = True, relative_paths: bool = True):
        """Write this STOS file to disk, preferring relative image paths when expressible."""
        OutLines = list()
        stos_dir = os.path.dirname(os.path.abspath(filename))

        def _stored_path(full_path: str) -> str:
            if relative_paths:
                return _path_for_stos_file(full_path, stos_dir)
            return _normalize_stos_path(os.path.abspath(full_path))

        # mosaic files to be warped
        OutLines.append(_stored_path(self.ControlImageFullPath))
        OutLines.append(_stored_path(self.MappedImageFullPath))

        # Write the header
        OutLines.append("0")
        OutLines.append("0")

        if self.ControlImageDim is not None:
            if len(self.ControlImageDim) == 2:
                self.ControlImageDim = [1.0, 1.0, int(self.ControlImageDim[0]), int(self.ControlImageDim[1])]
            elif len(self.ControlImageDim) == 4:
                pass
            else:
                raise ValueError("Unexpected number of dimensions of Control Image")
        elif os.path.exists(self.ControlImageFullPath):
            [ControlImageHeight, ControlImageWidth] = nornir_imageregistration.core.GetImageSize(
                self.ControlImageFullPath)
            self.ControlImageDim = [1.0, 1.0, int(ControlImageWidth), int(ControlImageHeight)]
        else:
            raise ValueError("Control Image not found and Control Image Dim is None")

        if self.MappedImageDim is None:
            [MappedImageHeight, MappedImageWidth] = nornir_imageregistration.core.GetImageSize(self.MappedImageFullPath)
            self.MappedImageDim = [1.0, 1.0, MappedImageWidth, MappedImageHeight]
        elif self.MappedImageDim is not None:
            if len(self.MappedImageDim) == 2:
                self.MappedImageDim = [1.0, 1.0, int(self.MappedImageDim[0]), int(self.MappedImageDim[1])]
            elif len(self.MappedImageDim) == 4:
                pass
            else:
                raise ValueError("Unexpected number of dimensions of Mapped Image")
        else:
            raise ValueError("Mapped Image not found and Mapped Image Dim is None")

        assert (self.ControlImageDim[2] >= 0)
        assert (self.ControlImageDim[3] >= 0)
        assert (self.MappedImageDim[2] >= 0)
        assert (self.MappedImageDim[3] >= 0)

        ControlDimStr = StosFile.__GetImageDimString(self.ControlImageDim)
        MappedDimStr = StosFile.__GetImageDimString(self.MappedImageDim)

        OutLines.append(ControlDimStr)
        OutLines.append(MappedDimStr)

        # OutLines.append(StosFile.CompressedTransformString(self.transform))
        OutLines.append(str(self.Transform))

        if AddMasks and (not (self.ControlMaskName is None or self.MappedMaskName is None)):
            OutLines.append('two_user_supplied_masks:')
            OutLines.append(_stored_path(self.ControlMaskFullPath))  # type: ignore[arg-type]
            OutLines.append(_stored_path(self.MappedMaskFullPath))  # type: ignore[arg-type]

        for i, val in enumerate(OutLines):
            if not val[-1] == '\n':
                # print str(val) + '\n'
                OutLines[i] = val + '\n'

        stos_parent = os.path.dirname(filename)
        if stos_parent:
            os.makedirs(stos_parent, exist_ok=True)
        with open(filename, "w") as OutFile:
            OutFile.writelines(OutLines)

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

    def ConvertPathsToAbsolute(self, stos_dir: str) -> None:
        """Resolve relative image/mask paths against the directory containing the STOS file."""
        if not stos_dir:
            return

        for attr in (
                'ControlImageFullPath',
                'MappedImageFullPath',
                'ControlMaskFullPath',
                'MappedMaskFullPath',
        ):
            try:
                full_path = getattr(self, attr)
            except ValueError:
                continue
            if full_path is None:
                continue
            if os.path.isabs(full_path):
                setattr(self, attr, os.path.normpath(full_path))
            else:
                setattr(self, attr, _path_from_stos_file(full_path, stos_dir))

    def ConvertPathsToRelative(self, stos_dir: str) -> None:
        """Convert image/mask paths to relative form when expressible; leave absolute otherwise."""
        if not stos_dir:
            return

        for attr in (
                'ControlImageFullPath',
                'MappedImageFullPath',
                'ControlMaskFullPath',
                'MappedMaskFullPath',
        ):
            try:
                full_path = getattr(self, attr)
            except ValueError:
                continue
            if full_path is None:
                continue
            if _can_express_relative(full_path, stos_dir):
                relative = _normalize_stos_path(
                    os.path.relpath(os.path.normpath(full_path), os.path.normpath(stos_dir)))
                setattr(self, attr, relative)

    def TryConvertRelativePathsToAbsolutePaths(self, stosDir: str) -> None:
        """Deprecated alias for :meth:`ConvertPathsToAbsolute`."""
        self.ConvertPathsToAbsolute(stosDir)

    def BlendWithLinear(self, linear_factor: float | None = None,
                        travel_limit: float | None = None,
                        ignore_rotation: bool = False,
                        reblend_iterations: int = 1,
                        reblend_tolerance: float | None = None,
                        ):
        '''
        Blends a stos file using a control point transform with a rigid linear approximation of
        the same transform (rotation, translation, scaling) with the passed blending factor
        :param linear_factor:  0 to 1.0, amount of weight to assign points passed through linear transform
        :param ignore_rotation: This was added for SEM data which is known to not have rotation between slices.  Defaults to false.
        :param travel_limit: Per-point distance scale for smooth blend toward rigid prediction.
        :param reblend_iterations: Iterative blend passes; values above 1 re-blend until convergence.
        :param reblend_tolerance: Stop iterating when max point movement falls below this threshold.
        :return:
        '''

        transformObj = nornir_imageregistration.transforms.LoadTransform(self.Transform, pixelSpacing=1.0)  # type: ignore[arg-type]
        assert (transformObj is not None)

        if isinstance(transformObj, nornir_imageregistration.IControlPoints):
            blend_kwargs: dict = {
                'linear_factor': linear_factor,
                'travel_limit': travel_limit,
                'ignore_rotation': ignore_rotation,
                'reblend_iterations': reblend_iterations,
            }
            if reblend_tolerance is not None:
                blend_kwargs['reblend_tolerance'] = reblend_tolerance
            blended_transform = nornir_imageregistration.transforms.utils.BlendWithLinear(transformObj, **blend_kwargs)
            updated_transform = blended_transform.ToITKString()
            transform_changed = updated_transform != self.Transform
            self.Transform = updated_transform
            return transform_changed

        return False

    def ChangeTransformPixelSpacing(self, oldspacing: int,
                                    newspacing: int,
                                    ControlImageFullPath: str,
                                    MappedImageFullPath: str,
                                    ControlMaskFullPath: str | None,
                                    MappedMaskFullPath: str | None,
                                    create_copy: bool = True):
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

                # PrettyOutput.Log("ChangeTransformPixelSpacing from " + str(oldspacing) + " to " + str(newspacing))
        scale = float(oldspacing) / float(newspacing)

        NewStosFile = StosFile()

        # NewStosFile.ControlImageDim = [x * scale for x in self.ControlImageDim]
        NewStosFile.ControlImageDim = copy.copy(self.ControlImageDim)
        NewStosFile.ControlImageDim[2] = self.ControlImageDim[2] * scale  # type: ignore[index]
        NewStosFile.ControlImageDim[3] = self.ControlImageDim[3] * scale  # type: ignore[index]
        # NewStosFile.MappedImageDim = [x * scale for x in self.MappedImageDim]

        # Update the filenames which are the first two lines of the file
        NewStosFile.MappedImageDim = copy.copy(self.MappedImageDim)
        NewStosFile.MappedImageDim[2] = self.MappedImageDim[2] * scale  # type: ignore[index]
        NewStosFile.MappedImageDim[3] = self.MappedImageDim[3] * scale  # type: ignore[index]

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
            transformObj = nornir_imageregistration.transforms.LoadTransform(self.Transform, pixelSpacing=1.0)  # type: ignore[arg-type]
            assert (transformObj is not None)

            if isinstance(transformObj, nornir_imageregistration.transforms.ITransformScaling):
                transformObj.Scale(scale)
            else:
                raise ValueError(
                    f"Transform needs to be scaled but does not support ITransformScaling interface {transformObj}")

            NewStosFile._Downsample = newspacing

            # if hasattr(transformObj, 'gridWidth'):
            # Save as a stos grid if we can
            #    bounds = (NewStosFile.MappedImageDim[1], NewStosFile.MappedImageDim[0], NewStosFile.MappedImageDim[3],
            #              NewStosFile.MappedImageDim[2])
            #    NewStosFile.transform = nornir_imageregistration.transforms.TransformToIRToolsString(transformObj,
            #                                                                                         bounds=bounds)
            # else:
            # NewStosFile.transform = nornir_imageregistration.transforms.TransformToIRToolsString(
            #        transformObj)  # , bounds=NewStosFile.MappedImageDim)

            NewStosFile.Transform = nornir_imageregistration.transforms.TransformToIRToolsString(transformObj)

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

        PrettyOutput.Log("ChangeTransformPixelSpacing from {0:d} to {1:d}".format(mapped_spacing, control_spacing))

        control_spacing = float(control_spacing)
        mapped_spacing = float(mapped_spacing)

        mapped_space_scalar = mapped_spacing / control_spacing

        NewStosFile = StosFile()

        NewStosFile.ControlImageDim = copy.copy(self.ControlImageDim)
        # NewStosFile.MappedImageDim = [x * scale for x in self.MappedImageDim]

        # Update the filenames which are the first two lines of the file
        NewStosFile.MappedImageDim = copy.copy(self.MappedImageDim)
        NewStosFile.MappedImageDim[2] = self.MappedImageDim[2] * mapped_space_scalar  # type: ignore[index]
        NewStosFile.MappedImageDim[3] = self.MappedImageDim[3] * mapped_space_scalar  # type: ignore[index]

        NewStosFile.ControlImageFullPath = self.ControlImageFullPath
        NewStosFile.ControlMaskFullPath = self.ControlMaskFullPath
        NewStosFile.MappedImageFullPath = MappedImageFullPath
        NewStosFile.MappedMaskFullPath = MappedMaskFullPath

        if os.path.exists(MappedImageFullPath):
            NewStosFile.MappedImageDim = StosFile.__GetImageDimsArray(MappedImageFullPath)

        # Adjust the transform points 
        transformObj = nornir_imageregistration.transforms.LoadTransform(self.Transform, pixelSpacing=1.0)  # type: ignore[arg-type]
        assert (transformObj is not None)
        if isinstance(transformObj, ITransformRelativeScaling):
            transformObj.ScaleWarped(scalar=mapped_space_scalar)

        NewStosFile._Downsample = control_spacing

        if hasattr(transformObj, 'gridWidth'):
            # Save as a stos grid if we can
            bounds = (NewStosFile.MappedImageDim[1], NewStosFile.MappedImageDim[0], NewStosFile.MappedImageDim[3],  # type: ignore[index]
                      NewStosFile.MappedImageDim[2])  # type: ignore[index]
            NewStosFile.Transform = nornir_imageregistration.transforms.TransformToIRToolsString(transformObj,
                                                                                                 bounds=bounds)
        else:
            NewStosFile.Transform = nornir_imageregistration.transforms.TransformToIRToolsString(
                transformObj)  # , bounds=NewStosFile.MappedImageDim)

        return NewStosFile

    def StomOutputExists(self, StosPath: str, OutputPath: str, StosMapPath=None):

        '''If we haven't written the stos file itself, return false'''
        stosfullname = os.path.join(StosPath, self.FormattedStosFileName)  # type: ignore[attr-defined]
        if not os.path.exists(stosfullname):
            return False

        '''Checks whether valid stom output exists for this file.  Returns true if all output files are valid'''
        predictedMappedOutputName = self.OutputMappedImageName  # type: ignore[attr-defined]

        predictedMappedOutputFullname = os.path.join(OutputPath, predictedMappedOutputName)
        if not os.path.exists(predictedMappedOutputFullname):
            return False

        if nornir_shared.files.RemoveOutdatedFile(stosfullname, predictedMappedOutputFullname):
            return False

        predictedControlOutputName = self.OutputControlImageName  # type: ignore[attr-defined]
        predictedControlOutputFullname = os.path.join(OutputPath, predictedControlOutputName)
        if not os.path.exists(predictedControlOutputFullname):
            return False

        if nornir_shared.files.RemoveOutdatedFile(stosfullname, predictedControlOutputFullname):
            return False

        if StosMapPath is not None:
            if nornir_shared.files.RemoveOutdatedFile(StosMapPath, predictedControlOutputFullname):
                return False

        return True


def IdentityRigidTransform() -> nornir_imageregistration.transforms.CenteredSimilarity2DTransform:
    """Return a centered similarity transform that maps coordinates to themselves."""
    return nornir_imageregistration.transforms.CenteredSimilarity2DTransform(
        target_offset=np.array([0.0, 0.0], dtype=np.float32),
        source_rotation_center=np.array([0.0, 0.0], dtype=np.float32),
        angle=0.0,
        scalar=1.0)


def RigidTransformFromStosPath(stos_path: str,
                               ignore_rotation: bool = False) -> ITransform:
    """Load a STOS file and estimate the best rigid/similarity transform for its control points."""
    loaded = StosFile.Load(stos_path)
    transform_obj = nornir_imageregistration.transforms.LoadTransform(loaded.Transform)  # type: ignore[arg-type]
    return nornir_imageregistration.transforms.converters.ConvertTransformToRigidTransform(
        transform_obj,
        ignore_rotation=ignore_rotation)


def AddStosTransforms(A_To_B,
                      B_To_C,
                      EnrichTolerance: float | None,
                      linear_factor: float | None = None,
                      travel_limit: float | None = None,
                      ignore_rotation: bool = False,
                      reblend_iterations: int = 1,
                      reblend_tolerance: float | None = None,
                      B_To_C_Linear: nornir_imageregistration.transforms.ITransform | None = None) -> StosFile:
    '''
    :param EnrichTolerance:
    :param A_To_B: Commonly a single section transform, "4->3"
    :param B_To_C: Commonly the transform to the center of a volume, "3->1"
    '''
    A_To_B_Stos = __argumentToStos(A_To_B)
    B_To_C_Stos = __argumentToStos(B_To_C)

    # I'll need to make sure I remember to set the downsample factor when I warp the .mosaic files
    A_To_B_Transform = nornir_imageregistration.transforms.LoadTransform(A_To_B_Stos.Transform)  # type: ignore[arg-type]
    B_To_C_Transform = nornir_imageregistration.transforms.LoadTransform(B_To_C_Stos.Transform)  # type: ignore[arg-type]

    # OK, I should use a rotation/translation only transform to regularize the added transforms to knock down accumulated warps/errors

    if linear_factor is None and travel_limit is None:
        A_To_C_Transform = nornir_imageregistration.transforms.addition.AddTransforms(B_To_C_Transform,
                                                                                      A_To_B_Transform, EnrichTolerance,  # type: ignore[arg-type]
                                                                                      create_copy=False)
    else:
        A_To_C_Transform = nornir_imageregistration.transforms.addition.AddTransformsWithLinearCorrection(
            B_To_C_Transform,
            A_To_B_Transform,  # type: ignore[arg-type]
            EnrichTolerance,
            create_copy=False,
            linear_factor=linear_factor,
            travel_limit=travel_limit,
            ignore_rotation=ignore_rotation,
            reblend_iterations=reblend_iterations,
            reblend_tolerance=reblend_tolerance,
            B_To_C_Linear=B_To_C_Linear)

    A_To_C_Stos = copy.deepcopy(A_To_B_Stos)
    A_To_C_Stos.TargetSectionNumber = B_To_C_Stos.TargetSectionNumber
    A_To_C_Stos.ControlImageFullPath = B_To_C_Stos.ControlImageFullPath
    A_To_C_Stos.ControlMaskFullPath = B_To_C_Stos.ControlMaskFullPath

    A_To_C_Stos.Transform = nornir_imageregistration.transforms.TransformToIRToolsString(A_To_C_Transform)  # type: ignore[arg-type]

    #     if hasattr(A_To_B_Transform, "gridWidth") and hasattr(A_To_B_Transform, "gridHeight"):
    #         A_To_C_Stos.transform = nornir_imageregistration.transforms.TransformToIRToolsGridString(A_To_C_Transform, A_To_B_Transform.gridWidth, A_To_B_Transform.gridHeight)
    #     else:
    #         A_To_C_Stos.transform = nornir_imageregistration.transforms.TransformToIRToolsString(A_To_C_Transform)

    A_To_C_Stos.ControlImageDim = B_To_C_Stos.ControlImageDim
    A_To_C_Stos.MappedImageDim = A_To_B_Stos.MappedImageDim

    return A_To_C_Stos
