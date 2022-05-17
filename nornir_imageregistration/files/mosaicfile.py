import os
import sys

from nornir_shared import checksum, prettyoutput
import nornir_shared.images as images


class MosaicFile(object):
    """description of class"""

    @classmethod
    def LoadChecksum(cls, path):
        #assert(os.path.exists(path))
        mosaicObj = MosaicFile.Load(path)
        if mosaicObj is None:
            return None
        
        return mosaicObj.Checksum

    @property
    def Checksum(self):
        # We need to sort the transforms so that the order of insertion does not change the checksum
        return checksum.DataChecksum(sorted(self.ImageToTransformString.values()))

    @property
    def NumberOfImages(self):
        if(self.ImageToTransformString is None):
            return 0

        return len(list(self.ImageToTransformString.keys()))

    def __init__(self):
        self.FileReportedNumberOfImages = None  # Sometimes files do not accurately report the number of images.  We populate this on load to know if we need to make a correction
        self.pixel_spacing = 1
        self.use_std_mask = 0
        self.format_version_number = 0

        self.ImageToTransformString = dict()
        pass

    def RemoveInvalidMosaicImages(self, TileDir):
        '''Runs a time consuming check that verifies all the image files are valid.  Returns true if invalid images were found.'''

        ImageList = []
        ImageList.extend(list(self.ImageToTransformString.keys()))
        InvalidImages = images.AreValidImages(ImageList, TileDir)

        FoundInvalid = False

        for InvalidImage in InvalidImages:
            if InvalidImage in self.ImageToTransformString:
                InvalidImageFullPath = os.path.join(TileDir, InvalidImage)
                if os.path.exists(InvalidImageFullPath):
                    prettyoutput.Log('Removing invalid image from disk:')

                    os.remove(InvalidImageFullPath)

                prettyoutput.Log('Removing invalid image from mosaic: %s' % InvalidImage)
                del self.ImageToTransformString[InvalidImage]
                FoundInvalid = True

               # prettyoutput.Log('Removing invalid image from mosaic: %s' % InvalidImage)
               # del self.ImageToTransformString[InvalidImage]

        return FoundInvalid


    @classmethod
    def GetInfo(cls, filename):
        '''Parses a filename and returns what we have learned if it follows the convention
           <SECTION>_<CHANNEL>_<MosaicSource>_<DOWNSAMPLE>.mosaic
           [SectionNumber, Channel, MosaicSource, Downsample]'''

        if(os.path.exists(filename) == False):
            prettyoutput.LogErr("mosaic file not found: " + filename)
            return

        baseName = os.path.basename(filename)
        [baseName, ext] = os.path.splitext(baseName)


        parts = baseName.split("_")

        try:
            SectionNumber = int(parts[0])
        except:
            # We really can't recover from this, so maybe an exception should be thrown instead
            SectionNumber = None
            prettyoutput.Log('Could not determine mosaic section number: ' + str(filename))

        try:
            MosaicType = parts[-2]
        except:
            MosaicType = None
            prettyoutput.Log('Could not determine mosaic source: ' + str(filename))

        if(len(parts) > 3):
            try:
                Channel = parts[1]
            except:
                Channel = None
                prettyoutput.Log('Could not determine mosaic channel: ' + str(filename))

        # If we don't have a valid downsample value we assume 1
        try:
            DownsampleStrings = parts[-1].split(".")
            Downsample = int(DownsampleStrings[0])
        except:
            Downsample = None
            prettyoutput.Log('Could not determine mosaic downsample: ' + str(filename))

        return [SectionNumber, Channel, MosaicType, Downsample]


    @classmethod
    def Load(cls, filename):
        
        lines = []
        try:
            with open(filename, 'r') as fMosaic:
                lines = fMosaic.readlines()
                fMosaic.close()
        except FileNotFoundError:
            prettyoutput.LogErr(f"Mosaic file not found: {filename}")
            return
        except Exception as error:
            prettyoutput.LogErr(f"Unexpected error {error} while opening Mosaic file {filename}")
            return

        obj = MosaicFile()

        iLine = 0
        try:
            while iLine < len(lines):
                line = lines[iLine]
                line = line.strip()
                [text, value] = line.split(':', 1)
    
                if(text.startswith('number_of_images')):
                    value.strip()
                    obj.FileReportedNumberOfImages = int(value)
                elif(text.startswith('pixel_spacing')):
                    value.strip()
                    obj.pixel_spacing = int(float(value))
                elif(text.startswith('use_std_mask')):
                    obj.use_std_mask = int(value)
                elif(text.startswith('format_version_number')):
                    obj.format_version_number = int(value)
                elif(text.startswith('image')):
                    if(obj.format_version_number == 0):
                        [filename, transform] = value.split(None, 1)
                        filename = filename.strip()
                        filename = os.path.basename(filename)
                        transform = transform.strip()
                        obj.ImageToTransformString[filename] = transform
                    else:
                        filename = os.path.basename(lines[iLine + 1].strip())
                        transform = lines[iLine + 2].strip()
                        obj.ImageToTransformString[filename] = transform
                        iLine = iLine + 2
    
                iLine = iLine + 1
        except:
            prettyoutput.LogErr(f"Error in {filename} on or above line #{iLine}\n{lines[iLine]}\n")
            raise 
            

        return obj

    @classmethod
    def Write(cls, OutfilePath, Entries, Flip=False, Flop=False, ImageSize=None, Downsample=1):
        '''
        Creates a .mosaic file in the specified directory containing the specified
        Dictionary (Entries) at the specified Downsample rate.  Entries should have the key=filename
        and values should be a tuple with pixel coordinates
    
        Setting Flip to True will invert all X coordinates
        Setting Flop to True will invert all Y coordinates
        '''

        if ImageSize is None:
            ImageSize = [(4080, 4080)] * len(Entries)
        elif not isinstance(ImageSize, list):
            assert (len(ImageSize) == 2),  "Expect tuple or list indicating image size"
            ImageSize = ImageSize * len(Entries)
        else:
            # A list of two entries for the size
            if len(ImageSize) == 2 and not isinstance(ImageSize[0], list):
                ImageSize = [(ImageSize[0], ImageSize[1])] * len(Entries)
            elif len(ImageSize) == len(Entries):
                ImageSize = ImageSize
            else:
                raise Exception("Unexpected list format for ImageSize argument")

        prettyoutput.CurseString('Stage', "WriteMosaicFile " + OutfilePath)


        # TODO - find min/max values and center image in .mosaic file
        minX = sys.maxsize
        minY = sys.maxsize
        maxX = -sys.maxsize - 1
        maxY = -sys.maxsize - 1

        keys = list(Entries.keys())
        keys.sort()

        for key in keys:
            Coord = Entries[key]
            if(Coord[0] < minX):
                minX = Coord[0]
            if(Coord[0] > maxX):
                maxX = Coord[0]
            if(Coord[1] < minY):
                minY = Coord[1]
            if(Coord[1] > maxY):
                maxY = Coord[1]

        prettyoutput.Log('MinX:\t' + str(minX) + '\tMaxX:\t' + str(maxX))
        prettyoutput.Log('MinY:\t' + str(minY) + '\tMaxY:\t' + str(maxY))

        # Write standard .mosiac file header
        with open(OutfilePath, 'w+') as OutFile:
            OutFile.write('format_version_number: 1\n')

            OutFile.write('number_of_images: ' + str(len(Entries)) + '\n')
            OutFile.write('pixel_spacing: ' + str(Downsample) + '\n')
            OutFile.write('use_std_mask: 0\n')

        #    prettyoutput.Log( "Keys: " + str(Entries.keys())
            keys = list(Entries.keys())
            keys.sort()

            for i, key in enumerate(keys):
                Coord = Entries[key]
                tilesize = ImageSize[i]

                # prettyoutput.Log( str(key) + " : " + str(Coord)

                # Centering is nice, but it seems to break multi mrc sections
        #            Coord = (Coord[0] - (minX + CenterXOffset), Coord[1] - (minY + CenterYOffset))
                Coord = (Coord[0], Coord[1])

                X = Coord[0]
                Y = Coord[1]
                if Flip:
                    Y = -Y
                if Flop:
                    X = -X

                # Remove dirname from key
                key = os.path.basename(key)

                outstr = 'image:\n' + key + '\n' + 'LegendrePolynomialTransform_double_2_2_1 vp 6 1 0 1 1 1 0 fp 4 ' + str(X) + ' ' + str(Y) + ' ' + str(tilesize[0] / 2) + ' ' + str(tilesize[1] / 2)
                OutFile.write(outstr + '\n')

            OutFile.close()

    def CompressTransforms(self, FloatTemplateStr=None):
        if FloatTemplateStr is None:
            FloatTemplateStr = '%g'

        for FileKey in list(self.ImageToTransformString.keys()):
            TransformStr = self.ImageToTransformString[FileKey]
            OutputTransformStr = ""
            parts = TransformStr.split()

            for i in range(0, len(parts)):
                part = parts[i]

                NeededSpace = " "
                # Figure out if we add a space after the line
                if(i == len(parts) - 1):
                    NeededSpace = ""

                floatVal = float()
                intVal = int()
                try:
                    floatVal = float(part)
                except:
     #               prettyoutput.Log( "Skip: " + part)
                    # Can't convert it to a number.  Write the old value and move on
                    # Don't leave a space after the last entry on a line.  This breaks
                    # filenames on the mac because Mac's can have a filename end in a
                    # space
                    OutputTransformStr = OutputTransformStr + part + NeededSpace
                    continue

                intVal = 0
                try:
                    intVal = int(part)
                except:
                    intVal = None
                    pass

                # Figure out if we can write number as an int
                outStr = ""
                if(not intVal is None):
                    OutputTransformStr = OutputTransformStr + str(intVal) + NeededSpace
                else:
                    # outStr = "%.2f" % (floatVal)

                    outStr = "%g" % (floatVal)
                    OutputTransformStr = OutputTransformStr + outStr + NeededSpace

            self.ImageToTransformString[FileKey] = OutputTransformStr

    def Save(self, filename):

        if(len(self.ImageToTransformString) <= 0):
            prettyoutput.LogErr("No tiles present in mosaic " + filename + "\nThe save request was aborted")
            return

        fMosaic = open(filename, 'w')
        fMosaic.write(self.MosaicStr())
        fMosaic.close()

    def MosaicStr(self):
        OutStrList = list()
        OutStrList.append('format_version_number: 1\n')
        OutStrList.append('number_of_images: ' + str(len(list(self.ImageToTransformString.keys()))) + '\n')
        OutStrList.append('pixel_spacing: ' + str(self.pixel_spacing) + '\n')
        OutStrList.append('use_std_mask: ' + str(self.use_std_mask) + '\n')

        for filename in sorted(self.ImageToTransformString.keys()):
            OutStrList.append('image:\n')
            transform = self.ImageToTransformString[filename]

            filename = os.path.basename(filename)
            OutStrList.append(filename + '\n')
            OutStrList.append(transform + '\n')

        OutStr = ''.join(OutStrList)
        return OutStr

