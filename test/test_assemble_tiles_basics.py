import math
import numpy.testing as npt
import nornir_imageregistration
import nornir_imageregistration.assemble_tiles as at
from nornir_imageregistration.mosaic import Mosaic
from nornir_imageregistration.distance import CreateDistanceImage, CreateDistanceImageBruteForce
import test_assemble_tiles


class BasicTests(test_assemble_tiles.TestMosaicAssemble):

    @property
    def TestName(self):
        return "PMG1"

    def test_CreateDistanceBuffer(self):
        firstShape = (10, 10)
        dMatrix = at.CreateDistanceImage(firstShape)
        npt.assert_array_equal(dMatrix.shape, firstShape, "Distance matrix shape incorrect")
        ten_corner_distance = math.sqrt((4.5 ** 2) * 2)
        self.assertAlmostEqual(dMatrix[0, 0], ten_corner_distance, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[9, 9], ten_corner_distance, 2, "Distance matrix incorrect")

        secondShape = (11, 11)
        eleven_corner_distance = math.sqrt((5 ** 2) * 2)
        dMatrix = at.CreateDistanceImage(secondShape)
        npt.assert_array_equal(dMatrix.shape, secondShape, "Distance matrix shape incorrect")

        self.assertAlmostEqual(dMatrix[0, 0], eleven_corner_distance, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[10, 10], eleven_corner_distance, 2, "Distance matrix incorrect")

        thirdShape = (10, 11)
        dMatrix = at.CreateDistanceImage(thirdShape)
        npt.assert_array_equal(dMatrix.shape, thirdShape, "Distance matrix shape incorrect")
        uneven_corner_distance = math.sqrt((4.5 ** 2) + (5 ** 2))

        self.assertAlmostEqual(dMatrix[0, 0], uneven_corner_distance, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[9, 0], uneven_corner_distance, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[9, 10], uneven_corner_distance, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[0, 10], uneven_corner_distance, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[0, 5], 4.5, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[4, 0], math.sqrt((5 ** 2) + (0.5 ** 2)), 2, "Distance matrix incorrect")

    def test_CreateDistanceBuffer2(self):
        #         zeroEvenShape = (2, 2)
        #         dMatrix = at.CreateDistanceImage2(zeroEvenShape)
        #
        #         zeroOddShape = (3, 3)
        #         dMatrix = at.CreateDistanceImage2(zeroOddShape)

        zeroOddShape = (5, 5)
        dMatrix = at.CreateDistanceImage(zeroOddShape)

        firstShape = (10, 10)
        dMatrixReference = CreateDistanceImageBruteForce(firstShape)
        dMatrix = at.CreateDistanceImage(firstShape)
        self.assertAlmostEqual(dMatrix[0, 0], math.sqrt((4.5 ** 2) * 2), 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[9, 9], math.sqrt((4.5 ** 2) * 2), 2, "Distance matrix incorrect")

        secondShape = (11, 11)
        dMatrix = at.CreateDistanceImage(secondShape)

        cornerDistance = math.sqrt((5 ** 2) * 2)
        self.assertAlmostEqual(dMatrix[0, 0], cornerDistance, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[10, 10], cornerDistance, 2, "Distance matrix incorrect")

        thirdShape = (10, 11)
        dMatrix = at.CreateDistanceImage(thirdShape)

        uneven_corner_distance = math.sqrt((4.5 ** 2) + (5 ** 2))

        self.assertAlmostEqual(dMatrix[0, 0], uneven_corner_distance, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[9, 0], uneven_corner_distance, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[9, 10], uneven_corner_distance, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[0, 10], uneven_corner_distance, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[0, 5], 4.5, 2, "Distance matrix incorrect")
        self.assertAlmostEqual(dMatrix[4, 0], math.sqrt((5 ** 2) + (0.5 ** 2)), 2, "Distance matrix incorrect")

    def test_MosaicBoundsEachMosaicType(self):
        downsamplePath = '004'
        tilesDir = self.GetTileFullPath(downsamplePath)
        for m in self.GetMosaicFiles():
            mosaic = Mosaic.LoadFromMosaicFile(m)
            mosaic_tileset = nornir_imageregistration.mosaic_tileset.CreateFromMosaic(mosaic, image_folder=tilesDir,
                                                                                      image_to_source_space_scale=int(
                                                                                          downsamplePath))

            self.assertIsNotNone(mosaic_tileset.SourceBoundingBox, "No bounding box returned for mosiac")

            self.Logger.info(m + " mapped bounding box: " + str(mosaic_tileset.SourceBoundingBox))

            self.assertIsNotNone(mosaic_tileset.TargetBoundingBox, "No bounding box returned for mosiac")

            self.Logger.info(m + " fixed bounding box: " + str(mosaic_tileset.TargetBoundingBox))
