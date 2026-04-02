import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.transformed_image_data import ITransformedImageData, TransformedImageDataState
from nornir_imageregistration.transformed_image_data_shared_memory import TransformedImageDataViaSharedMemory
from nornir_imageregistration.transformed_image_data_temp_files import TransformedImageDataViaTempFile


class TestTransformedImageDataContracts(unittest.TestCase):

    def _assert_common_contract(self, data: ITransformedImageData) -> None:
        self.assertIsInstance(data.image, np.ndarray)
        self.assertIsInstance(data.centerDistanceImage, np.ndarray)
        self.assertIsInstance(data.source_space_scale, float)
        self.assertIsInstance(data.target_space_scale, float)
        self.assertIsInstance(data.rendered_target_space_origin, np.ndarray)
        self.assertNotEqual(data.state, TransformedImageDataState.CLEARED)

    def test_create_contract_is_harmonized(self) -> None:
        image = np.ones((16, 16), dtype=np.float32)
        distance = np.ones((16, 16), dtype=np.float32)

        shared = TransformedImageDataViaSharedMemory.Create(
            image=image,
            centerDistanceImage=distance,
            transform=None,
            source_space_scale=1.0,
            target_space_scale=1.0,
            rendered_target_space_origin=(0.0, 0.0),
            SingleThreadedInvoke=True
        )
        temp = TransformedImageDataViaTempFile.Create(
            image=image,
            centerDistanceImage=distance,
            transform=None,
            source_space_scale=1.0,
            target_space_scale=1.0,
            rendered_target_space_origin=(0.0, 0.0),
            SingleThreadedInvoke=True
        )
        try:
            self._assert_common_contract(shared)
            self._assert_common_contract(temp)
        finally:
            shared.Clear()
            temp.Clear()

    def test_clear_moves_to_cleared_state_and_raises(self) -> None:
        image = np.ones((8, 8), dtype=np.float32)
        distance = np.ones((8, 8), dtype=np.float32)

        shared = TransformedImageDataViaSharedMemory.Create(
            image=image,
            centerDistanceImage=distance,
            transform=None,
            source_space_scale=1.0,
            target_space_scale=1.0,
            rendered_target_space_origin=(0.0, 0.0),
            SingleThreadedInvoke=True
        )
        temp = TransformedImageDataViaTempFile.Create(
            image=image,
            centerDistanceImage=distance,
            transform=None,
            source_space_scale=1.0,
            target_space_scale=1.0,
            rendered_target_space_origin=(0.0, 0.0),
            SingleThreadedInvoke=True
        )

        shared.Clear()
        temp.Clear()

        self.assertEqual(shared.state, TransformedImageDataState.CLEARED)
        self.assertEqual(temp.state, TransformedImageDataState.CLEARED)

        with self.assertRaises(ValueError):
            _ = shared.image
        with self.assertRaises(ValueError):
            _ = temp.image

    def test_tempfile_accepts_shared_memory_metadata(self) -> None:
        image = np.arange(64, dtype=np.float32).reshape(8, 8)
        distance = np.ones((8, 8), dtype=np.float32)
        metadata, shared_image = nornir_imageregistration.npArrayToSharedArray(image)

        temp = TransformedImageDataViaTempFile.Create(
            image=metadata,
            centerDistanceImage=distance,
            transform=None,
            source_space_scale=1.0,
            target_space_scale=1.0,
            rendered_target_space_origin=(0.0, 0.0),
            SingleThreadedInvoke=True
        )
        try:
            np.testing.assert_allclose(temp.image, shared_image)
            self.assertEqual(temp.state, TransformedImageDataState.IN_MEMORY)
        finally:
            temp.Clear()
            nornir_imageregistration.unlink_shared_memory(metadata)


if __name__ == "__main__":
    unittest.main()
