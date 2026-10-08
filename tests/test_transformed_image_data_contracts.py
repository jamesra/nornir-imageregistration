import dataclasses
import unittest

import numpy as np
from hypothesis import example, given, settings, strategies as st

import nornir_imageregistration
import nornir_imageregistration.assemble_tiles
from nornir_imageregistration import local_distortion_correction
from nornir_imageregistration.transforms.rigid import Rigid
from nornir_imageregistration.transformed_image_data import ITransformedImageData, TransformedImageDataState, \
    TransformedTileMetadata
from nornir_imageregistration.transformed_image_data_shared_memory import TransformedImageDataViaSharedMemory
from nornir_imageregistration.transformed_image_data_temp_files import TransformedImageDataViaTempFile

_UNIT_METADATA = TransformedTileMetadata(source_space_scale=1.0, target_space_scale=1.0,
                                         rendered_target_space_origin=(0.0, 0.0))

# Mosaic coordinates reach the hundreds of thousands of pixels; scales are 1/downsample.
_scaled_min = st.integers(min_value=-(2 ** 20), max_value=2 ** 20)
_target_space_scale = st.one_of(
    st.sampled_from([1.0, 0.5, 0.25, 0.125, 1.0 / 16, 1.0 / 32, 1.0 / 64, 1.0 / 256, 2.0]),
    st.floats(min_value=1e-3, max_value=4.0, allow_nan=False, allow_infinity=False))


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
            metadata=_UNIT_METADATA,
            SingleThreadedInvoke=True
        )
        temp = TransformedImageDataViaTempFile.Create(
            image=image,
            centerDistanceImage=distance,
            transform=None,
            metadata=_UNIT_METADATA,
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
            metadata=_UNIT_METADATA,
            SingleThreadedInvoke=True
        )
        temp = TransformedImageDataViaTempFile.Create(
            image=image,
            centerDistanceImage=distance,
            transform=None,
            metadata=_UNIT_METADATA,
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
            metadata=_UNIT_METADATA,
            SingleThreadedInvoke=True
        )
        try:
            np.testing.assert_allclose(temp.image, shared_image)
            self.assertEqual(temp.state, TransformedImageDataState.IN_MEMORY)
        finally:
            temp.Clear()
            nornir_imageregistration.unlink_shared_memory(metadata)


class TestTransformedTileMetadata(unittest.TestCase):

    def test_is_frozen_and_hashable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            _UNIT_METADATA.target_space_scale = 2.0  # type: ignore[misc]
        self.assertEqual(hash(_UNIT_METADATA),
                         hash(TransformedTileMetadata(1.0, 1.0, (0.0, 0.0))))

    @given(source_space_scale=_target_space_scale, target_space_scale=_target_space_scale,
           scaled_min_y=_scaled_min, scaled_min_x=_scaled_min)
    @example(source_space_scale=0.25, target_space_scale=0.125, scaled_min_y=454, scaled_min_x=0)
    @example(source_space_scale=1.0, target_space_scale=1.0 / 3.0, scaled_min_y=1, scaled_min_x=-7)
    def test_from_scaled_target_origin_matches_inline_rule(self, source_space_scale: float,
                                                           target_space_scale: float,
                                                           scaled_min_y: int, scaled_min_x: int) -> None:
        """The origin rule is the one TransformTile and grid refine computed inline, bit for bit."""
        metadata = TransformedTileMetadata.from_scaled_target_origin(
            source_space_scale, target_space_scale, scaled_min_y, scaled_min_x)
        self.assertIs(metadata.source_space_scale, source_space_scale)
        self.assertIs(metadata.target_space_scale, target_space_scale)
        expected = (scaled_min_y * (1.0 / target_space_scale), scaled_min_x * (1.0 / target_space_scale))
        self.assertEqual(expected, metadata.rendered_target_space_origin)
        self.assertEqual([type(v) for v in expected],
                         [type(v) for v in metadata.rendered_target_space_origin])

    @settings(max_examples=25, deadline=None)
    @given(source_space_scale=_target_space_scale, target_space_scale=_target_space_scale,
           scaled_min_y=_scaled_min, scaled_min_x=_scaled_min)
    def test_backends_expose_metadata_unchanged(self, source_space_scale: float, target_space_scale: float,
                                                scaled_min_y: int, scaled_min_x: int) -> None:
        """Every backend reports the metadata's scales and stores the origin as float32, as before."""
        metadata = TransformedTileMetadata.from_scaled_target_origin(
            source_space_scale, target_space_scale, scaled_min_y, scaled_min_x)
        image = np.ones((4, 4), dtype=np.float32)
        expected_origin = np.asarray(metadata.rendered_target_space_origin, dtype=np.float32)
        for backend in (TransformedImageDataViaSharedMemory, TransformedImageDataViaTempFile):
            data = backend.Create(image, image, None, metadata, SingleThreadedInvoke=True)
            try:
                self.assertIs(data.source_space_scale, source_space_scale)
                self.assertIs(data.target_space_scale, target_space_scale)
                self.assertEqual(np.float32, data.rendered_target_space_origin.dtype)
                np.testing.assert_array_equal(expected_origin, data.rendered_target_space_origin)
            finally:
                data.Clear()


class TestProducersRenderAtTheirTargetOrigin(unittest.TestCase):
    """TransformTile and the grid refine warp report the unscaled target-space origin of what they rendered."""

    @settings(max_examples=15, deadline=None)
    @given(offset_y=st.integers(min_value=-4000, max_value=4000),
           offset_x=st.integers(min_value=-4000, max_value=4000),
           downsample=st.sampled_from([1, 2, 4]))
    @example(offset_y=40, offset_x=104, downsample=2)
    def test_rendered_origin_is_target_region_min(self, offset_y: int, offset_x: int, downsample: int) -> None:
        offset = (float(offset_y * downsample), float(offset_x * downsample))
        image = np.random.default_rng(seed=7).random((32, 48)).astype(np.float32)
        tile = nornir_imageregistration.tile.Tile(Rigid(target_offset=offset), image,
                                                  image_to_source_space_scale=float(downsample), ID=0)
        whole = nornir_imageregistration.assemble_tiles.TransformTile(tile, SingleThreadedInvoke=True)
        try:
            np.testing.assert_array_equal(np.asarray(offset, dtype=np.float32), whole.rendered_target_space_origin)
        finally:
            whole.Clear()

        bbox = tile.TargetSpaceBoundingBox
        region = nornir_imageregistration.Rectangle.CreateFromBounds(
            (bbox.MinY + 8 * downsample, bbox.MinX + 16 * downsample, bbox.MaxY, bbox.MaxX))
        expected = np.asarray(region.BottomLeft, dtype=np.float32)
        source_rect = nornir_imageregistration.Rectangle.CreateFromBounds((0, 0, 32, 48))
        warped = local_distortion_correction._warp_overlap_for_grid_refine(
            tile, source_rect, region, 1.0 / downsample, single_threaded_invoke=True)
        cropped = nornir_imageregistration.assemble_tiles.TransformTile(
            tile, target_space_scale=1.0 / downsample, TargetRegion=region, SingleThreadedInvoke=True)
        try:
            np.testing.assert_array_equal(expected, warped.rendered_target_space_origin)
            np.testing.assert_array_equal(expected, cropped.rendered_target_space_origin)
        finally:
            warped.Clear()
            cropped.Clear()


if __name__ == "__main__":
    unittest.main()
