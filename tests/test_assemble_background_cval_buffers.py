"""TransformImage's enforce_background_cval path does not need a section copy.

The path did ``output = output.copy()`` before masking. The buffer being masked
is always locally allocated -- either the warp output from
SourceImageToTargetSpace or the tile-assembly ``np.zeros`` -- so the copy
protected nothing and cost a section-sized memcpy plus allocation.

It does *not* lower peak memory: the copy happens after the warp, whose
coordinate arrays already set a much higher peak (measured at roughly 10x the
section size). See issue #12 for that separate, architectural problem.

These tests pin the properties a copy *would* have protected, so removing it
stays safe: the caller's source image is not mutated, and the masked values are
correct. Both branches of TransformImage are covered, since the single-tile and
tile-assembly paths each had their own copy.
"""

from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration.transforms import rigid


def _identity_transform() -> rigid.RigidTranslation:
    return rigid.RigidTranslation(target_offset=(0, 0))


class TestEnforceBackgroundCvalDoesNotMutateSource(unittest.TestCase):

    def setUp(self) -> None:
        rng = np.random.default_rng(seed=1234)
        self.source = (rng.random((64, 64)) * 255).astype(np.float32)
        self.source_before = self.source.copy()

    def test_source_image_is_not_modified(self) -> None:
        nornir_imageregistration.assemble.TransformImage(
            _identity_transform(),
            np.array([64, 64]),
            self.source,
            CropUndefined=False,
            enforce_background_cval=0)

        np.testing.assert_array_equal(
            self.source, self.source_before,
            "TransformImage mutated the caller's source image")

    def test_masked_region_takes_the_background_value(self) -> None:
        """A shifted transform leaves part of the target unmapped."""
        transform = rigid.RigidTranslation(target_offset=(20, 20))

        output = nornir_imageregistration.assemble.TransformImage(
            transform,
            np.array([64, 64]),
            self.source,
            CropUndefined=False,
            enforce_background_cval=0)

        self.assertEqual(output.shape, (64, 64))
        # The shift guarantees unmapped output rows/columns, which must be
        # exactly the background value rather than interpolated garbage.
        self.assertTrue(np.any(output == 0), "No region received the background value")
        np.testing.assert_array_equal(
            self.source, self.source_before,
            "TransformImage mutated the caller's source image")

    def test_result_matches_a_copy_based_reference(self) -> None:
        """Removing the copy must not change any output pixel."""
        transform = rigid.RigidTranslation(target_offset=(7, -5))
        shape = np.array([64, 64])

        actual = nornir_imageregistration.assemble.TransformImage(
            transform, shape, self.source, CropUndefined=False,
            enforce_background_cval=0)

        # Reference: assemble without the background enforcement, then apply the
        # same mask to an explicit copy.
        raw = nornir_imageregistration.assemble.TransformImage(
            transform, shape, self.source, CropUndefined=False)
        mask = nornir_imageregistration.assemble.assembly_source_sample_mask(
            nornir_imageregistration.assemble.transform_for_host_assembly(transform),
            shape,
            self.source.shape[:2],
            extrapolate=True)
        expected = nornir_imageregistration.EnsureNumpyArray(raw).copy()
        expected[~mask] = 0

        np.testing.assert_array_equal(actual, expected)


class TestTileAssemblyBranch(unittest.TestCase):
    """Covers the second copy, on the multi-tile path.

    TransformImage picks its branch from the *source* image against a 2048 tile
    size, while the task loop iterates the *output* shape. A source larger than
    2048 with a small output therefore reaches the tile-assembly branch with a
    single task, which keeps this affordable for a unit test.
    """

    def test_tile_assembly_path_does_not_mutate_source(self) -> None:
        rng = np.random.default_rng(seed=4321)
        source = (rng.random((2176, 2176)) * 255).astype(np.float32)
        source_before = source.copy()

        transform = rigid.RigidTranslation(target_offset=(3, 4))
        shape = np.array([256, 256])

        grid_shape = nornir_imageregistration.TileGridShape(source.shape, [2048, 2048])
        self.assertFalse(
            np.all(np.asarray(grid_shape) == 1),
            "Fixture no longer reaches the tile-assembly branch")

        output = nornir_imageregistration.assemble.TransformImage(
            transform, shape, source, CropUndefined=False,
            enforce_background_cval=0)

        self.assertEqual(output.shape, (256, 256))
        np.testing.assert_array_equal(
            source, source_before,
            "Tile-assembly path mutated the caller's source image")


if __name__ == "__main__":
    unittest.main()
