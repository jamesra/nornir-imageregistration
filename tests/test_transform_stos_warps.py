"""
``TransformStos`` must warp the source image into target space.

The assemble call was commented out, so the function scaled the transform and then
returned (and saved) the *unwarped* source image. The existing size assertion in
``test_assemble.py`` could not catch it because the sample Fixed.png and Moving.png
are both 256x512, making the wrong answer the right shape. These tests use a
deliberately different target shape and a transform with a known displacement so
the warp itself is observable.

``stos`` was also left at ``None``, making both ``... is None`` fallbacks that read
image paths from the stos file dead code.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration import assemble
from nornir_imageregistration.transforms import factory

TARGET_H = 64
TARGET_W = 96
SOURCE_H = 32
SOURCE_W = 48
SHIFT_Y = 8
SHIFT_X = 12


def _source_image() -> np.ndarray:
    """A source with a single bright block, so its position is easy to locate."""
    image = np.zeros((SOURCE_H, SOURCE_W), dtype=np.float32)
    image[4:12, 6:18] = 1.0
    return image


def _translation_transform(shift_y: float = SHIFT_Y, shift_x: float = SHIFT_X):
    """A rigid translation mapping source points into target space."""
    return factory.CreateRigidTransform(
        target_image_shape=np.asarray((TARGET_H, TARGET_W), dtype=np.int64),
        source_image_shape=np.asarray((SOURCE_H, SOURCE_W), dtype=np.int64),
        rangle=0.0,
        warped_offset=(shift_y, shift_x))


def _write_stos(directory: str, fixed_path: str, moving_path: str) -> str:
    """Write a .stos file describing the translation, returning its path."""
    stos = nornir_imageregistration.StosFile()
    stos.ControlImageFullPath = fixed_path
    stos.MappedImageFullPath = moving_path
    stos.ControlImageDim = [TARGET_H, TARGET_W]
    stos.MappedImageDim = [SOURCE_H, SOURCE_W]
    stos.Transform = _translation_transform()
    stos_path = os.path.join(directory, 'translation.stos')
    stos.Save(stos_path)
    return stos_path


def test_output_has_target_shape_not_source_shape():
    """The result fills the target space, so returning the source is detectable."""
    warped = assemble.TransformStos(
        _translation_transform(),
        fixedImage=np.zeros((TARGET_H, TARGET_W), dtype=np.float32),
        warpedImage=_source_image(),
        CropUndefined=False)

    assert warped is not None
    assert warped.shape == (TARGET_H, TARGET_W), \
        'TransformStos returned the unwarped source image instead of warping it'


def _block_corner(image) -> tuple[int, int]:
    """Top-left corner of the bright block in *image*."""
    rows, cols = np.nonzero(np.asarray(image) > 0.5)
    assert rows.size > 0, 'the warped image lost all of the source content'
    return int(rows.min()), int(cols.min())


def _warp(shift_y: float, shift_x: float):
    return assemble.TransformStos(
        _translation_transform(shift_y=shift_y, shift_x=shift_x),
        fixedImage=np.zeros((TARGET_H, TARGET_W), dtype=np.float32),
        warpedImage=_source_image(),
        CropUndefined=False)


def test_content_is_displaced_by_the_transform():
    """The block moves by exactly the requested shift relative to no shift."""
    baseline_y, baseline_x = _block_corner(_warp(0, 0))
    shifted_y, shifted_x = _block_corner(_warp(SHIFT_Y, SHIFT_X))

    assert (shifted_y - baseline_y, shifted_x - baseline_x) == (SHIFT_Y, SHIFT_X)


def test_zero_shift_centers_source_in_target():
    """With no shift the source is centered in the larger target space.

    ``CreateRigidTransform`` centers the source, so the block is offset by half the
    size difference rather than staying at its source coordinates -- which is also
    why an unwarped return value is detectable here.
    """
    source = _source_image()
    warped = _warp(0, 0)

    source_y, source_x = _block_corner(source)
    warped_y, warped_x = _block_corner(warped)

    assert warped_y - source_y == (TARGET_H - SOURCE_H) // 2
    assert warped_x - source_x == (TARGET_W - SOURCE_W) // 2
    assert np.asarray(warped).shape == (TARGET_H, TARGET_W)


def test_saved_file_matches_returned_image(tmp_path):
    """What is written to disk is the warped image, not the source."""
    output_path = os.path.join(str(tmp_path), 'warped.png')
    warped = assemble.TransformStos(
        _translation_transform(),
        OutputFilename=output_path,
        fixedImage=np.zeros((TARGET_H, TARGET_W), dtype=np.float32),
        warpedImage=_source_image(),
        CropUndefined=False)

    assert os.path.exists(output_path)
    saved_size = nornir_imageregistration.GetImageSize(output_path)
    assert tuple(int(v) for v in saved_size) == (TARGET_H, TARGET_W)
    assert np.asarray(warped).shape == (TARGET_H, TARGET_W)


def test_image_paths_default_from_the_stos_file(tmp_path):
    """The ``fixedImage``/``warpedImage`` fallbacks read the stos file.

    Both branches were unreachable because ``stos`` was hardcoded to ``None``, so
    omitting either argument returned ``None`` instead of assembling.
    """
    directory = str(tmp_path)
    fixed_path = os.path.join(directory, 'fixed.png')
    moving_path = os.path.join(directory, 'moving.png')
    nornir_imageregistration.SaveImage(
        fixed_path, np.zeros((TARGET_H, TARGET_W), dtype=np.float32), bpp=8)
    nornir_imageregistration.SaveImage(moving_path, _source_image(), bpp=8)

    stos_path = _write_stos(directory, fixed_path, moving_path)

    warped = assemble.TransformStos(stos_path, CropUndefined=False)
    assert warped is not None, \
        'omitting both images returned None; the stos fallbacks are dead code'
    assert np.asarray(warped).shape == (TARGET_H, TARGET_W)


def test_stosfile_object_is_accepted():
    """A ``StosFile`` instance resolves to a transform.

    The branch called ``.transform`` on ``StosFile.Transform``, which is already a
    string, so passing a ``StosFile`` raised ``AttributeError``.
    """
    stos = nornir_imageregistration.StosFile()
    stos.Transform = _translation_transform()

    transform = assemble.ParameterToStosTransform(stos)
    assert transform is not None
    assert isinstance(transform, nornir_imageregistration.ITransform)


def test_scalar_scales_the_output():
    """``scalar`` scales the target shape the source is warped into."""
    warped = assemble.TransformStos(
        _translation_transform(),
        fixedImage=np.zeros((TARGET_H, TARGET_W), dtype=np.float32),
        warpedImage=_source_image(),
        scalar=2.0,
        CropUndefined=False)

    assert np.asarray(warped).shape == (TARGET_H * 2, TARGET_W * 2)
