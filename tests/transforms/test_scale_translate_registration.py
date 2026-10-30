"""Prove ScoreOneAngle peak + ToImageTransform stay consistent when scale ≠ 1.

If scale were applied about the origin in CenteredSimilarity, mapping the source
center after a known isotropic zoom would be wrong by about (s-1)*c pixels.
"""
from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage

import nornir_imageregistration
from nornir_imageregistration import stos_brute


def _gaussian_blob(shape: tuple[int, int], center_yx: tuple[float, float], sigma: float = 6.0) -> np.ndarray:
    """Return a float32 image with a single Gaussian blob."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    cy, cx = center_yx
    img = np.exp(-(((yy - cy) ** 2) + ((xx - cx) ** 2)) / (2.0 * sigma * sigma))
    return np.maximum(img, 1e-6).astype(np.float32)


def _shrink_about_center(image: np.ndarray, scale: float) -> np.ndarray:
    """Shrink content about the array center, then pad/crop back to the original shape."""
    assert scale > 0.0
    zoomed = ndimage.zoom(image, 1.0 / scale, order=1)
    out = np.full(image.shape, float(np.median(image)), dtype=np.float32)
    # Place zoomed content so geometric centers coincide.
    src_c = (np.asarray(zoomed.shape, dtype=float) - 1.0) / 2.0
    dst_c = (np.asarray(image.shape, dtype=float) - 1.0) / 2.0
    # Destination slice for the overlapping region.
    y0 = int(round(dst_c[0] - src_c[0]))
    x0 = int(round(dst_c[1] - src_c[1]))
    y1 = y0 + zoomed.shape[0]
    x1 = x0 + zoomed.shape[1]
    sy0 = max(0, -y0)
    sx0 = max(0, -x0)
    dy0 = max(0, y0)
    dx0 = max(0, x0)
    sy1 = sy0 + min(zoomed.shape[0] - sy0, out.shape[0] - dy0)
    sx1 = sx0 + min(zoomed.shape[1] - sx0, out.shape[1] - dx0)
    dy1 = dy0 + (sy1 - sy0)
    dx1 = dx0 + (sx1 - sx0)
    out[dy0:dy1, dx0:dx1] = zoomed[sy0:sy1, sx0:sx1]
    return out


class TestScaleTranslateRegistrationPath(unittest.TestCase):
    """Registration peak must not absorb origin-scale bias when writing CenteredSimilarity."""

    def test_score_one_angle_scale_then_to_image_transform_keeps_center(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

        shape = (128, 128)
        scale = 1.05
        translate_yx = np.asarray([4.0, -3.0], dtype=float)
        target_center = (np.asarray(shape, dtype=float) - 1.0) / 2.0
        blob_center = target_center + translate_yx

        target = _gaussian_blob(shape, (float(blob_center[0]), float(blob_center[1])))
        # Source content is the unscaled pattern at the image center; ScoreOneAngle zooms by *scale*.
        source_pattern = _gaussian_blob(shape, (float(target_center[0]), float(target_center[1])))
        source = _shrink_about_center(source_pattern, scale)

        target_stats = nornir_imageregistration.ImageStats.CalcStats(target)
        source_stats = nornir_imageregistration.ImageStats.CalcStats(source)

        scored = stos_brute.ScoreOneAngle(
            target_original=target,
            source_original=source,
            target_image_shape=shape,
            source_image_shape=shape,
            angle=0.0,
            target_stats=target_stats,
            source_stats=source_stats,
            target_image_prepadded=False,
            min_overlap=0.5,
            source_scale=scale,
        )

        record = nornir_imageregistration.AlignmentRecord(
            peak=scored.peak,
            weight=scored.weight,
            angle=0.0,
            scale=scale,
        )
        transform = record.ToImageTransform(shape, shape)

        mapped_center = nornir_imageregistration.EnsureNumpyArray(
            transform.Transform(target_center.reshape(1, 2)))[0]
        expected_center = target_center + np.asarray(record.peak, dtype=float)

        # Centered similarity: Transform(c) == c + t with t = peak for equal shapes.
        np.testing.assert_allclose(mapped_center, expected_center, atol=1e-3)

        # Peak should recover the known translate; allow a few pixels of correlation blur.
        peak = np.asarray(record.peak, dtype=float)
        peak_err = float(np.linalg.norm(peak - translate_yx))
        origin_scale_bias = abs(scale - 1.0) * float(np.linalg.norm(target_center))
        self.assertLess(
            peak_err,
            max(3.0, 0.25 * origin_scale_bias),
            msg=(
                f"peak {peak} vs expected translate {translate_yx}; "
                f"err={peak_err:.2f} should be << origin-scale bias {origin_scale_bias:.1f}"
            ),
        )

        # Mapped source center should land near the blob in target space.
        placement_err = float(np.linalg.norm(mapped_center - blob_center))
        self.assertLess(
            placement_err,
            max(3.0, 0.25 * origin_scale_bias),
            msg=(
                f"mapped center {mapped_center} vs blob {blob_center}; "
                f"err={placement_err:.2f} should be << origin-scale bias {origin_scale_bias:.1f}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
