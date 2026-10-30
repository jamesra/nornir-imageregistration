"""Tests for the STOS registration debug harness."""
from __future__ import annotations

import math
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import nornir_imageregistration.core as core
from nornir_imageregistration.alignment_record import AlignmentRecord
from nornir_imageregistration.files.stosfile import StosFile
from nornir_imageregistration.stos_registration_debug import (
    RC2_GRID16_239_240_MANUAL_STOS,
    RC2_GRID16_239_240_STOS,
    StosDebugAlgorithm,
    bbox_overlap_for_alignment,
    find_manual_reference_stos,
    is_no_peak_fallback,
    peak_exceeds_image_dimensions,
    pyre_logpolar_kwargs,
    rigid_report_for_alignment,
    run_stos_debug,
)
from nornir_imageregistration.transforms.rigid import Rigid, RigidTranslation

_MIN_TRANSFORM = (
    "FixedCenterOfRotationAffineTransform_double_2_2 vp 8 1 0 0 1 0 0 1 1 fp 2 2 2"
)


def _write_png(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    core.SaveImage(path, image)


def _identity_stos(stos_path: str, source_png: str, target_png: str) -> None:
    stos = StosFile.Create(
        target_image_fullpath=target_png,
        source_image_fullpath=source_png,
        transform=RigidTranslation(target_offset=(0.0, 0.0)),
    )
    stos.Transform = _MIN_TRANSFORM
    stos.Save(stos_path)


def _blob_pair(size: int = 64, shift: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Two images sharing a bright blob, with a known translation."""
    target = np.zeros((size, size), dtype=np.float32)
    cy = size // 2
    cx = size // 2
    target[cy - 8:cy + 8, cx - 8:cx + 8] = 1.0
    source = np.zeros_like(target)
    source[cy - 8 + shift:cy + 8 + shift, cx - 8 + shift:cx + 8 + shift] = 1.0
    return source, target


class TestStosRegistrationDebugHelpers(unittest.TestCase):
    """Helpers that do not load images or run registration."""

    def test_no_peak_fallback_on_zero_weight(self) -> None:
        record = AlignmentRecord(peak=(0.0, 0.0), weight=0.0, angle=-0.16)
        self.assertTrue(is_no_peak_fallback(record))
        self.assertFalse(is_no_peak_fallback(AlignmentRecord(peak=(1.0, 2.0), weight=1.0)))

    def test_peak_exceeds_image_dimensions(self) -> None:
        self.assertTrue(peak_exceeds_image_dimensions((5381.0, 5381.0), (100, 120), (110, 90)))
        self.assertFalse(peak_exceeds_image_dimensions((0.0, 0.0), (100, 120), (110, 90)))
        self.assertFalse(peak_exceeds_image_dimensions((50.0, 5381.0), (100, 120), (110, 90)))

    def test_bbox_overlap_identity_is_one(self) -> None:
        record = AlignmentRecord(peak=(0.0, 0.0), weight=1.0, angle=0.0)
        overlap = bbox_overlap_for_alignment(record, (10, 10), (10, 10))
        self.assertGreaterEqual(overlap, 0.99)
        self.assertLessEqual(overlap, 1.0)

    def test_bbox_overlap_huge_offset_is_zero(self) -> None:
        record = AlignmentRecord(peak=(5381.0, 5381.0), weight=0.0, angle=0.0)
        overlap = bbox_overlap_for_alignment(record, (100, 100), (100, 100))
        self.assertEqual(overlap, 0.0)

    def test_rigid_report_flags_no_peak_and_impossible_peak(self) -> None:
        record = AlignmentRecord(peak=(5381.0, 5381.0), weight=0.0, angle=-0.16)
        report = rigid_report_for_alignment(record, (200, 200), (200, 200))
        self.assertTrue(report.no_peak_fallback)
        self.assertTrue(report.peak_exceeds_image_dimensions)
        self.assertEqual(report.bbox_overlap, 0.0)

    def test_pyre_logpolar_preset(self) -> None:
        kwargs = pyre_logpolar_kwargs()
        self.assertEqual(kwargs["min_overlap"], 0.75)
        self.assertEqual(kwargs["larget_dimension"], 818)
        self.assertTrue(kwargs["try_flipped"])


class TestStosRegistrationDebugInspect(unittest.TestCase):
    """Load a synthetic STOS pair and inspect without registering."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = self._tmpdir.name
        grid_dir = os.path.join(self.root, "Grid16")
        os.makedirs(grid_dir, exist_ok=True)
        source, target = _blob_pair(size=32, shift=2)
        self.source_png = os.path.join(grid_dir, "source.png")
        self.target_png = os.path.join(grid_dir, "target.png")
        _write_png(self.source_png, source)
        _write_png(self.target_png, target)
        self.stos_path = os.path.join(grid_dir, "pair.stos")
        _identity_stos(self.stos_path, self.source_png, self.target_png)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_inspect_returns_shapes_and_does_not_register(self) -> None:
        with mock.patch(
                "nornir_imageregistration.stos_registration_debug."
                "SliceToSliceRigidRegistrationWithPreprocessedImages") as mocked:
            result = run_stos_debug(self.stos_path, StosDebugAlgorithm.inspect)
            mocked.assert_not_called()

        self.assertIsNone(result.alignment)
        self.assertIsNone(result.rigid)
        self.assertEqual(result.inspect.source_shape, (32, 32))
        self.assertEqual(result.inspect.target_shape, (32, 32))
        self.assertEqual(result.inspect.downsample_hint, 16)
        self.assertIsNotNone(result.inspect.stored_transform_type)
        self.assertGreater(len(result.inspect.overlap_masks), 0)
        for mask_stats in result.inspect.overlap_masks:
            self.assertGreaterEqual(mask_stats.eligible_fraction, 0.0)
            self.assertLessEqual(mask_stats.eligible_fraction, 1.0)

    def test_unknown_kwarg_raises(self) -> None:
        with self.assertRaises(TypeError) as raised:
            run_stos_debug(self.stos_path, StosDebugAlgorithm.inspect, not_a_real_kwarg=1)
        self.assertIn("not_a_real_kwarg", str(raised.exception))
        self.assertIn("Accepted:", str(raised.exception))

    def test_largest_dimension_alias_is_accepted(self) -> None:
        result = run_stos_debug(
            self.stos_path, StosDebugAlgorithm.inspect, largest_dimension=818)
        dims = {mask.larget_dimension for mask in result.inspect.overlap_masks}
        self.assertIn(818, dims)
        self.assertIn(None, dims)

    def test_auto_discovers_manual_reference_stos(self) -> None:
        manual_dir = os.path.join(os.path.dirname(self.stos_path), "Manual")
        os.makedirs(manual_dir, exist_ok=True)
        manual_path = os.path.join(manual_dir, os.path.basename(self.stos_path))
        center = (15.5, 15.5)
        stos = StosFile.Create(
            target_image_fullpath=self.target_png,
            source_image_fullpath=self.source_png,
            transform=Rigid(
                target_offset=(4.0, 3.0),
                source_rotation_center=center,
                angle=math.radians(12.0),
            ),
        )
        stos.Save(manual_path)

        self.assertEqual(find_manual_reference_stos(self.stos_path), os.path.abspath(manual_path))
        result = run_stos_debug(self.stos_path, StosDebugAlgorithm.inspect)
        self.assertIsNotNone(result.reference)
        assert result.reference is not None
        self.assertTrue(result.reference.auto_discovered)
        self.assertIsNotNone(result.reference.angle_deg)
        self.assertAlmostEqual(result.reference.angle_deg or 0.0, 12.0, delta=0.25)


class TestStosRegistrationDebugLogpolar(unittest.TestCase):
    """Log-polar on a tiny synthetic pair returns an AlignmentRecord."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = self._tmpdir.name
        source, target = _blob_pair(size=64, shift=3)
        self.source_png = os.path.join(self.root, "source.png")
        self.target_png = os.path.join(self.root, "target.png")
        _write_png(self.source_png, source)
        _write_png(self.target_png, target)
        self.stos_path = os.path.join(self.root, "pair.stos")
        _identity_stos(self.stos_path, self.source_png, self.target_png)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_logpolar_returns_alignment_and_bbox_in_unit_interval(self) -> None:
        result = run_stos_debug(
            self.stos_path,
            StosDebugAlgorithm.logpolar,
            min_overlap=0.25,
            larget_dimension=None,
            try_flipped=False,
        )
        self.assertIsNotNone(result.alignment)
        self.assertIsNotNone(result.rigid)
        assert result.rigid is not None
        self.assertGreaterEqual(result.rigid.bbox_overlap, 0.0)
        self.assertLessEqual(result.rigid.bbox_overlap, 1.0)
        self.assertEqual(result.rigid.peak_yx[0], result.alignment.peak[0])  # type: ignore[union-attr]


@unittest.skipUnless(
    os.path.isfile(RC2_GRID16_239_240_STOS),
    "RC2 Grid16 239-240 STOS is not present",
)
class TestStosRegistrationDebugRc2Live(unittest.TestCase):
    """Optional live inspect of the RC2 239-240 pair; skipped when the volume is absent."""

    def test_inspect_rc2_239_240(self) -> None:
        result = run_stos_debug(
            RC2_GRID16_239_240_STOS,
            StosDebugAlgorithm.inspect,
            min_overlap=0.75,
            larget_dimension=818,
        )
        self.assertEqual(result.inspect.downsample_hint, 16)
        self.assertEqual(len(result.inspect.source_shape), 2)
        fractions = {
            (mask.min_overlap, mask.larget_dimension): mask.eligible_fraction
            for mask in result.inspect.overlap_masks
        }
        self.assertIn((0.75, None), fractions)
        self.assertIn((0.75, 818), fractions)
        self.assertIn((0.5, None), fractions)
        self.assertIn((0.25, None), fractions)
        if os.path.isfile(RC2_GRID16_239_240_MANUAL_STOS):
            self.assertIsNotNone(result.reference)
            assert result.reference is not None
            self.assertTrue(result.reference.auto_discovered)
            self.assertEqual(
                os.path.normcase(result.reference.stos_path),
                os.path.normcase(os.path.abspath(RC2_GRID16_239_240_MANUAL_STOS)),
            )


if __name__ == "__main__":
    unittest.main()
