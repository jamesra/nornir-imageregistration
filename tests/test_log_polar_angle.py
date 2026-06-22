"""Tests for log-polar slice-to-slice angle recovery and 180-degree disambiguation."""

from __future__ import annotations

import os
import unittest
from typing import Sequence

import numpy as np
import scipy.ndimage

import nornir_imageregistration
import nornir_imageregistration.stos_brute as stos_brute
from nornir_imageregistration.files.stosfile import StosFile
from nornir_imageregistration.settings import SliceToSliceMethod
from nornir_imageregistration.transforms import LoadTransform
from mathfuncs.angles import assert_angles_equal_degrees

import setup_imagetest


def _wrap_angle_diff(measured: float, expected: float) -> float:
    """Return measured minus expected in (-180, 180] degrees."""
    diff = float(measured) - float(expected)
    while diff > 180.0:
        diff -= 360.0
    while diff <= -180.0:
        diff += 360.0
    return diff


def _create_rotated_and_offset_image(
    image: np.ndarray | str,
    mask: np.ndarray | str,
    angle: float,
    offset: tuple[int, int],
) -> nornir_imageregistration.ImagePermutationHelper:
    """Build a synthetic warped image by rotating and translating a source image."""
    input_image_data = nornir_imageregistration.ImagePermutationHelper(image, mask)

    if angle != 0:
        rotated_image = scipy.ndimage.rotate(
            input_image_data.Image.astype(np.float32), -angle, reshape=False
        )
        rotated_mask = scipy.ndimage.rotate(input_image_data.Mask, -angle, reshape=False)
    else:
        rotated_image = input_image_data.Image
        rotated_mask = input_image_data.Mask

    rotated_translated_image = nornir_imageregistration.CropImage(
        rotated_image,
        Xo=-offset[1],
        Yo=-offset[0],
        Width=input_image_data.Image.shape[1],
        Height=input_image_data.Image.shape[0],
        image_stats=input_image_data.Stats,
    )
    rotated_translated_mask = nornir_imageregistration.CropImage(
        rotated_mask,
        Xo=-offset[1],
        Yo=-offset[0],
        Width=input_image_data.Image.shape[1],
        Height=input_image_data.Image.shape[0],
        image_stats=input_image_data.Stats,
    )
    return nornir_imageregistration.ImagePermutationHelper(
        rotated_translated_image, rotated_translated_mask
    )


def _idoc_section016_paths(test_output_root: str | None = None) -> dict[str, str] | None:
    """Return IDoc 690/691 ds16 paths from bundled fixtures or repro output.

    With center section 693, section 690 is mapped (moving) and 691 is control (target).
    """
    candidates: list[str] = [
        os.path.join(os.path.dirname(__file__), "fixtures", "idoc_690_691"),
    ]
    if test_output_root is not None:
        candidates.append(
            os.path.join(test_output_root, "IDocBuildTestBootstrapDebugging", "TEM")
        )

    for base in candidates:
        paths = {
            "mapped_image": os.path.join(
                base, "0690", "TEM", "Leveled", "Images", "016", "0690_TEM_Leveled.png"
            ),
            "mapped_mask": os.path.join(
                base, "0690", "TEM", "Mask", "Images", "016", "0690_TEM_Mask.png"
            ),
            "control_image": os.path.join(
                base, "0691", "TEM", "Leveled", "Images", "016", "0691_TEM_Leveled.png"
            ),
            "control_mask": os.path.join(
                base, "0691", "TEM", "Mask", "Images", "016", "0691_TEM_Mask.png"
            ),
            "reference_stos": os.path.join(
                base,
                "StosBrute16",
                "690-691_ctrl-TEM_Leveled_map-TEM_Leveled.stos",
            ),
        }
        if all(os.path.isfile(p) for p in paths.values()):
            return paths
    return None


class TestLogPolarAngleConvention(setup_imagetest.ImageTestBase):
    """Synthetic and IDoc regression tests for log-polar angle estimation."""

    def setUp(self) -> None:
        super().setUp()
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy
        )
        self._warped_image_path = self.GetImagePath(
            "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png"
        )
        self._warped_mask_path = self.GetImagePath(
            "0017_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png"
        )

    def _run_log_polar(
        self,
        computation_lib: nornir_imageregistration.ComputationLib,
        source_image_data: nornir_imageregistration.ImagePermutationHelper,
        target_image_data: nornir_imageregistration.ImagePermutationHelper,
        min_overlap: float = 0.5,
    ) -> stos_brute.AngleScaleResult:
        """Run log-polar angle recovery with the requested computation backend."""
        nornir_imageregistration.SetActiveComputationLib(computation_lib)
        source_image = source_image_data.ImageWithMaskAsNoise
        target_image = target_image_data.ImageWithMaskAsNoise
        if computation_lib == nornir_imageregistration.ComputationLib.cupy:
            import cupy as cp  # type: ignore[import-not-found]

            source_image = cp.asarray(source_image)
            target_image = cp.asarray(target_image)
        return stos_brute._find_angle_and_scale_with_logpolar(
            source_image=source_image,
            target_image=target_image,
            source_stats=source_image_data.Stats,
            target_stats=target_image_data.Stats,
            min_overlap=min_overlap,
        )

    def _assert_angle_near(
        self,
        measured: float,
        expected: float,
        tolerance: float,
        msg: str,
    ) -> None:
        """Assert two angles match within tolerance using wrapped difference."""
        diff = abs(_wrap_angle_diff(measured, expected))
        self.assertLessEqual(
            diff,
            tolerance,
            f"{msg}: measured={measured:.3f} expected={expected:.3f} diff={diff:.3f}",
        )

    def _test_synthetic_angle(
        self,
        angle: float,
        tolerance: float,
        computation_lib: nornir_imageregistration.ComputationLib,
    ) -> None:
        """Shared synthetic rotation check for one angle."""
        source_image_data = nornir_imageregistration.ImagePermutationHelper(
            self._warped_image_path, self._warped_mask_path
        )
        # Synthetic ground-truth images are always built on CPU; scipy.ndimage.rotate
        # cannot consume CuPy arrays from ImagePermutationHelper under a GPU backend.
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy
        )
        target_image_data = _create_rotated_and_offset_image(
            self._warped_image_path, self._warped_mask_path, angle, (0, 0)
        )
        result = self._run_log_polar(computation_lib, source_image_data, target_image_data)
        self._assert_angle_near(
            result.angle,
            angle,
            tolerance,
            f"synthetic log-polar angle (lib={computation_lib.name})",
        )
        self.assertAlmostEqual(result.scale, 1.0, delta=0.15)

    def test_synthetic_angle_sweep_numpy(self) -> None:
        """Log-polar should track known synthetic rotations within tolerance."""
        cases: Sequence[tuple[float, float]] = (
            (0.0, 1.0),
            (45.0, 1.5),
            (90.0, 3.5),
            (132.0, 2.0),
            (-90.0, 3.5),
            (180.0, 1.5),
        )
        for angle, tolerance in cases:
            with self.subTest(angle=angle):
                self._test_synthetic_angle(
                    angle,
                    tolerance,
                    nornir_imageregistration.ComputationLib.numpy,
                )

    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "CuPy not available")
    def test_synthetic_angle_sweep_cupy(self) -> None:
        """CuPy path should match NumPy log-polar angle recovery on synthetic data."""
        for angle in (0.0, 45.0, 90.0, 132.0):
            with self.subTest(angle=angle):
                self._test_synthetic_angle(
                    angle,
                    3.5,
                    nornir_imageregistration.ComputationLib.cupy,
                )

    def test_idoc_690_691_matches_stos_brute_reference(self) -> None:
        """690 registers into 691; angle should match the StosBrute16 .stos transform."""
        paths = _idoc_section016_paths(os.environ.get("TESTOUTPUTPATH"))
        if paths is None:
            self.skipTest("IDoc 690/691 section 016 fixtures not found")

        mapped_image_data = nornir_imageregistration.ImagePermutationHelper(
            paths["mapped_image"], paths["mapped_mask"]
        )
        control_image_data = nornir_imageregistration.ImagePermutationHelper(
            paths["control_image"], paths["control_mask"]
        )

        reference_stos = StosFile.Load(paths["reference_stos"])
        reference_transform = LoadTransform(reference_stos.Transform)
        reference_angle = float(np.degrees(reference_transform.angle))
        self.assertEqual(
            os.path.basename(reference_stos.MappedImageFullPath),
            "0690_TEM_Leveled.png",
        )
        self.assertEqual(
            os.path.basename(reference_stos.ControlImageFullPath),
            "0691_TEM_Leveled.png",
        )

        pipeline_settings = nornir_imageregistration.settings.StosBruteSettings(
            min_overlap=0.75,
            method=SliceToSliceMethod.LogPolar,
            try_flipped=True,
            larget_dimension=818,
        )
        rigid_result = stos_brute.SliceToSliceRigidRegistrationWithPreprocessedImages(
            source_image_data=mapped_image_data,
            target_image_data=control_image_data,
            settings=pipeline_settings,
            SingleThread=True,
        )
        self._assert_angle_near(
            rigid_result.angle,
            reference_angle,
            tolerance=0.5,
            msg="690->691 log-polar (pipeline flip check) vs StosBrute16",
        )
    @unittest.skipUnless(nornir_imageregistration.HasCupy(), "CuPy not available")
    def test_idoc_690_691_cupy_matches_reference(self) -> None:
        """GPU log-polar path should match StosBrute16 for 690 mapped into 691 control."""
        paths = _idoc_section016_paths(os.environ.get("TESTOUTPUTPATH"))
        if paths is None:
            self.skipTest("IDoc 690/691 section 016 fixtures not found")

        mapped_image_data = nornir_imageregistration.ImagePermutationHelper(
            paths["mapped_image"], paths["mapped_mask"]
        )
        control_image_data = nornir_imageregistration.ImagePermutationHelper(
            paths["control_image"], paths["control_mask"]
        )
        reference_stos = StosFile.Load(paths["reference_stos"])
        reference_angle = float(np.degrees(LoadTransform(reference_stos.Transform).angle))

        result = self._run_log_polar(
            nornir_imageregistration.ComputationLib.cupy,
            mapped_image_data,
            control_image_data,
            min_overlap=0.75,
        )
        self._assert_angle_near(
            result.angle,
            reference_angle,
            tolerance=3.0,
            msg="690->691 CuPy log-polar vs StosBrute16 reference",
        )

    def test_no_constant_quadrature_offset(self) -> None:
        """Angle error should stay small across a coarse sweep (guards ~90-degree bias)."""
        source_image_data = nornir_imageregistration.ImagePermutationHelper(
            self._warped_image_path, self._warped_mask_path
        )
        for angle in range(-150, 151, 30):
            with self.subTest(angle=angle):
                target_image_data = _create_rotated_and_offset_image(
                    self._warped_image_path, self._warped_mask_path, angle, (0, 0)
                )
                result = self._run_log_polar(
                    nornir_imageregistration.ComputationLib.numpy,
                    source_image_data,
                    target_image_data,
                )
                diff = _wrap_angle_diff(result.angle, angle)
                self.assertLessEqual(
                    abs(diff),
                    4.0,
                    f"Unexpected quadrature offset at angle={angle}: diff={diff:.2f}",
                )


class TestLogPolarKnownRotationExtended(setup_imagetest.ImageTestBase):
    """Extend existing known-rotation checks with explicit 90-degree cases."""

    def setUp(self) -> None:
        super().setUp()
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy
        )
        self._warped_image_path = self.GetImagePath(
            "0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png"
        )
        self._warped_mask_path = self.GetImagePath(
            "0017_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png"
        )

    def _check_known_rotation(self, angle: float, tolerance: float) -> None:
        """Verify log-polar and full pipeline recover a synthetic rotation."""
        nornir_imageregistration.SetActiveComputationLib(
            nornir_imageregistration.ComputationLib.numpy
        )
        source_image_data = nornir_imageregistration.ImagePermutationHelper(
            self._warped_image_path, self._warped_mask_path
        )
        target_image_data = _create_rotated_and_offset_image(
            self._warped_image_path, self._warped_mask_path, angle, (34, 100)
        )
        results = stos_brute._find_angle_and_scale_with_logpolar(
            source_image=source_image_data.ImageWithMaskAsNoise,
            target_image=target_image_data.ImageWithMaskAsNoise,
            source_stats=source_image_data.Stats,
            target_stats=target_image_data.Stats,
        )
        assert_angles_equal_degrees(
            self, results.angle, angle, tolerance=tolerance, msg="Angle mismatch"
        )

        settings = nornir_imageregistration.settings.StosBruteSettings(
            min_overlap=0.5,
            method=SliceToSliceMethod.LogPolar,
            try_flipped=False,
        )
        rigid_results = stos_brute.SliceToSliceRigidRegistrationWithPreprocessedImages(
            source_image_data=source_image_data,
            target_image_data=target_image_data,
            settings=settings,
            SingleThread=True,
        )
        assert_angles_equal_degrees(
            self, rigid_results.angle, angle, tolerance=tolerance, msg="Rigid angle mismatch"
        )

    def test_known_rotation_90_degrees(self) -> None:
        self._check_known_rotation(angle=90.0, tolerance=3.5)

    def test_known_rotation_neg90_degrees(self) -> None:
        self._check_known_rotation(angle=-90.0, tolerance=3.5)

    def test_known_rotation_132_degrees(self) -> None:
        self._check_known_rotation(angle=132.0, tolerance=2.0)
