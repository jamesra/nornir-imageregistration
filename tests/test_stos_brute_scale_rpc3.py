"""Scale detection/correction regression on RPC3 Brute64 STOS pairs (INPUT_NORNIR_DATA)."""

from __future__ import annotations

import os
import unittest

import nornir_imageregistration
import nornir_imageregistration.stos_brute as stos_brute
from nornir_imageregistration.settings import SliceToSliceMethod

_STOS_BRUTE_SUBDIR = os.environ.get('NORNIR_STOS_BRUTE_DIR', 'Brute64')


def _input_nornir_join(*relative: str) -> str:
    root = os.environ.get('INPUT_NORNIR_DATA', '').strip()
    if not root:
        raise EnvironmentError('INPUT_NORNIR_DATA is not set')
    return os.path.join(root, *relative)


def _rpc3_stos_path() -> str:
    return _input_nornir_join(
        'RPC3', 'TEM', _STOS_BRUTE_SUBDIR, '86-87_ctrl_TEM_blob_map-TEM_blob.stos')


def _rpc3_fixture_available() -> bool:
    if 'INPUT_NORNIR_DATA' not in os.environ:
        return False
    try:
        return os.path.isfile(_rpc3_stos_path())
    except EnvironmentError:
        return False


@unittest.skipUnless(_rpc3_fixture_available(), 'RPC3 86-87 Brute64 STOS not available under INPUT_NORNIR_DATA')
class TestStosBruteScaleRpc38687(unittest.TestCase):
    """Regression tests for scale-aware STOS brute on a known RPC3 section pair."""

    @classmethod
    def setUpClass(cls):
        cls.stos_obj = nornir_imageregistration.files.stosfile.StosFile.Load(_rpc3_stos_path())
        cls.fixed_image = cls.stos_obj.ControlImageFullPath
        cls.warped_image = cls.stos_obj.MappedImageFullPath
        cls.fixed_mask = cls.stos_obj.ControlMaskFullPath
        cls.warped_mask = cls.stos_obj.MappedMaskFullPath

    def test_logpolar_reports_finite_scale(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        source = nornir_imageregistration.ImagePermutationHelper(self.warped_image, self.warped_mask)
        target = nornir_imageregistration.ImagePermutationHelper(self.fixed_image, self.fixed_mask)
        result = stos_brute._find_angle_and_scale_with_logpolar(
            source_image=source.ImageWithMaskAsNoise,
            target_image=target.ImageWithMaskAsNoise,
            source_stats=source.Stats,
            target_stats=target.Stats,
        )
        self.assertTrue(result.scale > 0.0)
        self.assertLess(result.scale, 2.0)

    def test_logpolar_registration_preserves_scale_on_record(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        record = stos_brute.SliceToSliceRigidRegistration(
            target_image=self.fixed_image,
            source_image=self.warped_image,
            target_mask=self.fixed_mask,
            source_mask=self.warped_mask,
            LargestDimension=1024,
            TestFlip=False,
            method=SliceToSliceMethod.LogPolar,
        )
        self.assertTrue(record.scale > 0.0)
        self.assertLess(record.scale, 2.0)
        transform = record.ToImageTransform(
            target_image_shape=nornir_imageregistration.core.GetImageSize(self.fixed_image),
            source_image_shape=nornir_imageregistration.core.GetImageSize(self.warped_image),
        )
        if not __import__('numpy').isclose(record.scale, 1.0):
            self.assertTrue(hasattr(transform, 'scalar'))
            self.assertFalse(__import__('numpy').isclose(transform.scalar, 1.0))  # type: ignore[attr-defined]

    def test_logpolar_scale_near_reference_transform(self):
        """Phase A9: log-polar scale should agree with the reference STOS transform scalar."""
        import numpy as np
        from nornir_imageregistration.transforms.factory import LoadTransform

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        reference_transform = LoadTransform(self.stos_obj.Transform)
        reference_scalar = float(getattr(reference_transform, 'scalar', 1.0))
        source = nornir_imageregistration.ImagePermutationHelper(self.warped_image, self.warped_mask)
        target = nornir_imageregistration.ImagePermutationHelper(self.fixed_image, self.fixed_mask)
        result = stos_brute._find_angle_and_scale_with_logpolar(
            source_image=source.ImageWithMaskAsNoise,
            target_image=target.ImageWithMaskAsNoise,
            source_stats=source.Stats,
            target_stats=target.Stats,
        )
        self.assertAlmostEqual(result.scale, reference_scalar, delta=0.05)

    def test_brute_force_with_scale_hint(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        record = stos_brute.SliceToSliceRigidRegistration(
            target_image=self.fixed_image,
            source_image=self.warped_image,
            target_mask=self.fixed_mask,
            source_mask=self.warped_mask,
            LargestDimension=1024,
            TestFlip=False,
            estimate_angle=True,
            method=SliceToSliceMethod.BruteForce,
        )
        self.assertGreater(record.weight, 0.0)


class TestSyntheticScaleDetection(unittest.TestCase):
    """Synthetic tissue-shrink detection without metadata (no INPUT_NORNIR_DATA)."""

    def test_logpolar_detects_isotropic_shrink(self):
        import numpy as np
        import scipy.ndimage

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        rng = np.random.default_rng(42)
        base = rng.standard_normal((256, 256)).astype(np.float32)
        base -= base.min()
        base /= base.max() + 1e-6
        shrink = 0.97
        warped = scipy.ndimage.zoom(base, shrink, order=1)
        pad_y = (base.shape[0] - warped.shape[0]) // 2
        pad_x = (base.shape[1] - warped.shape[1]) // 2
        source = np.zeros_like(base)
        source[pad_y:pad_y + warped.shape[0], pad_x:pad_x + warped.shape[1]] = warped

        source_h = nornir_imageregistration.ImagePermutationHelper(source)
        target_h = nornir_imageregistration.ImagePermutationHelper(base)
        result = stos_brute._find_angle_and_scale_with_logpolar(
            source_image=source_h.ImageWithMaskAsNoise,
            target_image=target_h.ImageWithMaskAsNoise,
            source_stats=source_h.Stats,
            target_stats=target_h.Stats,
        )
        self.assertAlmostEqual(result.scale, shrink, delta=0.10)
        self.assertAlmostEqual(result.angle, 0.0, delta=3.0)

    def test_scale_search_candidates_centers_on_user_hint(self):
        candidates = stos_brute._scale_search_candidates(1.0, 0.97, force_search=True)
        self.assertIn(0.97, candidates)
        self.assertTrue(any(abs(c - 0.97) <= 0.021 for c in candidates))

    def test_resolve_scale_search_params_user_beats_logpolar(self):
        from nornir_imageregistration.settings import StosBruteSettings

        settings = StosBruteSettings(initial_scale_hint=1.03)
        residual, force = stos_brute._resolve_scale_search_params(settings, 1.0, logpolar_residual_scale=0.91)
        self.assertAlmostEqual(residual, 1.03)
        self.assertTrue(force)

    def test_brute_force_with_initial_scale_hint(self):
        import numpy as np
        import scipy.ndimage

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        rng = np.random.default_rng(7)
        base = rng.standard_normal((256, 256)).astype(np.float32)
        base -= base.min()
        base /= base.max() + 1e-6
        shrink = 0.97
        warped = scipy.ndimage.zoom(base, shrink, order=1)
        pad_y = (base.shape[0] - warped.shape[0]) // 2
        pad_x = (base.shape[1] - warped.shape[1]) // 2
        source = np.zeros_like(base)
        source[pad_y:pad_y + warped.shape[0], pad_x:pad_x + warped.shape[1]] = warped

        record = stos_brute.SliceToSliceRigidRegistration(
            target_image=base,
            source_image=source,
            LargestDimension=256,
            TestFlip=False,
            estimate_angle=False,
            method=SliceToSliceMethod.BruteForce,
            initial_scale_hint=shrink,
        )
        self.assertAlmostEqual(record.scale, shrink, delta=0.06)
        self.assertGreater(record.weight, 0.0)

    def test_logpolar_with_initial_scale_hint(self):
        import numpy as np
        import scipy.ndimage

        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        rng = np.random.default_rng(11)
        base = rng.standard_normal((256, 256)).astype(np.float32)
        base -= base.min()
        base /= base.max() + 1e-6
        shrink = 0.97
        warped = scipy.ndimage.zoom(base, shrink, order=1)
        pad_y = (base.shape[0] - warped.shape[0]) // 2
        pad_x = (base.shape[1] - warped.shape[1]) // 2
        source = np.zeros_like(base)
        source[pad_y:pad_y + warped.shape[0], pad_x:pad_x + warped.shape[1]] = warped

        record = stos_brute.SliceToSliceRigidRegistration(
            target_image=base,
            source_image=source,
            LargestDimension=256,
            TestFlip=False,
            method=SliceToSliceMethod.LogPolar,
            initial_scale_hint=shrink,
        )
        self.assertAlmostEqual(record.scale, shrink, delta=0.06)
        self.assertGreater(record.weight, 0.0)


@unittest.skipUnless(_rpc3_fixture_available(), 'RPC3 86-87 Brute64 STOS not available under INPUT_NORNIR_DATA')
class TestStosBruteScaleRpc38687BruteForce(unittest.TestCase):
    """BruteForce scale path on RPC3 86-87 when corpus is available."""

    @classmethod
    def setUpClass(cls):
        cls.stos_obj = nornir_imageregistration.files.stosfile.StosFile.Load(_rpc3_stos_path())
        cls.fixed_image = cls.stos_obj.ControlImageFullPath
        cls.warped_image = cls.stos_obj.MappedImageFullPath
        cls.fixed_mask = cls.stos_obj.ControlMaskFullPath
        cls.warped_mask = cls.stos_obj.MappedMaskFullPath

    def test_brute_force_registration_preserves_finite_scale(self):
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        record = stos_brute.SliceToSliceRigidRegistration(
            target_image=self.fixed_image,
            source_image=self.warped_image,
            target_mask=self.fixed_mask,
            source_mask=self.warped_mask,
            LargestDimension=1024,
            TestFlip=False,
            estimate_angle=False,
            method=SliceToSliceMethod.BruteForce,
            initial_scale_hint=1.0,
        )
        self.assertTrue(record.scale > 0.0)
        self.assertLess(record.scale, 2.0)
        self.assertGreater(record.weight, 0.0)


if __name__ == '__main__':
    unittest.main()
