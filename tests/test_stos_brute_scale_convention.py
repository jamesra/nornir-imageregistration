"""StosBrute scale seeds use ScoreOneAngle's source_scale convention, and the measurement wins.

Two faults kept RC3 cross-camera pairs at exactly the metadata scale ratio (2026-11-02):

1. The log-polar radial estimate is the source's size relative to the target (a source
   shrunk to 0.92 reads 0.92), but it was handed on unchanged as a ``source_scale``,
   which is the factor that resamples the source onto the target (1/0.92 = 1.087).
   Near 1.0 the two barely differ; across RC3's Gatan/OneView cameras (about 5%) the
   seed went the wrong way. RC3 153-154 was seeded at 1.086 against a true 0.928.
2. The final refine started from the metadata default (residual 1.0) instead of the
   scale the search measured, and that refine only reaches a few percent, so it
   returned exactly the metadata ratio (266 of 766 RC3 brute results).

The image is the bundled real-tissue section (690, ds16); a synthetic ring pattern
does not move the radial estimate enough to tell the two conventions apart.
"""
from __future__ import annotations

import os
import unittest

import nornir_imageregistration
import numpy as np
from nornir_imageregistration import stos_brute as sb
from nornir_imageregistration.settings import StosBruteSettings

_SECTION = os.path.join(os.path.dirname(__file__), 'fixtures', 'idoc_690_691', '0690', 'TEM', 'Leveled',
                        'Images', '016', '0690_TEM_Leveled.png')


def _target() -> np.ndarray:
    return np.asarray(nornir_imageregistration.LoadImage(_SECTION, dtype=np.float32))


def _pair(source: np.ndarray, target: np.ndarray) -> sb._RegistrationPair:
    return sb._RegistrationPair(source, target,
                                nornir_imageregistration.ImageStats.CalcStats(source),
                                nornir_imageregistration.ImageStats.CalcStats(target), 0.5)


def _source_scaled_by(target: np.ndarray, size: float) -> np.ndarray:
    """The target resampled to *size* times its size; aligning it needs source_scale 1/size."""
    return np.asarray(nornir_imageregistration.ScaleImage(target, size), dtype=np.float32)


class _NumpyCase(unittest.TestCase):

    def setUp(self) -> None:
        self._lib = nornir_imageregistration.GetActiveComputationLib()
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)

    def tearDown(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(self._lib)


class TestLogPolarScaleConvention(_NumpyCase):

    def test_score_one_angle_aligns_at_the_inverse_size(self):
        """Pins the convention the log-polar result must follow."""
        target = _target()
        pair = _pair(_source_scaled_by(target, 0.92), target)

        def weight(scale: float) -> float:
            return float(sb.ScoreOneAngle(
                target_original=pair.target_image, source_original=pair.source_image,
                target_image_shape=pair.target_image.shape, source_image_shape=pair.source_image.shape,
                angle=0.0, target_stats=pair.target_stats, source_stats=pair.source_stats,
                min_overlap=0.5, source_scale=scale).weight)

        self.assertGreater(weight(1 / 0.92), 3 * weight(0.92))

    def test_logpolar_result_is_a_source_scale(self):
        target = _target()
        for size in (0.92, 0.95, 1.05, 1.08):
            with self.subTest(size=size):
                pair = _pair(_source_scaled_by(target, size), target)
                result = sb._find_angle_and_scale_with_logpolar(
                    pair.source_image, pair.target_image, pair.source_stats, pair.target_stats,
                    min_overlap=0.5)
                self.assertAlmostEqual(result.scale, 1 / size, delta=0.02,
                                       msg=f'{result.scale} is on the wrong side of 1.0 for size {size}')

    def test_raw_radial_estimate_is_still_the_size(self):
        """The estimator itself is unchanged; only its use converts it."""
        target = _target()
        for size in (0.95, 1.05):
            with self.subTest(size=size):
                source = _source_scaled_by(target, size)
                shape = (512, 512)
                padded = [nornir_imageregistration.phasecorrelation.pad_image_for_phase_correlation(
                    img, min_overlap=0.5, new_height=shape[0], new_width=shape[1]) for img in (target, source)]
                window = sb.HannWindowCache.GetOrCreate(shape)
                mags = [sb._logpolar_fft_magnitude(img, window, use_dog=True) for img in padded]
                seed, _ratio = sb._estimate_scale_radial_fft(
                    mags[0], mags[1], sb._radial_fft_max_radius(max(shape)), shape)
                self.assertAlmostEqual(seed, size, delta=0.03)


class TestRefineSeeds(unittest.TestCase):

    def _settings(self, hint: float | None = None) -> StosBruteSettings:
        return StosBruteSettings(initial_scale_hint=hint)

    def test_measured_scale_and_metadata_default_are_both_seeds(self):
        self.assertEqual(sb._refine_scale_seeds(self._settings(), 0.9848, 0.93, 1.0), [0.93, 1.0])

    def test_measured_scale_alone_without_a_resolved_hint(self):
        self.assertEqual(sb._refine_scale_seeds(self._settings(), 1.0, 0.97, None), [0.97])

    def test_close_seeds_are_not_duplicated(self):
        self.assertEqual(sb._refine_scale_seeds(self._settings(), 0.9848, 0.998, 1.0), [0.998])

    def test_caller_hint_is_the_only_seed(self):
        seeds = sb._refine_scale_seeds(self._settings(hint=0.9138), 0.9848, 0.93, 1.0)
        self.assertEqual(len(seeds), 1)
        self.assertAlmostEqual(seeds[0], 0.9138 / 0.9848)


class TestRefineFromSeeds(_NumpyCase):

    def test_measured_seed_beats_the_metadata_default(self):
        """Seeded at 1.0 alone the refine cannot reach 1.087; with the measured seed it does."""
        target = _target()
        pair = _pair(_source_scaled_by(target, 0.92), target)

        from_metadata_only = sb._refine_scale_from_seeds(pair, 0.0, [1.0], wide_search=False)
        from_both = sb._refine_scale_from_seeds(pair, 0.0, [1.08, 1.0], wide_search=False)

        self.assertLess(from_metadata_only, 1.07, 'precondition: 1.0 alone stays near the metadata')
        self.assertAlmostEqual(from_both, 1 / 0.92, delta=0.01)

    def test_a_poor_measured_seed_does_not_override_a_better_default(self):
        """Choosing by score: a wrong measurement costs time, not accuracy."""
        target = _target()
        pair = _pair(_source_scaled_by(target, 1.0), target)

        chosen = sb._refine_scale_from_seeds(pair, 0.0, [0.92, 1.0], wide_search=False)

        self.assertAlmostEqual(chosen, 1.0, delta=0.01)


if __name__ == '__main__':
    unittest.main()
