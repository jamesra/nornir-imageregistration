"""
Padding noise was unseeded, so stos_brute alignment did not reproduce run to run.

``pad_image_for_phase_correlation`` fills the padding region with noise drawn from
the image median and stddev. Filling with a constant would give phase correlation a
hard edge to lock onto, so the noise is deliberate -- but it was drawn from the
unseeded global generator, and everything downstream inherited that.

Three runs of ``_find_angle_and_scale_with_logpolar`` on identical input gave::

    angle=6.9328  scale=1.03798  weight=2.6237  translation=(-508.90, 508.18)
    angle=6.8582  scale=1.03761  weight=3.5682  translation=( 507.69, 506.96)
    angle=6.9711  scale=1.03879  weight=3.3303  translation=(-506.78, 508.89)

The translation *sign* flips, so the +180 degree disambiguation was being decided by
fill noise -- a discrete branch landing differently on identical input.

Padding a fixed image in three separate processes, before and after:

    before   131163.2162   131060.0124   131103.3426
    after    131167.5645   131167.5645   131167.5645

The subtlety worth preserving: a fixed seed *per call* would be wrong. Phase
correlation pads both the target and the source, and identical fill in both would
correlate the padding regions and offer the correlation a peak that is not in the
data. So the generator advances between calls and only its starting state is pinned.
"""

from __future__ import annotations

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration.phasecorrelation import pad_image_for_phase_correlation

SHAPE = (96, 96)


def _image(shift=(0, 0)):
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    base = ((np.sin(yy / 7.0) * np.cos(xx / 11.0) + 1.0) * 0.5).astype(np.float32)
    return np.roll(base, shift=shift, axis=(0, 1))


def _pad(image):
    return np.asarray(pad_image_for_phase_correlation(image, min_overlap=0.5))


# --- reproducibility ----------------------------------------------------------

def test_the_same_input_pads_the_same_way_after_reseeding():
    image = _image()

    nornir_imageregistration.seed_random_data()
    first = _pad(image)
    nornir_imageregistration.seed_random_data()
    second = _pad(image)

    assert np.array_equal(first, second)


def test_a_whole_sequence_of_pads_replays_identically():
    """A run is a sequence of calls, so the sequence is what has to reproduce."""
    target, source = _image(), _image(shift=(5, -3))

    nornir_imageregistration.seed_random_data()
    run_a = [_pad(target), _pad(source), _pad(target)]
    nornir_imageregistration.seed_random_data()
    run_b = [_pad(target), _pad(source), _pad(target)]

    for a, b in zip(run_a, run_b):
        assert np.array_equal(a, b)


def test_generated_noise_replays_identically():
    nornir_imageregistration.seed_random_data()
    first = nornir_imageregistration.GenRandomData(8, 8, 0.5, 0.1, 0.0, 1.0, xp=np)

    nornir_imageregistration.seed_random_data()
    second = nornir_imageregistration.GenRandomData(8, 8, 0.5, 0.1, 0.0, 1.0, xp=np)

    assert np.array_equal(first, second)


def test_two_fresh_processes_pad_a_fixed_image_identically():
    """The defect as reported: identical input, separate runs, different output.

    The in-process tests above all reach for the new seeding API, so pre-fix they
    fail on a missing attribute rather than on behaviour. This one uses only
    ``pad_image_for_phase_correlation`` and so fails pre-fix for the real reason.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent('''
        import numpy as np
        from nornir_imageregistration.phasecorrelation import pad_image_for_phase_correlation

        yy, xx = np.mgrid[0:96, 0:96]
        img = ((np.sin(yy / 7.0) * np.cos(xx / 11.0) + 1.0) * 0.5).astype(np.float32)
        padded = pad_image_for_phase_correlation(img, min_overlap=0.5)
        print(f"{float(np.asarray(padded, dtype=np.float64).sum()):.10f}")
        ''')

    def run():
        finished = subprocess.run([sys.executable, '-c', script],
                                  capture_output=True, text=True, timeout=300)
        assert finished.returncode == 0, finished.stderr
        return finished.stdout.strip().splitlines()[-1]

    assert run() == run()


# --- but the fill must not become correlated ----------------------------------

def test_successive_draws_differ():
    """A fixed seed per call would make target and source padding identical."""
    nornir_imageregistration.seed_random_data()

    first = nornir_imageregistration.GenRandomData(8, 8, 0.5, 0.1, 0.0, 1.0, xp=np)
    second = nornir_imageregistration.GenRandomData(8, 8, 0.5, 0.1, 0.0, 1.0, xp=np)

    assert not np.array_equal(first, second)


def test_target_and_source_padding_are_not_identical():
    nornir_imageregistration.seed_random_data()

    padded_target = _pad(_image())
    padded_source = _pad(_image(shift=(5, -3)))

    corner_t = padded_target[:12, :12]
    corner_s = padded_source[:12, :12]

    assert not np.array_equal(corner_t, corner_s), (
        'identical fill in both images would correlate the padding regions')


# --- the second noise source --------------------------------------------------
#
# A brute alignment fills noise twice: padding fills the frame around the image,
# and ``rotate_image`` fills the corners a rotation leaves empty via
# ``ImageStats.GenerateNoise``. Seeding only the padding left alignment as
# irreproducible as before -- five identical calls still flipped between two
# peaks (~3.51 and ~4.5) -- because the corner fill was still drawing from the
# unseeded global generator. Both have to share one generator.

def _stats():
    return nornir_imageregistration.ImageStats.CalcStats(_image())


def test_rotation_corner_fill_replays_identically():
    stats = _stats()

    nornir_imageregistration.seed_random_data()
    first = np.asarray(stats.GenerateNoise(256, dtype=np.float32, xp=np))
    nornir_imageregistration.seed_random_data()
    second = np.asarray(stats.GenerateNoise(256, dtype=np.float32, xp=np))

    assert np.array_equal(first, second)


def test_successive_corner_fills_differ():
    stats = _stats()
    nornir_imageregistration.seed_random_data()

    first = np.asarray(stats.GenerateNoise(256, dtype=np.float32, xp=np))
    second = np.asarray(stats.GenerateNoise(256, dtype=np.float32, xp=np))

    assert not np.array_equal(first, second)


def test_both_noise_sources_share_one_generator():
    """Seeding once has to cover padding and corner fill together."""
    stats = _stats()

    def draw_both():
        nornir_imageregistration.seed_random_data()
        pad = _pad(_image())
        corner = np.asarray(stats.GenerateNoise(64, dtype=np.float32, xp=np))
        return pad, corner

    pad_a, corner_a = draw_both()
    pad_b, corner_b = draw_both()

    assert np.array_equal(pad_a, pad_b)
    assert np.array_equal(corner_a, corner_b)


def test_a_rotated_pad_replays_identically():
    """The two sources combined, through the call brute alignment actually makes."""
    from nornir_imageregistration.stos_brute import pad_and_rotate_image

    image = _image()
    stats = nornir_imageregistration.ImageStats.CalcStats(image)

    def run():
        nornir_imageregistration.seed_random_data()
        return np.asarray(pad_and_rotate_image(
            image=image, angle=7.0, image_stats=stats,
            desired_shape=(160, 160), min_overlap=0.5), dtype=np.float64)

    assert np.array_equal(run(), run())


# --- the escape hatch ---------------------------------------------------------

def test_seeding_from_entropy_is_still_available():
    """Some callers genuinely want a fresh draw; None restores that."""
    nornir_imageregistration.seed_random_data(None)
    first = nornir_imageregistration.GenRandomData(16, 16, 0.5, 0.1, 0.0, 1.0, xp=np)

    nornir_imageregistration.seed_random_data(None)
    second = nornir_imageregistration.GenRandomData(16, 16, 0.5, 0.1, 0.0, 1.0, xp=np)

    assert not np.array_equal(first, second)

    nornir_imageregistration.seed_random_data()


def test_an_explicit_generator_overrides_the_default():
    a = nornir_imageregistration.GenRandomData(
        8, 8, 0.5, 0.1, 0.0, 1.0, xp=np, rng=np.random.default_rng(1234))
    b = nornir_imageregistration.GenRandomData(
        8, 8, 0.5, 0.1, 0.0, 1.0, xp=np, rng=np.random.default_rng(1234))

    assert np.array_equal(a, b)


# --- the noise must still be noise --------------------------------------------

def test_the_fill_still_matches_the_requested_statistics():
    nornir_imageregistration.seed_random_data()

    data = nornir_imageregistration.GenRandomData(
        400, 400, mean=0.5, standardDev=0.1, min_val=0.0, max_val=1.0, xp=np)

    # The default fill dtype is float16, which overflows a 160k-element reduction.
    data = np.asarray(data, dtype=np.float64)

    assert float(np.mean(data)) == pytest.approx(0.5, abs=0.01)
    assert float(np.std(data)) == pytest.approx(0.1, abs=0.01)


def test_the_fill_is_not_constant():
    """A constant fill is the edge artifact the noise exists to avoid."""
    nornir_imageregistration.seed_random_data()

    data = nornir_imageregistration.GenRandomData(64, 64, 0.5, 0.1, 0.0, 1.0, xp=np)

    assert float(np.std(data)) > 0.0
