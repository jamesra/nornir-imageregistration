"""
The deprecated assemble wrappers must forward their arguments.

``FixedImageToWarpedSpace`` and ``WarpedImageToFixedSpace`` accepted ``botleft``,
``area``, ``cval`` and ``extrapolate`` and then called the replacement function
with ``None``/``None``/``None``/``False`` literals. A caller asking for a specific
sub-region, fill value, or extrapolation silently got a full-bounding-box warp
filled with zeros instead -- no error, just different output.

Each test pins one argument by comparing the wrapper against the replacement
function called with the same argument, so a dropped argument is a failure rather
than a coincidence.
"""

from __future__ import annotations

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration import assemble
from nornir_imageregistration.transforms import factory

TARGET_H = 48
TARGET_W = 64
SOURCE_H = 24
SOURCE_W = 32

SUB_BOTLEFT = (6.0, 9.0)
SUB_AREA = (10.0, 14.0)
FILL_VALUE = 0.75


# These wrappers warn by design; most tests here are about their behavior, and
# the warning text itself is asserted explicitly in the last test.
pytestmark = pytest.mark.filterwarnings('ignore::DeprecationWarning')


def _source_image() -> np.ndarray:
    rng = np.random.default_rng(4)
    return (rng.random((SOURCE_H, SOURCE_W)) * 0.5 + 0.25).astype(np.float32)


def _transform():
    return factory.CreateRigidTransform(
        target_image_shape=np.asarray((TARGET_H, TARGET_W), dtype=np.int64),
        source_image_shape=np.asarray((SOURCE_H, SOURCE_W), dtype=np.int64),
        rangle=0.0,
        warped_offset=(0.0, 0.0))


def _as_array(result) -> np.ndarray:
    return np.asarray(nornir_imageregistration.EnsureNumpyArray(result))


def test_warped_to_fixed_forwards_area():
    """``area`` selects the output size instead of the full bounding box."""
    source = _source_image()
    wrapped = _as_array(assemble.WarpedImageToFixedSpace(
        _transform(), source, botleft=SUB_BOTLEFT, area=SUB_AREA))

    assert wrapped.shape == (int(SUB_AREA[0]), int(SUB_AREA[1])), \
        'area was discarded; the wrapper returned a full-bounding-box warp'


def test_warped_to_fixed_matches_replacement_for_subregion():
    """The wrapper and its replacement agree when given the same region."""
    source = _source_image()
    wrapped = _as_array(assemble.WarpedImageToFixedSpace(
        _transform(), source, botleft=SUB_BOTLEFT, area=SUB_AREA))
    direct = _as_array(assemble.SourceImageToTargetSpace(
        _transform(), source, output_botleft=SUB_BOTLEFT, output_area=SUB_AREA))

    np.testing.assert_array_equal(wrapped, direct)


def test_warped_to_fixed_forwards_botleft():
    """``botleft`` moves the sampled window, so content differs from origin."""
    source = _source_image()
    at_origin = _as_array(assemble.WarpedImageToFixedSpace(
        _transform(), source, botleft=(0.0, 0.0), area=SUB_AREA))
    offset = _as_array(assemble.WarpedImageToFixedSpace(
        _transform(), source, botleft=SUB_BOTLEFT, area=SUB_AREA))

    assert at_origin.shape == offset.shape
    assert not np.array_equal(at_origin, offset), \
        'botleft was discarded; both windows returned identical content'


def test_warped_to_fixed_forwards_cval():
    """``cval`` fills unmapped pixels instead of always using zero."""
    source = _source_image()
    # A window beyond the transformed source maps nothing, so it is all fill.
    far_botleft = (float(TARGET_H) + 20.0, float(TARGET_W) + 20.0)
    filled = _as_array(assemble.WarpedImageToFixedSpace(
        _transform(), source, botleft=far_botleft, area=SUB_AREA, cval=FILL_VALUE))

    assert np.allclose(filled, FILL_VALUE), \
        f'cval was discarded; unmapped pixels are {np.unique(filled)[:4]} not {FILL_VALUE}'


def test_fixed_to_warped_forwards_area():
    """The target-to-source wrapper forwards ``area`` as well."""
    source = _source_image()
    wrapped = _as_array(assemble.FixedImageToWarpedSpace(
        _transform(), source, botleft=SUB_BOTLEFT, area=SUB_AREA))

    assert wrapped.shape == (int(SUB_AREA[0]), int(SUB_AREA[1])), \
        'area was discarded by FixedImageToWarpedSpace'


def test_fixed_to_warped_matches_replacement_for_subregion():
    """``FixedImageToWarpedSpace`` agrees with ``TargetImageToSourceSpace``."""
    source = _source_image()
    wrapped = _as_array(assemble.FixedImageToWarpedSpace(
        _transform(), source, botleft=SUB_BOTLEFT, area=SUB_AREA))
    direct = _as_array(assemble.TargetImageToSourceSpace(
        _transform(), source, output_botleft=SUB_BOTLEFT, output_area=SUB_AREA))

    np.testing.assert_array_equal(wrapped, direct)


def test_fixed_to_warped_forwards_cval():
    """``FixedImageToWarpedSpace`` forwards ``cval``."""
    source = _source_image()
    far_botleft = (float(TARGET_H) + 20.0, float(TARGET_W) + 20.0)
    filled = _as_array(assemble.FixedImageToWarpedSpace(
        _transform(), source, botleft=far_botleft, area=SUB_AREA, cval=FILL_VALUE))

    assert np.allclose(filled, FILL_VALUE), 'cval was discarded'


def test_default_arguments_still_produce_a_full_warp():
    """Existing callers pass no optional arguments and must be unaffected."""
    source = _source_image()
    wrapped = _as_array(assemble.WarpedImageToFixedSpace(_transform(), source))
    direct = _as_array(assemble.SourceImageToTargetSpace(_transform(), source))

    np.testing.assert_array_equal(wrapped, direct)


@pytest.mark.parametrize('wrapper_name,expected_replacement', [
    ('FixedImageToWarpedSpace', 'TargetImageToSourceSpace'),
    ('WarpedImageToFixedSpace', 'SourceImageToTargetSpace'),
])
@pytest.mark.filterwarnings('always::DeprecationWarning')
def test_deprecation_message_names_the_right_functions(
        wrapper_name, expected_replacement):
    """The two warnings had their subjects and replacements crossed."""
    wrapper = getattr(assemble, wrapper_name)
    with pytest.warns(DeprecationWarning) as captured:
        wrapper(_transform(), _source_image())

    messages = [str(w.message) for w in captured]
    assert any(wrapper_name in m and expected_replacement in m for m in messages), \
        f'expected a warning naming {wrapper_name} and {expected_replacement}, got {messages}'
