"""
The four control-point bounding-box properties must resolve bounds the same way.

Before this change the helper choice was split across the two classes, and the host
class was split against itself:

    host ControlPointBase.TargetBoundingBox  BoundingPrimitiveFromPoints
    host ControlPointBase.SourceBoundingBox  BoundingRectangleFromPoints
    GPU  ...FixedBoundingBox                 BoundingPrimitiveFromPoints
    GPU  ...MappedBoundingBox                BoundingPrimitiveFromPoints

The two helpers differ only for 3D input: ``BoundingPrimitiveFromPoints`` returns a
6-value ``BoundingBox`` while ``BoundingRectangleFromPoints`` drops Z and returns a
4-value ``Rectangle``. Measured:

    Nx2 input: primitive=Rectangle   rectangle=Rectangle    equal_bounds=True
    Nx3 input: primitive=BoundingBox rectangle=Rectangle    (6 vs 4 values)

Control points cannot be 3D here. ``SourcePoints`` and ``TargetPoints`` are fixed
slices ``_points[:, 2:4]`` and ``_points[:, 0:2]`` of an Nx4 array, and
``BoundsArrayFromPoints`` raises outright on Nx4 input, so the 2D shape is load
bearing rather than incidental.

So the divergence was unreachable and no output was ever wrong. This was a latent
inconsistency: the properties all document ``:return: (minY, minX, maxY, maxX)``,
i.e. a Rectangle, but three of the four were resolved by a helper that would break
that contract the moment control points gained a third axis. They are now all
resolved by the helper that guarantees the documented type.

Because the divergence is unreachable, the behavioural tests here pass both before
and after; only ``test_all_four_sites_use_the_same_helper`` fails beforehand. It is
a drift guard, and that is the point.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration import spatial
from nornir_imageregistration.spatial.converters import (
    BoundingPrimitiveFromPoints, BoundingRectangleFromPoints)
from nornir_imageregistration.transforms import controlpointbase
from nornir_imageregistration.transforms.meshwithrbffallback import MeshWithRBFFallback
from nornir_imageregistration.transforms.triangulation import Triangulation


@pytest.fixture(autouse=True)
def _host_backend():
    nornir_imageregistration.SetActiveComputationLib(
        nornir_imageregistration.ComputationLib.numpy)


def _control_points():
    """A 3x3 lattice of [TargetY, TargetX, SourceY, SourceX] pairs."""
    rows = []
    for y in (0.0, 50.0, 100.0):
        for x in (0.0, 50.0, 100.0):
            rows.append([y + 3.0, x + 5.0, y, x])
    return np.array(rows, dtype=np.float64)


HOST_TRANSFORMS = [MeshWithRBFFallback, Triangulation]


# --- the invariant that makes the divergence unreachable ------------------------

@pytest.mark.parametrize('ctor', HOST_TRANSFORMS)
def test_control_point_axes_are_always_two_dimensional(ctor):
    t = ctor(_control_points())

    assert np.asarray(t.SourcePoints).shape[1] == 2
    assert np.asarray(t.TargetPoints).shape[1] == 2


def test_bounds_helper_rejects_the_raw_nx4_point_pairs():
    """Shows the Nx2 slicing is load bearing, not incidental."""
    with pytest.raises(Exception, match='Unexpected number of dimensions'):
        spatial.BoundsArrayFromPoints(_control_points())


# --- the two helpers -----------------------------------------------------------

@pytest.mark.parametrize('n', [3, 10, 500])
def test_helpers_agree_on_two_dimensional_points(n):
    rng = np.random.default_rng(44)
    pts = rng.random((n, 2)) * 1000.0

    primitive = BoundingPrimitiveFromPoints(pts)
    rectangle = BoundingRectangleFromPoints(pts)

    assert type(primitive) is type(rectangle)
    np.testing.assert_array_equal(primitive.BoundingBox, rectangle.BoundingBox)


def test_helpers_diverge_on_three_dimensional_points():
    """Records the trap the normalization removes.

    If this ever stops being true the two helpers have become interchangeable and
    the consistency requirement below is moot.
    """
    rng = np.random.default_rng(44)
    pts = rng.random((10, 3)) * 1000.0

    primitive = BoundingPrimitiveFromPoints(pts)
    rectangle = BoundingRectangleFromPoints(pts)

    assert len(primitive.BoundingBox) == 6
    assert len(rectangle.BoundingBox) == 4
    assert type(primitive) is not type(rectangle)


# --- the documented contract ---------------------------------------------------

@pytest.mark.parametrize('ctor', HOST_TRANSFORMS)
@pytest.mark.parametrize('prop', ['SourceBoundingBox', 'TargetBoundingBox',
                                  'MappedBoundingBox', 'FixedBoundingBox'])
def test_every_bounding_property_returns_a_rectangle(ctor, prop):
    """All four docstrings promise (minY, minX, maxY, maxX)."""
    t = ctor(_control_points())

    value = getattr(t, prop)

    assert isinstance(value, spatial.Rectangle), \
        f'{ctor.__name__}.{prop} returned {type(value).__name__}'
    assert len(value.BoundingBox) == 4


@pytest.mark.parametrize('ctor', HOST_TRANSFORMS)
def test_source_and_mapped_are_the_same_bounds(ctor):
    t = ctor(_control_points())

    np.testing.assert_array_equal(t.SourceBoundingBox.BoundingBox,
                                  t.MappedBoundingBox.BoundingBox)


@pytest.mark.parametrize('ctor', HOST_TRANSFORMS)
def test_target_and_fixed_are_the_same_bounds(ctor):
    t = ctor(_control_points())

    np.testing.assert_array_equal(t.TargetBoundingBox.BoundingBox,
                                  t.FixedBoundingBox.BoundingBox)


@pytest.mark.parametrize('ctor', HOST_TRANSFORMS)
def test_bounds_match_the_control_points_they_summarize(ctor):
    cp = _control_points()
    t = ctor(cp)

    # source axis is columns 2:4, target axis is columns 0:2
    np.testing.assert_allclose(
        t.SourceBoundingBox.BoundingBox,
        [cp[:, 2].min(), cp[:, 3].min(), cp[:, 2].max(), cp[:, 3].max()])
    np.testing.assert_allclose(
        t.TargetBoundingBox.BoundingBox,
        [cp[:, 0].min(), cp[:, 1].min(), cp[:, 0].max(), cp[:, 1].max()])


# --- the drift guard -----------------------------------------------------------

def test_all_four_sites_use_the_same_helper():
    """The actual fix: one helper choice, stated in four places that must agree."""
    sites = [
        controlpointbase.ControlPointBase.TargetBoundingBox,
        controlpointbase.ControlPointBase.SourceBoundingBox,
        controlpointbase.ControlPointBase_GPUComponent.FixedBoundingBox,
        controlpointbase.ControlPointBase_GPUComponent.MappedBoundingBox,
    ]

    for prop in sites:
        source = inspect.getsource(prop.fget)
        assert 'BoundingRectangleFromPoints' in source, \
            f'{prop.fget.__qualname__} does not use BoundingRectangleFromPoints'
        assert 'BoundingPrimitiveFromPoints' not in source, \
            f'{prop.fget.__qualname__} still uses BoundingPrimitiveFromPoints'
