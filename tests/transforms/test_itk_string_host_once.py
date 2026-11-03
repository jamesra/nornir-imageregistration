"""Transform strings are formatted from one host copy, byte-identical to the per-element loop.

Iterating a CuPy control-point array row by row read every element from the device:
``__MoveMosaicsToZeroOrigin`` spent ~4-5 s per RC3 section in ``ToITKString`` on srv
(2026-11-02). These tests pin the output to the old per-row formatting, on the host and on
the GPU components, and check that a volume of mosaics really moves to the origin.
"""

from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import mosaic, volume
from nornir_imageregistration.grid_subdivision import ITKGridDivision
from nornir_imageregistration.transforms import float_to_shortest_string, factory
from nornir_imageregistration.transforms.gridtransform import cuLinearNDInterpolator
from nornir_imageregistration.transforms.gridwithrbffallback import (
    GridWithRBFFallback,
    GridWithRBFFallback_GPUComponent,
)
from nornir_imageregistration.transforms.meshwithrbffallback import (
    MeshWithRBFFallback,
    MeshWithRBFFallback_GPUComponent,
)


def _legacy_grid_points(points) -> str:
    """The per-row loop ToITKString used before 2026-11-03."""
    template = " %(cx)s %(cy)s"
    return ''.join(template % {'cx': float_to_shortest_string(CX, 3), 'cy': float_to_shortest_string(CY, 3)}
                   for CY, CX, MY, MX in points)


def _legacy_mesh_string(transform) -> str:
    """The per-row ``_MeshTransformToIRToolsString`` used before 2026-11-03."""
    (bottom, left, top, right) = transform.MappedBoundingBox.ToTuple()
    numPoints = transform.points.shape[0]
    output = [f"MeshTransform_double_2_2 vp {numPoints * 4}"]
    template = " %(mx)s %(my)s %(cx)s %(cy)s"
    width = right - left
    height = top - bottom
    for CY, CX, MY, MX in transform.points:
        output.append(template % {'cx': float_to_shortest_string(CX, 3),
                                  'cy': float_to_shortest_string(CY, 3),
                                  'mx': float_to_shortest_string((MX - left) / width, 10),
                                  'my': float_to_shortest_string((MY - bottom) / height, 10)})
    output.append(f" fp 8 0 16 16 {left:g} {bottom:g} {(right - left) + 1:g} {(top - bottom) + 1:g} {numPoints:d}")
    return ''.join(output)


def _rigid() -> nornir_imageregistration.transforms.Rigid:
    return nornir_imageregistration.transforms.Rigid(
        target_offset=(-1234.5678, 987.6543), source_rotation_center=(150.25, 300.75), angle=0.137)


def _grid_division() -> ITKGridDivision:
    return ITKGridDivision(source_shape=(1000, 1000), cell_size=(256, 256), grid_spacing=(192, 192),
                           transform=_rigid())


def _mesh_points() -> np.ndarray:
    rng = np.random.default_rng(11)
    source = rng.uniform(0, 1000, size=(40, 2))
    target = source * 1.0137 + np.array([-512.3456789, 77.7777777]) + rng.normal(0, 3, size=(40, 2))
    return np.hstack((target, source))


class TestHostStrings(unittest.TestCase):

    def test_grid_string_matches_per_row_loop(self) -> None:
        transform = GridWithRBFFallback(_grid_division())
        text = factory.TransformToIRToolsString(transform)
        self.assertIn(_legacy_grid_points(transform.points), text)
        self.assertTrue(text.startswith(f"GridTransform_double_2_2 vp {transform.points.shape[0] * 2} "))

    def test_mesh_string_matches_per_row_loop(self) -> None:
        for dtype in (np.float64, np.float32):
            with self.subTest(dtype=dtype.__name__):
                transform = MeshWithRBFFallback(_mesh_points().astype(dtype))
                self.assertEqual(factory.TransformToIRToolsString(transform), _legacy_mesh_string(transform))


@unittest.skipUnless(nornir_imageregistration.HasCupy() and cuLinearNDInterpolator is not None,
                     'requires CuPy with cupyx LinearNDInterpolator')
class TestGpuStrings(unittest.TestCase):

    def setUp(self) -> None:
        self._previous = nornir_imageregistration.GetActiveComputationLib()
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        if not nornir_imageregistration.UsingCupy():
            self.skipTest('CuPy runtime probe failed (no usable GPU)')

    def tearDown(self) -> None:
        nornir_imageregistration.SetActiveComputationLib(self._previous)

    def test_gpu_grid_string_equals_host_and_per_row_loop(self) -> None:
        gpu = GridWithRBFFallback_GPUComponent(_grid_division())
        host = GridWithRBFFallback(_grid_division())
        text = factory.TransformToIRToolsString(gpu)
        self.assertEqual(text, factory.TransformToIRToolsString(host))
        self.assertIn(_legacy_grid_points(gpu.points), text)

    def test_gpu_mesh_string_equals_host_and_per_row_loop(self) -> None:
        gpu = MeshWithRBFFallback_GPUComponent(_mesh_points())
        host = MeshWithRBFFallback(_mesh_points())
        text = factory.TransformToIRToolsString(gpu)
        self.assertEqual(text, factory.TransformToIRToolsString(host))
        self.assertEqual(text, _legacy_mesh_string(gpu))


class TestVolumeOfMosaicsTranslatesToZero(unittest.TestCase):
    """``Volume.TranslateToZeroOrigin`` moves mosaic sections (it used to only report the offset)."""

    def _volume(self, offsets) -> volume.Volume:
        vol = volume.Volume()
        for i, offset in enumerate(offsets):
            grid = GridWithRBFFallback(ITKGridDivision(
                source_shape=(100, 100), cell_size=(32, 32), grid_spacing=(32, 32),
                transform=nornir_imageregistration.transforms.Rigid(target_offset=offset)))
            vol.AddSection(str(i), mosaic.Mosaic({'tile.png': grid}))
        return vol

    def test_negative_origin_is_moved_to_zero(self) -> None:
        vol = self._volume([(-300.5, 20.0), (40.0, -800.25)])
        offset = vol.TranslateToZeroOrigin()
        np.testing.assert_allclose(nornir_imageregistration.EnsureNumpyArray(offset), [300.5, 800.25])
        (minY, minX, _, _) = vol.VolumeBounds.ToTuple()
        self.assertAlmostEqual(minY, 0.0, places=3)
        self.assertAlmostEqual(minX, 0.0, places=3)

    def test_origin_already_at_zero_returns_none(self) -> None:
        vol = self._volume([(0.0, 5.0), (7.0, 0.0)])
        self.assertIsNone(vol.TranslateToZeroOrigin())


if __name__ == '__main__':
    unittest.main()
