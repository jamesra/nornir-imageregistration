"""Grid populate and GPU constructors keep Transform-backend points until packing."""

from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import HasCupy
from nornir_imageregistration.grid_subdivision import ITKGridDivision
from nornir_imageregistration.transforms.converters import ConvertTransformToGridTransform
from nornir_imageregistration.transforms.gridtransform import (
    GridTransform_GPU,
    GridTransform_GPUComponent,
    cuLinearNDInterpolator,
)
from nornir_imageregistration.transforms.gridwithrbffallback import (
    GridWithRBFFallback,
    GridWithRBFFallback_GPUComponent,
)
from nornir_imageregistration.transforms.meshwithrbffallback import MeshWithRBFFallback_GPUComponent


def _identity_gpu_mesh() -> MeshWithRBFFallback_GPUComponent:
    points = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 32.0, 0.0, 32.0],
            [32.0, 0.0, 32.0, 0.0],
            [32.0, 32.0, 32.0, 32.0],
        ],
        dtype=np.float64,
    )
    mesh = MeshWithRBFFallback_GPUComponent(points)
    mesh.InitializeDataStructures()
    return mesh


def _cpu_rigid() -> nornir_imageregistration.transforms.Rigid:
    return nornir_imageregistration.transforms.Rigid(
        target_offset=(0, 0),
        source_rotation_center=(16, 16),
        angle=0,
    )


class TestGridBackendPacking(unittest.TestCase):
    def test_cpu_populate_stays_numpy(self) -> None:
        grid = ITKGridDivision(
            source_shape=(32, 32),
            cell_size=(16, 16),
            grid_spacing=(16, 16),
            transform=_cpu_rigid(),
        )
        self.assertIs(nornir_imageregistration.cp.get_array_module(grid.TargetPoints), np)
        cpu = ConvertTransformToGridTransform(
            _cpu_rigid(), (32, 32), cell_size=(16, 16), grid_spacing=(16, 16), prefer_gpu=False)
        self.assertIsInstance(cpu, GridWithRBFFallback)
        self.assertIs(nornir_imageregistration.cp.get_array_module(cpu.TargetPoints), np)

    @unittest.skipUnless(HasCupy(), "requires CuPy")
    def test_populate_keeps_cupy_transform_output(self) -> None:
        import cupy as cupy_mod

        mesh = _identity_gpu_mesh()
        grid = ITKGridDivision(source_shape=(32, 32), cell_size=(16, 16), grid_spacing=(16, 16))
        mapped = grid.PopulateTargetPoints(mesh)
        self.assertIsNotNone(mapped)
        self.assertTrue(isinstance(mapped, cupy_mod.ndarray))
        self.assertTrue(isinstance(grid.TargetPoints, cupy_mod.ndarray))

    @unittest.skipUnless(
        HasCupy() and cuLinearNDInterpolator is not None, "requires cupyx LinearNDInterpolator")
    def test_prefer_gpu_packs_on_device(self) -> None:
        import cupy as cupy_mod

        previous = nornir_imageregistration.GetActiveComputationLib()
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
        try:
            mesh = _identity_gpu_mesh()
            gpu = ConvertTransformToGridTransform(
                mesh, (32, 32), cell_size=(16, 16), grid_spacing=(16, 16), prefer_gpu=True)
            self.assertIsInstance(gpu, GridWithRBFFallback_GPUComponent)
            self.assertTrue(isinstance(gpu.TargetPoints, cupy_mod.ndarray))

            cpu = ConvertTransformToGridTransform(
                mesh, (32, 32), cell_size=(16, 16), grid_spacing=(16, 16), prefer_gpu=False)
            self.assertIsInstance(cpu, GridWithRBFFallback)
            self.assertIsInstance(cpu.TargetPoints, np.ndarray)
            self.assertFalse(isinstance(cpu.TargetPoints, cupy_mod.ndarray))
        finally:
            nornir_imageregistration.SetActiveComputationLib(previous)

    @unittest.skipUnless(HasCupy(), "requires CuPy")
    def test_gpu_component_centroids_and_rotate_on_device(self) -> None:
        import cupy as cupy_mod

        grid = GridTransform_GPUComponent(ITKGridDivision(
            source_shape=(32, 32),
            cell_size=(16, 16),
            grid_spacing=(16, 16),
            transform=_cpu_rigid(),
        ))
        centroids = grid.GetFixedCentroids()
        self.assertTrue(isinstance(centroids, cupy_mod.ndarray))
        self.assertEqual(centroids.ndim, 2)
        grid.RotateTargetPoints(0.0, None)
        self.assertTrue(isinstance(grid.TargetPoints, cupy_mod.ndarray))

        gpu = GridTransform_GPU(ITKGridDivision(
            source_shape=(32, 32),
            cell_size=(16, 16),
            grid_spacing=(16, 16),
            transform=_cpu_rigid(),
        ))
        gpu.RotateTargetPoints(0.0, None)
        self.assertTrue(isinstance(gpu.TargetPoints, cupy_mod.ndarray))


if __name__ == "__main__":
    unittest.main()
