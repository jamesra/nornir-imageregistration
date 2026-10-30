"""
Micro-benchmark for Python ir-blob implementation.
"""

from __future__ import annotations

import argparse
import importlib
import time

import numpy as np

import nornir_imageregistration


def _blob_filter_module():
    return importlib.import_module("nornir_imageregistration.blob_filter")


def __CreateArgParser():
    parser = argparse.ArgumentParser(description="Benchmark Python blob filter on CPU/CuPy backends.")
    parser.add_argument("-shape", type=str, default="2048,2048", help="Image shape as rows,cols")
    parser.add_argument("-iterations", type=int, default=5, help="Benchmark iterations per backend")
    parser.add_argument("-radius", type=int, default=9, help="Blob local variance radius")
    parser.add_argument("-median", type=int, default=7, help="Median pre-filter radius")
    parser.add_argument("-max", type=float, default=3.0, dest="max_value", help="Blob response cap")
    parser.add_argument("-seed", type=int, default=7, help="Random seed")
    return parser


def _parse_shape(shape_text: str) -> tuple[int, int]:
    parts = [int(p.strip()) for p in shape_text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected shape as rows,cols, got: {shape_text}")
    return parts[0], parts[1]


def _run_case(image, mask, *, radius: int, median: int, max_value: float, iterations: int) -> float:
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        _blob_filter_module().BlobFilter(
            image,
            radius=radius,
            median_radius=median,
            max_value=max_value,
            mask=mask)
        durations.append(time.perf_counter() - start)
    return float(np.median(np.asarray(durations)))


def Execute(exec_args=None):
    parser = __CreateArgParser()
    args = parser.parse_args(exec_args)

    rows, cols = _parse_shape(args.shape)
    rng = np.random.default_rng(args.seed)
    image_np = rng.random((rows, cols), dtype=np.float32)
    mask_np = np.ones((rows, cols), dtype=bool)
    mask_np[::13, ::17] = False

    prev = nornir_imageregistration.GetActiveComputationLib()
    try:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
        cpu_sec = _run_case(
            image_np,
            mask_np,
            radius=args.radius,
            median=args.median,
            max_value=args.max_value,
            iterations=args.iterations)
        print(f"numpy_median_seconds={cpu_sec:.6f}")

        if nornir_imageregistration.HasCupy():
            import cupy as cp

            nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.cupy)
            image_cp = cp.asarray(image_np)
            mask_cp = cp.asarray(mask_np)
            gpu_sec = _run_case(
                image_cp,
                mask_cp,
                radius=args.radius,
                median=args.median,
                max_value=args.max_value,
                iterations=args.iterations)
            speedup = cpu_sec / gpu_sec if gpu_sec > 0 else float("inf")
            print(f"cupy_median_seconds={gpu_sec:.6f}")
            print(f"cupy_speedup_vs_numpy={speedup:.3f}")
    finally:
        nornir_imageregistration.SetActiveComputationLib(prev)


if __name__ == "__main__":
    Execute()
