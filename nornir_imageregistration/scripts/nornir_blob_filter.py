"""
Command-line wrapper for Python ir-blob equivalent.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import nornir_shared.misc

import nornir_imageregistration


def _blob_filter_module():
    return importlib.import_module("nornir_imageregistration.blob_filter")


def __CreateArgParser():
    parser = argparse.ArgumentParser(description="Generate a blob-filtered image.")
    parser.add_argument("-load", required=True, type=str, dest="load_path", help="Input image path")
    parser.add_argument("-save", required=True, type=str, dest="save_path", help="Output image path")
    parser.add_argument("-mask", required=False, type=str, default=None, dest="mask_path", help="Optional mask image path")
    parser.add_argument("-r", required=False, type=int, default=9, dest="radius", help="Radius for local variance window")
    parser.add_argument("-median", required=False, type=int, default=7, dest="median_radius", help="Radius for median pre-filter")
    parser.add_argument("-max", required=False, type=float, default=3.0, dest="max_value", help="Maximum blob response before normalization")
    parser.add_argument("-sh", required=False, type=int, default=1, dest="shared", help="Legacy compatibility argument (ignored)")
    parser.add_argument("-threads", required=False, type=int, default=0, dest="threads", help="Legacy compatibility argument (ignored)")
    parser.add_argument("-gpu_min_pixels", required=False, type=int, default=1024 * 1024, dest="gpu_min_pixels",
                        help="Minimum image area required to auto-enable CuPy processing")
    return parser


def ParseArgs(exec_args=None):
    if exec_args is None:
        exec_args = sys.argv[1:]

    parser = __CreateArgParser()
    return parser.parse_known_args(args=exec_args)


def Execute(exec_args=None):
    if exec_args is None:
        exec_args = sys.argv[1:]

    args, _extra = ParseArgs(exec_args)
    diagnostics = _blob_filter_module().BlobFilterImageFile(
        args.load_path,
        args.save_path,
        radius=args.radius,
        median_radius=args.median_radius,
        max_value=args.max_value,
        mask_path=args.mask_path,
        min_pixels_for_gpu=args.gpu_min_pixels,
        return_diagnostics=True)

    print(
        "Wrote: {path} (backend={backend}, median_variance={median:.6g}, fallback={fallback})".format(
            path=args.save_path,
            backend=diagnostics.backend,
            median=diagnostics.global_median_variance,
            fallback=diagnostics.used_numpy_fallback))


if __name__ == "__main__":
    args, _extra = ParseArgs()
    log_path = os.path.join(os.path.dirname(args.save_path), "Logs")
    nornir_shared.misc.SetupLogging(OutputPath=log_path)
    Execute()
