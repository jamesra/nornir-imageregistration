"""Shared path defaults for the nornir_imageregistration command-line scripts."""
import argparse
import os
from collections.abc import Callable
from typing import NoReturn


def mosaic_default_tile_dir(mosaic_path: str) -> str:
    """Return the tile folder used when no tile path is given: the mosaic file's directory, as an absolute path.

    ``abspath`` comes first so a bare mosaic filename gives the working directory, not ``''``.
    """
    return os.path.dirname(os.path.abspath(mosaic_path))


def ensure_output_parent_dir(output_path: str) -> None:
    """Create the folder that will hold ``output_path`` if it does not exist yet.

    ``abspath`` comes first so a bare output filename uses the working directory instead of ``makedirs('')``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)


def validate_mosaic_tile_args(args: argparse.Namespace, on_use_error: Callable[[str], NoReturn]) -> None:
    """Validate ``inputpath``/``outputpath``/``tilepath`` of a script that reads a .mosaic and its tiles.

    Calls ``on_use_error`` (which must exit) when the input mosaic or the tile folder is missing, creates the
    output folder, and replaces a ``None`` tile path with the mosaic's directory.
    """
    if not os.path.exists(args.inputpath):
        on_use_error("Input mosaic file not found: " + args.inputpath)

    ensure_output_parent_dir(args.outputpath)

    if args.tilepath is None:
        args.tilepath = mosaic_default_tile_dir(args.inputpath)

    if not os.path.exists(args.tilepath):
        on_use_error("Tile path not found: " + args.tilepath)
