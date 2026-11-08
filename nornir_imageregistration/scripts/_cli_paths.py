"""Shared path defaults for the nornir_imageregistration command-line scripts."""
import os


def mosaic_default_tile_dir(mosaic_path: str) -> str:
    """Return the tile folder used when no tile path is given: the mosaic file's directory, as an absolute path.

    ``abspath`` comes first so a bare mosaic filename gives the working directory, not ``''``.
    """
    return os.path.dirname(os.path.abspath(mosaic_path))
