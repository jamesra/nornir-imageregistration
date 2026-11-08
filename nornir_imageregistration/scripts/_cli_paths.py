"""Shared path defaults for the nornir_imageregistration command-line scripts."""
import os


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
