"""Shared exceptions for nornir-imageregistration."""

from __future__ import annotations

__all__ = ["MissingTilesetInputError", "as_missing_tileset_error"]


class MissingTilesetInputError(FileNotFoundError):
    """Raised when required pyramid tile paths are absent (input files or output directories)."""

    output_path: str
    missing_paths: list[str]

    def __init__(
        self,
        output_path: str,
        missing_paths: list[str],
        *,
        for_output: bool = False,
    ) -> None:
        self.output_path = output_path
        self.missing_paths = missing_paths
        if for_output:
            message = f"Cannot write tile output {output_path}; missing: {missing_paths}"
        else:
            message = f"No input tiles for {output_path}; missing: {missing_paths}"
        super().__init__(message)


def as_missing_tileset_error(
    output_path: str,
    missing_paths: list[str],
    cause: BaseException,
    *,
    for_output: bool = False,
) -> MissingTilesetInputError:
    """Build MissingTilesetInputError chained from a save/copy ENOENT."""
    error = MissingTilesetInputError(output_path, missing_paths, for_output=for_output)
    error.__cause__ = cause
    return error
