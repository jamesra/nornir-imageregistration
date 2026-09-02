"""Generate tileset image pyramid levels and manage temp paths for tiles.

Created on Sep 10, 2019. The network implementation
of these functions copies the images locally and writes the output locally before
moving it to the final output directory.  This saves trips over the network as
we build the pyramid, which tends to be slow for sometimes hundreds of thousands
of small files.  This also helps the image I/O, which at this time is implemented
by pillow as lots of small I/O requests against the image file.
"""
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import logging
import re

from PIL import Image
import numpy

# Disable decompression bomb protection since we are dealing with huge images on purpose
Image.MAX_IMAGE_PIXELS = None
import os
import enum
import errno
import shutil
import nornir_pools
import nornir_shared.files
import nornir_imageregistration.temporaryfiles as temporaryfiles
from nornir_imageregistration.exceptions import MissingTilesetInputError, as_missing_tileset_error

logger = logging.getLogger(__name__)

_GRID_TILE_COORD_PATTERNS: dict[tuple[str, str], re.Pattern[str]] = {}



class Quadrant(enum.IntEnum):
    """An enum describing quadrants"""
    TopLeft = 1
    TopRight = 2
    BottomLeft = 3
    BottomRight = 4


# import nornir_shared.prettyoutput as prettyoutput


def _grid_tile_coord_pattern(file_prefix: str, file_postfix: str) -> re.Pattern[str]:
    """Return a compiled regex that extracts X/Y from a grid tile filename."""
    key = (file_prefix, file_postfix)
    pattern = _GRID_TILE_COORD_PATTERNS.get(key)
    if pattern is None:
        pattern = re.compile(
            rf"{re.escape(file_prefix)}X(\d+)_Y(\d+){re.escape(file_postfix)}$"
        )
        _GRID_TILE_COORD_PATTERNS[key] = pattern
    return pattern


def parse_grid_tile_xy(filename: str, file_prefix: str, file_postfix: str) -> tuple[int, int] | None:
    """Return (X, Y) grid coordinates parsed from a tile filename."""
    match = _grid_tile_coord_pattern(file_prefix, file_postfix).match(os.path.basename(filename))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def list_grid_tile_coords(directory: str, file_prefix: str, file_postfix: str) -> set[tuple[int, int]]:
    """Return the set of (X, Y) coordinates for tile files in a directory."""
    coords: set[tuple[int, int]] = set()
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                parsed = parse_grid_tile_xy(entry.name, file_prefix, file_postfix)
                if parsed is not None:
                    coords.add(parsed)
    except FileNotFoundError:
        pass
    return coords


def expected_parent_coords_from_source(
    source_directory: str, file_prefix: str, file_postfix: str,
) -> set[tuple[int, int]]:
    """Return parent (X, Y) coordinates required at the next pyramid level."""
    child_coords = list_grid_tile_coords(source_directory, file_prefix, file_postfix)
    return {(x // 2, y // 2) for x, y in child_coords}


def find_missing_lineage_parent_tiles(
    source_directory: str,
    dest_directory: str,
    file_prefix: str,
    file_postfix: str,
) -> list[tuple[int, int]]:
    """Return sorted parent coordinates that should exist in dest but do not."""
    expected = expected_parent_coords_from_source(source_directory, file_prefix, file_postfix)
    actual = list_grid_tile_coords(dest_directory, file_prefix, file_postfix)
    return sorted(expected - actual)


def _copy_source_tile_local(src: str, dst: str) -> None:
    """Copy a present source tile into the local cache, with network retries."""
    if not os.path.isfile(src):
        return
    if os.path.isfile(dst) and not nornir_shared.files.IsOutdated(src, dst):
        return
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    nornir_shared.files.copy_file(src, dst)


def ClearTempDirectories(level_paths: Sequence[str] | None) -> None:
    """Delete temporary directories used to generate pyramid levels. Returns None."""

    if level_paths is None:
        return

    if len(level_paths) == 0:
        return

    temp_dir = temporaryfiles.gettempdir()

    pool = nornir_pools.GetGlobalThreadPool()
    for level_path in level_paths:
        LevelDir = os.path.join(temp_dir, os.path.basename(level_path))
        pool.add_task("Remove temp directory {0}".format(LevelDir), nornir_shared.files.rmtree, LevelDir,
                      ignore_errors=True)

    pool.wait_completion()


def GetTempPathForTile(fullpath: str):
    """Return the temporary directory path for a tile given its full path."""
    LevelDir = os.path.basename(os.path.dirname(fullpath))
    return os.path.join(temporaryfiles.gettempdir(), LevelDir)


def GetTempDirForLevelDir(fullpath: str):
    """Return the temporary level directory path for a tileset level given its full path."""
    return os.path.join(temporaryfiles.gettempdir(), os.path.basename(fullpath))


def CreateOneTilesetTileWithPillowOverNetwork(TileDims: tuple[int, int],
                                              TopLeft: str, TopRight: str,
                                              BottomLeft: str, BottomRight: str,
                                              OutputFileFullPath: str,
                                              temp_input_dir: str | None,
                                              output_level_temp_dir: str | None,
                                              executor: ThreadPoolExecutor | None = None):
    """Copy files to a local temp directory before access to improve IO over the network since Pillow tends to issue lots
       of small IO calls instead of reading the entire file.
       The temporary files are not removed so the next tileset level can utilize the local data.
       
       Use ClearTempDirectories to clean up the temporary data

       The temporary and output directories are assumed to exist.  If they do not exist, an exception will be raised.

       Do not use the temporary local cache if the local input cached files exist.  If they do not, use the remote files
       and write the output to the temp cache for the next level.

       :param input_level_temp_dir: Input temporary directory.  If passed, it is assumed the directory exists
       :param output_level_temp_dir: Output temporary directory.  If passed, it is assumed the directory exists
       """

    should_cleanup_executor = executor is None
    if executor is None:
        executor = ThreadPoolExecutor()

    try:
        LevelDir = os.path.basename(os.path.dirname(TopLeft))
        if temp_input_dir is not None:
            temp_level_input_dir = temp_input_dir
            input_temp_dir_exists = True
        else:
            input_temp_dir_exists = False

        if output_level_temp_dir is None:
            output_level_dir = os.path.basename(os.path.dirname(OutputFileFullPath))
            temp_output_dir = os.path.join(temporaryfiles.gettempdir(), output_level_dir)
        else:
            temp_output_dir = output_level_temp_dir
 

        TopLeftBase = os.path.basename(TopLeft)
        TopRightBase = os.path.basename(TopRight)
        BottomLeftBase = os.path.basename(BottomLeft)
        BottomRightBase = os.path.basename(BottomRight)

        source_paths = (TopLeft, TopRight, BottomLeft, BottomRight)

        use_temp_dir = False
        if input_temp_dir_exists:
            cached_paths = tuple(
                os.path.join(temp_level_input_dir, base)
                for base in (TopLeftBase, TopRightBase, BottomLeftBase, BottomRightBase))
            use_temp_dir = any(os.path.exists(path) for path in cached_paths)

        if use_temp_dir:
            temp_TopLeft, temp_TopRight, temp_BottomLeft, temp_BottomRight = cached_paths
        else:
            # Read straight from the source when the local cache holds none of this tile's
            # inputs. The cache paths were previously left in place in that case -- they were
            # only replaced by the source paths when the cache *directory* was absent -- so a
            # tile whose inputs had not been cached was handed four paths that do not exist.
            # Nothing could be assembled from them, so the function logged "Pyramid tile not
            # written although source tiles exist" and returned, silently leaving a hole in the
            # level even though every source tile was present on disk.
            #
            # The directory exists but is missing a tile's inputs whenever a level is rebuilt
            # after an interrupted run: the previous run consumed and deleted the cached inputs
            # it had already used, and the caller only falls back to None when the directory is
            # gone entirely. (#256)
            temp_TopLeft, temp_TopRight, temp_BottomLeft, temp_BottomRight = source_paths
        any_source_exists = any(os.path.isfile(path) for path in source_paths)

        # Verify the contents of the temporary directory if they exist
        if use_temp_dir:
            copy_task_iter = executor.map(
                _copy_source_tile_local,
                source_paths,
                [temp_TopLeft, temp_TopRight, temp_BottomLeft, temp_BottomRight],
            )

            for _ in copy_task_iter:
                pass

        outputbase = os.path.basename(OutputFileFullPath)
        temp_output = os.path.join(temp_output_dir, outputbase)

        CreateOneTilesetTileWithPillow(TileDims, temp_TopLeft, temp_TopRight, temp_BottomLeft, temp_BottomRight,
                                       temp_output, executor=executor)

        if not os.path.exists(temp_output):
            if any_source_exists:
                existing_sources = [path for path in source_paths if os.path.isfile(path)]
                logger.warning(
                    "Pyramid tile not written although source tiles exist: output=%s sources=%s",
                    OutputFileFullPath,
                    existing_sources,
                )
            return

        # Copy synchronously so we do not queue unbounded CIFS copies on the shared executor.
        try:
            nornir_shared.files.copy_file(temp_output, OutputFileFullPath)
        except MissingTilesetInputError:
            raise
        except (FileNotFoundError, OSError) as e:
            if isinstance(e, FileNotFoundError) or e.errno == errno.ENOENT:
                dest_dir = os.path.dirname(OutputFileFullPath)
                missing_paths = [dest_dir] if dest_dir else []
                raise as_missing_tileset_error(
                    OutputFileFullPath, missing_paths, e, for_output=True) from e
            raise

        # Remove the input because this function is used to generate levels, and once we generate the next level we don't need the source level
        if use_temp_dir:
            def remove_temp_file(temp_file: str):
                """Remove a temporary file, ignoring errors if it does not exist."""
                try:
                    os.remove(temp_file)
                except IOError as e:
                    # prettyoutput.Log(f"Error removing temporary file {temp_file}: {e}")
                    pass

            remove_tasks = executor.map(remove_temp_file,
                                        [temp_TopLeft, temp_TopRight, temp_BottomLeft, temp_BottomRight])
            for _ in remove_tasks:
                pass

            try:
                if temp_input_dir is not None:
                    os.rmdir(temp_input_dir)
            except IOError:
                pass

    except Exception as e:
        raise
    finally:
        if should_cleanup_executor:
            executor.shutdown(wait=True)


def CreateOneTilesetTileWithPillow(TileDims: tuple[int, int], TopLeft: str, TopRight: str, BottomLeft: str,
                                   BottomRight: str,
                                   OutputFileFullPath: str,
                                   executor: ThreadPoolExecutor | None = None):
    """Create a single tile by merging four tiles from a higher resolution and downsampling
    :param TileDims: (Height, Width) of tiles
    :param TopLeft: Path to top-left tile
    :param TopRight: Path to top-right tile
    :param BottomLeft: Path to bottom-left tile
    :param BottomRight: Path to bottom-right tile
    :param OutputFileFullPath: Path to save the output tile
    """

    should_cleanup_executor = executor is None
    if executor is None:
        executor = ThreadPoolExecutor()

    try:
        TileSize = numpy.asarray((TileDims[1], TileDims[0]),
                                 dtype=numpy.int64)  # Pillow uses the opposite ordering of axis
        DoubleTileSize = TileSize * 2  # Double the size

        def load_and_validate_tile(tile_path: str, position: Quadrant) -> Image.Image | None:
            """
            Load a tile image and validate its size
            :param tile_path: Path to the tile image
            :param position: Description of tile position for error messages
            :return: Loaded PIL Image or None if file is missing or unreadable
            """
            # Absent quadrants are normal at mosaic edges; only warn when a path exists
            # but cannot be decoded (handled after the parallel load).
            if not os.path.isfile(tile_path):
                return None
            try:
                with Image.open(tile_path) as img:
                    if img.size[0] != TileSize[0] or img.size[1] != TileSize[1]:
                        raise ValueError(
                            f"Existing tile {tile_path} with size {img.size} does not match requested size {TileSize} at {position}")

                    # Keep pixels after the file handle closes; copy() is one buffer,
                    # unlike frombytes(tobytes()) which allocates bytes + a second image.
                    return img.copy()
            except OSError:
                return None

        # Dictionary mapping tile positions to their coordinates in the composite
        tile_positions = {
            Quadrant.TopLeft: ((0, 0), TopLeft),
            Quadrant.TopRight: ((TileSize[0], 0), TopRight),
            Quadrant.BottomLeft: ((0, TileSize[1]), BottomLeft),
            Quadrant.BottomRight: ((TileSize[0], TileSize[1]), BottomRight)
        }  # type: dict[Quadrant, tuple[tuple[int, int], str]]

        source_paths = (TopLeft, TopRight, BottomLeft, BottomRight)
        existing_sources = [path for path in source_paths if os.path.isfile(path)]

        load_futures = {
            executor.submit(load_and_validate_tile, path, quadrant): quadrant
            for quadrant, (_coords, path) in tile_positions.items()
        }
        loaded_tiles: dict[Quadrant, Image.Image | None] = {}
        for future in as_completed(load_futures):
            quadrant = load_futures[future]
            loaded_tiles[quadrant] = future.result()

        failed_existing = [
            path
            for quadrant, (_coords, path) in tile_positions.items()
            if path in existing_sources and loaded_tiles.get(quadrant) is None
        ]

        imComposite = None
        for quadrant, (coords, _path) in tile_positions.items():
            img = loaded_tiles.get(quadrant)
            if img is None:
                continue
            if imComposite is None:
                imComposite = Image.new(img.mode, size=(int(DoubleTileSize[0]), int(DoubleTileSize[1])), color=0)
            imComposite.paste(img, box=coords)
            del img

        if failed_existing:
            if imComposite is None:
                logger.warning(
                    "Pyramid tile not written; %d source file(s) present but none loaded: output=%s paths=%s",
                    len(failed_existing),
                    OutputFileFullPath,
                    failed_existing,
                )
                return
            logger.warning(
                "Pyramid tile incomplete; %d of %d existing source(s) failed to load: output=%s paths=%s",
                len(failed_existing),
                len(existing_sources),
                OutputFileFullPath,
                failed_existing,
            )
        elif imComposite is None:
            return

        resize_size = (int(TileSize[0]), int(TileSize[1]))  # Convert numpy array to tuple of ints
        imFinal = imComposite.resize(resize_size, resample=Image.Resampling.LANCZOS)  # type: ignore[union-attr]
        try:
            imFinal.save(OutputFileFullPath, optimize=True)
        except FileExistsError:
            pass
        except (FileNotFoundError, OSError) as e:
            if e.errno == errno.ENOENT or isinstance(e, FileNotFoundError):
                output_dir = os.path.dirname(OutputFileFullPath)
                missing_paths = [output_dir] if output_dir else []
                raise as_missing_tileset_error(
                    OutputFileFullPath, missing_paths, e, for_output=True) from e
            raise

        del imComposite

        return
    finally:
        if should_cleanup_executor:
            executor.shutdown(wait=True)


if __name__ == '__main__':
    pass
