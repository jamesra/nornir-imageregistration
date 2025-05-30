"""
Created on Sep 10, 2019

@author: u0490822

These functions generate tileset image pyramid levels.  The network implementation
of these functions copies the images locally and writes the output locally before
moving it to the final output directory.  This saves trips over the network as
we build the pyramid, which tends to be slow for sometimes hundreds of thousands
of small files.  This also helps the image I/O, which at this time is implemented
by pillow as lots of small I/O requests against the image file.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import concurrent.futures

from PIL import Image
import numpy

# Disable decompression bomb protection since we are dealing with huge images on purpose
Image.MAX_IMAGE_PIXELS = None
import os
import enum
import shutil
from functools import partial
import nornir_pools
import nornir_shared.files
import nornir_imageregistration.temporaryfiles as temporaryfiles


class Quadrant(enum.IntEnum):
    """An enum describing quadrants"""
    TopLeft = 1
    TopRight = 2
    BottomLeft = 3
    BottomRight = 4


# import nornir_shared.prettyoutput as prettyoutput

def ClearTempDirectories(level_paths):
    """Deletes temporary directories used to generate levels"""

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
    """
    Given a tileset image, return the temporary filename for the tile
    """
    LevelDir = os.path.basename(os.path.dirname(fullpath))
    return os.path.join(temporaryfiles.gettempdir(), LevelDir)


def GetTempDirForLevelDir(fullpath: str):
    """
    Given a tileset level, return the temporary level directory
    """
    return os.path.join(temporaryfiles.gettempdir(), os.path.basename(fullpath))


def CreateOneTilesetTileWithPillowOverNetwork(TileDims: tuple[int, int],
                                              TopLeft: str, TopRight: str,
                                              BottomLeft: str, BottomRight: str,
                                              OutputFileFullPath: str,
                                              output_level_temp_dir: str | None,
                                              executor: ThreadPoolExecutor | None = None):
    """Copy files to a local temp directory before access to improve IO over the network since Pillow tends to issue lots
       of small IO calls instead of reading the entire file.
       The temporary files are not removed so the next tileset level can utilize the local data.
       Use ClearTempDirectories to clean up the temporary data

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
        temp_input_dir = os.path.join(temporaryfiles.gettempdir(), LevelDir)
        input_temp_dir_exists = os.path.exists(temp_input_dir)

        if output_level_temp_dir is None:
            output_level_dir = os.path.basename(os.path.dirname(OutputFileFullPath))
            temp_output_dir = os.path.join(temporaryfiles.gettempdir(), output_level_dir)
            os.makedirs(temp_output_dir, exist_ok=True)
        else:
            temp_output_dir = output_level_temp_dir

        TopLeftBase = os.path.basename(TopLeft)
        TopRightBase = os.path.basename(TopRight)
        BottomLeftBase = os.path.basename(BottomLeft)
        BottomRightBase = os.path.basename(BottomRight)

        temp_TopLeft = os.path.join(temp_input_dir, TopLeftBase)
        temp_TopRight = os.path.join(temp_input_dir, TopRightBase)
        temp_BottomLeft = os.path.join(temp_input_dir, BottomLeftBase)
        temp_BottomRight = os.path.join(temp_input_dir, BottomRightBase)

        use_temp_dir = input_temp_dir_exists and any(
            os.path.exists(x) for x in [temp_TopLeft, temp_TopRight, temp_BottomLeft, temp_BottomRight])

        if not use_temp_dir:
            temp_TopLeft = TopLeft
            temp_TopRight = TopRight
            temp_BottomLeft = BottomLeft
            temp_BottomRight = BottomRight

        def try_copy_local(src: str, dst: str):
            """Try to copy a file locally, ignoring errors if the file does not exist."""
            try:
                # If the file does not exist, or is older than the source, copy it
                if nornir_shared.files.IsOutdated(src, dst):
                    shutil.copyfile(src, dst)
                    return True

                return False
            except IOError as e:
                # prettyoutput.Log(f"Missing input file {src}: {e}")
                return False

        # Verify the contents of the temporary directory if they exist
        if use_temp_dir:
            copy_task_iter = executor.map(try_copy_local,
                                          [TopLeft, TopRight, BottomLeft, BottomRight],
                                          [temp_TopLeft, temp_TopRight, temp_BottomLeft, temp_BottomRight])

            for copied in copy_task_iter:
                pass

        outputbase = os.path.basename(OutputFileFullPath)
        temp_output = os.path.join(temp_output_dir, outputbase)

        CreateOneTilesetTileWithPillow(TileDims, temp_TopLeft, temp_TopRight, temp_BottomLeft, temp_BottomRight,
                                       temp_output, executor=executor)

        # Copy the file, but leave the temp in case we genereate the next level
        executor.submit(shutil.copyfile, temp_output, OutputFileFullPath)

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
                os.rmdir(temp_input_dir)
            except IOError:
                pass

    except Exception as e:
        raise
    finally:
        if should_cleanup_executor:
            executor.shutdown(wait=False)  # Copies and deletes finish in the background


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
            :return: Loaded PIL Image or None if file is missing
            """
            try:
                with Image.open(tile_path) as img:
                    if img.size[0] != TileSize[0] or img.size[1] != TileSize[1]:
                        raise ValueError(
                            f"Existing tile {tile_path} with size {img.size} does not match requested size {TileSize} at {position}")

                    # Create a new PIL image from the array, ensuring it's in the right format
                    return Image.frombytes(img.mode, img.size, img.tobytes())
            except IOError:
                return None

        # Create a composite image to hold all tiles
        imComposite = None

        # Dictionary mapping tile positions to their coordinates in the composite
        tile_positions = {
            Quadrant.TopLeft: ((0, 0), TopLeft),
            Quadrant.TopRight: ((TileSize[0], 0), TopRight),
            Quadrant.BottomLeft: ((0, TileSize[1]), BottomLeft),
            Quadrant.BottomRight: ((TileSize[0], TileSize[1]), BottomRight)
        }  # type: dict[Quadrant, tuple[tuple[int, int], str]]

        imComposite = None

        def load_tile_done_callback(future: Future, quadrant: Quadrant):
            # Process tiles as they complete
            nonlocal imComposite
            nonlocal DoubleTileSize
            img = future.result()
            coords = tile_positions[quadrant][0]
            if img is not None:
                if imComposite is None:
                    imComposite = Image.new(img.mode, size=(int(DoubleTileSize[0]), int(DoubleTileSize[1])), color=0)
                imComposite.paste(img, box=coords)
                del img  # Explicitly delete the image to free memory

        # Load all tiles in parallel using ThreadPoolExecutor
        # Create a map of futures to their positions
        future_to_position = {}
        for quadrant, (coords, path) in tile_positions.items():
            future = executor.submit(load_and_validate_tile, path, quadrant)
            callback_partial = partial(load_tile_done_callback, quadrant=quadrant)
            future.add_done_callback(callback_partial)
            future_to_position[future] = quadrant

        # Append the callback to write images to imComposite.
        for future in as_completed(list(future_to_position.keys())):
            future.result()  # Trigger any exceptions from futures to be raised

        if imComposite is not None:
            resize_size = (int(TileSize[0]), int(TileSize[1]))  # Convert numpy array to tuple of ints
            with imComposite.resize(resize_size, resample=Image.LANCZOS) as imFinal:
                try:
                    imFinal.save(OutputFileFullPath, optimize=True)
                except FileExistsError:
                    pass

            del imComposite

        return
    finally:
        if should_cleanup_executor:
            executor.shutdown(wait=True)


if __name__ == '__main__':
    pass
