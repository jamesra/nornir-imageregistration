'''
Created on Jul 18, 2019

@author: u0490822

A helper class to marshal large images using the file system instead of in-memory.
'''
from __future__ import annotations

import atexit
import collections
import logging
import os
import shutil
import tempfile
import threading
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
import nornir_pools
from nornir_imageregistration.shared_mem_metadata import Shared_Mem_Metadata
from nornir_imageregistration.transformed_image_data import ITransformedImageData, TransformedImageDataState


# When porting to Python 3.10 there was a regression where
# concurrent.futures.ThreadPoolExecutor system could not function in atext calls
# So I reverted to shutil until it is fixed
# atexit.register(nornir_shared.files.rmtree, _sharedTempRoot)
# atexit.register(shutil.rmtree, _sharedTempRoot, ignore_errors=True)


class TransformedImageDataViaTempFile(ITransformedImageData):
    """
    Returns data from multiprocessing thread processes.  Uses memory mapped files when there is too much data for pickle to be efficient
    """
    _image_path: str | None
    _centerDistanceImage_path: str | None
    _image: NDArray[np.floating] | None
    _centerDistanceImage: NDArray[np.floating] | None
    _source_space_scale: float
    _target_space_scale: float
    _image_state: TransformedImageDataState
    _center_distance_image_state: TransformedImageDataState
    _transform: Any | None
    _errmsg: str | None
    _rendered_target_space_origin: NDArray[np.float32]

    _temp_folder_created = False
    # Declared as sharedTempRoot while every read and write used _sharedTempRoot, so this
    # default applied to a name nothing referenced and the name that was referenced did not
    # exist until ConvertToTempFileIfLarge created it. (#106)
    _sharedTempRoot = None
    # Serializes creation of the shared folder. Without it every thread that reached the
    # check before any of them set the flag created its own root. (#107)
    _temp_folder_lock = threading.Lock()

    # Paths whose deletion failed because a memmap was still open, to retry later. (#108)
    _pending_deletions: collections.deque[str] = collections.deque()
    _pending_deletions_lock = threading.Lock()
    _max_deletion_retries_per_call = 8

    tempfile_threshold = 64 * 64

    @property
    def errormsg(self) -> str | None:
        return self._errmsg

    @property
    def state(self) -> TransformedImageDataState:
        if self._image_state == TransformedImageDataState.CLEARED or \
                self._center_distance_image_state == TransformedImageDataState.CLEARED:
            return TransformedImageDataState.CLEARED
        if self._image_state == TransformedImageDataState.TEMP_FILE or \
                self._center_distance_image_state == TransformedImageDataState.TEMP_FILE:
            return TransformedImageDataState.TEMP_FILE
        return TransformedImageDataState.IN_MEMORY

    #
    #     def __getstate__(self):
    #         odict = {}
    #         odict["_image"] = self._image
    #         odict["_centerDistanceImage"] = self._centerDistanceImage
    #         odict["_source_space_scale"] = self._source_space_scale
    #         odict["_transform"] = self._transform
    #         odict["_errmsg"] = self._errmsg
    #         odict["_image_path"] = self._image_path
    #         odict["_centerDistanceImage_path"] = self._centerDistanceImage_path
    #         odict["_tempdir"] = self._tempdir
    #         odict["_image_shape"] = self._image_shape
    #         odict["_centerDistanceImage_shape"] = self._centerDistanceImage_shape
    #         odict["_image_dtype"] = self._image_dtype
    #         odict["_centerDistance_dtype"] = self._centerDistance_dtype
    #         return odict
    #
    #     def __setstate__(self, dictionary):
    #         self.__dict__.update(dictionary)
    #

    @property
    def image(self) -> NDArray:
        if self._image_state == TransformedImageDataState.CLEARED:
            raise ValueError("No image associated with TransformedImageData")

        image = self._image
        if image is None:
            if self._image_path is None:
                raise ValueError("No image associated with TransformedImageData")

            image = np.load(self._image_path,
                            mmap_mode='r')  # np.memmap(self._image_path, mode='c', shape=self._image_shape, dtype=self._image_dtype)
            self._image = image

        return image

    @property
    def centerDistanceImage(self) -> NDArray:
        if self._center_distance_image_state == TransformedImageDataState.CLEARED:
            raise ValueError("No distance image associated with TransformedImageData")

        distance_image = self._centerDistanceImage
        if distance_image is None:
            if self._centerDistanceImage_path is None:
                raise ValueError("No distance image associated with TransformedImageData")

            distance_image = np.load(self._centerDistanceImage_path,
                                     mmap_mode='r')  # np.memmap(self._centerDistanceImage_path, mode='c', shape=self._centerDistanceImage_shape, dtype=self._centerDistance_dtype)
            self._centerDistanceImage = distance_image

        return distance_image

    @property
    def source_space_scale(self) -> float:
        return self._source_space_scale

    @property
    def target_space_scale(self) -> float:
        return self._target_space_scale

    @property
    def rendered_target_space_origin(self) -> NDArray[np.float32]:
        """
        The bottom left origin of the transformed data.  When requesting an assembled image for a target region
        rounding sometimes can occur this property contains the actual bottom left coordinate of the image data
        in target space.
        :return:
        """
        return self._rendered_target_space_origin

    # @property
    # def transform(self):
    #    return self._transform

    @classmethod
    def Create(cls, image: NDArray | Shared_Mem_Metadata, centerDistanceImage: NDArray | Shared_Mem_Metadata,
               transform,
               source_space_scale: float, target_space_scale: float,
               rendered_target_space_origin: Tuple[float, float], SingleThreadedInvoke: bool):
        o = TransformedImageDataViaTempFile(source_space_scale=source_space_scale,
                                            target_space_scale=target_space_scale,
                                            rendered_target_space_origin=rendered_target_space_origin)
        o._image = nornir_imageregistration.ImageParamToImageArray(image)
        o._centerDistanceImage = nornir_imageregistration.ImageParamToImageArray(centerDistanceImage)
        o._image_state = TransformedImageDataState.IN_MEMORY
        o._center_distance_image_state = TransformedImageDataState.IN_MEMORY
        o._transform = transform

        o._image_path = None
        o._centerDistanceImage_path = None
        # o._transform = transform

        if not SingleThreadedInvoke:
            o.ConvertToTempFileIfLarge()

        return o

    @staticmethod
    def _EnsureSharedTempFolder() -> str:
        """Create this process's shared temporary folder if it does not exist yet.

        Extracted so SaveArrayToTemporaryFile can guarantee the folder before writing.  Every
        file written here has delete=False and is reclaimed only by the atexit handler
        registered below, which covers this folder alone -- so a file written outside it would
        never be cleaned up.  (#106)

        Creation is serialized. The check and the set used to be separate, so every thread that
        reached the check before any of them set the flag created its own root -- measured 8 of
        8 with 8 threads, with 7 of 8 saved files landing in a root other than the one finally
        published. Each root was registered for cleanup, so a clean exit still removed them
        all; the costs were a directory per caller for the process lifetime, files scattered
        across roots rather than in the shared one, and 8 roots instead of 1 left behind if the
        process dies without running atexit handlers.

        The lock is held for the whole function rather than double-checked outside it. It is
        uncontended after the first call, and every caller goes on to write a .npy file, so the
        acquire is not measurable here -- not worth reasoning about unsynchronized reads of the
        two attributes, especially on a free-threaded build. (#107)

        :return: path of the shared temporary folder
        """
        with TransformedImageDataViaTempFile._temp_folder_lock:
            if not TransformedImageDataViaTempFile._temp_folder_created:
                temp_dir = nornir_imageregistration.gettempdir()
                shared_temp_root = tempfile.mkdtemp(
                    prefix="nornir-imageregistration.transformed_image_data.", dir=temp_dir)

                # Registered against the local, not the class attribute. The original re-read
                # the attribute to build this argument, so a thread preempted between
                # publishing its path and evaluating the argument would have registered
                # whichever path another thread published -- leaving its own unregistered and
                # double-registering the other. Narrow enough that it never reproduced, but the
                # local removes it rather than relying on scheduling. (#107)
                atexit.register(shutil.rmtree, shared_temp_root, ignore_errors=True)

                # Registered after the rmtree so it runs before it: atexit is LIFO, and a
                # deferred file is worth one last individual attempt (and a warning) while the
                # directory still exists. (#108)
                atexit.register(TransformedImageDataViaTempFile._FlushPendingDeletions)

                TransformedImageDataViaTempFile._sharedTempRoot = shared_temp_root
                TransformedImageDataViaTempFile._temp_folder_created = True

            return TransformedImageDataViaTempFile._sharedTempRoot

    @staticmethod
    def SaveArrayToTemporaryFile(name: str, image: NDArray) -> str:
        """
        Save the image to a temporary file and return the name of the temporary file
        :param name: Suffix to prepend to the filename
        :param image: NDArray to save
        :return: name of temporary file
        """
        if image is None:
            raise ValueError("image cannot be None")

        # Called unconditionally rather than relying on ConvertToTempFileIfLarge having run
        # first. This is a public staticmethod, and reaching it directly raised AttributeError
        # on the missing _sharedTempRoot; with the declaration corrected it would instead fall
        # back to dir=None, writing an undeleted file into the system temp dir that the atexit
        # cleanup does not cover. (#106)
        shared_temp_root = TransformedImageDataViaTempFile._EnsureSharedTempFolder()

        with tempfile.NamedTemporaryFile(suffix=name + '.npy', dir=shared_temp_root,
                                         delete=False) as tfile:
            np.save(tfile, image)
            return tfile.name

    def ConvertToTempFileIfLarge(self):
        '''
        Save our image data into files.  This gets it out of memory, lowering our footprint.  When we return
        to the calling process we do not need to marshal the images across a pipe.  This was replaced by
        use of SharedMemory, but that implementation seems to destroy the sharedmemory before it can be
        returned to the caller.
        :return:
        '''
        image = self.image
        center_distance_image = self.centerDistanceImage
        if np.prod(image.shape) > TransformedImageDataViaTempFile.tempfile_threshold:
            _image_path_task = None
            _centerDistanceImage_path_task = None

            # Create the temporary directory if it doesn't exist.  Done here as well as inside
            # SaveArrayToTemporaryFile so it happens once on this thread rather than racing
            # between the two pool tasks submitted below.
            TransformedImageDataViaTempFile._EnsureSharedTempFolder()

            # TODO: Replace with a task group once we are on Python 3.11
            pool = nornir_pools.GetGlobalThreadPool()

            _image_path_task = pool.add_task("Image", self.SaveArrayToTemporaryFile, "Image", image)
            self._image = None
            self._image_state = TransformedImageDataState.TEMP_FILE

            _centerDistanceImage_path_task = pool.add_task("Distance",
                                                           self.SaveArrayToTemporaryFile, "Distance",
                                                           center_distance_image)
            self._centerDistanceImage = None
            self._center_distance_image_state = TransformedImageDataState.TEMP_FILE

            self._image_path = _image_path_task.wait_return()
            self._centerDistanceImage_path = _centerDistanceImage_path_task.wait_return()

        return

    def Clear(self):
        """Release loaded arrays and any temporary files."""
        self._image = None
        self._centerDistanceImage = None
        self._image_state = TransformedImageDataState.CLEARED
        self._center_distance_image_state = TransformedImageDataState.CLEARED
        self._transform = None

        # It is hard to delete these temporary files because it is ambiguous on when
        # numpy releases the underlying file
        if self._centerDistanceImage_path is not None or self._image_path is not None:
            pool = nornir_pools.GetGlobalThreadPool()
            pool.add_task(str(self._image_path), TransformedImageDataViaTempFile._RemoveTempFiles,
                          self._centerDistanceImage_path,
                          self._image_path)
            self._centerDistanceImage_path = None
            self._image_path = None

    @staticmethod
    def _TryRemoveTempFile(path: str | None) -> bool:
        """Attempt to delete one temporary file.

        :return: True if the path is gone (deleted, already absent, or None), False if it is
            still held and should be retried.
        """
        if path is None:
            return True

        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            # On Windows an open memmap makes this a sharing violation. Caught as IOError
            # before, which is the same exception -- IOError is an alias for OSError and
            # PermissionError derives from it -- so the only thing that changes here is that
            # the caller now learns it failed. (#108)
            return False

    @staticmethod
    def _RemoveTempFiles(_centerDistanceImage_path, _image_path):
        """Delete this instance's temporary files, and retry a bounded slice of earlier failures.

        Deletion can legitimately fail: the arrays are handed out by the image and
        centerDistanceImage properties as memmaps, and while a caller still holds one -- or a
        view of one, which keeps it alive through .base -- the file cannot be removed. Clearing
        the instance's own reference does not help in that case. Measured: the live assemble
        path drops its arrays before calling Clear and deletes both files successfully, but a
        retained reference or view leaves both behind.

        Previously the failure was logged and dropped, so the file stayed for the lifetime of
        the process even though it becomes deletable the moment the caller lets go -- confirmed
        by retrying by hand. Failures are now re-queued and retried on subsequent calls.

        Only a bounded number of pending paths are retried per call so that a run which
        accumulates many undeletable files does not turn each Clear into a sweep of all of
        them. Anything still pending is attempted once more at exit. (#108)
        """
        cls = TransformedImageDataViaTempFile

        deferred = [path for path in (_centerDistanceImage_path, _image_path)
                    if not cls._TryRemoveTempFile(path)]

        with cls._pending_deletions_lock:
            retry_count = min(len(cls._pending_deletions), cls._max_deletion_retries_per_call)
            retries = [cls._pending_deletions.popleft() for _ in range(retry_count)]

        deferred.extend(path for path in retries if not cls._TryRemoveTempFile(path))

        if len(deferred) > 0:
            with cls._pending_deletions_lock:
                cls._pending_deletions.extend(deferred)

            # Debug rather than warning: a first failure is expected whenever the consumer still
            # holds the array, and it resolves itself. Residue that outlives the process is
            # reported once by _FlushPendingDeletions instead of once per tile.
            logging.getLogger(__name__).debug(
                'Deferred deletion of %d temporary file(s); %d now pending',
                len(deferred), len(cls._pending_deletions))

    @staticmethod
    def _FlushPendingDeletions():
        """Final attempt at any deferred deletions, registered atexit.

        Runs before the shared root's rmtree, which is registered earlier and so runs later.
        The rmtree would remove these files anyway on a clean exit; the value here is the
        warning, which is the only signal that something held memmaps for the whole run. (#108)
        """
        cls = TransformedImageDataViaTempFile
        with cls._pending_deletions_lock:
            pending = list(cls._pending_deletions)
            cls._pending_deletions.clear()

        still_held = [path for path in pending if not cls._TryRemoveTempFile(path)]

        if len(still_held) > 0:
            logging.getLogger(__name__).warning(
                '%d temporary file(s) could not be deleted; a memmap was still open. '
                'They are under %s and are removed with it. First: %s',
                len(still_held), cls._sharedTempRoot, still_held[0])

    def __init__(self,
                 source_space_scale: float = 0.0,
                 target_space_scale: float = 0.0,
                 rendered_target_space_origin: Tuple[float, float] = (0.0, 0.0),
                 errorMsg: str | None = None):
        self._image = None
        self._centerDistanceImage = None
        self._source_space_scale = source_space_scale
        self._target_space_scale = target_space_scale
        self._rendered_target_space_origin = np.asarray(rendered_target_space_origin, dtype=np.float32)
        self._image_state = TransformedImageDataState.CLEARED
        self._center_distance_image_state = TransformedImageDataState.CLEARED
        self._transform = None
        self._errmsg = errorMsg
        self._image_path = None
        self._centerDistanceImage_path = None
        self._tempdir = None
        # self._image_shape = None
        # self._centerDistanceImage_shape = None
        # self._image_dtype = None
        # self._centerDistance_dtype = None
