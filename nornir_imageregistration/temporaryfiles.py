"""
Functions to create temporary files and directories for image registration.
The environment variable NORNIR_TEMP_DIR is used to specify the location of the temporary files.
If it is not specified the output of tempfile.TemporaryDirectory() is used.
"""

import os
import tempfile

__tempdir = None


def gettempdir() -> str:
    """
    Get the temporary directory for the current user.
    If the environment variable NORNIR_TEMP_DIR is set, use that directory.
    Otherwise, use the default temporary directory.
    :return:
    """
    global __tempdir

    if __tempdir is not None:
        return __tempdir

    if "NORNIR_TEMP_DIR" in os.environ:
        __tempdir = os.environ["NORNIR_TEMP_DIR"]
    else:
        __tempdir = tempfile.gettempdir()

    return __tempdir
