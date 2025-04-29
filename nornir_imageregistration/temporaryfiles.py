"""
Functions to create temporary files and directories for image registration.
The environment variable NORNIR_TEMP_DIR is used to specify the location of the temporary files.
If it is not specified the output of tempfile.TemporaryDirectory() is used.
"""

import os
import tempfile

__first_call = True


def gettempdir() -> str:
    """
    Get the temporary directory for the current user.
    If the environment variable NORNIR_TEMP_DIR is set, use that directory.
    Otherwise, use the default temporary directory.
    :return:
    """
    tempdir = os.environ.get("NORNIR_TEMP_DIR", tempfile.gettempdir())
    global __first_call
    if __first_call:

        if not os.path.exists(tempdir):
            os.makedirs(tempdir, exist_ok=True)

        __first_call = False

    return tempdir
