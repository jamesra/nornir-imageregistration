import warnings
import logging

from typing import Sequence, Type

import nornir_imageregistration.debugging

import numpy as np
import scipy.linalg


class IgnoreRuntimeWarnings:
    """A context manager that swallows warnings with set keywords in release and sends them to the log in debug mode."""
    _original_filters: Sequence[tuple[str, str | None, type[Warning], str | None, int]] = None
    _record: warnings.catch_warnings | None = None
    _warnings: list[Warning] | None = None
    _warnings_to_filter: list[str | Type[Warning]]
    _log_msg: str | None = None

    @property
    def warnings(self) -> list[Warning] | None:
        """Returns the list of warnings that were caught."""
        return self._warnings

    def __init__(self, warnings_to_filter: list[str] | str | Type[Warning], log_msg: str | None = None):
        self._log_msg = log_msg if log_msg is not None else ""
        if isinstance(warnings_to_filter, list):
            self._warnings_to_filter = warnings_to_filter
        else:
            self._warnings_to_filter = [warnings_to_filter]

    def __enter__(self):
        self._original_filters = warnings.filters[:]
        warnings.simplefilter("always", RuntimeWarning)
        self._record = warnings.catch_warnings(record=True)
        self._warnings = self._record.__enter__()
        return self._warnings

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._record.__exit__(exc_type, exc_val, exc_tb)
        for warning in self._warnings:
            if issubclass(warning.category, RuntimeWarning):
                if warning.message and warning.message.args and len(warning.message.args) > 0:
                    message = warning.message.args[0]
                    for warning_to_filter in self._warnings_to_filter:
                        if isinstance(warning_to_filter, str):
                            if warning_to_filter in message:
                                if nornir_imageregistration.debugging.in_debug_mode():
                                    logging.warning(f'{message}: {self._log_msg}')
                                else:
                                    pass
                                break
                        elif issubclass(warning.category, warning_to_filter):
                            if nornir_imageregistration.debugging.in_debug_mode():
                                logging.warning(f'{message}: {self._log_msg}')
                            else:
                                pass
                            break
                        else:
                            warnings.warn(message, RuntimeWarning)

        warnings.filters = self._original_filters


class IgnoreUnderflow(IgnoreRuntimeWarnings):
    def __init__(self, log_msg: str | None = None):
        super().__init__('underflow', log_msg)


class IgnoreOverflow(IgnoreRuntimeWarnings):
    def __init__(self, log_msg: str | None = None):
        super().__init__('overflow', log_msg)


class IgnoreUnderAndOverflow(IgnoreRuntimeWarnings):
    def __init__(self, log_msg: str | None = None):
        super().__init__(['underflow', 'overflow'], log_msg)


class IgnoreLinAlgWarning(IgnoreRuntimeWarnings):
    def __init__(self, log_msg: str | None = None):
        super().__init__(scipy.linalg.LinAlgWarning, log_msg)
