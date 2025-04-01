import warnings
import logging

from typing import Sequence

import nornir_imageregistration.debugging


class IgnoreRuntimeWarnings:
    """A context manager that swallows warnings with set keywords in release and sends them to the log in debug mode."""
    _original_filters: Sequence[tuple[str, str | None, type[Warning], str | None, int]] = None
    _record: warnings.catch_warnings | None = None
    _warnings: list[Warning] | None = None
    _warnings_to_filter: list[str]
    _log_msg: str | None = None

    def __init__(self, warnings_to_filter: list[str] | str, log_msg: str | None = None):
        self._log_msg = log_msg if log_msg is not None else ""
        self._warnings_to_filter = warnings_to_filter if isinstance(warnings_to_filter, list) else [warnings_to_filter]

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
                        if warning_to_filter in message:
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
