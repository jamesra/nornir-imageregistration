"""Retry file reads that a network share refuses for a moment.

SMB/CIFS shares answer some opens with a transient error instead of data: ``EAGAIN``
when too many opens are in flight (the client is out of credits), ``EBUSY`` when the
server reports a sharing violation from another process on the NAS. The same file reads
fine a moment later, so failing the whole import on the first refusal is wrong
(RC2 raw TIFs on ``/backup1``, a Synology over SMB 3.1.1, 2026-11-05).
"""
from __future__ import annotations

import errno
import logging
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar('T')

#: errno values that mean "try again", not "this file is broken".
TRANSIENT_ERRNOS = frozenset({errno.EAGAIN, errno.EWOULDBLOCK, errno.EBUSY})

_logger = logging.getLogger(__name__)


def is_transient(error: BaseException) -> bool:
    """True for an OSError a share raises when it will likely succeed on a retry."""
    return isinstance(error, OSError) and getattr(error, 'errno', None) in TRANSIENT_ERRNOS


def call_with_io_retry(read: Callable[[], T], attempts: int, base_delay: float, description: str = '',
                       sleep: Callable[[float], None] = time.sleep, log_retries: bool = True) -> T:
    """Call ``read``, retrying transient share errors with doubling delays.

    :param read: the read to attempt; called with no arguments.
    :param attempts: total calls, the first included.
    :param base_delay: seconds before the first retry; each later wait doubles.
    :param description: what is being read, for the retry log line.
    :param sleep: injectable for tests.
    :param log_retries: log a warning before each retry.
    :raises OSError: the last transient error after ``attempts`` tries, or any other error at once.
    """
    for attempt in range(attempts):
        try:
            return read()
        except OSError as error:
            if not is_transient(error) or attempt == attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            if log_retries:
                _logger.warning('Transient read error on %s (%s); retry %d/%d in %.2f s',
                                description, error, attempt + 1, attempts - 1, delay)
            sleep(delay)
    raise AssertionError('unreachable: the loop returns or raises')
