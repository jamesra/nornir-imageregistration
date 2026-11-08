"""Shared invalid-argument exit for the nornir_imageregistration command-line scripts."""
import argparse
import logging
import sys
from typing import NoReturn

USE_ERROR_EXIT_CODE = 2
"""Same status argparse uses for invalid arguments, so wrappers checking the exit code see a failure."""


def exit_on_use_error(parser: argparse.ArgumentParser, logger_name: str, message: str) -> NoReturn:
    """Print the parser's usage, log ``message`` as an error on ``logger_name``, and exit with status 2."""
    parser.print_usage()
    logging.getLogger(logger_name).error(message)
    sys.exit(USE_ERROR_EXIT_CODE)
