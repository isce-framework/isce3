from __future__ import annotations

import logging
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime


def get_logger() -> logging.Logger:
    """
    Get the Static Layers logger.

    Create or get the singleton instance of the logger used for modules within the
    NISAR Static Layers workflows.

    Returns
    -------
    logging.Logger
        The global Static Layers logger.

    See Also
    --------
    set_logger_handler
    """
    logger = logging.getLogger("STATIC")
    if not logger.handlers:
        set_logger_handler(logger)
    return logger


def set_logger_handler(
    logger: logging.Logger | None = None,
    log_file: os.PathLike | str | None = None,
    *,
    mode: str = "w",
    verbose: bool = False,
) -> None:
    """
    Configure the input logger for the NISAR Static Layers workflow.

    Defines the log message format and adds handlers for logging to console and/or to an
    output text file. Any previously existing handlers will be removed.

    Parameters
    ----------
    logger : logging.Logger or None, optional
        The logger instance to configure. If None, uses the global Static Layers logger.
        Defaults to None.
    log_file : path-like or None, optional
        Optional output text file to log messages to. If None, log messages will be
        directed to the console. Defaults to None.
    mode : {'w', 'a'}, optional
        The mode for logging to a file. Ignored if `log_file` is None.

        'w':
          The default mode. Create the file, truncate if it exists.

        'a':
          Append to the file if it exists. Otherwise, create the file.
    verbose : bool, optional
        If True, direct log messages to the console as well as the log file. Ignored if
        `log_file` is None. Defaults to False.

    See Also
    --------
    get_logger
    """
    if mode not in ("w", "a"):
        raise ValueError(f"{mode=}, must be either 'w' or 'a'.")

    # Remove any existing handlers.
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # Set the minimum log level to `DEBUG`.
    log_level = logging.DEBUG
    logger.setLevel(log_level)

    # Configure the log message format to use the NISAR CSV log format, defined by the
    # L0B PGE Design Document (Section 9). Each SAS workflow is supposed to have a
    # unique integer code associated with its log messages. Currently, RSLC uses 999999
    # and QA uses 999998. Here we set the Static Layers code to 999997.
    msgfmt = (
        f"%(asctime)s.%(msecs)03d, %(levelname)s, STATIC, "
        f'999997, %(pathname)s:%(lineno)d, "%(message)s"'
    )
    fmt = logging.Formatter(msgfmt, "%Y-%m-%d %H:%M:%S")

    # If requested, add a stream handler for logging to console.
    if (log_file is None) or verbose:
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    # Optionally add a file handler for logging to an output file.
    if log_file is not None:
        handler = logging.FileHandler(filename=log_file, mode=mode)
        handler.setLevel(log_level)
        handler.setFormatter(fmt)
        logger.addHandler(handler)


@contextmanager
def log_elapsed_time(
    log_func: Callable[[str], None],
    what: str,
) -> Generator[None, None, None]:
    """
    Log the elapsed time of a `with` block.

    When used as a context manager, measures and logs the elapsed time between when the
    context manager's `__enter__` and `__exit__` methods were invoked, with up-to
    microsecond precision (depending on the precision of the underlying clock).

    Parameters
    ----------
    log_func : callable
        A function object that writes a message to the log.
    what : str
        Prefix for the log message. The body of logged message will be
        '<what> took <elapsed>'.
    """
    tic = datetime.now()
    try:
        yield
    finally:
        toc = datetime.now()
        log_func(f"{what} took {toc - tic}")
