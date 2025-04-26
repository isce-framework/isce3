from __future__ import annotations

import logging
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime


def get_logger() -> logging.Logger:
    """ """
    logger = logging.getLogger("STATIC")
    if not logger.handlers:
        set_logger_handler(logger)
    return logger


def set_logger_handler(
    logger: logging.Logger,
    log_file: os.PathLike | str | None = None,
    *,
    mode: str = "w",
    verbose: bool = False,
) -> None:
    """ """
    if mode not in ("w", "a"):
        raise ValueError(f"{mode=}, must be either 'w' or 'a'.")

    for handler in logger.handlers:
        logger.removeHandler(handler)

    log_level = logging.DEBUG
    logger.setLevel(log_level)

    msgfmt = (
        f"%(asctime)s.%(msecs)03d, %(levelname)s, STATIC, "
        f'999997, %(pathname)s:%(lineno)d, "%(message)s"'
    )
    fmt = logging.Formatter(msgfmt, "%Y-%m-%d %H:%M:%S")

    if (log_file is None) or verbose:
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

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
    yield
    toc = datetime.now()
    log_func(f"{what} took {toc - tic}")
