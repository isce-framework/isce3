from __future__ import annotations

import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def create_tmp_text_file(
    contents: str, suffix: str | None = None
) -> Generator[Path, None, None]:
    """
    A context manager that creates a temporary text file with the specified contents.

    The file is automatically removed from the file system when the context block exits.

    Parameters
    ----------
    contents : str
        The contents of the text file.
    suffix : str or None, optional
        An optional file name suffix. If None, there will be no suffix. Defaults to
        None.

    Yields
    ------
    pathlib.Path
        The file system path of the temporary file.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        filepath = Path(f.name)
        filepath.write_text(contents)
        yield filepath
