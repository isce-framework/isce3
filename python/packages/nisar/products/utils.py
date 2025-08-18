from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def to_bytes(s: str | ArrayLike) -> np.ndarray:
    """
    Convert a Unicode string or array of strings into bytes.
    Parameters
    ----------
    s : str or array_like
        A Unicode string or array of Unicode strings.
    Returns
    -------
    numpy.ndarray
        The input string(s) converted to bytestrings with 'utf-8' encoding.
    """
    return np.char.encode(s, encoding="utf-8")