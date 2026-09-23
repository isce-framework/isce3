import numpy as np
from typing import Iterator

def cola_windows(n: int, window_size: int = 64) -> Iterator[tuple[slice, np.ndarray]]:
    """
    Generate windows with the constant-overlap-add (COLA) property.
    Hann windows are generated with 50% overlap and sum to one.
    The first segment is a half window, and final two windows may
    be partial.

    Parameters
    ----------
    n : int
        Length of array to analyze.
    window_size : int, optional
        Length of analysis blocks. Must be even-valued. Defaults to 64.

    Yields
    -------
    selection : slice
        Segment to index in data array.
    window : numpy.ndarray
        Hann window (possibly partial) for block analysis/synthesis.
    """
    if window_size % 2 != 0:
        raise ValueError("window_size must be even")
    w0 = np.hanning(window_size + 1)[:window_size]
    half = window_size // 2
    for i in range(0, n + half - 1, half):
        dstart = i - half
        dend = dstart + window_size
        wstart = 0
        wend = window_size
        if dstart < 0:
            wstart = -dstart
            dstart = 0
        if dend > n:
            dend = n
            wend = dend - dstart
        yield slice(dstart, dend), w0[wstart:wend]
