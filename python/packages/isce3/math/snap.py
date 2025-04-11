import numpy as np
from isce3.core import DateTime, TimeDelta


def snap(x, interval, round_func=np.ceil):
    """
    Round a location to the nearest grid point.

    Parameters
    ----------
    x : float
        An arbitary location on the real line.
    interval : float
        The spacing of a grid with origin at x=0.
        Same units as x.
    round_func : Callable[[float], float], optional
        Function to use for rounding.  Defaults to `numpy.ceil`.

    Returns
    -------
    xg : float
        The nearest grid point to x according to the given rounding mode.
        That is, `xg = interval * i` for some integer `i`.
    """
    return round_func(x / interval) * interval


def snap_datetime(t, interval, round_func=np.ceil):
    """
    Round a time to the nearest grid point (counted from midnight).

    Parameters
    ----------
    t : isce3.core.DateTime
        An arbitary time point.
    interval : float
        The spacing (in seconds) of a time grid with origin at the midnight on
        or before `t`.
    round_func : Callable[[float], float], optional
        Function to use for rounding.  Defaults to `numpy.ceil`.

    Returns
    -------
    tg : float
        The nearest grid point to t according to the given rounding mode.
        That is, `tg = midnight + interval * i` for some integer `i`.

    Notes
    -----
    If there are an integer number of intervals per second (Δ⁻¹ ∈ ℤ), then the
    grid is well-defined even across days since there are an integer number of
    seconds in each UTC day.
    """
    midnight = DateTime(t.year, t.month, t.day)
    seconds = (t - midnight).total_seconds()
    return midnight + TimeDelta(snap(seconds, interval, round_func))
