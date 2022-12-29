"""
Some functionalities for analyzing block of multi-channel echos such as
digital beam forming (DBF)
"""
import numpy as np
from isce3.antenna import ant2rgdop


def form_single_tap_dbf_echo(raw_dset, slice_line, el_trans,
                             az_trans, pos, vel, quat, sr, dem):
    """
    Form a 1-tap DBFed (composite) range lines for a desired range lines
    and at a center of azimuth block.

    The respective channel boundaries are determined by el/az angles at
    transition points, the geometry (orbit plus DEM) and attitude data of
    the platform

    Parameters
    ----------
    raw_dset : nisar.products.readers.Raw.DataDecoder
    slice_line : slice
        slice object of desired range lines
    el_trans : np.ndarray(float)
        EL angles at beam transitions in radians
    az_trans : float
        azimuth angle at beams transition in radians
    pos : np.ndarray(float)
        3-element position vector of the spacecraft in ECEF
    vel : np.ndarray(float)
        3-element velocity vector of the spacecraft in ECEF
    quat : isce3.core.Quaternion
        Contains 4-element quaternion vector of the spacecraft attitude
    sr : isce3.core.Linspace
        Slant ranges of the echo
    dem : isce3.geometry.DEMInterpolator

    Returns
    -------
    np.ndarray(complex64)
        2-D complex composite echo data with shape (range lines, range bins)

    Raises
    ------
    RuntimeError
        If computed slant ranges at beams transition is out of echo range bins

    """
    # Get slant ranges at beams transition
    sr_trans, _, _ = ant2rgdop(el_trans, az_trans, pos, vel, quat, 1, dem)
    # convert slant ranges to range bins for beam limits
    rgb_trans = np.int_(np.round((sr_trans - sr.first) / sr.spacing))
    # check range bins at beam transition to be within echo range.
    if (np.sum(rgb_trans < 1) != 0 or np.sum(rgb_trans >= (sr.size - 1)) != 0):
        raise RuntimeError(
            'Slant ranges at beams transition is out of echo range!')
    # build range bin limits used in pair for each channel
    rgb_limits = np.zeros(el_trans.size + 2, dtype=int)
    rgb_limits[-1] = sr.size
    rgb_limits[1:-1] = rgb_trans
    # initialize composite one-tap DBFed echo array
    num_lines = slice_line.stop - slice_line.start
    echo = np.zeros((num_lines, sr.size), dtype=raw_dset.dtype)
    # loop over channels
    for cc in range(el_trans.size + 1):
        # get range bin slices
        slice_rgb = slice(*rgb_limits[cc:cc+2])
        # get decoded raw echo
        echo[:, slice_rgb] = raw_dset[cc, slice_line, slice_rgb]
    return echo
