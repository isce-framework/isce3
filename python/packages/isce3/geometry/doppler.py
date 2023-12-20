import numpy as np


def los2doppler(look, v, wvl):
    """
    Compute Doppler given line-of-sight vector

    Parameters
    ----------
    look : array_like
        ECEF line of sight vector in m
    v : array_like
        ECEF velocity vector in m/s
    wvl : float
        Radar wavelength in m

    Returns
    -------
    float
        Doppler frequency in Hz
    """
    return 2 / wvl * np.asarray(v).dot(look) / np.linalg.norm(look)
