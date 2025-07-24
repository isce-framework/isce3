from isce3.signal.fir_filter_func import design_shaped_bandpass_filter
from .logic import Band
import numpy as np


def get_common_band_filter(band_in, band_out, fs, attenuation=40.0, width=0.15,
                           window=None):
    """
    Get common band filter for NISAR mixed-mode processing.

    Parameters
    ----------
    band_in : Band
        Input band.  Data is assumed basebanded to band_in.center
    band_out : Band
        Output band.  Must overlap input.
    fs : float
        Sample rate of input data in Hz.
    attenuation : float
        Minimum attenuation in filter stop band in dB.
    width : float
        Width of filter transitions as a fraction of common bandwidth.
    window : Tuple[str, float]
        Desired shape of passband.  Either None for flat or a tuple of
        (window_name, window_shape) such as ("kaiser", 1.6) or ("cosine", 0.7)

    Returns
    -------
    coeffs : ndarray
        Filter coefficients to apply to input data.  Length N will be odd, and
        group delay will be (N-1)/2.
    shift : float
        Phase ramp in rad/sample needed to apply to input data to make it
        baseband with respect to band_out.
    """
    common = band_in & band_out
    if not common:
        raise ValueError(f"{band_in} does not overlap {band_out}")
    h = design_shaped_bandpass_filter(common.width,
                                      common.center - band_in.center,
                                      window=window,
                                      stopatt=attenuation,
                                      transition_width=width,
                                      force_odd_len=True, fs=fs)
    shift = (band_in.center - band_out.center) * 2 * np.pi / fs
    return h, shift
