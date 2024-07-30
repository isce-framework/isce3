"""
Generate arbitrary FIR , LPF or BPF, Filter Coefficients
"""
import numpy as np
import scipy.signal as spsg


def cheby_equi_ripple_filter(samprate, bandwidth, rolloff=1.2, ripple=0.1,
                             stopatt=40, centerfreq=0.0, force_odd_len=False):
    """
    Generate an arbitrary FIR equi-ripple Chebyshev , Low Pass Filter (LPF)
    or Band Pass Filter (BPF) coefficients.

    It uses 'remez' optmization algorithm for designing Chebyshev filter
    with equal pass-band and stop-band ripples.
    The min length of the filter is determined based on 'Kaiser' formula.

    Parameters
    ----------
    samprate : float
        Sampling frequency in Hz, MHz, etc.
    bandwidth : float
        Bandwidth in same unit as samprate
    rollfoff : float, default=1.2
        Roll-off factor or shaping factor of the filter. This must be > 1.0.
    ripple : float, default=0.1
        Pass-band ripples in dB.
    stopatt : float, default=40.0
        Minimum Stopband attenuation in dB.
    centerfreq : float, default=0.0
        Center frequency in the same unit as samprate.
    force_odd_len : bool, default=False
        Whether or to not to force the filter length to be an odd value.

    Returns
    -------
    numpy.ndarray
        Filter coefficients.

    Raises
    ------
    ValueError
        For bad inputs.

    """
    if samprate <= 0.0:
        raise ValueError('samprate must be a positive value')
    if bandwidth <= 0.0 or bandwidth >= samprate:
        raise ValueError(
            'bandwidth must be a positive value less than samprate')
    max_rolloff = samprate / bandwidth
    if rolloff <= 1 or rolloff > max_rolloff:
        raise ValueError(
            'rolloff must be a value greater than 1 and equal or less'
            f' than {max_rolloff}'
        )
    if ripple <= 0:
        raise ValueError('rippler must be a positive value')
    if stopatt <= 0:
        raise ValueError('stopatt must be a positive value')

    # LPF params
    delta_pas = 10.**(ripple/20.) - 1
    delta_stp = 10.**(-stopatt/20.)
    weight_fact = delta_pas / delta_stp
    max_rolloff = samprate / bandwidth

    # get LPF length
    fstop = rolloff * bandwidth
    deltaf = (fstop - bandwidth) / samprate / 2.0
    len_flt = np.int_(np.ceil((-20. * np.log10(np.sqrt(delta_stp*delta_pas))
                               - 13.) / 14.6 / deltaf) + 1)

    if (force_odd_len and len_flt % 2 == 0):
        len_flt += 1

    # get LPF coeffs
    coeffs = spsg.remez(len_flt, 0.5 / samprate
                        * np.array([0, bandwidth, fstop, samprate]),
                        np.array([1.0, 0.0]), np.array([1, weight_fact]),
                        fs=1, type='bandpass', maxiter=50)

    # up/down conversion
    if abs(centerfreq) > 0.0:
        return _lowpass2bandpass(coeffs, centerfreq / samprate)
    return coeffs


def _kaiser_design(stopatt, transition_width, force_odd_len):
    """Return length, shape, and time samples for Kaiser filter design method.
    """
    n, beta = spsg.kaiserord(stopatt, transition_width)
    if force_odd_len and n % 2 == 0:
        n += 1
    t = np.arange(n) - (n - 1) / 2
    return n, beta, t


def _kaiser_irf(t, beta):
    """Impulse response (Fourier transform) of Kaiser window.
    """
    alpha = beta / np.pi
    x2 = t**2 - alpha**2
    # complex to get hyperbolic sine for negative values
    x = np.sqrt(x2.astype(complex))
    return (np.sinc(x) / np.i0(beta)).real


def _coswin_irf(t, eta):
    """Impulse response (Fourier transform) of cosine window.
    """
    # Could write with a single sinc(t) but singularities at ±1 require
    # high-order Taylor expansion to maintain accuracy.  Simpler to just use
    # three sinc calls.
    a = (1 + eta) / 2
    b = (1 - eta) / 4
    return a * np.sinc(t) + b * (np.sinc(t - 1) + np.sinc(t + 1))


# Tuple for window follows scipy API, e.g.,
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html

def design_shaped_lowpass_filter(bandwidth, window=None, stopatt=40.0,
                                 transition_width=0.15, force_odd_len=False,
                                 fs=1.0):
    """
    Design a low pass filter having a passband shaped like a window using the
    Kaiser method.

    Parameters
    ----------
    bandwidth : float
        Width (FWHM) of the filter in same units as `fs`.
        (Double the cut-off frequency.)
    window : Tuple[str, float] or None
        Window defining shape of pass band.  Pair of (window_name, shape).
        Valid names are "kaiser" or "cosine", for example ("kaiser", 4.5).
        `None` for unit passband.
    stopatt : float
        Minimum stopband attenuation in dB.  Also bounds approximation error
        in passband.
    transition_width : float
        Total width of both transition regions as a fraction of `bandwidth`.
        Note that transitions are split evenly between pass and stop bands,
        centered on ± bandwidth / 2.
    force_odd_len : bool
        Require odd filter length.
    fs : float
        Sample rate in same units as `bandwidth`.

    Returns
    -------
    coefficients : ndarray
        Filter coefficients.  Will have unit DC gain.
    """
    name, shape = window if window is not None else ("kaiser", 0.0)
    name = name.lower()
    designs = {"kaiser": _kaiser_irf, "cosine": _coswin_irf}
    if name not in designs:
        raise ValueError(f"Require window name in {designs.keys()} got {name}.")
    irf = designs[name]
    if bandwidth <= 0.0:
        raise ValueError(f"Require bandwidth > 0 but got {bandwidth}")
    if transition_width <= 0.0:
        raise ValueError(
            f"Require transition_width > 0 but got {transition_width}")
    if fs <= 0.0:
        raise ValueError(f"Require fs > 0 but got {fs}")
    # Normalized bandwidth.
    bw = bandwidth / fs
    # Transition width is specified in terms of output bandwidth, so scale to
    # get width at sample rate of filter.
    tw = transition_width * bw
    if (bw + tw / 2) > 1.0:
        raise ValueError("Passband + transition cannot exceed Nyquist")
    n, beta, t = _kaiser_design(stopatt, tw, force_odd_len)
    return bw * irf(bw * t, shape) * np.kaiser(n, beta)


def _lowpass2bandpass(h, fc):
    """
    Turn a low pass filter into a band pass filter by applying a phase ramp.
    Phase will be zero at center of filter.
    """
    n = len(h)
    t = np.arange(n) - (n - 1) / 2.0
    return h * np.exp(2j * np.pi * fc * t)


def design_shaped_bandpass_filter(bandwidth, centerfreq=0.0, window=None,
                                  stopatt=40.0, transition_width=0.15,
                                  force_odd_len=False, fs=1.0):
    """
    Design a band pass filter having a passband shaped like a window using the
    Kaiser method.

    Parameters
    ----------
    bandwidth : float
        Width (FWHM) of the filter in same units as `fs`.
        (Double the cut-off frequency of the prototype lowpass filter.)
    centerfreq : float
        Center frequency of passband in same units as `fs`.
    window : Tuple[str, float] or None
        Window defining shape of pass band.  Pair of (window_name, shape).
        Valid names are "kaiser" or "cosine", for example ("kaiser", 4.5).
        `None` for unit passband.
    stopatt : float
        Minimum stopband attenuation in dB.  Also bounds approximation error
        in passband.
    transition_width : float
        Total width of both transition regions as a fraction of pass band width.
        Note that transitions are split evenly between pass and stop bands.
    force_odd_len : bool
        Require odd filter length.
    fs : float
        Sample rate in same units as `bandwidth`.

    Returns
    -------
    coefficients : ndarray
        Filter coefficients.  Will have unit gain at center of passband.
    """
    h = design_shaped_lowpass_filter(bandwidth, window, stopatt,
                                     transition_width, force_odd_len, fs)
    return _lowpass2bandpass(h, centerfreq / fs)
