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
                        Hz=1, type='bandpass', maxiter=50)

    # up/down conversion
    if abs(centerfreq) > 0.0:
        return coeffs * np.exp(2j * np.pi * centerfreq / samprate *
                               np.arange(len_flt))
    return coeffs
