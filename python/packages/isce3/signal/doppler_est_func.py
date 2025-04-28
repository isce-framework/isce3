"""
Collection of functions for doppler centroid estimation.
"""
import functools
import numbers
import collections as cl
import numpy as np
from scipy import fft


def corr_doppler_est(echo, prf, lag=1, axis=None):
    """Estimate Doppler centroid based on complex correlator.

    It uses the Correlation Doppler Estimator (CDE) approach
    proposed by [MADSEN1989]_

    Parameters
    ----------
    echo : np.ndarray(complex)
        1-D or 2-D numpy complex array
    prf : float
        Pulse-repetition frequency or sampling rate in the azimuth
        direction in (Hz).
    lag : int, default=1
        Lag of the correlator, a positive value.
    axis : None or int, optional
        Axis along which the correlator is performed.
        If None it will be the first axis.

    Returns
    -------
    float
        Ambiguous Doppler centroid within [-0.5*prf, 0.5*prf]
    float
        Correlation coefficient, a value within [0, 1]

    Raises
    ------
    ValueError
        For bad input arguments
    TypeError
        If echo is not numpy array
    RuntimeError:
        Mismtach between lag and number of elements of echo used in correlator
    np.AxisError:
        Mismtach between axis value and echo dimension

    See Also
    --------
    sign_doppler_est : Sign-Doppler estimator
    wavelen_diversity_doppler_est

    References
    ----------
    .. [MADSEN1989]  S. Madsen, 'Estimating The Doppler Centroid of SAR Data',
       IEEE Transaction On Aerospace and Elect Sys, March 1989

    """
    if prf <= 0.0:
        raise ValueError('prf must be a positive value')
    if not isinstance(echo, np.ndarray):
        raise TypeError('echo must be a numpy array')
    if echo.ndim > 2:
        raise ValueError('Max dimension of echo must be 2')
    if lag < 1:
        raise ValueError('Lag must be equal or larger than 1')
    if axis is None:
        axis = 0
    else:
        if axis > (echo.ndim - 1):
            raise np.AxisError(
                f'axis {axis} is out of bound for dimension {echo.ndim}')

    if axis == 0:
        if echo.shape[0] < (lag + 1):
            raise RuntimeError(
                f'Not enough samples for correlator along axis {axis}')
        xcor_cmp = (echo[lag:] * echo[:-lag].conj()).mean()
        # get mag of product of auto correlations
        acor_mag = np.sqrt((abs(echo[lag:])**2).mean())
        acor_mag *= np.sqrt((abs(echo[:-lag])**2).mean())
    else:
        if echo.shape[1] < (lag + 1):
            raise RuntimeError(
                f'Not enough samples for correlator along axis {axis}')
        xcor_cmp = (echo[:, lag:] * echo[:, :-lag].conj()).mean()
        # get mag of product of auto correlations
        acor_mag = np.sqrt((abs(echo[:, lag:])**2).mean())
        acor_mag *= np.sqrt((abs(echo[:, :-lag])**2).mean())

    # calculate correlation coefficient
    if acor_mag > 0:
        corr_coef = abs(xcor_cmp) / acor_mag
    else:
        corr_coef = 0.0

    return prf / (2.0 * np.pi * lag) * np.angle(xcor_cmp), corr_coef


def sign_doppler_est(echo, prf, lag=1, axis=None):
    """Estimate Doppler centroid based on sign of correlator coeffs.

    It uses Sign-Doppler estimator (SDE) approach proposed by [MADSEN1989]_

    Parameters
    ----------
    echo : np.ndarray(complex)
        1-D or 2-D numpy complex array
    prf : float
        Pulse-repetition frequency or sampling rate in the azimuth
        direction in (Hz).
    lag : int, default=1
        Lag of the correlator, a positive value.
    axis : None or int, optional
        Axis along which the correlator is perform.
        If None it will be the first axis.

    Returns
    -------
    float
        Ambiguous Doppler centroid within [-0.5*prf, 0.5*prf]

    Raises
    ------
    ValueError
        For bad input arguments
    TypeError
        If echo is not numpy array
    RuntimeError:
        Mismtach between lag and number of elements of echo used in correlator
    np.AxisError:
        Mismtach between Axis value and echo dimension

    See Also
    --------
    corr_doppler_est : Correlation Doppler Estimator (CDE)
    wavelen_diversity_doppler_est

    References
    ----------
    .. [MADSEN1989]  S. Madsen, 'Estimating The Doppler Centroid of SAR Data',
       IEEE Transaction On Aerospace and Elect Sys, March 1989

    """
    if prf <= 0.0:
        raise ValueError('prf must be a positive value')
    if not isinstance(echo, np.ndarray):
        raise TypeError('echo must be a numpy array')
    if echo.ndim > 2:
        raise ValueError('Max dimension of echo must be 2')
    if lag < 1:
        raise ValueError('Lag must be equal or larger than 1')
    if axis is None:
        axis = 0
    else:
        if axis > (echo.ndim - 1):
            raise np.AxisError(
                f'axis {axis} is out of bound for dimension {echo.ndim}')

    sgn_i = _sgn(echo.real)
    sgn_q = _sgn(echo.imag)

    if axis == 0:
        if echo.shape[0] < (lag + 1):
            raise RuntimeError(
                f'Not enough samples for correlator along axis {axis}')
        xcor_ii = (sgn_i[lag:] * sgn_i[:-lag]).mean()
        xcor_qq = (sgn_q[lag:] * sgn_q[:-lag]).mean()
        xcor_iq = (sgn_i[lag:] * sgn_q[:-lag]).mean()
        xcor_qi = (sgn_q[lag:] * sgn_i[:-lag]).mean()
    else:
        if echo.shape[1] < (lag + 1):
            raise RuntimeError(
                f'Not enough samples for correlator along axis {axis}')
        xcor_ii = (sgn_i[:, lag:] * sgn_i[:, :-lag]).mean()
        xcor_qq = (sgn_q[:, lag:] * sgn_q[:, :-lag]).mean()
        xcor_iq = (sgn_i[:, lag:] * sgn_q[:, :-lag]).mean()
        xcor_qi = (sgn_q[:, lag:] * sgn_i[:, :-lag]).mean()

    r_sinlaw = np.sin(0.5 * np.pi * np.asarray([xcor_ii, xcor_qq,
                                                xcor_qi, -xcor_iq]))
    xcor_cmp = 0.5 * complex(r_sinlaw[:2].sum(), r_sinlaw[2:].sum())

    return prf / (2.0 * np.pi * lag) * np.angle(xcor_cmp)


def wavelen_diversity_doppler_est(echo, prf, samprate, bandwidth,
                                  centerfreq):
    """Estimate Doppler based on wavelength diversity.

    It uses slope of phase of range frequency along with single-lag
    time-domain correlator approach proposed by [BAMLER1991]_.

    Parameters
    ----------
    echo : np.ndarray(complex)
        2-D complex basebanded echo, azimuth by range in time domain.
    prf : float
        Pulse repetition frequency in (Hz)
    samprate : float
        Sampling rate in range , second dim, in (Hz)
    bandwidth : float
        RF/chirp bandiwdth in (Hz)
    centerfreq : float
        RF center frequency of chirp in (Hz)

    Returns
    -------
    float
        Unambiguous Doppler centroid at center frequency in (Hz)

    Raises
    ------
    ValueError
        For bad input
    TypeError
        If echo is not numpy array

    See Also
    --------
    corr_doppler_est : Correlation Doppler Estimator (CDE)
    sign_doppler_est : Sign-Doppler estimator (SDE)

    References
    ----------
    .. [BAMLER1991]  R. Bamler and H. Runge, 'PRF-Ambiguity Resolving by
        Wavelength Diversity', IEEE Transaction on GeoSci and Remote Sensing,
        November 1991.

    """
    if prf <= 0:
        raise ValueError('PRF must be positive value!')
    if samprate <= 0:
        raise ValueError('samprate must be positive value!')
    if bandwidth <= 0 or bandwidth >= samprate:
        raise ValueError('badnwidth must be positive less than samprate!')
    if centerfreq <= 0.0:
        raise ValueError('centerfreq must be positive value!')
    if not isinstance(echo, np.ndarray):
        raise TypeError('echo must be a numpy array')
    if echo.ndim != 2:
        raise ValueError('echo must have two dimensions')
    num_azb, num_rgb = echo.shape
    if num_azb <= 2:
        raise ValueError('The first dimension of echo must be larger than 2')
    if num_rgb > 2:
        raise ValueError('The second dimension of echo must be larger than 2!')

    # FFT along range
    nfft = fft.next_fast_len(num_rgb)
    echo_fft = fft.fft(echo, nfft, axis=1)

    # one-lag correlator along azimuth
    az_corr = (echo_fft[1:] * echo_fft[:-1].conj()).mean(axis=0)

    # Get the unwrapped phase of range spectrum within +/-bandwidth/2.
    df = samprate / nfft
    half_bw = 0.5 * bandwidth
    idx_hbw = nfft // 2 - int(half_bw / df)
    unwrap_phs_rg = np.unwrap(np.angle(fft.fftshift(az_corr)
                                       [idx_hbw: -idx_hbw]))  # (rad)

    # perform linear regression in range freq within bandwidth
    freq_bw = -half_bw + df * np.arange(nfft - 2 * idx_hbw)
    pf_coef = np.polyfit(freq_bw, unwrap_phs_rg, deg=1)

    # get the doppler centroid at center freq based on slope
    dop_slope = prf / (2. * np.pi) * pf_coef[0]

    return centerfreq * dop_slope


@functools.singledispatch
def unwrap_doppler(dop, prf):
    """Unwrap doppler value(s)

    Parameters
    ----------
    dop : float or np.ndarray(float) or Sequence[float]
        Doppler centroid value(s) in (Hz)
    prf : float
        Pulse repetition frequency in (Hz).

    Returns
    -------
    float or np.ndarray(float)
        Unwrapped Doppler values the same format as input in (Hz)

    Raises
    ------
    ValueError
        For non-positive prf
    TypeError:
        Bad data types for dop

    """
    raise TypeError('Unsupported data type for doppler')


@unwrap_doppler.register(numbers.Real)
def _unwrap_doppler_scalar(dop: float, prf: float) -> float:
    """Returns single doppler as it is"""
    if prf <= 0.0:
        raise ValueError('prf must be a positive value')
    return dop


@unwrap_doppler.register(np.ndarray)
def _unwrap_doppler_array(dop: np.ndarray, prf: float) -> np.ndarray:
    """Unwrap doppler values stored as numpy array.

    For 2-D array, unwrap operates along the first axis.

    """
    if prf <= 0.0:
        raise ValueError('prf must be a positive value')
    frq2phs = 2 * np.pi / prf
    phs2frq = 1. / frq2phs
    return phs2frq * np.unwrap(frq2phs * dop, axis=0)


@unwrap_doppler.register(cl.abc.Sequence)
def _unwrap_doppler_sequence(dop: cl.abc.Sequence, prf: float) -> np.ndarray:
    """Unwrap doppler values stored as Sequence """
    if prf <= 0.0:
        raise ValueError('prf must be a positive value')
    frq2phs = 2 * np.pi / prf
    phs2frq = 1. / frq2phs
    return phs2frq * np.unwrap(frq2phs * np.asarray(dop))

# List of helper functions


def _sgn(x: np.ndarray) -> np.ndarray:
    """Wrapper around numpy.sign.

    It replaces zero values with one.

    """
    s = np.sign(x)
    s[s == 0] = 1
    return s
