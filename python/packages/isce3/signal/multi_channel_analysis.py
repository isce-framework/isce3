"""
Some functionalities for analyzing block of multi-channel echos such as
digital beam forming (DBF)
"""
import os
import numpy as np
from scipy import fft
from scipy.signal import fftconvolve

from isce3.antenna import ant2rgdop


def raise_cosine_win(size: int, ped: float = 1) -> np.ndarray:
    """Raise cosine window function"""
    return (1 + ped)/2. - (1 - ped)/2. * np.cos(
        2.0 * np.pi/(size - 1) * np.arange(0, size))


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


def dbf_onetap_from_dm2(
        dset, az_time, el_trans, az_trans, sr, orbit, attitude,
        dem, cal_coefs=None):
    """
    Form a 1-tap DBFed (composite) range lines for an AZ block of
    DM2 3-D echo dataset.

    This method has discontinuity at the beam transition!

    The respective channel boundaries are determined by el/az angles at
    transition points, the geometry (orbit plus DEM) and attitude data of
    the platform

    Parameters
    ----------
    dset : np.ndarray
        3-D decoded raw echo of DM2 with shape (channels, pulses, range bins)
    az_time : float
        AZ time w.r.t. to orbit epoch in seconds
    el_trans : np.ndarray(float)
        EL angles at beam transitions in radians
    az_trans : float
        azimuth angle at beams transition in radians
    sr : isce3.core.Linspace
        Slant ranges of the echo
    orbit : isce3.core.Orbit
    attitude : isce3.core.Attitude
    dem : isce3.geometry.DEMInterpolator
    cal_coefs : array of float or complex, optional
        Calibration coefficients mutiplied to echo from each channel
        while mosaicking. Must be the same size as number of RX channels!

    Returns
    -------
    np.ndarray(complex64)
        2-D complex composite echo data with shape (pulses, range bins)

    Raises
    ------
    RuntimeError
        If computed slant ranges at beams transition is out of echo range bins

    See Also
    --------
    dbf_onetap_from_dm2_seamless

    """
    # check input size and shape
    n_chnl, n_lines, n_rgb = dset.shape
    if sr.size != n_rgb:
        raise ValueError('Mismatch in number of range bins between '
                         f'slant range {sr.size} and raw data {n_rgb}!')
    if cal_coefs is not None and cal_coefs.size != n_chnl:
        raise ValueError(f'Size of Cal Coeffs {cal_coefs.size} does not '
                         f'match number of channels in raw {n_chnl}')
    # build range bin limits used in pair for each channel
    rgb_limits = _beams_transition_rangebin_limits(
        el_trans, az_trans, az_time, sr, orbit, attitude, dem)
    # initialize composite one-tap DBFed echo array
    echo = np.zeros((n_lines, sr.size), dtype=dset.dtype)
    # loop over channels
    for cc in range(n_chnl):
        # get range bin slices
        slice_rgb = slice(*rgb_limits[cc:cc+2])
        # get decoded raw echo w/ or w/o scaling by cal coeffs
        if cal_coefs is None:
            echo[:, slice_rgb] = dset[cc, :, slice_rgb]
        else:  # apply calib
            echo[:, slice_rgb] = cal_coefs[cc] * dset[cc, :, slice_rgb]
    return echo


def dbf_onetap_from_dm2_seamless(
        dset, chirp, az_time, el_trans, az_trans, sr, orbit, attitude, dem,
        num_cpu=None, ped_win=1.0, cal_coefs=None):
    """
    Form a 1-tap DBFed (composite) range lines for an AZ block of
    DM2 3-D echo dataset.

    This method take into account pulse extension to allow seamless
    composite rangeline formation from target viewpoint!

    The respective channel boundaries are determined by el/az angles at
    transition points, the geometry (orbit plus DEM) and attitude data of
    the platform

    Parameters
    ----------
    dset : np.ndarray
        3-D decoded raw echo of DM2 with shape (channels, pulses, range bins)
    chirp : np.ndarray
        1-D array of complex TX chirp signal.
    az_time : float
        AZ time w.r.t. to orbit epoch in seconds
    el_trans : np.ndarray(float)
        EL angles at beam transitions in radians
    az_trans : float
        azimuth angle at beams transition in radians
    sr : isce3.core.Linspace
        Slant ranges of the echo
    orbit : isce3.core.Orbit
    attitude : isce3.core.Attitude
    dem : isce3.geometry.DEMInterpolator
    num_cpu : int, optional
        Number of CPU/workers used in FFT-related process over AZ blocks.
        A positive value. Default is all cores.
        If provided, the value will be confined within [1, # OS CPUs]
    ped_win : float, default=1.0
        Window pedestal of raised-cosine window used in for chirp compression.
    cal_coefs : array of float or complex, optional
        Calibration coefficients mutiplied to echo from each channel
        while mosaicking. Must be the same size as number of RX channels!

    Returns
    -------
    np.ndarray(complex64)
        2-D complex composite echo data with shape (pulses, range bins)

    Raises
    ------
    RuntimeError
        If computed slant ranges at beams transition is out of echo range bins

    See Also
    --------
    dbf_onetap_from_dm2

    """
    # check input size and shape
    n_chnl, n_lines, n_rgb = dset.shape
    if sr.size != n_rgb:
        raise ValueError('Mismatch in number of range bins between '
                         f'slant range {sr.size} and raw data {n_rgb}!')
    if cal_coefs is not None and cal_coefs.size != n_chnl:
        raise ValueError(f'Size of Cal Coeffs {cal_coefs.size} does not '
                         f'match number of channels in raw {n_chnl}')
    # build range bin limits used in pair for each channel
    rgb_limits = _beams_transition_rangebin_limits(
        el_trans, az_trans, az_time, sr, orbit, attitude, dem)
    # get max range bins and fft size
    nrgb_max = np.diff(rgb_limits).max()
    nfft = fft.next_fast_len(nrgb_max + chirp.size)
    win_func = raise_cosine_win(chirp.size, ped_win)
    chp_ref_win = win_func * chirp[::-1].conj()
    # make a unit-energy chirp ref
    chp_ref_win /= np.linalg.norm(chp_ref_win)
    # adjust for the lost full compression gain due to non-rectangular
    # window function given chirp (LFM) signal!
    win_scalar = np.sqrt(win_func.size) / np.linalg.norm(win_func)
    chp_ref_win *= win_scalar
    # fft of the chirp ref to be used for range compression
    chp_ref_fft = fft.fft(chp_ref_win, n=nfft)
    # initialize composite one-tap DBFed echo array
    echo = np.zeros((n_lines, sr.size), dtype=dset.dtype)
    nrgb_ext = chirp.size - 1
    # get number of cpu workers if not provided
    if num_cpu is None:
        num_cpu = os.cpu_count() or 1
    else:
        num_cpu = min(os.cpu_count() or 1, max(num_cpu, 1))
    with fft.set_workers(num_cpu):
        # loop over channels
        for cc in range(n_chnl):
            # get range bin slices
            slice_raw = slice(rgb_limits[cc], rgb_limits[cc + 1] + nrgb_ext)
            data = dset[cc, :, slice_raw]
            # perform valid-mode range comp per channel
            slice_rgc = slice(nrgb_ext,  slice_raw.stop - slice_raw.start)
            slice_out = slice(*rgb_limits[cc:cc+2])
            data_fft = fft.fft(data, n=nfft, axis=1)
            data_rgc = fft.ifft(data_fft * chp_ref_fft, axis=1)[:, slice_rgc]
            if cal_coefs is None:
                echo[:, slice_out] = data_rgc
            else:  # apply calib
                echo[:, slice_out] = cal_coefs[cc] * data_rgc
        # perform full convolution to add pulse extension back w/o group-delay
        # adjustment but keep the total range bins!
        echo[...] = fftconvolve(echo, chirp[np.newaxis, :],
                                mode='full', axes=1)[:, :sr.size]
    return echo


def _beams_transition_rangebin_limits(
        el_trans, az_trans, az_time, sr, orbit, attitude, dem):
    """
    Arrays of range bin limits used for transitioning from
    one antenna EL beam to another.

    Parameters
    ----------
    el_trans : np.ndarray(float)
        EL angles at beam transitions in radians
    az_trans : float
        azimuth angle at beams transition in radians
    az_time : float
        AZ time w.r.t. to orbit epoch in seconds
    sr : isce3.core.Linspace
        Slant ranges of the echo
    orbit : isce3.core.Orbit
    attitude : isce3.core.Attitude
    dem : isce3.geometry.DEMInterpolator

    Returns
    -------
    np.ndarray(int)
        1-D array of integer with size `el_trans.size` + 2

    """
    # Get slant ranges at beams transition
    pos, vel = orbit.interpolate(az_time)
    quat = attitude.interpolate(az_time)
    # Pass a dummy wavelength=1 since it doesn't affect the result.
    sr_trans, _, _ = ant2rgdop(el_trans, az_trans, pos, vel, quat, 1, dem)
    # convert slant ranges to range bins for beam limits
    rgb_trans = np.int_(np.round((sr_trans - sr.first) / sr.spacing))
    # check range bins at beam transition to be within echo range.
    if (np.any(rgb_trans < 1) or np.any(rgb_trans >= (sr.size - 1))):
        print(f'(first, last) slant ranges (km) -> ({sr.first * 1e-3:.3f}'
              f', {sr.last * 1e-3:.3f})')
        print(f'Slant range at beam transitions (km) ->\n{sr_trans * 1e-3}')
        raise RuntimeError(
            'Slant ranges at beams transition is out of echo range!')
    # build range bin limits used in pair for each channel
    rgb_limits = np.zeros(el_trans.size + 2, dtype=int)
    rgb_limits[-1] = sr.size
    rgb_limits[1:-1] = rgb_trans
    return rgb_limits
