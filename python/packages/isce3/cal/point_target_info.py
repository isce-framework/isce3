#!/usr/bin/env python3
"""
Analyze a point target in a complex*8 file.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from warnings import warn

import numpy as np

from isce3.core import DateTime, LUT2d
from isce3.image.v2 import resample_to_coords
from isce3.io.dataset import DatasetReader
from isce3.product import RadarGridParameters

desc = __doc__


class MissingNull(Exception):
    """Raised when mainlobe null(s) cannot be determined"""

    pass

class UnsupportedWindow(Exception):
    """Raised if window_type input is not supported."""

    pass

def get_chip(x: DatasetReader, i: float, j: float, nchip: int = 64) -> np.ndarray:
    """
    Get a chip from a given raster.

    This function does not check whether or not (i, j) is a valid sampling
    position on the input dataset.

    Parameters
    ----------
    x : DatasetReader
        The dataset to read.
    i : float
        The row index on the input dataset to acquire.
    j : float
        The column index on the input dataset to acquire.
    nchip : int, optional
        The length of both dimensions of the chip. Defaults to 64
        (e.g., if nchip=64, the chip shape will be (64, 64).)

    Returns
    -------
    np.ndarray
        The chip, extracted from the dataset.
    """
    i = int(i)
    j = int(j)
    chip = np.zeros((nchip, nchip), dtype=x.dtype)
    nchip2 = nchip // 2
    i0 = i - nchip2 + 1
    i1 = i0 + nchip
    j0 = j - nchip2 + 1
    j1 = j0 + nchip
    # FIXME handle edge cases by zero-padding
    chip[:, :] = x[i0:i1, j0:j1]
    return i0, j0, chip


def estimate_frequency(z):
    cx = np.sum(z[:, 1:] * z[:, :-1].conj())
    cy = np.sum(z[1:, :] * z[:-1, :].conj())
    return np.angle([cx, cy])


def shift_frequency(z, fx, fy):
    x = np.arange(z.shape[1])
    y = np.arange(z.shape[0])
    z *= np.exp(1j * fx * x)[None,:]
    z *= np.exp(1j * fy * y)[:,None]
    return z


def measure_location_from_spectrum(x):
    """
    Estimate location of a target, assuming there's just one.

    Parameters
    ----------
    x : array_like
        Two-dimensional chip of data containing a single point-like target.

    Returns
    -------
    xy : tuple of float
        (column, row) location of target, in samples.
    """
    X = np.fft.fft2(x)
    # Estimate location of target, assuming there's just one.
    tx, ty = estimate_frequency(X)
    # scale to pixels
    tx *= -X.shape[1] / (2 * np.pi)
    ty *= -X.shape[0] / (2 * np.pi)
    # ensure positive
    tx %= X.shape[1]
    ty %= X.shape[0]
    return tx, ty


def oversample(x, nov, baseband=False, return_slopes=False):
    m, n = x.shape
    assert m == n

    if not baseband:
        # shift the data to baseband
        fx, fy = estimate_frequency(x)
        x = shift_frequency(x, -fx, -fy)

    X = np.fft.fft2(x)
    # Zero-pad high frequencies in the spectrum.
    Y = np.zeros((n * nov, n * nov), dtype=X.dtype)
    n2 = n // 2
    Y[:n2, :n2] = X[:n2, :n2]
    Y[-n2:, -n2:] = X[-n2:, -n2:]
    Y[:n2, -n2:] = X[:n2, -n2:]
    Y[-n2:, :n2] = X[-n2:, :n2]
    # Split Nyquist bins symmetrically.
    assert n % 2 == 0
    Y[:n2, n2] = Y[:n2, -n2] = 0.5 * X[:n2, n2]
    Y[-n2:, n2] = Y[-n2:, -n2] = 0.5 * X[-n2:, n2]
    Y[n2, :n2] = Y[-n2, :n2] = 0.5 * X[n2, :n2]
    Y[n2, -n2:] = Y[-n2, -n2:] = 0.5 * X[n2, -n2:]
    Y[n2, n2] = Y[n2, -n2] = Y[-n2, n2] = Y[-n2, -n2] = 0.25 * X[n2, n2]
    # Back to time domain.
    y = np.fft.ifft2(Y)
    # NOTE account for scaling of different-sized DFTs.
    y *= nov ** 2

    if not baseband:
        # put the phase back on
        y = shift_frequency(y, fx / nov, fy / nov)

    y = np.asarray(y, dtype=x.dtype)
    if return_slopes:
        return (y, fx, fy)
    return y


def estimate_resolution(x, dt=1.0):
    # Find the peak.
    y = abs(x) ** 2
    i = np.nanargmax(y)
    # Construct a function with zeros at the -3dB points.
    u = y - 0.5 * y[i]
    # Make sure the interval contains a peak.  If not, return interval width.
    if (u[0] >= 0.0) or (u[-1] >= 0.0):
        print(
            "Warning: Interval does not contain a well-defined peak.", file=sys.stderr
        )
        return dt * len(x)
    # Take its absolute value so can search for minima instead of intersections.
    z = abs(u)
    # Find the points on each side of the peak.
    left = z[:i]
    ileft = np.nanargmin(left)
    right = z[i:]
    iright = i + np.nanargmin(right)
    # Return the distance between -3dB crossings, scaled by the sample spacing.
    return dt * (iright - ileft)


def comp_kaiserwin_peak_to_nth_null_dist(beta, num_nulls_main=2):
    """
    Compute distance between peak to nth null for a Kaiser window

    Parameters:
    -----------
    beta: float
        Kaiser window parameter
    num_nulls_main: int
        number of nulls included in the mainlobe from each side of mainlobe peak
        num_nulls_main = 1: mainlobe extends out to 1st null
        num_nulls_main = 2: mainlobe extends out to 2nd null

    Returns:
    --------
    peak_to_nth_null_dist: float
        Distance from the peak to the nth null, relative to the signal resolution (not the 
        sample bin spacing)
    
    References:
    -----------
    [1] A.Nuttall, "Some windows with very good sidelobe behavior," in IEEE Transactions
           on Acoustics, Speech, and Signal Processing, vol.29, no.1, pp.84 - 91, February
           1981, doi: 10.1109 / TASSP.1981.1163506.
    """

    peak_to_nth_null_dist = np.sqrt(num_nulls_main ** 2 + (beta / np.pi) ** 2)

    return peak_to_nth_null_dist


def comp_coswin_peak_to_2nd_null_dist(eta):
    r"""
    Compute distance between peak to 2nd null for a Raised Cosine window

    The cosine-on-pedestal weighting function :math:`c_p(f,\eta)` is given by
  
    .. math:: 
        c_p(f,\eta) = \frac{1+ \eta}{2} + \frac{1 - \eta}{2}\cos \left( \frac{2 \pi f}{B}\right)

    where :math:`f` is frequency in Hertz, :math:`\eta` is the window pedestal height, 
    :math:`B` is bandwidth in Hertz, :math:`\frac{-B}{2}\le f \le \frac{B}{2}` and 
    :math: `0\le \eta \le 1`.     


    Parameters:
    -----------
    eta: float
        Raised Cosine window parameter

    Returns:
    --------
    peak_to_2nd_null_dist: float
        Distance from the peak to the 2nd null, relative to the signal resolution (not the 
        sample bin spacing)

    References:
    -----------
    S. Hensley, S. Oveisgharan, S. Saatchi, M. Simard, R. Ahmed and Z. Haddad,
    "An Error Model for Biomass Estimates Derived From Polarimetric Radar Backscatter,"
    in IEEE Transactions on Geoscience and Remote Sensing, vol. 52, no. 7, pp. 4065-4082,
    July 2014, doi: 10.1109/TGRS.2013.2279400.
    """

    # Closed form solution for Raised Cosine window null position is only available for mainlobe
    # width which includes first sidelobes, i.e. num_nulls_main=2
    num_nulls_main = 2

    if eta <= 1 / 7:
        peak_to_2nd_null_dist = (num_nulls_main + 1)
    else:
        peak_to_2nd_null_dist = num_nulls_main

    return peak_to_2nd_null_dist


def locate_null(t, half, n=0):
    """Locate null in impulse response.

    Parameters:
    -----------
    t : array_like
        Time axis of data
    half : array_like
        Impulse response magnitude, starting at peak and decreasing,
        e.g., the right half or the reversed left half.
    n : int
        Which null to find, n=0 for first null after peak (default).

    Returns:
    --------
    location : t.dtype
        Time axis indexed at location of null
    """
    assert len(t) == len(half)
    if np.any(half > half[0]):
        raise ValueError("IRF not sorted correctly")
    dx = np.diff(half)
    # Exclude any points where adjacent values are equal.
    # Note that len(dx) == len(x) - 1.  If points i and i+1 are equal, exclude
    # point i in order to err in favor of the largest possible null location.
    unequal = np.where(dx != 0.0)[0]
    t = t[unequal]
    dx = np.diff(half[unequal])
    # Nulls occur where derivative goes from negative to positive.
    nulls = np.where(np.diff(np.sign(dx)) == 2)[0] + 1
    if len(nulls) <= n:
        raise MissingNull("Insufficient nulls found in impulse response.")
    return t[nulls[n]]


def search_first_null_pair(matched_output, mainlobe_peak_idx):
    """Compute mainlobe null locations as sample index for ISLR and PSLR.

    Null locations are first nulls to the left and right of the mainlobe.

    Parameters:
    -----------
    matched_output: array of float
        Range or Azimuth cuts of point target or 2-D antenna pattern
    mainlobe_peak_idx: int
        index of mainlobe peak

    Returns:
    --------
    null_left_idx: int
        null location left of mainlobe peak in sample index
    null_right_idx: int
        null location right of mainlobe peak in sample index
    """
    t = np.arange(len(matched_output))
    left = slice(mainlobe_peak_idx, 0, -1)
    right = slice(mainlobe_peak_idx, None)
    ileft = locate_null(t[left], matched_output[left])
    iright = locate_null(t[right], matched_output[right])
    return ileft, iright


def compute_islr_pslr(
    data_in_linear,
    fs_bw_ratio=1.2,
    num_sidelobes=10,
    predict_null=False,
    window_type='rect',
    window_parameter=0
):
    """
    Computes integrated sidelobe ratio (ISLR) and peak to sidelobe ratio (PSLR) of a point 
    target impulse response.

    ISLR mainlobe nulls can be located based on provided Fs/BW ratio and window_type/window_parameter
    or it can be computed based on null search. If former is selected (predict_null=True),
    first sidelobes are included in the mainlobe for ISLR calculations.
    If latter is selected, first sidelobes are not included as part of mainlobe in
    ISLR calculations. PSLR calculation is based on null search only. It does not include first
    sidelobe as part of mainlobe in its calculations.


    Parameters:
    -----------
    data_in_linear: array of complex
        Range or azimuth cut (as linear amplitude) through a point target
    fs_bw_ratio: float
        optional, sampling frequency to bandwidth ratio
        fs_bw_ratio = Fs / BW
    num_sidelobes: float
        optional total number of sidelobes for ISLR computation,
        default is 10.
    predict_null: boolean
        optional, if predict_null is True, mainlobe null locations are computed based 
        on Fs/bandwidth ratio and winodw_type/window_parameter for ISLR calculations.
        i.e, mainlobe null is located at Fs/B * peak-to-nth-null-dist.  
        Otherwise, mainlobe null locations are computed based on null search algorithm.
        PSLR Exception: mainlobe does not include first sidelobes, search is always
        conducted to find the locations of first null regardless of predict_null parameter.
    window_type: str
        optional, user provided window types used for tapering
        
        'rect': 
		Rectangular window is applied
        'cosine': 
		Raised-Cosine window
        'kaiser': 
		Kaiser Window
    window_parameter: float
        optional window parameter. For a Kaiser window, this is the beta
        parameter. For a raised cosine window, it is the pedestal height.
        It is ignored if `window_type = 'rect'`.

    Returns:
    --------
    islr_db: float
        ISLR in dB
    pslr_db: float
        PSLR in dB
    """

    if (window_type != 'rect') and (window_type != 'kaiser') and (window_type != 'cosine'):
        raise UnsupportedWindow(
            'The input window type is not supported. Only rect, kaiser, and cosine windows are supported'
        )

    data_in_pwr_linear = np.abs(data_in_linear) ** 2
    data_in_pwr_db = 10 * np.log10(data_in_pwr_linear)
    peak_idx = np.nanargmax(data_in_pwr_linear)

    # Theoretical nulls are based on Fs/BW ratio and window_type/window_parameter
    if predict_null:
        # If predict option is selected, first sidelobes are always included in the mainlobe
        num_nulls_main = 2
        if window_type == "rect":
            samples_null_to_peak = int(np.round(num_nulls_main * fs_bw_ratio))
        elif window_type == "kaiser":
            peak_2_null_dist = comp_kaiserwin_peak_to_nth_null_dist(
                window_parameter,
                num_nulls_main
                )
            samples_null_to_peak = int(np.round(peak_2_null_dist * fs_bw_ratio))
        elif window_type == "cosine":
            peak_2_null_dist = comp_coswin_peak_to_2nd_null_dist(window_parameter)
            samples_null_to_peak = int(np.round(peak_2_null_dist * fs_bw_ratio))

        # Compute number of samples between mainlobe peak and first null to its left and right
        null_main_left_idx = peak_idx - samples_null_to_peak
        null_main_right_idx = peak_idx + samples_null_to_peak
        num_samples_side_total = int(np.round(num_sidelobes * samples_null_to_peak))

        # PSLR is always computed based on manual null search
        null_first_left_idx, null_first_right_idx = search_first_null_pair(
            data_in_pwr_db, peak_idx
        )

        sidelobe_left_idx = null_main_left_idx - num_samples_side_total
        sidelobe_right_idx = null_main_right_idx + num_samples_side_total
    else:  
       # Search for mainlobe nulls
        null_first_left_idx, null_first_right_idx = search_first_null_pair(
            data_in_pwr_db, peak_idx
        )
        null_main_left_idx = null_first_left_idx
        null_main_right_idx = null_first_right_idx

        # Compute number of samples between mainlobe peak and first null to its left
        samples_null_to_peak = peak_idx - null_first_left_idx
        num_samples_side_total = int(np.round(num_sidelobes * samples_null_to_peak))

        sidelobe_left_idx = null_first_left_idx - num_samples_side_total
        sidelobe_right_idx = null_first_right_idx + num_samples_side_total

    # ISLR
    islr_mainlobe = data_in_pwr_linear[null_main_left_idx : null_main_right_idx + 1]
    
    # Check if index is out of bounds
    if sidelobe_left_idx<0: 
        sidelobe_left_idx = 0

    # Check if index is out of bounds
    if sidelobe_right_idx>len(data_in_pwr_linear)-1:
        sidelobe_right_idx = len(data_in_pwr_linear)-1    
         
    islr_sidelobe_range = np.r_[
        sidelobe_left_idx:null_main_left_idx,
        null_main_right_idx + 1 : sidelobe_right_idx + 1,
    ]
    islr_sidelobe = data_in_pwr_linear[islr_sidelobe_range]

    pwr_total = np.nansum(data_in_pwr_linear)
    islr_main_pwr = np.nansum(islr_mainlobe)
    islr_side_pwr = np.nansum(islr_sidelobe)

    islr_db = 10 * np.log10(islr_side_pwr / islr_main_pwr)

    # PSLR
    pslr_sidelobe_range = np.r_[
        sidelobe_left_idx:null_first_left_idx,
        null_first_right_idx + 1 : sidelobe_right_idx + 1,
    ]
    pslr_mainlobe = data_in_pwr_linear[null_first_left_idx:null_first_right_idx + 1]
    pslr_sidelobe = data_in_pwr_linear[pslr_sidelobe_range]

    pwr_mainlobe_max = np.nanmax(pslr_mainlobe)
    pwr_sidelobe_max = np.nanmax(pslr_sidelobe)

    pslr_db = 10 * np.log10(pwr_sidelobe_max / pwr_mainlobe_max)

    return islr_db, pslr_db


def dB(x):
    return 20.0 * np.log10(abs(x))


def plot_profile(t, x, title=None):
    import matplotlib.pyplot as plt

    peak = abs(x).max()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(t, dB(x) - dB(peak), "-k")
    ax1.set_ylim((-40, 0.3))
    ax1.set_ylabel("Power (dB)")
    ax2 = ax1.twinx()
    phase_color = "0.75"
    ax2.plot(t, np.angle(x), color=phase_color)
    ax2.set_ylim((-np.pi, np.pi))
    ax2.set_ylabel("Phase (rad)")
    ax2.spines["right"].set_color(phase_color)
    ax2.tick_params(axis="y", colors=phase_color)
    ax2.yaxis.label.set_color(phase_color)
    ax1.set_xlim((-15, 15))
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    if title:
        ax1.set_title(title)
    return fig


@dataclass(frozen=True)
class IPRCrossSection:
    """
    A 1-D cross-section of a point-like target impulse response along azimuth or range.

    Attributes
    ----------
    data : np.ndarray of complex
        The cross-section
    i_indices : np.ndarray of float
        The i indices at which the cross-section was sampled
    j_indices : np.ndarray of float
        The j indices at which the cross-section was sampled.
    """
    data: np.ndarray
    i_indices: np.ndarray
    j_indices: np.ndarray


def analyze_point_target(
    slc: DatasetReader,
    i: float,
    j: float,
    nov: int = 32,
    plot: bool = False,
    cuts: bool = False,
    chipsize: int = 64,
    fs_bw_ratio: float = 1.2,
    num_sidelobes: int = 10,
    predict_null: bool = False,
    window_type: str = "rect",
    window_parameter: float = 0.0,
    shift_domain: str = "time",
    geo_heading: float | None = None,
    pixel_spacing: tuple[float, float] = (1.0, 1.0)
) -> tuple[dict, list["matplotlib.figure.Figure"] | None]:
    """
    Measure point-target attributes.

    Parameters
    ----------
    slc : DatasetReader
        complex 2D image.
    i, j : float
        Row and column indices where point-target is expected.
        (Need not be integer.)
    nov : int, optional
        Amount of oversampling. Defaults to 32
    plot : bool, optional
        Generate interactive plots if True. Defaults to False
    cuts : bool, optional
        Include cuts through the peak in the output dictionary if True.
        Defaults to False
    chipsize : int, optional
        number of samples around the point target to be included for point target metric
        analysis. Defaults to 64
    fs_bw_ratio : float, optional
        Sampling frequency to bandwidth ratio, for use only when predict_null is True.
        Defaults to 1.2
    num_sidelobes : int, optional
        The total number of sidelobes for ISLR computation. Defaults to 10
    predict_null : bool, optional
        if True, mainlobe null locations are computed based on Fs/bandwidth ratio and
        window_type/window_parameter, when computing ISLR calculations. Otherwise,
        mainlobe null locations are computed based on null search algorithm.
        PSLR Exception: mainlobe does not include first sidelobes, search is always
        conducted to find the locations of first null regardless of predict_null
        parameter. Defaults to False
    window_type : str, optional
        User-provided window type, used for tapering, one of
        {"rect", "cosine", "kaiser"}. Defaults to 'rect'.
        "rect": Rectangular window is applied
        "cosine": Raised-Cosine window
        "kaiser": Kaiser Window.
    window_parameter : float, optional
        For a Kaiser window, this is the beta parameter.
        For a Raised Cosine window, it is the pedestal height.
        Ignored if window_type is "rect". Defaults to 0.0
    shift_domain : str, optional
        The domain to estimate peak location in, one of {"time", "frequency"}.
        If "time" then estimate peak location using max of oversampled data.
        If "frequency" then estimate a phase ramp in the frequency domain; useful when
        target is well focused, has high SNR, and is the only target in the neighborhood
        (often the case in point target simulations). Defaults to "time"
    geo_heading : float or None, optional
        Only for geocoded data: The heading of the satellite in the geodetic coordinate
        system of input data, in radians east of north, for use only if the SLC is
        geocoded. None if it is in radar coordinates. Defaults to None
    pixel_spacing : tuple[float, float], optional
        Only for geocoded data: The pixel spacing of the image in the
        (y_spacing, x_spacing) directions, for use in adjusting the heading angle to the
        raster dimensions. Defaults to (1.0, 1.0)

    Returns
    -------
    dict:
        Dictionary of point target attributes.
    list[matplotlib.figure.Figure] or None
        If plot=true then return the dictionary and a list of figures.
        Else, return None.
    """
    # Notation: For all references to i and j in this function, i represents row
    # indices on the chip or its parent raster, and j represents column indices
    # on the same.

    chip, chip_min_i, chip_min_j = generate_chip_on_slc(slc, i, j, chipsize=chipsize)

    return analyze_point_target_chip(
        chip=chip,
        chip_min_i=chip_min_i,
        chip_min_j=chip_min_j,
        i_pos=i,
        j_pos=j,
        nov=nov,
        plot=plot,
        cuts=cuts,
        fs_bw_ratio=fs_bw_ratio,
        num_sidelobes=num_sidelobes,
        predict_null=predict_null,
        window_type=window_type,
        window_parameter=window_parameter,
        shift_domain=shift_domain,
        geo_heading=geo_heading,
        pixel_spacing=pixel_spacing,
    )


def generate_chip_on_slc(
    slc: DatasetReader,
    i: float,
    j: float,
    chipsize: int = 64
) -> np.ndarray:
    """
    Given an SLC dataset and a position, generate a data chip.

    The specified position will be at the center of the resulting chip. If `chipsize`
    is an even number, then it will be at the location one row above and one
    column to the left of the center. See `get_chip()` for specific dimensions
    of the generated chip.

    Parameters
    ----------
    slc : DatasetReader
        complex 2D image.
    i, j : float
        Row and column indices where point-target is expected.
        (Need not be integer.)
    chipsize : int, optional
        number of samples around the point target to be included for point
        target metric analysis. Defaults to 64

    Returns
    -------
    chip : np.ndarray
        The generated chip.
    chip_min_i, chip_min_j : int
        The row and column index of the upper-left corner of the chip in the input
        raster.
    """
    # Notation: For all references to i and j in this function, i represents row
    # indices on the chip or its parent raster, and j represents column indices
    # on the same.
    
    # Check if i or j indices are out of bounds w.r.t slc image
    if i > slc.shape[0] or i < 0 or j > slc.shape[1] or j < 0:
        raise ValueError(
            "User provided target location points to a spot outside of image array. "
            "This could be due to residual azimuth/range delays or incorrect geometry "
            "info or incorrect user provided target location info."
        )

    # Raise an exception if the target position was too near the border of the image,
    # with insufficient margin to extract a "chip" around it.
    near_border = (
        (i < chipsize // 2)
        or (i > slc.shape[0] - chipsize // 2)
        or (j < chipsize // 2)
        or (j > slc.shape[1] - chipsize // 2)
    )
    if near_border:
        raise RuntimeError(
            "target is too close to image border -- consider reducing chipsize (nchip)"
        )

    chip_min_i, chip_min_j, chip0 = get_chip(slc, i, j, nchip=chipsize)

    # Casting the chip to complex64 makes it compatible with the resampler which allows
    # PTA for real-valued data using this method.
    return np.asanyarray(chip0, dtype=np.complex64), chip_min_i, chip_min_j


def analyze_point_target_chip(
    chip: np.ndarray,
    chip_min_i: int,
    chip_min_j: int,
    i_pos: float,
    j_pos: float,
    nov: int = 32,
    plot: bool = False,
    cuts: bool = False,
    fs_bw_ratio: float = 1.2,
    num_sidelobes: int = 10,
    predict_null: bool = False,
    window_type: str = "rect",
    window_parameter: float = 0.0,
    shift_domain: str = "time",
    geo_heading: float | None = None,
    pixel_spacing: tuple[float, float] = (1.0, 1.0)
) -> tuple[dict, list["matplotlib.figure.Figure"] | None]:
    """
    Measure point-target attributes on a given chip.
    
    Presumes the point target is expected at the centerpoint of the chip. (e.g.
    the chip was generated by the `get_chip()` function.)

    Parameters
    ----------
    chip : numpy.ndarray
        complex 2D image.
    chip_i_min, chip_j_min : int
        The minimum row and column indices of the chip on the broader SLC scene.
    i_pos, j_pos : float
        Row and column indices where point-target is expected on the overall dataset.
        (Need not be integer.)
    nov : int, optional
        Amount of oversampling. Defaults to 32
    plot : bool, optional
        Generate interactive plots if True. Defaults to False
    cuts : bool, optional
        Include cuts through the peak in the output dictionary if True.
        Defaults to False
    fs_bw_ratio : float, optional
        Sampling frequency to bandwidth ratio, for use only when predict_null is True.
        Defaults to 1.2
    num_sidelobes : int, optional
        The total number of sidelobes for ISLR computation. Defaults to 10
    predict_null : bool, optional
        if True, mainlobe null locations are computed based on Fs/bandwidth ratio and
        window_type/window_parameter, when computing ISLR calculations. Otherwise,
        mainlobe null locations are computed based on null search algorithm.
        PSLR Exception: mainlobe does not include first sidelobes, search is always
        conducted to find the locations of first null regardless of predict_null
        parameter. Defaults to False
    window_type : str, optional
        User-provided window type, used for tapering, one of
        {"rect", "cosine", "kaiser"}. Defaults to 'rect'.
        "rect": Rectangular window is applied
        "cosine": Raised-Cosine window
        "kaiser": Kaiser Window.
    window_parameter : float, optional
        For a Kaiser window, this is the beta parameter.
        For a Raised Cosine window, it is the pedestal height.
        Ignored if window_type is "rect". Defaults to 0.0
    shift_domain : str, optional
        The domain to estimate peak location in, one of {"time", "frequency"}.
        If "time" then estimate peak location using max of oversampled data.
        If "frequency" then estimate a phase ramp in the frequency domain; useful when
        target is well focused, has high SNR, and is the only target in the neighborhood
        (often the case in point target simulations). Defaults to "time"
    geo_heading : float or None, optional
        Only for geocoded data: The heading of the satellite in the geodetic coordinate
        system of input data, in radians east of north, for use only if the SLC is
        geocoded. None if it is in radar coordinates. Defaults to None
    pixel_spacing : tuple[float, float], optional
        Only for geocoded data: The pixel spacing of the image in the
        (y_spacing, x_spacing) directions, for use in adjusting the heading angle to the
        raster dimensions. Defaults to (1.0, 1.0)

    Returns
    -------
    dict:
        Dictionary of point target attributes.
    list[matplotlib.figure.Figure] or None
        If plot=true then return the dictionary and a list of figures.
        Else, return None.
    """
    # Notation: For all references to i and j in this function, i represents row
    # indices on the chip or its parent raster, and j represents column indices
    # on the same.

    upsampled_chip, fx, fy = oversample(chip, nov, return_slopes=True)

    # The chip must be contiguous in order to be resampled.
    upsampled_chip = np.ascontiguousarray(upsampled_chip)

    k = np.nanargmax(np.abs(upsampled_chip))
    ichip, jchip = np.unravel_index(k, upsampled_chip.shape)
    chipmax = upsampled_chip[ichip, jchip]

    if shift_domain == 'time':
        imax = chip_min_i + ichip * 1.0 / nov
        jmax = chip_min_j + jchip * 1.0 / nov
    elif shift_domain == 'frequency':
        chip_jmax, chip_imax = measure_location_from_spectrum(chip)
        imax = chip_min_i + chip_imax
        jmax = chip_min_j + chip_jmax
        if (np.abs(chip_imax - ichip / nov) > 1) or (abs(chip_jmax - jchip / nov) > 1):
            warn(f"Spectral estimate of chip max at ({chip_imax}, {chip_jmax}) differs "
                f"from time domain estimate at ({ichip / nov}, {jchip / nov}) by "
                "more than one pixel.  Magnitude and phase is reported using "
                "time-domain estimate.")
    else:
        raise ValueError("Expected shift_domain in {'time', 'frequency'} but"
                        f" got {shift_domain}.")

    if geo_heading is not None:
        spacing_i, spacing_j = pixel_spacing

        az_cross_section = sample_geocoded_side_lobe(
            chip=upsampled_chip,
            heading=geo_heading,
            cr_position=(ichip, jchip),
            pixel_spacing=(spacing_i / nov, spacing_j / nov),
            deramp=True,
        )

        # For now, assume that the collection geometry was not squinted such that the
        # azimuth & range axes are orthogonal.
        # TODO: add squint angle to the interface to handle squinted cases
        rg_cross_section = sample_geocoded_side_lobe(
            chip=upsampled_chip,
            heading=geo_heading + np.pi / 2,
            cr_position=(ichip, jchip),
            pixel_spacing=(spacing_i / nov, spacing_j / nov),
            deramp=True,
        )

        az_slice = az_cross_section.data
        rg_slice = rg_cross_section.data

    else:
        az_slice = upsampled_chip[:, jchip]
        rg_slice = upsampled_chip[ichip, :]

    # Acquire resolution, in pixels, in the range and azimuth directions.
    range_resolution = estimate_resolution(rg_slice, 1.0 / nov)
    azimuth_resolution = estimate_resolution(az_slice, 1.0 / nov)

    # Find PSLR and ISLR of range and azimuth cuts
    fs_bw_ratio_ov = nov * fs_bw_ratio
    range_islr_db, range_pslr_db = compute_islr_pslr(
        rg_slice,
        fs_bw_ratio=fs_bw_ratio_ov,
        num_sidelobes=num_sidelobes,
        predict_null=predict_null,
        window_type=window_type,
        window_parameter=window_parameter
    )

    azimuth_islr_db, azimuth_pslr_db = compute_islr_pslr(
        az_slice,
        fs_bw_ratio=fs_bw_ratio_ov,
        num_sidelobes=num_sidelobes,
        predict_null=predict_null,
        window_type=window_type,
        window_parameter=window_parameter
    )

    return_dict = {
        "magnitude": np.abs(chipmax),
        "phase": np.angle(chipmax),
        "azimuth": {
            "ISLR": azimuth_islr_db,
            "PSLR": azimuth_pslr_db,
            "resolution": azimuth_resolution,
        },
        "range": {
            "ISLR": range_islr_db,
            "PSLR": range_pslr_db,
            "resolution": range_resolution,
        },
    }

    if geo_heading is not None:
        return_dict["x"] = {
            "index": jmax,
            "offset": jmax - j_pos,
            "phase ramp": fx,
        }
        return_dict["y"] = {
            "index": imax,
            "offset": imax - i_pos,
            "phase ramp": fy,
        }
        # In the case of a geocoded dataset, the range and azimuth slices will have the
        # peak power position located directly at the center of the slice.
        def get_slice_indices(n: int) -> np.ndarray:
            return (np.arange(n, dtype=np.float64) - (n - 1) / 2) / nov

        rg_indices = get_slice_indices(len(rg_slice))
        az_indices = get_slice_indices(len(az_slice))

    else:
        return_dict["azimuth"]["index"] = imax
        return_dict["azimuth"]["offset"] = imax - i_pos
        return_dict["azimuth"]["phase ramp"] = fy
        return_dict["range"]["index"] = jmax
        return_dict["range"]["offset"] = jmax - j_pos
        return_dict["range"]["phase ramp"] = fx

        # In the case of a radar-coordinate dataset, the range and azimuth slices will
        # have the peak power position located at an arbitrary point within the slice.
        idx_rg = np.arange(rg_slice.shape[0], dtype=float)
        idx_az = np.arange(az_slice.shape[0], dtype=float)
        rg_indices = chip_min_j + idx_rg / nov - j_pos
        az_indices = chip_min_i + idx_az / nov - i_pos

    if cuts:
        return_dict["azimuth"]["cut"] = az_indices.tolist()
        return_dict["azimuth"]["magnitude cut"] = list(np.abs(az_slice))
        return_dict["azimuth"]["phase cut"] = list(np.angle(az_slice))

        return_dict["range"]["cut"] = rg_indices.tolist()
        return_dict["range"]["magnitude cut"] = list(np.abs(rg_slice))
        return_dict["range"]["phase cut"] = list(np.angle(rg_slice))
    
        for x in ["azimuth", "range"]:
            assert len(return_dict[x]["cut"]) == len(return_dict[x]["magnitude cut"])
            assert len(return_dict[x]["cut"]) == len(return_dict[x]["phase cut"])

    if plot:
        figs = [
            plot_profile(az_indices, az_slice, title="Azimuth"),
            plot_profile(rg_indices, rg_slice, title="Range"),
        ]
        return return_dict, figs
    return return_dict, None


def sample_geocoded_side_lobe(
    chip: np.ndarray,
    heading: float,
    cr_position: tuple[int, int],
    pixel_spacing: tuple[float, float],
    deramp: bool = True,
) -> IPRCrossSection:
    """
    Given a chip of geocoded data, a CR position on the chip, and a heading, return
    side lobe slices interpolated over the data and the positions of the slices.

    This function does not preserve phase data in the returned slices if basebanding
    is used.

    Parameters
    ----------
    chip : np.ndarray of complex64
        A square region of geocoded data that contains a corner reflector.
    heading : float
        The heading of the sensor on the chip with respect to the corner
        reflector.
    cr_position : tuple[int, int]
        The position of the corner reflector on the chip.
    pixel_spacing : tuple[float, float]
        The (north, east) pixel spacing of pixels on the chip.
    deramp : bool, optional
        If True, estimate and deramp the carrier frequency of the data. If `deramp` is
        False, the data is assumed to be already base-banded (de-ramped).

    Returns
    -------
    IPRCrossSection
        An object containing information about a lobe cross section as described by this
        class.
    """

    chip_size, chip_size_x = chip.shape
    if chip_size != chip_size_x:
        raise ValueError(f"Chip must be square. Dimensions given as {chip.shape}.")
    
    if deramp:
        # Baseband the chip for resampling
        fx, fy = estimate_frequency(chip)
        chip = shift_frequency(chip, -fx, -fy)
    
    spacing_north, spacing_east = pixel_spacing

    # Adjust the heading from degrees east of north in spatial units to east of north
    # in units of pixels on the chip. This allows accurate prediction of the angle
    # at which the side lobes will appear on the geolocated image.
    # This is taken by taking the east and north components of the heading in units of
    # pixels per unit time, then taking the arctangent of the east over the north to get
    # the angle in the raster coordinate system.
    heading_north = np.cos(heading)
    heading_east = np.sin(heading)
    adjusted_heading = np.arctan2(
        heading_east / spacing_east, heading_north / spacing_north
    )

    pos_i, pos_j = cr_position
    heading_north_az = np.cos(adjusted_heading)
    heading_east_az = np.sin(adjusted_heading)

    indices_arange = np.arange(chip_size, dtype=np.float64) - (chip_size - 1) / 2

    # resample_to_coords is usually used for rectangular datasets and thus expects
    # rectangular arrays of indices and a rectangular output array. This can be done
    # by creating arrays that are one-by-chip-size pixels in dimension.
    sample_indices_i = np.empty((1, chip_size), dtype=np.float64)
    sample_indices_i[0,:] = indices_arange * heading_north_az + pos_i

    sample_indices_j = np.empty((1, chip_size), dtype=np.float64)
    sample_indices_j[0,:] = indices_arange * heading_east_az + pos_j

    # The interpolation will happen with a dummy grid, as the chip should already
    # be baseband and RadarGridParameters is only required for Doppler correction.
    dummy_grid: RadarGridParameters = RadarGridParameters(
        sensing_start=1,
        wavelength=1,
        prf=1,
        starting_range=1,
        range_pixel_spacing=1,
        look_side="right",
        length=chip_size,
        width=chip_size,
        ref_epoch=DateTime(),
    )

    # Sample all of the sample indices to get the slice.
    checking_sample = resample_to_coords(
        input_data_block=chip,
        range_input_indices=sample_indices_j,
        azimuth_input_indices=sample_indices_i,
        input_radar_grid=dummy_grid,
        native_doppler=LUT2d(),
        fill_value=np.nan + 1.0j * np.nan,
    )

    lobe_slice = checking_sample[0]

    if deramp:
        # Re-apply the phase ramp to the resampled data.
        # `fx` and `fy` have units of 1/samples
        lobe_slice *= np.exp(1j * (fy * sample_indices_i + fx * sample_indices_j))[0,:]

    return IPRCrossSection(
        data=lobe_slice,
        i_indices=sample_indices_i[0,:],
        j_indices=sample_indices_j[0,:],
    )


def tofloatvals(x):
    """Map all values in a (possibly nested) dictionary to Python floats.

    Modifies the dictionary in-place and returns None.
    """
    for k in x:
        if type(x[k]) == dict:
            tofloatvals(x[k])
        elif type(x[k]) == list:
            x[k] = [float(xki) for xki in x[k]]
        else:
            x[k] = float(x[k])


def main(argv):
    import argparse
    import matplotlib.pyplot as plt
    import json

    parser = argparse.ArgumentParser(description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-1",
        action="store_true",
        dest="one_based",
        help="Use one-based (Fortran) indexes.",
    )
    parser.add_argument(
        "-i", 
        action="store_true", 
        help="Interactive plots."
    )
    parser.add_argument(
        "--cuts",
        action="store_true", 
        help="Add range/azimuth slices to output JSON."
    )
    parser.add_argument(
        "--chipsize", 
        type=int, 
        default=64,
        required=False,
        help="Number of samples around the point target to be analyzed."
    )
    parser.add_argument("filename")
    parser.add_argument("n", type=int, help='number of range bins in a range line')
    parser.add_argument("row", type=float, help='point target azimuth bin location')
    parser.add_argument("column", type=float, help='point target range bin loction')
    parser.add_argument(
        "--nov",
        type=int,
        default=32,
        required=False,
        help="Point target samples oversampling factor",
    )
    parser.add_argument(
        "--fs-bw-ratio",
        type=float,
        default=1.2,
        required=False,
        help="Input data oversampling factor. Only used when --predict-null requested.",
    )
    parser.add_argument(
        "--num-sidelobes",
        type=float,
        default=10,
        required=False,
        help="total number of lobes, including mainlobe, default=10",
    )
    parser.add_argument(
        "-n",
        "--predict-null",
        action="store_true",
        default=False,
        help="default is false. If true, locate mainlobe nulls based on  Fs/BW ratio instead of search",
    )
    parser.add_argument(
        "--window-type",
        type=str,
        default='rect',
        required=False,
        help="Type of window used to taper impulse response sidelobes: 'rect', 'kaiser', 'cosine'. Only used when --predict-null requested.",
    )
    parser.add_argument(
        "--window-parameter",
        type=float,
        default=0,
        required=False,
        help="Window parameter for Kaiser and Raised Cosine windows",
    )
    parser.add_argument(
        "--shift-domain",
        choices=("time", "frequency"),
        default="time",
        help="Estimate shift in time domain or frequency domain."
    )
    args = parser.parse_args(argv[1:])

    n, i, j = [getattr(args, x) for x in ("n", "row", "column")]
    if args.one_based:
        i, j = i - 1, j - 1

    x = np.memmap(args.filename, dtype="complex64", mode="r")
    m = len(x) // n
    x = x.reshape((m, n))

    info, _ = analyze_point_target(
        x,
        i,
        j,
        nov = args.nov,
        plot=args.i,
        cuts=args.cuts,
        chipsize=args.chipsize,
        fs_bw_ratio=args.fs_bw_ratio,
        num_sidelobes=args.num_sidelobes,
        predict_null=args.predict_null,
        window_type=args.window_type,
        window_parameter=args.window_parameter,
        shift_domain=args.shift_domain
    )

    tofloatvals(info)

    print(json.dumps(info, indent=2))

    if args.i:
        plt.show()


if __name__ == "__main__":
    main(sys.argv)
