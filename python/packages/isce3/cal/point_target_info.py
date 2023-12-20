#!/usr/bin/env python3
"""
Analyze a point target in a complex*8 file.
"""
import sys
import numpy as np
from warnings import warn

desc = __doc__


class MissingNull(Exception):
    """Raised when mainlobe null(s) cannot be determined"""

    pass

class UnsupportedWindow(Exception):
    """Raised if window_type input is not supported."""

    pass

def get_chip(x, i, j, nchip=64):
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
    i = np.argmax(y)
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
    ileft = np.argmin(left)
    right = z[i:]
    iright = i + np.argmin(right)
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
    peak_idx = np.argmax(data_in_pwr_linear)

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

    pwr_total = np.sum(data_in_pwr_linear)
    islr_main_pwr = np.sum(islr_mainlobe)
    islr_side_pwr = np.sum(islr_sidelobe)

    islr_db = 10 * np.log10(islr_side_pwr / islr_main_pwr)

    # PSLR
    pslr_sidelobe_range = np.r_[
        sidelobe_left_idx:null_first_left_idx,
        null_first_right_idx + 1 : sidelobe_right_idx + 1,
    ]
    pslr_mainlobe = data_in_pwr_linear[null_first_left_idx:null_first_right_idx + 1]
    pslr_sidelobe = data_in_pwr_linear[pslr_sidelobe_range]

    pwr_mainlobe_max = np.amax(pslr_mainlobe)
    pwr_sidelobe_max = np.amax(pslr_sidelobe)

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


def analyze_point_target(
    slc,
    i,
    j,
    nov=32,
    plot=False,
    cuts=False,
    chipsize=64,
    fs_bw_ratio=1.2,
    num_sidelobes=10,
    predict_null=False,
    window_type='rect',
    window_parameter=0,
    shift_domain='time',
):
    """Measure point-target attributes.

    Parameters
    ----------
        slc: array of 2D
            complex image (2D array).
        i, j: float
            Row and column indices where point-target is expected.
            (Need not be integer.)
        nov: int
            Amount of oversampling.
        plot: bool
            Generate interactive plots.
        cuts: bool
            Include cuts through the peak in the output dictionary.
        chipsize: int
            number of samples around the point target to be included for
            point target metric analysis
        fs_bw_ratio: float
            optional, sampling frequency to bandwidth ratio
            Only used when predict_null=True
            fs_bw_ratio = Fs / BW
        num_sidelobes: float
            optional total number of sidelobes for ISLR computation,
            default is 10.
        predict_null: boolean
            optional, if predict_null is True, mainlobe null locations are computed based 
            on Fs/bandwidth ratio and window_type/window_parameter, when computing
            ISLR calculations. Otherwise, mainlobe null locations are computed based on null
            search algorithm.
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
            optional, window parameter. For a Kaiser window, this is the beta parameter.
            For a Raised Cosine window, it is the pedestal height.
            It is ignored if `window_type='rect'`.
        shift_domain: {time, frequency}
            If 'time' then estimate peak location using max of oversampled data.
            If 'frequency' then estimate a phase ramp in the frequency domain.
            Default is 'time' but 'frequency' is useful when target is well
            focused, has high SNR, and is the only target in the neighborhood
            (often the case in point target simulations).

    Returns:
    --------
        Dictionary of point target attributes.  If plot=true then return the
        dictionary and a list of figures.
    """
    
    # Check if i or j indices are out of bounds w.r.t slc image
    if i > slc.shape[0] or i < 0 or j > slc.shape[1] or j < 0:
        raise ValueError('User provided target location (lon/lat/height) points to a spot '
        'outside of RSLC image. This could be due to residual azimuth/range delays or '
        'incorrect geometry info or incorrect user provided target location info.')

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

    chip_i0, chip_j0, chip0 = get_chip(slc, i, j, nchip=chipsize)

    chip, fx, fy = oversample(chip0, nov=nov, return_slopes=True)

    k = np.argmax(np.abs(chip))
    ichip, jchip = np.unravel_index(k, chip.shape)
    chipmax = chip[ichip, jchip]

    if shift_domain == 'time':
        imax = chip_i0 + ichip * 1.0 / nov
        jmax = chip_j0 + jchip * 1.0 / nov
    elif shift_domain == 'frequency':
        tx, ty = measure_location_from_spectrum(chip0)
        imax = chip_i0 + ty
        jmax = chip_j0 + tx
        if (abs(tx - jchip / nov) > 1) or (abs(ty - ichip / nov) > 1):
            warn(f"Spectral estimate of chip max at ({ty}, {tx}) differs from "
                 f"time domain estimate at ({ichip / nov}, {jchip / nov}) by "
                 "more than one pixel.  Magnitude and phase is reported using "
                 "time-domain estimate.")
    else:
        raise ValueError("Expected shift_domain in {'time', 'frequency'} but"
                         f" got {shift_domain}.")

    az_slice = chip[:, jchip]
    rg_slice = chip[ichip, :]

    dr = estimate_resolution(rg_slice, 1.0 / nov)
    da = estimate_resolution(az_slice, 1.0 / nov)

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

    d = {
        "magnitude": abs(chipmax),
        "phase": np.angle(chipmax),
        "range": {
            "index": jmax,
            "offset": jmax - j,
            "phase ramp": fx,
            "resolution": dr,
            "ISLR": range_islr_db,
            "PSLR": range_pslr_db,
        },
        "azimuth": {
            "index": imax,
            "offset": imax - i,
            "phase ramp": fy,
            "resolution": da,
            "ISLR": azimuth_islr_db,
            "PSLR": azimuth_pslr_db,
        },
    }

    idx = np.arange(chip.shape[0], dtype=float)
    ti = chip_i0 + idx / nov - i
    tj = chip_j0 + idx / nov - j
    if cuts:
        d["range"]["magnitude cut"] = list(np.abs(rg_slice))
        d["range"]["phase cut"] = list(np.angle(rg_slice))
        d["range"]["cut"] = list(tj)
        d["azimuth"]["magnitude cut"] = list(np.abs(az_slice))
        d["azimuth"]["phase cut"] = list(np.angle(az_slice))
        d["azimuth"]["cut"] = list(ti)
    if plot:
        figs = [
            plot_profile(tj, rg_slice, title="Range"),
            plot_profile(ti, az_slice, title="Azimuth"),
        ]
        return d, figs
    return d


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

    info = analyze_point_target(
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
    if args.i:
        info = info[0]

    tofloatvals(info)

    print(json.dumps(info, indent=2))

    if args.i:
        plt.show()


if __name__ == "__main__":
    main(sys.argv)
