from __future__ import annotations

import warnings
from collections.abc import Collection, Mapping
from typing import Optional

import numpy as np
from numpy.linalg import norm
from numpy.typing import ArrayLike

import isce3
from isce3.cal import point_target_info


def pow2db(x: ArrayLike) -> np.ndarray:
    """Converts a power quantity from linear units to decibels (dB)."""
    return 10.0 * np.log10(x)


def db2pow(x: ArrayLike) -> np.ndarray:
    """Converts a power quantity from to decibels (dB) to linear units."""
    return 10.0 ** np.divide(x, 10.0)


def find_quadratic_poly2d_peak(coeffs: Collection[float]) -> tuple[float, float]:
    r"""
    Find the location of the peak of a quadratic bivariate polynomial.

    Determine the location of the peak of the quadratic bivariate polynomial
    :math:`z = ax^2 + by^2 + cxy + dx + ey + f` by solving for the common real root of
    its partial derivatives with respect to :math:`x` and :math:`y`.

    Solves the system of equations

    .. math::

        \begin{cases}
            \frac{\partial z}{\partial x} = 2ax + cy + d = 0\\
            \frac{\partial z}{\partial y} = 2by + cx + e = 0
        \end{cases}

    whose solution :math:`(x_p, y_p)` is

    .. math::

        \begin{align*}
            x_p &= \frac{2bd - ce}{c^2 - 4ab}\\
            y_p &= \frac{2ae - cd}{c^2 - 4ab}
        \end{align*}.

    Parameters
    ----------
    coeffs : array_like
        Coefficients `[a, b, c, d, e, f]` of the quadratic bivariate polynomial
        expression :math:`z = ax^2 + by^2 + cxy + dx + ey + f`.

    Returns
    -------
    xp, yp : float
        x- and y-coordinates of the peak.

    Raises
    ------
    ValueError
        If too few or too many polynomial coefficients were provided.
    RuntimeError
        If the critical point is not a local maxima.
    """
    if len(coeffs) < 6:
        raise ValueError(
            "too few polynomial coefficients were supplied: expected 6, got"
            f" {len(coeffs)}"
        )
    elif len(coeffs) > 6:
        raise ValueError(
            "too many polynomial coefficients were supplied: expected 6, got"
            f" {len(coeffs)}"
        )

    # Unpack coefficients.
    a, b, c, d, e, _ = coeffs

    # Check the second partial derivatives to make sure that the function is concave so
    # that the critical point is a local maxima.
    if (a >= 0.0) or (b >= 0.0):
        raise RuntimeError("polynomial does not contain a peak")

    # Solve for the peak location.
    denom = c ** 2 - 4.0 * a * b
    xp = (2.0 * b * d - e * c) / denom
    yp = (2.0 * a * e - c * d) / denom

    return xp, yp


def estimate_peak_value(data: ArrayLike) -> float:
    """
    Estimate the peak value of a 2-D discrete-time signal by fitting a quadratic.

    Parameters
    ----------
    data : array_like
        A real-valued 2-D array of samples containing a single peak-like signal that may
        be approximated by a 2-D quadratic polynomial. The data is assumed to be
        uniformly sampled.

    Returns
    -------
    peak : float
        The peak value.

    Raises
    ------
    RuntimeError
        If the input data does not contain a well-formed peak.
    """
    data = np.asanyarray(data)

    if data.ndim != 2:
        raise ValueError("input data must be a 2-D array")
    if np.issubdtype(data.dtype, np.complexfloating):
        raise TypeError("input data must be real-valued")

    # Get the pixel coordinates of the 2-D grid that the input data lies on.
    num_rows, num_cols = data.shape
    xcoords = np.arange(num_cols, dtype=np.float64)
    ycoords = np.arange(num_rows, dtype=np.float64)

    # Fit a quadratic 2-D polynomial to the peak.
    poly2d = isce3.core.fit_bivariate_polynomial(xcoords, ycoords, data, degree=2)

    # Get the coefficients of the quadratic bivariate polynomial.
    coeffs = [
        poly2d.coeffs[0, 2],  # x^2
        poly2d.coeffs[2, 0],  # y^2
        poly2d.coeffs[1, 1],  # xy
        poly2d.coeffs[0, 1],  # x
        poly2d.coeffs[1, 0],  # y
        poly2d.coeffs[0, 0],  # 1
    ]

    # Get the (x,y) location of the peak.
    try:
        xp, yp = find_quadratic_poly2d_peak(coeffs)
    except RuntimeError as e:
        raise RuntimeError("the input array did not contain a well-formed peak") from e

    # Estimate the peak power by evaluating the polynomial expression at the peak
    # location.
    return poly2d.eval(yp, xp)


def zero_crossings(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Find the positions of zero-crossings in y(x).

    If the data contained no zero-crossings, an empty array is returned.

    A zero crossing is considered to have occurred only when two adjacent samples have
    opposing signs (e.g. if the `y` array is all zeros, the data does not contain any
    zero-crossings).

    Parameters
    ----------
    x, y : array_like
        1-dimensional arrays containing x and y samples.

    Returns
    -------
    xz : numpy.ndarray
        Output array containing x-positions of zero-crossings.
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)

    if (x.ndim != 1) or (y.ndim != 1):
        raise ValueError("input arrays must be 1-dimensional")

    # For each pair of adjacent samples in the input `x`, check whether a sign change
    # occurred between them (either negative-to-nonnegative or nonnegative-to-negative).
    # Returns a boolean array where nonzero values indicate that a sign changed occurred
    # between the corresponding element and its right-hand neighbor in `x`. Zeros are
    # considered nonnegative.
    def sign_change(x):
        return np.diff(np.signbit(x))

    # Get pairs of indices of elements in `y` bounding each zero-crossing point.
    (left_indices,) = np.nonzero(sign_change(y))
    right_indices = left_indices + 1

    # Computes the x-intercept of the line through (`x0`, `y0`) and (`x1`, `y1`).
    def x_intercept(x0, y0, x1, y1):
        assert np.all(y1 != y0)
        return x0 - y0 * (x1 - x0) / (y1 - y0)

    # Estimate zero-crossing points with sub-sample precision by linear interpolation
    # between neighboring samples.
    x0 = x[left_indices]
    y0 = y[left_indices]
    x1 = x[right_indices]
    y1 = y[right_indices]
    return x_intercept(x0, y0, x1, y1)


def estimate_peak_width(data: ArrayLike, threshold: float) -> float:
    """
    Estimate the width of a peak in a 1-D discrete-time signal.

    Parameters
    ----------
    data : array_like
        A real-valued 1-D array of samples containing a single peak-like signal that
        exceeds `threshold`. The data is assumed to be uniformly sampled.
    threshold : float
        The threshold signal value at which to measure the peak width. The signal data
        should cross the threshold exactly twice, once at the rising edge and once at
        the falling edge of the peak.

    Returns
    -------
    width : float
        The width, in samples, of the peak.

    Raises
    ------
    RuntimeError
        If the input data did not contain a well-formed peak, or if it contained
        multiple peaks.
    """
    data = np.asanyarray(data)

    if data.ndim != 1:
        raise ValueError("input data must be a 1-D array")
    if np.issubdtype(data.dtype, np.complexfloating):
        raise TypeError("input data must be real-valued")

    # Get the approximate locations, in samples, where the input signal data intersects
    # with the horizontal line y=threshold.
    coords = np.arange(len(data), dtype=np.float64)
    edges = zero_crossings(coords, data - threshold)

    # There should be exactly two such locations found (the rising edge & falling edge
    # of the peak).
    if len(edges) < 2:
        raise RuntimeError("the input array did not contain a well-formed peak")
    elif len(edges) > 2:
        raise RuntimeError("the input array contained multiple peaks")

    # Compute the inner width between the two edges, in samples.
    return edges[1] - edges[0]


def measure_target_rcs(
    target_llh: isce3.core.LLH,
    img_data: ArrayLike,
    radar_grid: isce3.product.RadarGridParameters,
    orbit: isce3.core.Orbit,
    doppler: isce3.core.LUT2d,
    ellipsoid: isce3.core.Ellipsoid = isce3.core.WGS84_ELLIPSOID,
    *,
    nchip: int = 64,
    upsample_factor: int = 32,
    peak_find_domain: str = "time",
    nfit: int = 5,
    power_method: str = "box",
    pthresh: float = 3.0,
    geo2rdr_params: Optional[Mapping[str, float]] = None,
) -> float:
    r"""
    Estimate a point-like target's radar cross-section (RCS) using the provided echo
    data.

    Parameters
    ----------
    target_llh : isce3.core.LLH
        The target position expressed as longitude, latitude, and height above the
        reference ellipsoid in radians, radians, and meters respectively.
    img_data : array_like
        The input radar domain (azimuth time by slant range) image data. A 2-D array
        with shape (num_lines, num_range_bins). The image should be focused and
        normalized such that its intensity represents :math:`\beta_0` values\ [1]_. The
        data is assumed to be uniformly-sampled.
    radar_grid : isce3.product.RadarGridParameters
        The radar coordinates of the grid on which `img_data` is sampled.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center.
    doppler : isce3.core.LUT2d
        The Doppler centroid, in hertz, of the image grid of the focused data, expressed
        as a function of azimuth time, in seconds relative to the epoch of `radar_grid`,
        and slant range, in meters. Note that this should be the Doppler associated with
        the image grid, which may in general be different from the native Doppler of the
        aquired echo data.
    ellipsoid : isce3.core.Ellipsoid, optional
        The geodetic reference ellipsoid, with dimensions in meters. Defaults to the
        WGS 84 ellipsoid.
    nchip : int, optional
        The width, in pixels, of the square block of image data to extract centered
        around the target position for oversampling and peak finding. Must be >= 1.
        Defaults to 64.
    upsample_factor : int, optional
        The upsampling ratio. Must be >= 1. Defaults to 32.
    peak_find_domain : {'time', 'freq'}, optional
        Option controlling how the target peak position is estimated.

        'time':
          The default mode. The peak location is found in the time domain by detecting
          the maximum value within a square block of image data around the expected
          target location. The signal data is upsampled to improve precision.

        'freq':
          The peak location is found by estimating the phase ramp in the frequency
          domain. This mode is useful when target is well-focused, has high SNR, and is
          the only target in the neighborhood (often the case in point target
          simulations).
    nfit : int, optional
        The width, in *oversampled* pixels, of the square sub-block of image data
        (centered around the target position) to extract for fitting a quadratic
        polynomial to the peak. Note that this is the size in pixels *after upsampling*.
        Must be >= 3. Defaults to 5.
    power_method : {'box', 'integrated'}, optional
        The method for estimating the target signal power.

        'box':
          The default mode. Measures power using the rectangular box method, which
          assumes that the target response can be approximated by a 2-D rectangular
          function. The total power is estimated by multiplying the peak power by the
          3dB response widths in along-track and cross-track directions.

        'integrated':
          Measures power using the integrated power method. The total power is measured
          by summing the power of bins whose power exceeds a predefined minimum power
          threshold.
    pthresh : float, optional
        The minimum power threshold, measured in dB below the peak power, for estimating
        the target signal power using the integrated power method. This parameter is
        ignored if `power_method` is not 'integrated'. Defaults to 3.
    geo2rdr_params : dict or None, optional
        An optional dict of parameters configuring the behavior of the root-finding
        routine used in geo2rdr. The following keys are supported:

        'tol_aztime':
          Azimuth time convergence tolerance in seconds.

        'time_start':
          Start of search interval, in seconds. Defaults to ``orbit.start_time``.

        'time_end':
          End of search interval, in seconds. Defaults to ``orbit.end_time``.

    Returns
    -------
    sigma : float
        The measured radar cross-section, in meters squared (linear scale -- not dB).

    Raises
    ------
    RuntimeError
        If the target position was outside of the image grid or too near the border.
    RuntimeError
        If the data block surrounding the target position did not contain a well-formed
        peak.

    References
    ----------
    .. [1] R. K. Raney, T. Freeman, R. W. Hawkins, and R. Bamler, “A plea for radar
       brightness,” Proceedings of IGARSS '94 - 1994 IEEE International Geoscience and
       Remote Sensing Symposium.
    """
    if np.shape(img_data) != (radar_grid.length, radar_grid.width):
        raise ValueError(
            "shape mismatch: img_data and radar_grid must have compatible shapes"
        )

    # Ensure that time tags of orbit & radar grid are w.r.t the same epoch.
    if orbit.reference_epoch != radar_grid.ref_epoch:
        warnings.warn(
            "input orbit and radar_grid are referenced to different UTC epochs -- a"
            " copy of the orbit data will be created with updated time tags",
            RuntimeWarning,
        )
        orbit = orbit.copy()
        orbit.update_reference_epoch(radar_grid.ref_epoch)

    if nchip < 1:
        raise ValueError("nchip must be >= 1")
    if upsample_factor < 1:
        raise ValueError("upsample_factor must be >= 1")
    if nfit < 3:
        raise ValueError("nfit must be >= 3 in order to fit a quadratic polynomial")

    if upsample_factor < 2:
        warnings.warn(
            "upsample_factor should be >= 2 to ensure that power-detecting the signal"
            " data does not cause aliasing and that there are sufficient samples to fit"
            " to the signal peak",
            RuntimeWarning,
        )

    # Convert LLH object to an array containing [lon, lat, height].
    target_llh = target_llh.to_vec3()

    # Get target (x,y,z) position in ECEF coordinates.
    target_xyz = ellipsoid.lon_lat_to_xyz(target_llh)

    if geo2rdr_params is None:
        geo2rdr_params = {}

    wavelength = radar_grid.wavelength
    look_side = radar_grid.lookside

    # Run geo2rdr to get the target position in radar coordinates.
    aztime, srange = isce3.geometry.geo2rdr_bracket(
        xyz=target_xyz,
        orbit=orbit,
        doppler=doppler,
        wavelength=wavelength,
        side=look_side,
        **geo2rdr_params,
    )

    # Check if target position was outside of the image grid.
    out_of_bounds = (
        (aztime < radar_grid.sensing_start)
        or (aztime > radar_grid.sensing_stop)
        or (srange < radar_grid.starting_range)
        or (srange > radar_grid.end_range)
    )
    if out_of_bounds:
        raise RuntimeError(
            "target position in radar coordinates was outside of the supplied image"
            " grid"
        )

    # Convert target position (aztime,srange) coordinates to (row,col) pixel
    # coordinates within the radar grid.
    i = (aztime - radar_grid.sensing_start) * radar_grid.prf
    j = (srange - radar_grid.starting_range) / radar_grid.range_pixel_spacing

    # Raise an exception if the target position was too near the border of the image,
    # with insufficient margin to extract a "chip" around it.
    near_border = (
        (i < nchip // 2)
        or (i > radar_grid.length - nchip // 2)
        or (j < nchip // 2)
        or (j > radar_grid.width - nchip // 2)
    )
    if near_border:
        raise RuntimeError(
            "target is too close to image border -- consider reducing nchip"
        )

    # Extract a small square block of image data centered around the expected target
    # location.
    _, _, chip = point_target_info.get_chip(img_data, i, j, nchip=nchip)

    # Upsample.
    chip_ups = point_target_info.oversample(chip, nov=upsample_factor)

    # Get chip power in linear and dB.
    chip_ups_pwr = np.abs(chip_ups) ** 2
    chip_ups_pwr_db = pow2db(chip_ups_pwr)

    # Estimate the peak position within the upsampled chip, in pixel coordinates.
    if peak_find_domain == "time":
        ichip, jchip = np.unravel_index(np.argmax(chip_ups_pwr), chip_ups.shape)
    elif peak_find_domain == "freq":
        jchip, ichip = point_target_info.measure_location_from_spectrum(chip_ups)
    else:
        raise ValueError(
            f"peak_find_domain must be 'time' or 'freq', got {peak_find_domain!r}"
        )

    # Estimate the peak power of the process, in dB, by fitting a quadratic polynomial
    # to a small subset of the upsampled chip. Note that if the target response is
    # approximately Gaussian-shaped in linear scale, it will approximate a quadratic
    # function in dB.
    _, _, small_chip_pwr_db = point_target_info.get_chip(
        chip_ups_pwr_db, ichip, jchip, nchip=nfit
    )
    peak_pwr_db = estimate_peak_value(small_chip_pwr_db)

    # Convert dB back to linear units.
    peak_pwr = db2pow(peak_pwr_db)

    # Estimate total target signal power using rectangular box method or integrated
    # power method.
    if power_method == "box":
        # Estimate target response width in azimuth bins.
        az_cut = chip_ups_pwr_db[:, int(round(jchip))]
        az_response_width = estimate_peak_width(az_cut, peak_pwr_db - 3.0)

        # Estimate target response width in range bins.
        rg_cut = chip_ups_pwr_db[int(round(ichip)), :]
        rg_response_width = estimate_peak_width(rg_cut, peak_pwr_db - 3.0)

        # Get total power assuming a rectangular-shaped response function.
        target_pwr = peak_pwr * az_response_width * rg_response_width
    elif power_method == "integrated":
        # Get a mask of pixels above the specified threshold.
        mask = chip_ups_pwr_db >= (peak_pwr_db - pthresh)

        # Get the total power of pixels above the threshold.
        target_pwr = np.sum(chip_ups_pwr[mask])
    else:
        raise ValueError(
            f"power_method must be 'box' or 'integrated', got {power_method!r}"
        )

    # Get the approximate ground track speed of the platform, assuming that the platform
    # trajectory can be locally approximated by a circular geocentric orbit.
    platform_pos, platform_vel = orbit.interpolate(aztime)
    target_pos = ellipsoid.lon_lat_to_xyz(target_llh)
    platform_ground_speed = norm(platform_vel) * (norm(target_pos) / norm(platform_pos))

    # Get the approximate azimuth pixel spacing (of the original -- not oversampled --
    # pixels), in meters, at the target location.
    az_pixel_spacing = platform_ground_speed / radar_grid.prf

    # Convert from pixels^2 to meters^2.
    pixel_area = az_pixel_spacing * radar_grid.range_pixel_spacing
    ups_pixel_area = pixel_area / upsample_factor ** 2
    return target_pwr * ups_pixel_area
