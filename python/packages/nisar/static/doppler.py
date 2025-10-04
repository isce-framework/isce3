from __future__ import annotations

import numpy as np

import isce3
from isce3.core import DataInterpMethod, normalize_data_interp_method

from .util import ceil_divide


def make_native_doppler_lut(
    radar_grid: isce3.product.RadarGridParameters,
    orbit: isce3.core.Orbit,
    attitude: isce3.core.Attitude,
    dem: isce3.geometry.DEMInterpolator,
    az_spacing: float,
    rg_spacing: float,
    *,
    interp_method: DataInterpMethod | str = DataInterpMethod.BILINEAR,
    bounds_error: bool = True,
) -> isce3.core.LUT2d:
    """
    Estimate the Doppler centroid of a SAR acquisition from ephemeris data.

    Parameters
    ----------
    radar_grid : isce3.product.RadarGridParameters
        Azimuth time / slant range coordinate grid of the SAR data.
    orbit : isce3.core.Orbit
        The path of the radar antenna phase center over a time interval that spans
        the azimuth time extent of `radar_grid`. Must have the same reference epoch as
        `radar_grid`.
    attitude : isce3.core.Attitude
        The orientation of the antenna over a time interval that spans the azimuth time
        extent of `radar_grid`. Represents the rotation from the radar antenna
        coordinate system to ECEF coordinates as a function of time. The antenna
        coordinate system is a Cartesian coordinate system with +Z axis pointing along
        the mechanical boresight of the antenna, +X axis pointing in the direction of
        increasing elevation angle, and +Y axis pointing in the direction of increasing
        azimuth angle. Must have the same reference epoch as `radar_grid`.
    dem : DEMInterpolator
        Digital elevation model specifying the height of topography, in meters above
        some reference ellipsoid, covering a region that spans the footprint of
        `radar_grid` on the ground.
    az_spacing : float
        The azimuth time spacing, in seconds, of the output LUT coordinate grid.
        Must be > 0.
    rg_spacing : float
        The slant range spacing, in meters, of the output LUT coordinate grid.
        Must be > 0.
    interp_method : isce3.core.DataInterpMethod or str, optional
        Interpolation method used by the output LUT. Defaults to bilinear interpolation.
    bounds_error : bool, optional
        Whether to raise an exception when attempting to evaluate the output LUT outside
        of its valid domain. Defaults to True.

    Returns
    -------
    isce3.core.LUT2d
        A 2-D LUT that may be used to compute the estimated Doppler centroid of the SAR
        acquisition, in hertz, as a function of azimuth time and slant range, spanning
        the input radar grid. The azimuth time tags are referenced to the same epoch as
        `radar_grid`, `orbit`, and `attitude`.
    """
    if radar_grid.ref_epoch != orbit.reference_epoch:
        raise ValueError(
            f"radar grid reference epoch ({radar_grid.ref_epoch}) must match orbit"
            f" reference epoch ({orbit.reference_epoch})"
        )
    if radar_grid.ref_epoch != attitude.reference_epoch:
        raise ValueError(
            f"radar grid reference epoch ({radar_grid.ref_epoch}) must match attitude"
            f" reference epoch ({attitude.reference_epoch})"
        )

    if not (az_spacing > 0.0):
        raise ValueError(f"{az_spacing=}, must be > 0")
    if not (rg_spacing > 0.0):
        raise ValueError(f"{rg_spacing=}, must be > 0")

    # Create a 1-D array with uniform spacing `step` that contains the interval
    # [`start`, `stop`].
    def make_step_range(start, stop, step) -> np.ndarray:
        num = ceil_divide(stop - start, step) + 1
        return start + step * np.arange(num)

    # Choose azimuth time and slant range coordinates of the Doppler LUT grid such that
    # it fully contains the input radar grid.
    start_time = radar_grid.sensing_start
    stop_time = radar_grid.sensing_stop
    az_time = make_step_range(start_time, stop_time, az_spacing)

    near_range = radar_grid.starting_range
    far_range = radar_grid.end_range
    slant_range = make_step_range(near_range, far_range, rg_spacing)

    return isce3.geometry.make_doppler_lut_from_attitude(
        az_time=az_time,
        slant_range=slant_range,
        orbit=orbit,
        attitude=attitude,
        wavelength=radar_grid.wavelength,
        dem=dem,
        interp_method=normalize_data_interp_method(interp_method),
        bounds_error=bounds_error,
    )
