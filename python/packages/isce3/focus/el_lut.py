import isce3
import numpy as np


def make_el_lut(orbit, attitude, side, doppler, wavelength,
                dem=isce3.geometry.DEMInterpolator(),
                rdr2geo_params=dict(),
                interp_method=None):
    """
    Generate a look-up table of antenna elevation angle (EL) over.
    Parameters
    ----------
    orbit : isce3.core.Orbit
        Trajectory of antenna phase center
    Attitude : isce3.core.Attitude
        Orientation of the antenna (RCS) frame wrt ECEF frame
    side : isce3.core.LookSide or str in {"left", "right"}
        Side the radar looks
    doppler : isce3.core.LUT2d
        Doppler centroid (in Hz) corresponding to radar boresight.
        The output EL LUT2d will have identical (time, range) postings.
    wavelength : float
        Wavelength associated with Doppler LUT (to convert to squint angle).
    dem : isce3.geometry.DEMInterpolator, optional
        Digital elevation model. Surface heights (in m) above ellipsoid.
    rdr2geo_params : dict, optional
        Root finding parameters for isce3.geometry.rdr2geo_bracket.
        Keys among {"tol_height", "look_min", "look_max"}.
    interp_method : str, optional
        Interpolation method to be used for the resulting LUT
        Defaults to the same interpolation method as the doppler LUT
    Returns
    -------
    el_lut : isce3.core.LUT2d
        EL angle (in rad) vs native-Doppler radar coordinates (time, range).
    """
    # Use same coords in the provided Doppler LUT
    # Note that RSLC metadata requires all LUTs to be on the same grid anyhow.
    az_times = doppler.y_start + doppler.y_spacing * np.arange(doppler.length)
    slant_ranges = doppler.x_start + doppler.x_spacing * np.arange(doppler.width)

    # Get XYZ coord for all (az_time, slant_range) coords.
    target_xyz = np.zeros((len(az_times), len(slant_ranges), 3))
    if side == "left":
        side = isce3.core.LookSide.Left
    elif side == "right":
        side = isce3.core.LookSide.Right
    for itime, az_time in enumerate(az_times):
        for irange, slant_range in enumerate(slant_ranges):
            dop = doppler.eval(az_time, slant_range)
            target_xyz[itime, irange] = isce3.geometry.rdr2geo_bracket(
                    az_time, slant_range, orbit, side, dop, wavelength, dem=dem,
                    **rdr2geo_params
            )

    # Given XYZs we can easily compute the angle layer
    elevation = np.zeros(target_xyz.shape[:2])
    # There are several conventions for antenna angle coordinates.
    # The NISAR patterns are provided in the "EL AND AZ" convention,
    # which is the default in isce3.  See JPL D-80882 and REE manual.
    frame = isce3.antenna.Frame()
    for i in range(elevation.shape[0]):
        # position and orientation don't depend on slant range
        ti = az_times[i]
        radar_pos, radar_vel = orbit.interpolate(ti)
        # RCS is the "(Radar Antenna) Reflector Coordinate System"
        # The attitude data tells us how it's oriented relative to
        # ECEF XYZ coordinates.  See JPL D-80882, JPL D-102264
        q_rcs2xyz = attitude.interpolate(ti)
        for j in range(elevation.shape[1]):
            los_xyz = target_xyz[i,j] - radar_pos
            los_xyz *= 1.0 / np.linalg.norm(los_xyz)
            los_rcs = q_rcs2xyz.conjugate().rotate(los_xyz)
            el, az = frame.cart2sph(los_rcs)
            elevation[i, j] = el

    if interp_method is None:
        interp_method = doppler.interp_method

    return isce3.core.LUT2d(doppler.x_start, doppler.y_start,
                            doppler.x_spacing, doppler.y_spacing, elevation,
                            interp_method, doppler.bounds_error)
