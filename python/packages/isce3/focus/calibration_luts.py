from enum import Enum, unique
import isce3
from isce3.core import LUT2d
import numpy as np


def make_los_luts(orbit, attitude, side, doppler, wavelength,
                  dem=isce3.geometry.DEMInterpolator(),
                  rdr2geo_params=dict(),
                  interp_method=None):
    """
    Generate look-up tables related to radar line-of-sight.

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
    inc_lut : isce3.core.LUT2d
        Ellipsoidal incidence angle (in rad) relative to ellipsoid normal at
        target vs native-Doppler radar coordinates (time, range), where
        ellipsoid is inferred from DEM.
    hgt_lut : isce3.core.LUT2d
        Terrain height (in m) vs native-Doppler radar coordinates (time, range).
    """
    # Use same coords in the provided Doppler LUT
    # Note that RSLC metadata requires all LUTs to be on the same grid anyhow.
    az_times = doppler.y_start + doppler.y_spacing * np.arange(doppler.length)
    slant_ranges = doppler.x_start + doppler.x_spacing * np.arange(doppler.width)

    # Get XYZ coord for all (az_time, slant_range) coords.
    target_xyz = np.zeros((len(az_times), len(slant_ranges), 3))
    for itime, az_time in enumerate(az_times):
        for irange, slant_range in enumerate(slant_ranges):
            dop = doppler.eval(az_time, slant_range)
            target_xyz[itime, irange] = isce3.geometry.rdr2geo_bracket(
                    az_time, slant_range, orbit, side, dop, wavelength, dem=dem,
                    **rdr2geo_params
            )

    ellipsoid = dem.ellipsoid

    # Given XYZs we can easily compute the angle layer
    elevation = np.zeros(target_xyz.shape[:2])
    incidence = np.zeros(target_xyz.shape[:2])
    height = np.zeros(target_xyz.shape[:2])

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

            lon, lat, hgt = ellipsoid.xyz_to_lon_lat(target_xyz[i, j])
            normal = ellipsoid.n_vector(lon, lat)
            incidence[i, j] = np.arccos(-los_xyz.dot(normal))
            height[i, j] = hgt

    if interp_method is None:
        interp_method = doppler.interp_method

    return tuple(LUT2d(doppler.x_start, doppler.y_start,
                       doppler.x_spacing, doppler.y_spacing, x,
                       interp_method, doppler.bounds_error)
                 for x in (elevation, incidence, height))


@unique
class AreaConvention(str, Enum):  # StrEnum in Python 3.11+
    """
    Flags denoting pixel area normalization convention for backscatter values.
    """

    BETA0 = "beta0"
    """
    Area in slant plane (line-of-sight cross velocity).
    """

    SIGMA0 = "sigma0"
    """
    Area in ground plane.
    """

    GAMMA0 = "gamma0"
    """
    Ground area projected into a plane perpendicular to radar line-of-sight.
    """


def make_cal_luts(inc_lut: LUT2d, abscal: float = 1.0,
                  input_convention: AreaConvention = "beta0") -> tuple[LUT2d]:
    """
    Generate calibration lookup tables for converting between backscatter
    area normalization conventions.

    Parameters
    ----------
    inc_lut : isce3.core.LUT2d
        Ellipsoidal incidence angle (in rad) lookup table.
    abscal : float
        Absolute calibration factor to divide from image power (units of DN^2)
    input_convention : isce3.focus.AreaConvention, optional
        Area normalization convention of the input data.

    Returns
    -------
    beta0 : isce3.core.LUT2d
    sigma0 : isce3.core.LUT2d
    gamma0 : isce3.core.LUT2d
        Lookup tables for scaling input SLC data to all three backscatter
        area normalization conventions according to the formula
            backscatter = abs(z)**2 / lut**2
        where z is the complex SLC DN value and lut is the interpolated LUT
        value.  Coordinate axes and other settings are identical to the input
        incidence angle LUT.

    Notes
    -----
    The units and sense of the LUTs follow the conventions established for
    Sentinel-1 data [1]_ and NISAR data [2]_.

    .. [1] Piantanida, et al., "Sentinel-1 Level 1 Detailed Algorithm
       Definition", ESA DI-MPC-IPFDPM, Rev 2.5, p. 9-28, 2022.
    .. [2] Hawkins, "NASA SDS Product Specification Level-1 Range Doppler
       Single Look Complex," JPL D-102268, Version 1.1.0, p. 17, 2024.
    """
    # validate input
    input_convention = AreaConvention(input_convention)

    # grab incidence angle
    inc = inc_lut.data
    one = np.ones_like(inc)

    # XXX will include abscal when constructing LUT2d below.
    if input_convention == "beta0":
        beta0 = one
        sigma0 = np.sqrt(1.0 / np.sin(inc))
        gamma0 = np.sqrt(1.0 / np.tan(inc))
    elif input_convention == "sigma0":
        beta0 = np.sqrt(np.sin(inc))
        sigma0 = one
        gamma0 = np.sqrt(np.cos(inc))
    elif input_convention == "gamma0":
        beta0 = np.sqrt(np.tan(inc))
        sigma0 = np.sqrt(1.0 / np.cos(inc))
        gamma0 = one
    else:
        assert False, f"unhandled area convention {input_convention}"

    return tuple(LUT2d(inc_lut.x_start, inc_lut.y_start, inc_lut.x_spacing,
                       inc_lut.y_spacing, backscatter * np.sqrt(abscal),
                       inc_lut.interp_method, inc_lut.bounds_error)
                 for backscatter in (beta0, sigma0, gamma0))
