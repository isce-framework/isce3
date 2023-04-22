"""
Some antenna-related geometry functions
"""
import numpy as np

from isce3.antenna import Frame
from isce3.core import Ellipsoid
from isce3.geometry import DEMInterpolator, rdr2geo


def geo2ant(tg_llh, pos_sc_ecef, quat_ant2ecef, ant_frame=Frame(),
            ellips=Ellipsoid()):
    """Convert Geodetic location to Antenna frame (EL,AZ).

    Parameters
    ----------
    tg_llh : sequence or 1-D array of three floats
        Geodetic longitude, latitude, and height of a target w.r.t
        Ellipsoid in (rad, rad, m)
    pos_sc_ecef : sequence or 1-D array of three floats
        Position of spacecraft in XYZ ECEF coordinate in meters
    quat_ant2ecef : isce3.core.Quaternion
        Quaternion attitude of spacecraft from antenna frame to ECEF
    ant_frame : isce3.antenna.Frame, default=EL-AND-AZ
        This object defines antenna spherical coordinate type.
    ellips : isce3.core.Ellipsoid, default=WGS84
        Ellipsoid model of the spheroid planet on where the target is located.

    Returns
    -------
    float
        Elevation (EL) angle in antenna frame in radians
    float
        Azimuth (AZ) angle in antenna frame in radians

    Raises
    ------
    RuntimeError
        For zero slant range from spacecraft to target

    """
    # convert target LLH to XYZ of ECEF
    tg_ecef = ellips.lon_lat_to_xyz(tg_llh)
    # form verctor from S/C to target
    vec_sc2tg = tg_ecef - np.asarray(pos_sc_ecef)
    sr = np.linalg.norm(vec_sc2tg)
    if np.isclose(sr, 0):
        raise RuntimeError('Near-zero slant range is encountered!')
    # pointing unit vector in ECEF
    pnt_ecef = vec_sc2tg / sr
    # pointing vector in antenna XYZ
    pnt_ant = quat_ant2ecef.conjugate().rotate(pnt_ecef)
    # XYZ to EL-AZ in (rad)
    el_ant, az_ant = ant_frame.cart2sph(pnt_ant)
    return el_ant, az_ant


def rdr2ant(az_time, slant_range, orbit, attitude, side, wavelength,
            doppler=0.0, dem=DEMInterpolator(), ellipsoid=Ellipsoid(),
            ant_frame=Frame(), **kwargs):
    """Transformation from radar grid geometry to antenna frame.

    Parameters
    ----------
    az_time : float
        Azimuth time of doppler intersection in (sec) wrt
        to reference epoch of orbit/attitude
    slant_range : float
        slant range in (m).
    orbit : isce.core.Orbit
    attitude : isce.core.Attitude
    side :  str or isce3.core.LookSide
        Radar antenna look direction.
    wavelength : float
        Radar wavelength in (m)
    doppler : float, default=0.0
        Doppler that defines radar grid geometry in (Hz)
    dem : isce3.geometry.DEMInterpolator, default=0.0
        It covers DEM heights w.r.t. ellipsoid.
    ellipsoid : isce3.core.Ellipsoid, default='WGS84'
        Ellipsoid model of the spheroid planet on where the target is located.
    ant_frame : isce3.antenna.Frame, default='EL-AND-AZ'
        This object defines antenna spherical coordinate type.
    **kwargs : dict, optional
        Extra arguments to `rdr2geo` as follows
        threshold : float, optional
            Range convergence threshold in (m).
        maxiter : int, optional
            Maximum iterations.
        extraiter : int, optional
            Additional iterations.

    Returns
    -------
    float
        Elevation (EL) angle in antenna frame in radians
    float
        Azimuth (AZ) angle in antenna frame in radians

    """
    llh = rdr2geo(az_time, slant_range, orbit, side, doppler=doppler,
                  wavelength=wavelength, dem=dem, ellipsoid=ellipsoid,
                  **kwargs)

    pos_sc_ecef, _ = orbit.interpolate(az_time)
    quat_ant2ecef = attitude.interpolate(az_time)

    return geo2ant(llh, pos_sc_ecef, quat_ant2ecef, ant_frame=ant_frame,
                   ellips=ellipsoid)
