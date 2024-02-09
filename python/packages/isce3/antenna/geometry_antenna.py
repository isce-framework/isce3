"""
Some antenna-related geometry functions
"""
import numpy as np

from isce3.antenna import Frame
from isce3.core import Ellipsoid, make_projection
from isce3.geometry import DEMInterpolator, rdr2geo
from scipy.optimize import root_scalar
from warnings import warn


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


def sphere_range_az_to_xyz(slant_range, az, pos_ecef, quat, a):
    """
    Compute target position on sphere given range and AZ angle (EL_AND_AZ frame)
    using Newton iteration.  See [1]_ for definition of AZ angle.

    Parameters
    ----------
    slant_range : float
        Range to target in m
    az : float
        AZ angle in radians
    pos_ecef : array_like
        ECEF XYZ position of radar in m
    quat : isce3.core.Quaternion
        Orientation of the antenna (RCS to ECEF quaternion)
    a : float
        Radius of spherical planet in m.  Center is at (0, 0, 0).

    Returns
    -------
    xyz : array_like
        Target position in ECEF in m

    References
    ----------
    .. [1] Shen, "NISAR Project Coordinate Systems Definition", JPL D-80882
    """
    # Use law of cosines to solve the intersection of the range sphere and
    # planet spheres.  This is a circle, and LOS vectors to that circle all
    # make a constant angle (look angle) wrt the nadir vector.
    center_distance = np.linalg.norm(pos_ecef)
    coslook = ((slant_range**2 + center_distance**2 - a**2) /
        (2 * slant_range * center_distance))

    rcs2xyz = quat.to_rotation_matrix()
    xyz2rcs = rcs2xyz.transpose()
    nadir_rcs = xyz2rcs @ (-pos_ecef / center_distance)
    sinc = lambda x: np.sinc(x / np.pi)

    if abs(1 - abs(nadir_rcs[2])) < np.finfo(float).eps:
        raise ValueError("radar is looking straight up or down")

    # Look angle and EL angle are typically quite similar (just a constant
    # offset when AZ=0), but the exact relationship is nonlinear for the general
    # (AZ != 0) case.  Since we have fixed AZ, we vary EL until
    #   los.dot(nadir) == cos(look)
    # The equation is continuously differentiable, so Newton's method is
    # appropriate.

    # Equation lookvec.dot(nadir) - cos(lookangle) == 0
    # and its derivative.  For derivations of formulas see notes at
    # https://github.jpl.nasa.gov/bhawkins/nisar-notebooks/blob/master/EL_AND_AZ.ipynb
    def f_df(el):
        t = np.sqrt(az**2 + el**2)
        sinct = sinc(t)
        cost = np.cos(t)
        # The look vector here is parameterized a little differently but gives
        # the same result as Frame(EL_AND_AZ)::sphToCart
        f = np.array([el * sinct, az * sinct, cost]).dot(nadir_rcs) - coslook
        if t > np.sqrt(np.finfo(t).eps):
            # Derivative of the look vector wrt EL
            dlook = np.array([
                (el / t)**2 * (cost - sinct) + sinct,
                az * el / t**2 * (cost - sinct),
                -el * sinct])
        else:
            # Taylor expansion to avoid dividing by small t.
            dlook = np.array([1 - el**2 / 3, -az * el / 3, -el])
        # Derivative wrt EL of (look.dot(nadir) - cos(theta)).
        # Only the look vector depends on EL.
        df = dlook.dot(nadir_rcs)
        return f, df

    # There are two solutions (left/right ambiguity), but using x0=0 should
    # give us the one closest to the boresight.
    sol = root_scalar(f_df, x0=0.0, fprime=True, method="newton")
    if not sol.converged:
        raise RuntimeError("newton iterations for EL did not converge")

    look_rcs = Frame().sph2cart(sol.root, az)
    look_xyz = rcs2xyz @ look_rcs
    return pos_ecef + slant_range * look_xyz


def get_approx_el_bounds(slant_range, az, pos_ecef, quat, dem=DEMInterpolator(),
                         f=1.1):
    """
    Estimate elevation (EL in an EL_AND_AZ frame) bounds based on
    ellipsoid and height extrema.

    Parameters
    ----------
    slant_range : float
        Range to target in m
    az : float
        AZ angle in radians
    pos_ecef : array_like
        ECEF XYZ position of radar in m
    quat : isce3.core.Quaternion
        Orientation of the antenna (RCS to ECEF quaternion)
    dem : isce3.geometry.DEMInterpolator, optional
        Digital elevation model, heights in m above ellipsoid.
        Defaults to WGS84.  Raises exception if dem.have_stats is False.
    f : float
        Dilation factor to use to expand the estimated EL interval.

    Returns
    -------
    el_min, el_max : float
        Approximate EL bounds on the solution, radians
    """
    ellipsoid = make_projection(dem.epsg_code).ellipsoid

    # Bounding radii of ellipsoid
    amin, amax = sorted([ellipsoid.a, ellipsoid.b])

    # TODO If we have DEM data then we can tighten this up based on its
    # latitude extent.  This is a little complicated because we care about the
    # absolute lat extrema (not signed) and also because the DEM may not be in a
    # lat/lon projection.  Alternatively, we could compute a radius at the scene
    # center and just rely on the fudge factor to sufficiently widen the EL
    # bounds.

    # Incorporate terrain extrema.
    if not dem.have_stats:
        raise ValueError("Provided DEMInterpolator does not have stats")
    amin += dem.min_height
    amax += dem.max_height

    # Warn if range is small relative to Earth surface bounds.  For example,
    # the 20 km difference between equatorial and polar radii would be a
    # problem for an airborne radar flying at 10 km altitude.
    if slant_range / (amax - amin) < 2:
        warn(f"Range ({slant_range} m) is small relative to "
            f"surface variation ({amax - amin} m).")

    # Calculate solution on sphere.
    xyz1 = sphere_range_az_to_xyz(slant_range, az, pos_ecef, quat, amin)
    xyz2 = sphere_range_az_to_xyz(slant_range, az, pos_ecef, quat, amax)

    # Convert XYZ to EL.
    frame = Frame()
    xyz2rcs = quat.to_rotation_matrix().transpose()

    el1, _ = frame.cart2sph(xyz2rcs @ (xyz1 - pos_ecef))
    el2, _ = frame.cart2sph(xyz2rcs @ (xyz2 - pos_ecef))
    el1, el2 = sorted([el1, el2])

    # Widen the bounds a bit.
    elmid = (el1 + el2) / 2
    elstep = f * (el2 - el1) / 2
    return elmid - elstep, elmid + elstep
