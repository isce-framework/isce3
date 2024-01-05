from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Optional

import numpy as np
import shapely
from numpy.typing import ArrayLike

import isce3


@dataclass(frozen=True)
class TriangularTrihedralCornerReflector:
    """
    A triangular trihedral corner reflector (CR).

    Parameters
    ----------
    id : str
        The unique identifier of the corner reflector.
    llh : isce3.core.LLH
        The geodetic coordinates of the corner reflector: the geodetic longitude and
        latitude in radians and the height above the WGS 84 ellipsoid in meters.
    elevation : float
        The elevation angle, in radians. This is the tilt angle of the vertical axis of
        the corner reflector with respect to the ellipsoid normal vector.
    azimuth : float
        The azimuth angle, in radians. This is the heading angle of the corner
        reflector, or the angle that the corner reflector boresight makes w.r.t
        geographic East, measured clockwise positive in the E-N plane.
    side_length : float
        The length of each leg of the trihedral, in meters.
    """

    id: str
    llh: isce3.core.LLH
    elevation: float
    azimuth: float
    side_length: float


def parse_triangular_trihedral_cr_csv(
    csvfile: os.PathLike,
) -> Iterator[TriangularTrihedralCornerReflector]:
    """
    Parse a CSV file containing triangular trihedral corner reflector (CR) data.

    Returns an iterator over corner reflectors within the file.

    The CSV file is assumed to be in the format used by the `UAVSAR Rosamond Corner
    Reflector Array (RCRA) <https://uavsar.jpl.nasa.gov/cgi-bin/calibration.pl>`_, which
    contains the following columns:

    1. Corner reflector ID
    2. Latitude (deg)
    3. Longitude (deg)
    4. Height above ellipsoid (m)
    5. Azimuth (deg)
    6. Tilt / Elevation angle (deg)
    7. Side length (m)

    Parameters
    ----------
    csvfile : path_like
        The CSV file path.

    Yields
    ------
    cr : TriangularTrihedralCornerReflector
        A corner reflector.
    """
    dtype = np.dtype(
        [
            ("id", np.object_),
            ("lat", np.float64),
            ("lon", np.float64),
            ("height", np.float64),
            ("az", np.float64),
            ("el", np.float64),
            ("side_length", np.float64),
        ]
    )

    # Parse CSV data.
    # Treat the header row ("Corner reflector ID, ...") as a comment so that it will be
    # ignored if present.
    # Any additional columns beyond those mentioned above will be ignored so that new
    # additions to the file spec don't break compatibility.
    cols = range(len(dtype))
    crs = np.loadtxt(
        csvfile,
        dtype=dtype,
        delimiter=",",
        usecols=cols,
        ndmin=1,
        comments=["#", "Corner reflector ID,", '"Corner reflector ID",'],
    )

    # Convert lat, lon, az, & el angles to radians.
    for attr in ["lat", "lon", "az", "el"]:
        crs[attr] = np.deg2rad(crs[attr])

    for cr in crs:
        id, lat, lon, height, az, el, side_length = cr
        llh = isce3.core.LLH(lon, lat, height)
        yield TriangularTrihedralCornerReflector(id, llh, el, az, side_length)


def cr_to_enu_rotation(el: float, az: float) -> isce3.core.Quaternion:
    """
    Get a quaternion to rotate from a corner reflector (CR) intrinsic coordinate system
    to East, North, Up (ENU) coordinates.

    The CR coordinate system has three orthogonal axes aligned with the three legs of
    the trihedral. The coordinate system is defined such that, when the elevation and
    azimuth angles are each 0 degrees:

    * the x-axis points 45 degrees South of the East-axis
    * the y-axis points 45 degrees North of the East-axis
    * the z-axis is aligned with the Up-axis

    Parameters
    ----------
    el : float
        The elevation angle, in radians. This is the tilt angle of the vertical axis of
        the corner reflector with respect to the ellipsoid normal vector.
    az : float
        The azimuth angle, in radians. This is the heading angle of the corner
        reflector, or the angle that the corner reflector boresight makes w.r.t
        geographic East, measured clockwise positive in the E-N plane.

    Returns
    -------
    q : isce3.core.Quaternion
        A unit quaternion representing the rotation from CR to ENU coordinates.
    """
    q1 = isce3.core.Quaternion(angle=el, axis=[1.0, -1.0, 0.0])
    q2 = isce3.core.Quaternion(angle=0.25 * np.pi + az, axis=[0.0, 0.0, -1.0])
    return q2 * q1


def enu_to_cr_rotation(el: float, az: float) -> isce3.core.Quaternion:
    """
    Get a quaternion to rotate from East, North, Up (ENU) coordinates to a corner
    reflector (CR) intrinsic coordinate system.

    The CR coordinate system has three orthogonal axes aligned with the three legs of
    the trihedral. The coordinate system is defined such that, when the elevation and
    azimuth angles are each 0 degrees:

    * the x-axis points 45 degrees South of the East-axis
    * the y-axis points 45 degrees North of the East-axis
    * the z-axis is aligned with the Up-axis

    Parameters
    ----------
    el : float
        The elevation angle, in radians. This is the tilt angle of the vertical axis of
        the corner reflector with respect to the ellipsoid normal vector.
    az : float
        The azimuth angle, in radians. This is the heading angle of the corner
        reflector, or the angle that the corner reflector boresight makes w.r.t
        geographic East, measured clockwise positive in the E-N plane.

    Returns
    -------
    q : isce3.core.Quaternion
        A unit quaternion representing the rotation from ENU to CR coordinates.
    """
    return cr_to_enu_rotation(el, az).conjugate()


def normalize_vector(v: ArrayLike) -> np.ndarray:
    """
    Normalize a vector.

    Compute the unit vector pointing in the direction of the input vector.

    Parameters
    ----------
    v : array_like
        The input vector. Must be nonzero.

    Returns
    -------
    u : numpy.ndarray
        The normalized vector.
    """
    return np.asanyarray(v) / np.linalg.norm(v)


def target2platform_unit_vector(
    target_llh: isce3.core.LLH,
    orbit: isce3.core.Orbit,
    doppler: isce3.core.LUT2d,
    wavelength: float,
    look_side: str,
    ellipsoid: isce3.core.Ellipsoid = isce3.core.WGS84_ELLIPSOID,
    *,
    geo2rdr_params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """
    Compute the target-to-platform line-of-sight (LOS) unit vector.

    Parameters
    ----------
    target_llh : isce3.core.LLH
        The target position expressed as longitude, latitude, and height above the
        reference ellipsoid in radians, radians, and meters respectively.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center.
    doppler : isce3.core.LUT2d
        The Doppler centroid of the data, in hertz, expressed as a function of azimuth
        time, in seconds relative to the orbit epoch, and slant range, in meters. Note
        that this should be the native Doppler of the data acquisition, which may in
        general be different than the Doppler associated with the radar grid that the
        focused image was projected onto.
    wavelength : float
        The radar wavelength, in meters.
    look_side : {"Left", "Right"}
        The radar look direction.
    ellipsoid : isce3.core.Ellipsoid, optional
        The geodetic reference ellipsoid, with dimensions in meters. Defaults to the
        WGS 84 ellipsoid.
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
    u : (3,) numpy.ndarray
        A unit vector pointing from the target position to the platform position in
        Earth-Centered, Earth-Fixed (ECEF) coordinates.
    """
    # Convert LLH object to an array containing [lon, lat, height].
    target_llh = target_llh.to_vec3()

    # Get target (x,y,z) position in ECEF coordinates.
    target_xyz = ellipsoid.lon_lat_to_xyz(target_llh)

    if geo2rdr_params is None:
        geo2rdr_params = {}

    # Run geo2rdr to get the target azimuth time coordinate in seconds since the orbit
    # epoch.
    aztime, _ = isce3.geometry.geo2rdr_bracket(
        xyz=target_xyz,
        orbit=orbit,
        doppler=doppler,
        wavelength=wavelength,
        side=look_side,
        **geo2rdr_params,
    )

    # Get platform (x,y,z) position in ECEF coordinates.
    platform_xyz, _ = orbit.interpolate(aztime)

    return normalize_vector(platform_xyz - target_xyz)


def predict_triangular_trihedral_cr_rcs(
    cr: TriangularTrihedralCornerReflector,
    orbit: isce3.core.Orbit,
    doppler: isce3.core.LUT2d,
    wavelength: float,
    look_side: str,
    *,
    geo2rdr_params: Optional[Mapping[str, float]] = None,
) -> float:
    r"""
    Predict the radar cross-section (RCS) of a triangular trihedral corner reflector.

    Calculate the predicted monostatic RCS of a triangular trihedral corner reflector,
    given the corner reflector dimensions and imaging geometry\ [1]_.

    Parameters
    ----------
    cr : TriangularTrihedralCornerReflector
        The corner reflector position, orientation, and size.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center.
    doppler : isce3.core.LUT2d
        The Doppler centroid of the data, expressed as a function of azimuth time, in
        seconds relative to the orbit epoch, and slant range, in meters.
    wavelength : float
        The radar wavelength, in meters.
    look_side : {"Left", "Right"}
        The radar look direction.
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
        The predicted radar cross-section of the corner reflector, in meters squared
        (linear scale -- not dB).

    References
    ----------
    .. [1] R. R. Bonkowski, C. R. Lubitz, and C. E. Schensted, “Studies in Radar
       Cross-Sections - VI. Cross-sections of corner reflectors and other multiple
       scatterers at microwave frequencies,” University of Michigan Radiation
       Laboratory, Tech. Rep., October 1953.
    """
    # Get the target-to-platform line-of-sight vector in ECEF coordinates.
    los_vec_ecef = target2platform_unit_vector(
        target_llh=cr.llh,
        orbit=orbit,
        doppler=doppler,
        wavelength=wavelength,
        look_side=look_side,
        ellipsoid=isce3.core.WGS84_ELLIPSOID,
        geo2rdr_params=geo2rdr_params,
    )

    # Convert to ENU coordinates and then to CR-intrinsic coordinates.
    los_vec_enu = isce3.geometry.enu_vector(
        cr.llh.longitude, cr.llh.latitude, los_vec_ecef
    )
    los_vec_cr = enu_to_cr_rotation(cr.elevation, cr.azimuth).rotate(los_vec_enu)

    # Get the CR boresight unit vector in the same coordinates.
    boresight_vec = normalize_vector([1.0, 1.0, 1.0])

    # Get the direction cosines between the two vectors, sorted in ascending order.
    p1, p2, p3 = np.sort(los_vec_cr * boresight_vec)

    # Compute expected RCS.
    a = p1 + p2 + p3
    if (p1 + p2) > p3:
        b = np.sqrt(3.0) * a - 2.0 / (np.sqrt(3.0) * a)
    else:
        b = 4.0 * p1 * p2 / a

    return 4.0 * np.pi * cr.side_length ** 4 * b ** 2 / wavelength ** 2


def get_target_observation_time_and_elevation(
    target_llh: isce3.core.LLH,
    orbit: isce3.core.Orbit,
    attitude: isce3.core.Attitude,
    wavelength: float,
    look_side: str,
    frame: isce3.antenna.Frame = isce3.antenna.Frame("EL_AND_AZ"),
    ellipsoid: isce3.core.Ellipsoid = isce3.core.WGS84_ELLIPSOID,
    *,
    geo2rdr_params: Optional[Mapping[str, float]] = None,
) -> tuple[isce3.core.DateTime, float]:
    """
    Get zero-Doppler observation time and antenna elevation angle of a geodetic target.

    Parameters
    ----------
    target_llh : isce3.core.LLH
        The target position expressed as geodetic longitude, latitude, and height above
        the reference ellipsoid in radians, radians, and meters respectively.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center.
    wavelength : float
        The radar wavelength, in meters.
    look_side : {"Left", "Right"}
        The radar look direction.
    frame : isce3.antenna.Frame, optional
        Antenna frame which defines the type of spherical coordinate. Defaults to an
        'EL_AND_AZ' frame.
    ellipsoid : isce3.core.Ellipsoid, optional
        The geodetic reference ellipsoid, with dimensions in meters. Defaults to the
        WGS 84 ellipsoid.
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
    az_datetime : isce3.core.DateTime
        The target's zero-Doppler observation time (the time of the platform's closest
        approach to the target) as a UTC datetime.
    el_angle : float
        The elevation angle of the target, in radians. Elevation is measured in the
        cross-track direction w.r.t antenna boresight, increasing toward far-range.
    """
    # Convert LLH object to an array containing [lon, lat, height].
    target_llh = target_llh.to_vec3()

    # Get target (x,y,z) position in ECEF coordinates.
    target_xyz = ellipsoid.lon_lat_to_xyz(target_llh)

    if geo2rdr_params is None:
        geo2rdr_params = {}

    zero_doppler = isce3.core.LUT2d()

    # Run geo2rdr to get the target azimuth time coordinate, in seconds since the orbit
    # epoch.
    aztime, _ = isce3.geometry.geo2rdr_bracket(
        xyz=target_xyz,
        orbit=orbit,
        doppler=zero_doppler,
        wavelength=wavelength,
        side=look_side,
        **geo2rdr_params,
    )

    # Convert `aztime` to a UTC timepoint.
    az_datetime = orbit.reference_epoch + isce3.core.TimeDelta(aztime)

    # Interpolate orbit & attitude to get platform position in ECEF coordinates and
    # reflector coordinate system (RCS) to ECEF quaternion.
    platform_ecef, _ = orbit.interpolate(aztime)
    q_rcs2ecef = attitude.interpolate(aztime)

    # Get antenna elevation angle.
    el_angle, _ = isce3.antenna.geo2ant(
        tg_llh=target_llh,
        pos_sc_ecef=platform_ecef,
        quat_ant2ecef=q_rcs2ecef,
        ant_frame=frame,
        ellips=ellipsoid,
    )

    return az_datetime, el_angle


def get_crs_in_polygon(
    crs: Iterable[TriangularTrihedralCornerReflector],
    polygon: shapely.Polygon,
    buffer: float | None = None,
) -> Iterator[TriangularTrihedralCornerReflector]:
    """
    Filter out corner reflectors located outside of a Lon/Lat polygon.

    For each input corner reflector, check whether it is contained within `polygon`. An
    optional buffer may be added to the polygon in order accept corner reflectors
    slightly outside its extents.

    Returns an iterator over corner reflectors found within the polygon. The relative
    order of the corner reflectors is preserved.

    Parameters
    ----------
    crs : iterable of TriangularTrihedralCornerReflector
        Input iterable of corner reflector data.
    polygon : shapely.Polygon
        A convex polygon, in geodetic Lon/Lat coordinates w.r.t the WGS 84 ellipsoid,
        enclosing the area of interest. Longitude (x) coordinates should be specified in
        degrees in the range [-180, 180]. Latitude (y) coordinates should be specified
        in degrees in the range [-90, 90].
    buffer : float or None, optional
        An optional additional margin that extends the region of interest. Corner
        reflectors located within the buffer region are considered to be contained
        within the polygon. The units of `buffer` should be the same as `polygon`. Must
        be >= 0. If None, no buffer is applied. Defaults to None.

    Yields
    ------
    cr : TriangularTrihedralCornerReflector
        A corner reflector from the input iterable that was contained within the
        polygon.

    Notes
    -----
    This function uses a point-in-polygon algorithm that assumes a Euclidean geometry.
    The results may therefore be inaccurate due to the curvature of the reference
    surface if the polygon points are spaced far apart, or if the points lie around a
    discontinuity in the coordinate space (such as the antimeridian or poles).
    """
    # If additional buffer was requested, dilate the polygon by the specified margin,
    # which must be nonnegative.
    if buffer is not None:
        if buffer < 0:
            raise ValueError(f"buffer must be >= 0 (or None), got {buffer}")
        polygon = polygon.buffer(buffer)

    # Wraps the input angle (in radians) to the interval [-pi, pi).
    def wrap(phase: float) -> float:
        return (phase + np.pi) % (2.0 * np.pi) - np.pi

    # Filter out corner reflectors not contained within the specified polygon.
    for cr in crs:
        # Get corner reflector lon/lat coordinates in radians, wrap the longitude
        # coordinate to the expected interval, and convert to degrees.
        lon, lat, _ = cr.llh.to_vec3()
        lon_lat_deg = np.rad2deg([wrap(lon), lat])

        # Check whether the corner reflector is in the polygon.
        point = shapely.Point(lon_lat_deg)
        if polygon.contains(point):
            yield cr
