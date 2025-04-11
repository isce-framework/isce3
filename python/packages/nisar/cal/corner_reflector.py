from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path

import numpy as np

import isce3
from isce3.cal import TriangularTrihedralCornerReflector


class CRValidity(IntFlag):
    r"""
    The validity of a surveyed corner reflector (CR) for particular applications.

    Flags can be combined using bitwise OR (`|`) operations if a corner reflector is
    suitable for multiple applications.

    The integer validity codes are defined by the NISAR Corner Reflector Software
    Interface Specification (SIS) document\ [1]_.

    References
    ----------
    .. [1] B. Hawkins, "Corner Reflector Software Interface Specification," JPL
       D-107698 (2023).
    """

    INVALID = 0
    """Not valid for any usage (out of service)."""
    IPR = 1
    """Usable for assessing shape of impulse response (ISLR, PSLR, resolution)."""
    RAD_POL = 2
    """Usable for radiometric and polarimetric calibration."""
    GEOM = 4
    """Usable for geometric calibration."""


@dataclass(frozen=True)
class CornerReflector(TriangularTrihedralCornerReflector):
    """
    A triangular trihedral corner reflector (CR) used for NISAR calibration.

    Extends `isce3.cal.TriangularTrihedralCornerReflector` with additional information
    required for NISAR Science Calibration and Validation (Cal/Val) activities.

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
        Equivalently, this is the heading of the base of the CR w.r.t North in
        the clockwise direction.
    side_length : float
        The length of each leg of the trihedral, in meters.
    survey_date : isce3.core.DateTime
        UTC date and time when the corner reflector survey was conducted.
    validity : CRValidity
        Corner reflector validity for NISAR Science Cal/Val applications.
    velocity : (3,) array_like
        Corner reflector velocity, in meters per second (m/s), due to tectonic plate
        motion. Velocity should be provided in a local East-North-Up (ENU) coordinate
        system with respect to the WGS 84 reference ellipsoid with its origin at the CR
        location.

    See Also
    --------
    isce3.cal.TriangularTrihedralCornerReflector
    parse_corner_reflector_csv
    """

    survey_date: isce3.core.DateTime
    validity: CRValidity
    velocity: np.ndarray

    def __post_init__(self) -> None:
        # Check that `validity` was in the range of valid values.
        # TODO(Python>=3.11) Use `boundary=STRICT` instead (see
        # https://docs.python.org/3/library/enum.html#enum.FlagBoundary)
        if (self.validity < 0) or (self.validity > sum(CRValidity)):
            raise ValueError(f"validity flag has invalid value {self.validity}")

        # Normalize velocity to an array of doubles and check that it's a 3-vector.
        velocity = np.asarray(self.velocity, dtype=np.float64)
        if velocity.shape != (3,):
            raise ValueError(
                "expected velocity vector with shape (3,), instead got shape"
                f" {velocity.shape}"
            )

        # XXX Workaround for `frozen=True`.
        object.__setattr__(self, "velocity", velocity)


def parse_corner_reflector_csv(csvfile: str | os.PathLike) -> Iterator[CornerReflector]:
    r"""
    Parse a CSV file containing NISAR corner reflector (CR) survey data.

    Returns an iterator over corner reflectors in the CSV file.

    The CSV format is defined by the NISAR Corner Reflector Software Interface
    Specification (SIS) document\ [1]_. It contains the following fields:

    1. Corner reflector ID
    2. Latitude (deg)
    3. Longitude (deg)
    4. Height above ellipsoid (m)
    5. Azimuth (deg)
    6. Tilt / Elevation angle (deg)
    7. Side length (m)
    8. Survey Date
    9. Validity
    10. Velocity East (m/s)
    11. Velocity North (m/s)
    12. Velocity Up (m/s)

    Parameters
    ----------
    csvfile : path-like
        The CSV file path.

    Yields
    ------
    cr : CornerReflector
        Corner reflector data corresponding to a single entry in the CSV file.

    Notes
    -----
    This function outputs the full survey history for each corner reflector found in the
    CSV file. It does not filter out any invalid corner reflectors or outdated survey
    data.

    See Also
    --------
    CornerReflector
    get_latest_cr_data_before_epoch
    get_valid_crs

    References
    ----------
    .. [1] B. Hawkins, "Corner Reflector Software Interface Specification," JPL
       D-107698 (2023).
    """
    csvfile = Path(csvfile)

    # Corner reflector CSV must be an existing file.
    if not csvfile.is_file():
        raise FileNotFoundError(f"corner reflector CSV file not found: {csvfile}")

    dtype = np.dtype(
        [
            ("id", np.object_),
            ("lat", np.float64),
            ("lon", np.float64),
            ("height", np.float64),
            ("az", np.float64),
            ("el", np.float64),
            ("side_length", np.float64),
            ("survey_date", np.object_),
            ("validity", np.int_),
            ("vel_e", np.float64),
            ("vel_n", np.float64),
            ("vel_u", np.float64),
        ]
    )

    # Parse CSV data.
    # Treat the header row ("Corner reflector ID, ...") as a comment so that it will be
    # ignored if present.
    try:
        data = np.loadtxt(
            csvfile,
            dtype=dtype,
            delimiter=",",
            ndmin=1,
            comments=["#", "Corner reflector ID,", '"Corner reflector ID",'],
        )
    except ValueError as e:
        errmsg = f"error parsing NISAR corner reflector CSV file {csvfile}"
        raise RuntimeError(errmsg) from e

    # Convert lat, lon, az, & el angles to radians.
    for attr in ["lat", "lon", "az", "el"]:
        data[attr] = np.deg2rad(data[attr])

    for d in data:
        llh = isce3.core.LLH(longitude=d[2], latitude=d[1], height=d[3])
        # Careful to strip strings in case there are spaces between columns.
        # No problem with numeric fields since int() and float() already strip.
        corner_id = d[0].strip()
        survey_date = isce3.core.DateTime(d[7].strip())
        validity = CRValidity(int(d[8]))
        velocity = np.asarray([d[9], d[10], d[11]], dtype=np.float64)

        yield CornerReflector(
            id=corner_id,
            llh=llh,
            elevation=d[5],
            azimuth=d[4],
            side_length=d[6],
            survey_date=survey_date,
            validity=validity,
            velocity=velocity,
        )


def get_latest_cr_data_before_epoch(
    crs: Iterable[CornerReflector], epoch: isce3.core.DateTime
) -> Iterator[CornerReflector]:
    """
    Filter corner reflector data based on survey date.

    For each unique corner reflector ID, get the corner reflector data from the most
    recent survey date not later than the specified epoch.

    Parameters
    ----------
    crs : iterable of CornerReflector
        The survey history of one or more corner reflectors. May contain data from
        multiple survey dates for each unique corner reflector.
    epoch : isce3.core.DateTime
        The date and time of the radar observation. Data from corner reflector surveys
        after this epoch are ignored.

    Yields
    ------
    cr : CornerReflector
        Corner reflector survey data from the closest survey date before the specified
        epoch for a given corner reflector ID.

    See Also
    --------
    CornerReflector
    parse_corner_reflector_csv
    get_valid_crs
    """
    # Create a dict whose keys are unique corner reflector IDs and values are the
    # corresponding `CornerReflector` objects with the most recent survey date among
    # input corner reflectors with the same ID (excluding survey dates that came after
    # `epoch`).
    latest_crs_by_id: dict[str, CornerReflector] = {}

    for cr in crs:
        # Ignore this corner reflector data if its survey date was after `epoch`.
        if cr.survey_date > epoch:
            continue

        try:
            # Check if the dict contains an existing corner reflector with the same ID.
            other_cr = latest_crs_by_id[cr.id]
        except KeyError:
            # No corner reflector with the same ID was found -- insert this corner
            # reflector.
            latest_crs_by_id[cr.id] = cr
        else:
            # A corner reflector with the same ID was found -- replace it in the dict if
            # the new corner reflector had a more recent survey date.
            if cr.survey_date > other_cr.survey_date:
                latest_crs_by_id[cr.id] = cr

    # Return an iterator over values in the dict.
    yield from latest_crs_by_id.values()


def get_valid_crs(
    crs: Iterable[CornerReflector], flags: CRValidity | None = None
) -> Iterator[CornerReflector]:
    """
    Filter out invalid corner reflector data.

    Corner reflectors may rotate into and out of service. This may be due to either
    planned maintenance or unplanned disruptions (e.g., blown over due to strong winds).
    Furthermore, a corner reflector may be suitable for some applications but not
    others. For example, a corner whose location was not precisely surveyed would be
    inappropriate for use in geometric calibration but could still be used for
    radiometric calibration.

    This function may be used to filter out out-of-service corner reflectors or corner
    reflectors that are not valid for a particular application (or applications). It
    returns an iterator over valid corner reflectors from the input iterable. The
    relative order of the corner reflectors is preserved.

    Parameters
    ----------
    crs : iterable of CornerReflector
        Input iterable of corner reflector data.
    flags : CRValidity or None, optional
        Validity flag(s) to check for. If None, only corner reflectors that are out of
        service (i.e. with validity code == 0) are filtered out. Otherwise, corner
        reflectors that do not have any of the specified validity flags set will be
        filtered out. For example, using ``flags=CRValidity.RAD_POL | CRValidity.GEOM``
        would yield only those corner reflectors that were valid for either
        radiometric/polarimetric calibration or geometric calibration activities.
        Defaults to None.

    Yields
    ------
    cr : CornerReflector
        A corner reflector from the input iterable with one or more validity flags set.

    See Also
    --------
    CornerReflector
    parse_corner_reflector_csv
    get_latest_cr_data_before_epoch
    """
    # If `flags` was not None, the predicate checks if any of the specified flags were
    # set. Otherwise, it checks if any bit flags at all were set.
    if flags is not None:
        pred = lambda cr: cr.validity & flags != CRValidity.INVALID
    else:
        pred = lambda cr: cr.validity != CRValidity.INVALID

    # Get only those corner reflectors for which `pred` was true.
    return filter(pred, crs)


def parse_and_filter_corner_reflector_csv(
    csvfile: str | os.PathLike,
    observation_date: isce3.core.DateTime,
    validity_flags: CRValidity | None = None,
) -> Iterator[CornerReflector]:
    r"""
    Parse the input corner reflector CSV file and filter the CR data based on survey
    date and validity flags.

    Returns an iterator over corner reflectors within the file, excluding outdated
    survey data (i.e. data for which there was a more recent survey of the same corner
    reflector prior to the radar observation) and corner reflectors that were not valid
    for particular calibration/validation activities. The relative order of corner
    reflectors is preserved.

    Parameters
    ----------
    csvfile : path-like
        The path to a CSV file containing the survey history of zero or more corner
        reflectors. The file format is assumed to conform to the specification described
        by the NISAR Corner Reflector Software Interface Specification (SIS) document\
        [1]_.
    observation_date : isce3.core.DateTime
        The date (and time) of the radar observation. Data from corner reflector surveys
        after this epoch are ignored.
    validity_flags : CRValidity or None, optional
        Validity flag(s) to check for. If None, only corner reflectors that are out of
        service (i.e. with validity code == 0) are filtered out. Otherwise, corner
        reflectors that do not have any of the specified validity flags set will be
        filtered out. For example, using ``flags=CRValidity.RAD_POL | CRValidity.GEOM``
        would yield only those corner reflectors that were valid for either
        radiometric/polarimetric calibration or geometric calibration activities.
        Defaults to None.

    Yields
    ------
    cr : nisar.cal.CornerReflector
        The most recent survey data before the radar observation for a given corner
        reflector that was valid for radiometric calibration.

    References
    ----------
    .. [1] B. Hawkins, "Corner Reflector Software Interface Specification," JPL
       D-107698 (2023).
    """
    # Get all corner reflector (CR) data from the input CSV file.
    all_crs = parse_corner_reflector_csv(csvfile)

    # The NISAR CSV spec allows for the full survey history of each corner reflector
    # to be stored in the same CSV file. Filter out any outdated corner reflector
    # survey data as well as survey data collected after the radar observation.
    latest_crs = get_latest_cr_data_before_epoch(all_crs, epoch=observation_date)

    # The CSV may also contain corner reflectors that are not usable for the specific
    # Cal/Val application(s) defined by `flags`. These CRs should be filtered out. It's
    # important that this step occurs after the previous filtering step in order to
    # correctly handle cases where a corner reflector was previously in-service, but the
    # most recent survey reports that it is out-of-service.
    valid_crs = get_valid_crs(latest_crs, flags=validity_flags)

    return valid_crs


def filter_crs_per_az_heading(crs, az_heading, az_atol=np.deg2rad(20.0)):
    """
    Filter corner reflectors per desired azimuth (AZ) orientation within
    a desired absolute tolerance.

    Parameters
    ----------
    crs : iterable of type CornerReflector or
        TriangularTrihedralCornerReflector.
    az_heading : float
        Desired AZ/heading angle in radians w.r.t. geographic North.
    az_atol : float, default=20.0 degrees
        Absolute tolerance in radians when comapring AZ of CRs with
        `az_heading`. The default is around 0.5 * HBPW of an ideal
        triangular trihedral CR (HPBW ~ 40 deg).

    Yields
    ------
    cr : CornerReflector or TriangularTrihedralCornerReflector
        The datatype of cr depends on type of items in `crs`.

    """
    for cr in crs:
        if abs(_wrap_phase(cr.azimuth - az_heading)) <= az_atol:
            yield cr


def _wrap_phase(phs, deg=False):
    """
    Wrap phase(s) within [-pi, pi] (default) or [-180, 180] (deg=True).
    Input and output phases have the same unit determined by `deg`.

    Parameters
    ----------
    phs : array_like float
        phase(s) whose unit set by `deg`.
    deg : bool, default=False
        This will determine the units for input/output phases.

    Returns
    -------
    float or numpy.ndarray of float
        Phase(s) whose value(s) are wrapped within [-pi, pi] or [-180, 180]
        depending on value of `deg`.

    """
    if deg:
        phs = np.deg2rad(phs)
    return np.angle(np.exp(np.multiply(1j, phs)), deg=deg)
