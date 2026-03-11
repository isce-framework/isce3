from __future__ import annotations

import os
from datetime import datetime

from nisar.products.readers.attitude import load_attitude_from_xml
from nisar.products.readers.orbit import load_orbit_from_xml

import isce3

from .logging import get_logger
from .util import truncate_datetime_to_integer_seconds


def get_cropped_orbit_and_attitude(
    orbit_xml_file: str | os.PathLike,
    pointing_xml_file: str | os.PathLike,
    start_time: str | datetime | None,
    end_time: str | datetime | None,
    *,
    padding: float = 0.0,
) -> tuple[isce3.core.Orbit, isce3.core.Attitude]:
    r"""
    Parse and crop orbit and attitude data.

    Parse the input orbit and pointing XML files and crop their contents to a common
    interval.

    Ensures that the resulting orbit and attitude time tags are referenced to a common
    epoch and that the reference epoch has integer seconds precision.

    Parameters
    ----------
    orbit_xml_file : path-like
        Path to the input orbit ephemeris XML file. Must be an existing XML file
        conforming to the NISAR Orbit Ephemeris Product Specification\ [1]_.
    pointing_xml_file : path-like
        Path to the input radar pointing XML file. Must be an existing XML file
        conforming to the NISAR Radar Pointing Product Specification\ [2]_.
    start_time : str or datetime.datetime or None
        UTC date and time of the start of the radar observation, as a
        `datetime.datetime` object or a string in ISO 8601 format. Must be <=
        `end_time`. If None, defaults to the later of the start time of the orbit data
        in `orbit_xml_file` and the start time of the attitude data in
        `pointing_xml_file`.
    end_time : str or datetime.datetime or None
        UTC date and time of the end of the radar observation, as a `datetime.datetime`
        object or a string in ISO 8601 format. Must be >= `start_time`. If None,
        defaults to the earlier of the end time of the orbit data in `orbit_xml_file`
        and the end time of the attitude data in `pointing_xml_file`.
    padding : float, optional
        Additional padding, in seconds, beyond the specified `start_time` and `end_time`
        to retain when cropping orbit and attitude data. Ignored if `start_time` and
        `end_time` are None. Must be >= 0. Defaults to 0.

    Notes
    -----
    NISAR orbit and attitude files are expected contain 30 hours of state vectors for 24
    hours of radar observation data, with 3 hours of padding on either side.

    References
    ----------
    .. [1] H. Fattahi, S. Buckley. "NASA SDS Orbit Ephemeris Product Software Interface
        Specification". JPL D-102253. 2024.
    .. [2] H. Fattahi, B. Hawkins, S. Buckley. "NASA SDS Radar Pointing Product
        Software Interface Specification". JPL D-102264. 2024.
    """
    if not (padding >= 0.0):
        raise ValueError(f"{padding=}, must be >= 0")

    logger = get_logger()

    # Load ephemeris data from input XML files.
    logger.info(f"Load orbit data from file {orbit_xml_file}")
    orbit_full = load_orbit_from_xml(orbit_xml_file)
    logger.info(
        "Original orbit data spans time interval"
        f" [{orbit_full.start_datetime, orbit_full.end_datetime}]"
    )

    logger.info(f"Load attitude data from file {pointing_xml_file}")
    attitude_full = load_attitude_from_xml(pointing_xml_file)
    logger.info(
        "Original attitude data spans time interval"
        f" [{attitude_full.start_datetime, attitude_full.end_datetime}]"
    )

    # Normalize the argument to an `isce3.core.DateTime` object.
    def normalize_datetime(t: str | datetime) -> isce3.core.DateTime:
        if isinstance(t, datetime):
            t = t.isoformat()
        return isce3.core.DateTime(t)

    # Convert `padding` to an `isce3.core.TimeDelta` object.
    padding = isce3.core.TimeDelta(padding)

    # Get the start & end of the time interval to crop the orbit and attitude data to.
    if start_time is None:
        start_time = max(orbit_full.start_datetime, attitude_full.start_datetime)
    else:
        start_time = normalize_datetime(start_time)
        start_time -= padding

    if end_time is None:
        end_time = min(orbit_full.end_datetime, attitude_full.end_datetime)
    else:
        end_time = normalize_datetime(end_time)
        end_time += padding

    if not (start_time <= end_time):
        raise ValueError(
            f"start_time must be <= end_time, got {start_time=} and {end_time=}"
        )

    # Crop orbit. Choose `npad` such that at least 4 state vectors are available for
    # Hermite interpolation.
    orbit_cropped = orbit_full.crop(start_time, end_time, npad=2)
    logger.info(
        "Cropped orbit data spans time interval"
        f" [{orbit_cropped.start_datetime, orbit_cropped.end_datetime}]"
    )

    # Crop attitude. Choose `npad` such that at least 2 points are available for slerp.
    attitude_cropped = attitude_full.crop(start_time, end_time, npad=1)
    logger.info(
        "Cropped attitude data spans time interval"
        f" [{attitude_cropped.start_datetime, attitude_cropped.end_datetime}]"
    )

    # Ensure the reference epoch has integer seconds precision.
    epoch = orbit_cropped.reference_epoch
    if epoch.frac != 0.0:
        epoch = truncate_datetime_to_integer_seconds(epoch)
        orbit_cropped.update_reference_epoch(epoch)

    # Ensure the orbit & attitude have the same reference epoch.
    if attitude_cropped.reference_epoch != epoch:
        attitude_cropped.update_reference_epoch(epoch)

    return orbit_cropped, attitude_cropped
