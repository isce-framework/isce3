import string
from datetime import datetime
from osgeo import osr

import isce3


def int_to_n_digit_string(i: int, n: int = 3) -> str:
    """
    Format an integer as an n-digit zero-padded string.

    Parameters
    ----------
    i : int
        The input integer. Must be >= 0 and <= 10^n - 1.

    n : int
        The number of digits in the output string.

    Returns
    -------
    str
        An n-character string representation of the input integer, left-padded
        with zeros.
    """
    max_value = 10 ** n - 1
    if (i < 0) or (i > max_value):
        raise ValueError(f"argument must be in the range [0, {max_value}],"
                         f" got {i}")
    return f"{i:0{n}d}"


def orbit_direction_to_char_code(direction: isce3.core.OrbitPassDirection) -> str:
    """
    Get a single-character mnemonic representing the input orbit pass direction.

    Parameters
    ----------
    direction : isce3.core.OrbitPassDirection
        The input orbit pass direction.

    Returns
    -------
    str
        'A' for ascending orbits or 'D' for descending orbits.
    """
    if direction == isce3.core.OrbitPassDirection.ASCENDING:
        return "A"
    if direction == isce3.core.OrbitPassDirection.DESCENDING:
        return "D"
    raise ValueError(f"unexpected orbit pass direction: {direction}")


def datetime_to_yyyymmddthhmmss(t: datetime) -> str:
    """
    Convert a datetime to a string in 'YYYYmmddTHHMMSS' format.

    Parameters
    ----------
    t : datetime.datetime
        The input datetime object. Must not contain a fractional seconds component.

    Returns
    -------
    str
        A string representation of the input datetime object, consisting of 8 digits
        representing the year-month-day component, followed by a literal 'T', followed
        by 6 digits representing the hour-minute-second component.
    """
    if t.microsecond != 0:
        raise ValueError(
            f"input datetime must not contain a fractional seconds component; got {t}"
        )
    return t.strftime("%Y%m%dT%H%M%S")


def processing_center_to_char_code(processing_center: str) -> str:
    """
    Get a single-character mnemonic representing the processing center.

    Parameters
    ----------
    processing_center : str
        A string denoting the data processing center of a NISAR Static Layers granule,
        e.g. 'JPL'.

    Returns
    -------
    str
        'J' for 'JPL'; 'X' otherwise.
    """
    if processing_center == "JPL":
        return "J"
    return "X"


def form_granule_id(
    *,
    mission_id: str,
    radar_band: str,
    product_level: int,
    product_type: str,
    relative_orbit_number: int,
    orbit_pass_direction: isce3.core.OrbitPassDirection,
    frame_number: int,
    x_posting: float,
    y_posting: float,
    epsg_code: int,
    validity_start_datetime: datetime,
    composite_release_id: str,
    processing_center: str,
    product_counter: int,
) -> str:
    """
    Form a NISAR Static Layers granule ID string.

    The granule ID is the file name of a particular Static Layers product, according to
    the official naming convention (excluding the '.h5' file extension).

    Parameters
    ----------
    mission_id : str
        Mission identifier, e.g. 'NISAR'.
    radar_band : str
        Radio frequency band, e.g. 'L'.
    product_level : int
        Product level, e.g. 2.
    product_type : str
        Product type, e.g. 'STATIC'.
    relative_orbit_number : int
        Track identifier. Must be in the range [0, 999].
    orbit_pass_direction : isce3.core.OrbitPassDirection
        Orbit pass direction (ascending or descending).
    frame_number : int
        Frame identifier. Must be in the range [0, 999].
    x_posting, y_posting : float
        X and Y spacing of the raster coordinate grid, in the units of the grid's native
        coordinate system. Must be > 0 and <= 999.
    epsg_code : int
        EPSG code of the spatial reference system in which the raster coordinate grid is
        defined.
    validity_start_datetime : datetime.datetime
        UTC date and time of the start of the granule's validity date range. Must not
        contain a fractional seconds component.
    composite_release_id : str
        Unique 6-digit alphanumeric product version identifier in the science data
        production system.
    processing_center : str
        Data processing center, e.g. 'JPL'.
    product_counter : int
        Product counter used to distinguish multiple products generated with the same
        inputs and software version. Must be in the range [0, 999].

    Returns
    -------
    str
        The granule ID. If the spatial reference system specified by
        epsg_code is projected, the X and Y postings are multiplied
        by 10 and formatted as four-digit, zero-padded integers.
        If the spatial reference system is geographic, the X and Y
        postings are formatted directly (without multiplication) as
        four-digit, zero-padded integers, with the decimal point
        removed.

    References
    ----------
    .. [1] S. Niemoeller, "NASA SDS Product Specification Level-2 Static Layers", JPL
        D-107727, 2025.
    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)

    # The granule ID for projected coordinate systems encodes the X and Y
    # postings in decimeters as 4-digit zero-padded integers,
    # while the granule ID for geographic coordinate systems encodes the
    # X and Y postings in degrees as 4-digit zero-padded integers with the
    # decimal point removed.
    if not srs.IsGeographic():
        template = string.Template(
            "${MISSION}_${I}${L}_${PROD}_${REL}_${P}_${FRM}"
            "_${XpostingDecimeters}_${YpostingDecimeters}"
            "_${ValidityStartDateTime}_${CRID}_${LOC}_${CTR}"
        )
    else:
        template = string.Template(
            "${MISSION}_${I}${L}_${PROD}_${REL}_${P}_${FRM}"
            "_${Xposting}_${Yposting}"
            "_${ValidityStartDateTime}_${CRID}_${LOC}_${CTR}"
        )
    return template.substitute(
        MISSION=mission_id,
        I=radar_band,
        L=product_level,
        PROD=product_type,
        REL=int_to_n_digit_string(relative_orbit_number, n=3),
        P=orbit_direction_to_char_code(orbit_pass_direction),
        FRM=int_to_n_digit_string(frame_number, 3),
        XpostingDecimeters=int_to_n_digit_string(int(10*round(x_posting)),
                                                 n=4),
        YpostingDecimeters=int_to_n_digit_string(int(10*round(y_posting)),
                                                 n=4),
        Xposting=int_to_n_digit_string(int(round(x_posting)), n=4),
        Yposting=int_to_n_digit_string(int(round(y_posting)), n=4),
        ValidityStartDateTime=datetime_to_yyyymmddthhmmss(
            validity_start_datetime),
        CRID=composite_release_id,
        LOC=processing_center_to_char_code(processing_center),
        CTR=int_to_n_digit_string(product_counter, n=3),
    )
