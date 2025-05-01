# import re
import string
from datetime import datetime

import isce3


def int_to_3_digit_string(i: int) -> str:
    """ """
    if (i < 0) or (i > 999):
        raise ValueError  # FIXME
    return f"{i:03d}"


def orbit_direction_to_char_code(direction: isce3.core.OrbitPassDirection) -> str:
    """ """
    if direction == isce3.core.OrbitPassDirection.ASCENDING:
        return "A"
    if direction == isce3.core.OrbitPassDirection.DESCENDING:
        return "D"
    raise ValueError  # FIXME


def datetime_to_yyyymmddthhmmss(t: datetime) -> str:
    """ """
    if t.microsecond != 0:
        raise ValueError
    return t.strftime("%Y%m%dT%H%M%S")


def processing_center_to_char_code(processing_center: str) -> str:
    """ """
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
    validity_start_datetime: datetime,
    composite_release_id: str,
    processing_center: str,
    product_counter: int,
) -> str:
    """ """
    template = string.Template(
        "${MISSION}_${I}${L}_${PROD}_${REL}_${P}_${FRM}_${Xposting}_${Yposting}"
        "_${ValidityStartDateTime}_${CRID}_${LOC}_${CTR}"
    )
    return template.substitute(
        MISSION=mission_id,
        I=radar_band,
        L=product_level,
        PROD=product_type,
        REL=int_to_3_digit_string(relative_orbit_number),
        P=orbit_direction_to_char_code(orbit_pass_direction),
        FRM=int_to_3_digit_string(frame_number),
        Xposting=int_to_3_digit_string(int(round(x_posting))),
        Yposting=int_to_3_digit_string(int(round(y_posting))),
        ValidityStartDateTime=datetime_to_yyyymmddthhmmss(validity_start_datetime),
        CRID=composite_release_id,
        LOC=processing_center_to_char_code(processing_center),
        CTR=int_to_3_digit_string(product_counter),
    )
