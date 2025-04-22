from __future__ import annotations

from isce3.geometry import RtcAlgorithm, RtcAreaBetaMode


def normalize_rtc_algorithm(algorithm: str | RtcAlgorithm) -> RtcAlgorithm:
    """
    Normalize the argument to an `RtcAlgorithm` object.

    If the input is a string, it will be converted to the corresponding enum value. If
    it's an `RtcAlgorithm` object, it will be returned unmodified.

    Parameters
    ----------
    method : str or RtcAlgorithm
        The object to be converted to an `RtcAlgorithm` value.

    Returns
    -------
    RtcAlgorithm
        The input converted to an `RtcAlgorithm`.
    """
    if isinstance(algorithm, RtcAlgorithm):
        return algorithm

    if algorithm == "area_projection":
        return RtcAlgorithm.RTC_AREA_PROJECTION
    if algorithm == "bilinear_distribution":
        return RtcAlgorithm.RTC_BILINEAR_DISTRIBUTION

    raise ValueError(f"unexpected RtcAlgorithm argument: {algorithm!r}")


def normalize_rtc_area_beta_mode(mode: str | RtcAreaBetaMode | None) -> RtcAreaBetaMode:
    """
    Normalize the argument to a `RtcAreaBetaMode` object.

    If the input is a string, it will be converted to the corresponding enum value. If
    it's a `RtcAreaBetaMode` object, it will be returned unmodified. Returns
    `RtcAreaBetaMode.Auto` if the input is `None`.

    Parameters
    ----------
    method : str or RtcAreaBetaMode or None
        The object to be converted to a `RtcAreaBetaMode` value.

    Returns
    -------
    RtcAreaBetaMode
        The input converted to a `RtcAreaBetaMode`.
    """
    if isinstance(mode, RtcAreaBetaMode):
        return mode

    if mode == "pixel_area":
        return RtcAreaBetaMode.PIXEL_AREA
    if mode == "projection_angle":
        return RtcAreaBetaMode.PROJECTION_ANGLE
    if (mode == "auto") or (mode is None):
        return RtcAreaBetaMode.AUTO

    raise ValueError(f"unexpected RtcAreaBetaMode argument: {mode!r}")
