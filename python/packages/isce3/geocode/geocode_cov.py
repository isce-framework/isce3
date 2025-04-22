from __future__ import annotations

from isce3.geocode import GeocodeOutputMode


def normalize_geocode_output_mode(mode: str | GeocodeOutputMode) -> GeocodeOutputMode:
    """
    Normalize the argument to a `GeocodeOutputMode` object.

    If the input is a string, it will be converted to the corresponding enum value. If
    it's a `GeocodeOutputMode` object, it will be returned unmodified.

    Parameters
    ----------
    method : str or GeocodeOutputMode
        The object to be converted to a `GeocodeOutputMode` value.

    Returns
    -------
    GeocodeOutputMode
        The input converted to a `GeocodeOutputMode`.
    """
    if isinstance(mode, GeocodeOutputMode):
        return mode

    if mode == "area_projection":
        return GeocodeOutputMode.AREA_PROJECTION
    if mode == "interp":
        return GeocodeOutputMode.INTERP

    raise ValueError(f"unexpected GeocodeOutputMode argument: {mode!r}")
