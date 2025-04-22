from __future__ import annotations

from isce3.core import DataInterpMethod, GeocodeMemoryMode, LookSide


def normalize_look_side(look_side: str | LookSide) -> LookSide:
    """
    Normalize the argument to a `LookSide` object.

    If the input is a string, it will be converted to the corresponding enum value. If
    it's a `LookSide` object, it will be returned unmodified.

    Parameters
    ----------
    look_side : str or LookSide
        The object to be converted to a `LookSide` value.

    Returns
    -------
    LookSide
        The input converted to a `LookSide`.
    """
    if isinstance(look_side, LookSide):
        return look_side

    if look_side == "left":
        return LookSide.Left
    if look_side == "right":
        return LookSide.Right

    raise ValueError(f"unexpected LookSide argument: {look_side!r}")


def normalize_data_interp_method(method: str | DataInterpMethod) -> DataInterpMethod:
    """
    Normalize the argument to a `DataInterpMethod` object.

    If the input is a string, it will be converted to the corresponding enum value. If
    it's a `DataInterpMethod` object, it will be returned unmodified.

    Parameters
    ----------
    method : str or DataInterpMethod
        The object to be converted to a `DataInterpMethod` value.

    Returns
    -------
    DataInterpMethod
        The input converted to a `DataInterpMethod`.
    """
    if isinstance(method, DataInterpMethod):
        return method

    if (method == "biquintic") or (method == "BIQUINTIC"):
        return DataInterpMethod.BIQUINTIC
    if (method == "sinc") or (method == "SINC"):
        return DataInterpMethod.SINC
    if (method == "bilinear") or (method == "BILINEAR"):
        return DataInterpMethod.BILINEAR
    if (method == "bicubic") or (method == "BICUBIC"):
        return DataInterpMethod.BICUBIC
    if (method == "nearest") or (method == "NEAREST"):
        return DataInterpMethod.NEAREST

    raise ValueError(f"unexpected DataInterpMethod argument: {method!r}")


def normalize_geocode_memory_mode(
    mode: str | GeocodeMemoryMode | None
) -> GeocodeMemoryMode:
    """
    Normalize the argument to a `GeocodeMemoryMode` object.

    If the input is a string, it will be converted to the corresponding enum value. If
    it's a `GeocodeMemoryMode` object, it will be returned unmodified. Returns
    `GeocodeMemoryMode.Auto` if the input is `None`.

    Parameters
    ----------
    method : str or GeocodeMemoryMode or None
        The object to be converted to a `GeocodeMemoryMode` value.

    Returns
    -------
    GeocodeMemoryMode
        The input converted to a `GeocodeMemoryMode`.
    """
    if isinstance(mode, GeocodeMemoryMode):
        return mode

    if mode == "single_block":
        return GeocodeMemoryMode.SingleBlock
    if mode == "geogrid":
        return GeocodeMemoryMode.BlocksGeogrid
    if mode == "geogrid_and_radargrid":
        return GeocodeMemoryMode.BlocksGeogridAndRadarGrid
    if (mode == "auto") or (mode is None):
        return GeocodeMemoryMode.Auto

    raise ValueError(f"unexpected GeocodeMemoryMode argument: {mode!r}")
