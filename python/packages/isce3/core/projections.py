def is_utm(epsg: int) -> bool:
    """
    Check if the input EPSG code represents a UTM zone.

    Parameters
    ----------
    epsg : int
        The input EPSG code.

    Returns
    -------
    bool
        True if the coordinate reference system represented by `epsg` is a Universal
        Transverse Mercator (UTM) projection; otherwise False.
    """
    return (32600 < epsg <= 32660) or (32700 < epsg <= 32760)
