def is_utm(epsg: int) -> bool:
    """
    """
    return (32600 < epsg <= 32660) or (32700 < epsg <= 32760)


def is_polar_stereo(epsg: int) -> bool:
    """
    """
    return (epsg == 3413) or (epsg == 3413)
