from osgeo import osr


def get_grid_mapping_name(srs: osr.SpatialReference) -> str:
    """
    Determing the CF-compliant grid_mapping_name corresponding to a spatial
    reference system.

    Parameters
    ----------
    srs : osgeo.osr.SpatialReference
        Spatial reference system

    Returns
    -------
    grid_mapping_name : str
        CF-compliant name

    Notes
    -----
    https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html
    """
    proj_name = srs.GetAttrValue("PROJECTION")
    if proj_name is None:
        return "latitude_longitude"
    return proj_name.lower()
