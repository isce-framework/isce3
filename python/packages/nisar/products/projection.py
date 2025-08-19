import numpy as np
from osgeo import osr
from numpy.typing import ArrayLike

import isce3
from isce3.product.cf_conventions import get_grid_mapping_name


def build_projection_dataset_attrs_dict(epsg: int) -> dict[str, ArrayLike]:
    """
    Get attributes describing the spatial reference system of NISAR L2 products.

    Returns a dict that may be used to populate the Attributes of `projection` Datasets
    in NISAR Level 2 products in a way that's compliant with Climate and Forecast (CF)
    Metadata Conventions.

    Parameters
    ----------
    epsg : int
        The EPSG code associated with the spatial reference system.

    Returns
    -------
    dict
        A dict containing attributes describing the spatial reference system.
        String-valued attributes are represented by bytestrings with 'utf-8' encoding.
    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)

    # Common attributes for all spatial reference systems.
    # FIXME: Should `spatial_ref` be replaced with `crs_wkt` for CF-1.7 compliance? See
    # https://github-fn.jpl.nasa.gov/NISAR-ADT/NISAR_PIX/issues/318.
    attrs = dict(
        epsg_code=epsg,
        spatial_ref=np.bytes_(srs.ExportToWkt()),
        grid_mapping_name=np.bytes_(get_grid_mapping_name(srs)),
        semi_major_axis=6378137.0,
        inverse_flattening=298.257223563,
        ellipsoid=np.bytes_("WGS84"),
    )

    if epsg == 4326:  # WGS 84
        attrs["longitude_of_prime_meridian"] = 0.0
        return attrs

    # Common attributes for all projected spatial reference systems.
    attrs["false_easting"] = srs.GetProjParm(osr.SRS_PP_FALSE_EASTING)
    attrs["false_northing"] = srs.GetProjParm(osr.SRS_PP_FALSE_NORTHING)
    attrs["longitude_of_projection_origin"] = srs.GetProjParm(
        osr.SRS_PP_LONGITUDE_OF_ORIGIN
    )

    if epsg == 3413:  # Polar Stereographic (North)
        attrs["latitude_of_projection_origin"] = 90.0
        attrs["standard_parallel"] = 70.0
        attrs["straight_vertical_longitude_from_pole"] = -45.0
        return attrs

    if epsg == 3031:  # Polar Stereographic (South)
        attrs["latitude_of_projection_origin"] = -90.0
        attrs["standard_parallel"] = -71.0
        attrs["straight_vertical_longitude_from_pole"] = 0.0
        return attrs

    # Attribute for non-Polar-Stereographic spatial reference systems.
    attrs["latitude_of_projection_origin"] = srs.GetProjParm(
        osr.SRS_PP_LATITUDE_OF_ORIGIN
    )

    if isce3.core.is_utm(epsg):
        attrs["utm_zone_number"] = epsg % 100
        attrs["longitude_of_central_meridian"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN
        )
        attrs["scale_factor_at_central_meridian"] = srs.GetProjParm(
            osr.SRS_PP_SCALE_FACTOR
        )
        return attrs

    if epsg == 6933:  # EASE-Grid 2.0
        attrs["longitude_of_central_meridian"] = 0.0
        attrs["standard_parallel"] = 30.0
        return attrs

    # FIXME: This CRS uses the GRS80 reference ellipsoid -- not WGS 84. The ellipsoid
    # parameters defined above are not valid for this EPSG code. LAEA Europe is not
    # supported anywhere else in ISCE3. Should we remove this? See
    # https://github.com/isce-framework/isce3/issues/72.
    if epsg == 3035:  # LAEA Europe
        attrs["standard_parallel"] = -71.0
        attrs["straight_vertical_longitude_from_pole"] = 0.0
        return attrs

    raise NotImplementedError(
        f"EPSG {epsg} waiting for implementation / not supported in ISCE3"
    )
