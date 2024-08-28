from osgeo import osr
from isce3.product.cf_conventions import get_grid_mapping_name


def test_grid_mapping_names():
    # https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#_transverse_mercator
    epsg_names = {
        4326: "latitude_longitude",
        32611: "transverse_mercator",
        3413: "polar_stereographic",
    }
    for epsg, expected in epsg_names.items():
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        assert expected == get_grid_mapping_name(srs)
