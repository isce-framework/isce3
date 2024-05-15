import numpy as np

from osgeo import osr

def transform_xy_to_latlon(epsg, x, y, margin = 0.1):
    '''
    Convert the x, y coordinates in the source projection to WGS84 lat/lon

    Parameters
     ----------
     epsg: int
         epsg code
     x: numpy.ndarray
         x coordinates
     y: numpy.ndarray
         y coordinates
     margin: float
         data cube margin, default is 0.1 degrees

     Returns
     -------
     lat_datacube: numpy.ndarray
         latitude of the datacube
     lon_datacube: numpy.ndarray
         longitude of the datacube
     cube_extent: tuple
         extent of the datacube in south-north-west-east convention
     '''

    # x, y to Lat/Lon
    srs_src = osr.SpatialReference()
    srs_src.ImportFromEPSG(epsg)

    srs_wgs84 = osr.SpatialReference()
    srs_wgs84.ImportFromEPSG(4326)

    if epsg != 4326:
        # Transform the xy to lat/lon
        transformer_xy_to_latlon = osr.CoordinateTransformation(srs_src, srs_wgs84)

        # Stack the x and y
        x_y_pnts_radar = np.stack((x.ravel(), y.ravel()), axis=-1)

        # Transform to lat/lon
        lat_lon_radar = np.array(
            transformer_xy_to_latlon.TransformPoints(x_y_pnts_radar))

        # Lat lon of data cube
        lat_datacube = lat_lon_radar[:, 0].reshape(x.shape)
        lon_datacube = lat_lon_radar[:, 1].reshape(x.shape)
    else:
        lat_datacube = y.copy()
        lon_datacube = x.copy()

    # Extent of the data cube
    cube_extent = (np.nanmin(lat_datacube) - margin, np.nanmax(lat_datacube) + margin,
                   np.nanmin(lon_datacube) - margin, np.nanmax(lon_datacube) + margin)

    return lat_datacube, lon_datacube, cube_extent
