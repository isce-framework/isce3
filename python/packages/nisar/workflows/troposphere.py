#!/usr/bin/env python3
import copy
import journal
import os
import pathlib
import time

from datetime import datetime
import pyproj

import h5py
import numpy as np
from osgeo import gdal, osr

import pyaps3 as pa
import RAiDER

from RAiDER.llreader import BoundingBox
from RAiDER.losreader import Zenith, Conventional, Raytracing

from RAiDER.models.ecmwf import ECMWF
from RAiDER.models.era5 import ERA5
from RAiDER.models.gmao import GMAO
from RAiDER.models.hres import HRES
from RAiDER.models.hrrr import HRRR
from RAiDER.models.merra2 import MERRA2
from RAiDER.models.ncmr import NCMR
from RAiDER.delay import tropo_delay


from nisar.workflows import h5_prep
from nisar.workflows.troposphere_runconfig import InsarTroposphereRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse


def transform_xy_to_latlon(epsg, x, y):
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

     Returns 
     ------- 
     lat_datacube: numpy.ndarray
         latitude of the datacube 
     lon_datacube: numpy.ndarray
         longitude of the datacube
     cube_extent: tuple
         extent of the datacube in south-north-west-east convention
     '''

    # X, y to Lat/Lon
    srs_src = osr.SpatialReference()
    srs_src.ImportFromEPSG(epsg)

    srs_wgs84 = osr.SpatialReference()
    srs_wgs84.ImportFromEPSG(4326)

    # Transformer
    transformer_xy_to_latlon = osr.CoordinateTransformation(srs_src, srs_wgs84)

    # Stack the x and y
    x_y_pnts_radar = np.stack((x.flatten(), y.flatten()), axis=-1)

    # Transform to lat/lon
    lat_lon_radar = np.array(
        transformer_xy_to_latlon.TransformPoints(x_y_pnts_radar))

    # Lat lon of data cube
    lat_datacube = lat_lon_radar[:, 0].reshape(x.shape)
    lon_datacube = lat_lon_radar[:, 1].reshape(x.shape)

    # 0.1 degrees is aded  to make sure the weather model cover the entire image
    margin = 0.1

    # Extent of the data cube
    cube_extent = (np.nanmin(lat_datacube)-margin, np.nanmax(lat_datacube)+margin,
                   np.nanmin(lon_datacube)-margin, np.nanmax(lon_datacube)+margin)

    return lat_datacube, lon_datacube, cube_extent


def run(cfg: dict, gunw_hdf5: str):
    '''
    Compute the troposphere delay datacube and added to the  GUNW product

    Parameters
     ----------
     cfg: dict
        runconfig dictionary
     gunw_hdf5: str
        gunw hdf5 file

    Returns
     -------
       None
    '''

    # Create error and info channels
    info_channel = journal.info("troposphere.run")
    info_channel.log("starting insar_troposphere_delay computation")

    # Fetch the configurations
    tropo_weather_model_cfg = cfg['dynamic_ancillary_file_group']['troposphere_weather_model']
    tropo_cfg = cfg['processing']['troposphere_delay']

    weather_model_type = tropo_cfg['weather_model_type'].upper()
    reference_weather_model_file = tropo_weather_model_cfg['reference_file_path']
    secondary_weather_model_file = tropo_weather_model_cfg['secondary_file_path']

    tropo_package = tropo_cfg['package'].lower()
    tropo_delay_direction = tropo_cfg['delay_direction'].lower()
    tropo_delay_product = [product.lower()
                           for product in tropo_cfg['delay_product']]

    t_all = time.time()

    with h5py.File(gunw_hdf5, 'a', libver='latest', swmr=True) as f:

        # Fetch the GUWN Incidence Angle Datacube
        ia_cube = np.array(
            f['science/LSAR/GUNW/metadata/radarGrid/incidenceAngle'])
        xcoord_radar_grid = np.array(
            f['science/LSAR/GUNW/metadata/radarGrid/xCoordinates'])
        ycoord_radar_grid = np.array(
            f['science/LSAR/GUNW/metadata/radarGrid/yCoordinates'])
        height_radar_grid = np.array(
            f['science/LSAR/GUNW/metadata/radarGrid/heightAboveEllipsoid'])

        expected_ia_cube_shape = (
            height_radar_grid.shape[0], ycoord_radar_grid.shape[0], xcoord_radar_grid.shape[0])

        # EPSG code
        epsg = int(np.array(f['science/LSAR/GUNW/metadata/radarGrid/epsg']))

        # Wavelenth in meters
        wavelength = 299792458.0 / \
            float(
                np.array(f['/science/LSAR/GUNW/grids/frequencyA/centerFrequency']))

        # X and y for the entire datacube
        y_2d_radar = np.tile(ycoord_radar_grid, (len(xcoord_radar_grid), 1)).T
        x_2d_radar = np.tile(xcoord_radar_grid, (len(ycoord_radar_grid), 1))

        # Lat/lon coordinates
        lat_datacube, lon_datacube, _ = transform_xy_to_latlon(
            epsg, x_2d_radar, y_2d_radar)

        for delay_product in tropo_delay_product:

            # pyaps package
            if tropo_package == 'pyaps':

                delay_type = delay_product
                if delay_type == 'hydro':
                    delay_type = 'dry'

                tropo_delay_datacube_list = []
                for index, hgt in enumerate(height_radar_grid):
                    if tropo_delay_direction == 'zenith':
                        inc_datacube = np.zeros(x_2d_radar.shape)
                    else:
                        inc_datacube = ia_cube[index, :, :]

                    dem_datacube = np.full(inc_datacube.shape, hgt)

                    # Delay for the reference image
                    ref_aps_estimator = pa.PyAPS(reference_weather_model_file,
                                                 dem=dem_datacube,
                                                 inc=inc_datacube,
                                                 lat=lat_datacube,
                                                 lon=lon_datacube,
                                                 grib=weather_model_type,
                                                 humidity='Q',
                                                 verb=False,
                                                 Del=delay_type)

                    phs_ref = ref_aps_estimator.getdelay()

                    # Delay for the secondary image
                    second_aps_estimator = pa.PyAPS(secondary_weather_model_file,
                                                    dem=dem_datacube,
                                                    inc=inc_datacube,
                                                    lat=lat_datacube,
                                                    lon=lon_datacube,
                                                    grib=weather_model_type,
                                                    humidity='Q',
                                                    verb=False,
                                                    Del=delay_type)

                    phs_second = second_aps_estimator.getdelay()

                    # Convert the delay in meters to radians
                    tropo_delay_datacube_list.append(
                        (phs_ref - phs_second)*4.0*np.pi/wavelength)

                # Tropo delay datacube
                tropo_delay_datacube = np.stack(tropo_delay_datacube_list)
                tropo_delay_datacube_list = None


            # raider package
            else:

                # Acquisition time for reference and secondary images
                acquisition_time_ref = str(np.array(
                    f['science/LSAR/identification/referenceZeroDopplerStartTime']).astype(str))
                acquisition_time_second = str(np.array(
                    f['science/LSAR/identification/secondaryZeroDopplerStartTime']).astype(str))

                datetime_reference = datetime.strptime(
                    acquisition_time_ref, '%Y-%M-%DT%HH:%%MM:%SS')
                dateimte_secondary = datetime.strptime(
                    acquisition_time_second, '%Y-%M-%DT%HH:%%MM:%SS')

                x_spacing = float(
                    np.array(f['science/LSAR/GUNW/grids/frequencyA/xCoordinateSpacing']))
                y_spacing = float(
                    np.array(f['science/LSAR/GUNW/grids/frequencyA/yCoordinateSpacing']))

                if x_spacing != y_spacing:
                    err_str = f'x = {x_spacing} and y = {y_spacing} spacing should be equal'
                    raise ValueError(err_str)

                # Cube Spacing
                cube_spacing = x_spacing

                # AOI bounding box
                min_lat = np.min(lat_datacube)
                max_lat = np.max(lat_datacube)
                min_lon = np.min(lon_datacube)
                max_lon = np.max(lon_datacube)

                aoi = BoundingBox([min_lat, maxx_lat, min_lon, max_lon])

                # Line of sight
                los = Zenith()

                # Line of sight mapping direction
                if tropo_delay_direction == 'line_of_sight_mapping':
                    los = Conventional()
                if tropo_delay_direction == 'line_of_sight_raytracing':
                    los = Raytracing()

                # Height levels
                height_levels = list(height_radar_grid)

                # Tropodelay computation
                tropo_delay_reference, _ = tropo_delay(dt=datetime_reference,
                                                       weather_model_file=reference_weather_model_file,
                                                       aoi=aoi,
                                                       los=los,
                                                       height_levels=height_levels,
                                                       out_proj=epsg,
                                                       cube_spacing_m=cube_spacing)

                tropo_delay_secondary, _ = tropo_delay(dt=datetime_scondary,
                                                       weather_model_file=secondary_weather_model_file,
                                                       aoi=aoi,
                                                       los=los,
                                                       height_levels=height_levels,
                                                       out_proj=epsg,
                                                       cube_spacing_m=cube_spacing)

                # Troposphere delay by raider package
                tropo_delay = tropo_delay_reference[delay_product] - \
                    tropo_delay_secondary[delay_product]

                # Covert it to radians
                tropo_delay_datacube = tropo_delay*4.0*np.pi/wavelength

            # Write to GUWN product
            radar_grid = f.get('science/LSAR/GUNW/metadata/radarGrid')
            radar_grid.create_dataset(f'tropoDelay_{tropo_delay_direction}_{delay_product}',
                                      data=tropo_delay_datacube, dtype=np.float32, compression='gzip')

            f.close()

    t_all_elapsed = time.time() - t_all
    info_channel.log(
        f"successfully ran troposhere delay  in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":

    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    tropo_runcfg = InsarTroposphereRunConfig(args)
    _, out_paths = h5_prep.get_products_and_paths(tropo_runcfg.cfg)
    run(tropo_runcfg.cfg, gunw_hdf5=out_paths['GUNW'])
