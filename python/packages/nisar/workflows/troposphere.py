#!/usr/bin/env python3
import h5py
import journal
import numpy as np
import os
from osgeo import gdal, osr
import time

import pyaps3 as pa

import isce3
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
    cube_extent = (np.nanmin(lat_datacube) - margin, np.nanmax(lat_datacube) + margin,
                   np.nanmin(lon_datacube) - margin, np.nanmax(lon_datacube) + margin)

    return lat_datacube, lon_datacube, cube_extent


def compute_troposphere_delay(cfg: dict, gunw_hdf5: str):
    '''
    Compute the troposphere delay datacube

    Parameters
     ----------
     cfg: dict
        runconfig dictionary
     gunw_hdf5: str
        NISAR GUNW hdf5 file

    Returns
     -------
     troposphere_delay_datacube: dict
        troposphere delay datacube dictionary
    '''

    error_channel = journal.error('troposphere.run')

    # Fetch the configurations
    tropo_weather_model_cfg = cfg['dynamic_ancillary_file_group']['troposphere_weather_model']
    tropo_cfg = cfg['processing']['troposphere_delay']

    weather_model_type = tropo_cfg['weather_model_type'].upper()
    reference_weather_model_file = tropo_weather_model_cfg['reference_troposphere_file']
    secondary_weather_model_file = tropo_weather_model_cfg['secondary_troposphere_file']

    tropo_package = tropo_cfg['package'].lower()
    tropo_delay_direction = tropo_cfg['delay_direction'].lower()
    tropo_delay_product = [product.lower()
                           for product in tropo_cfg['delay_product']]

    # Troposphere delay datacube
    troposphere_delay_datacube = dict()

    with h5py.File(gunw_hdf5, 'r', libver='latest', swmr=True) as f:

        # Fetch the GUWN Incidence Angle Datacube
        rdr_grid_path = 'science/LSAR/GUNW/metadata/radarGrid'

        inc_angle_cube = np.array(f[f'{rdr_grid_path}/incidenceAngle'])
        xcoord_radar_grid = np.array(f[f'{rdr_grid_path}/xCoordinates'])
        ycoord_radar_grid = np.array(f[f'{rdr_grid_path}/yCoordinates'])
        height_radar_grid = np.array(f[f'{rdr_grid_path}/heightAboveEllipsoid'])

        expected_inc_angle_cube_shape = (
            height_radar_grid.shape[0], ycoord_radar_grid.shape[0], xcoord_radar_grid.shape[0])

        # EPSG code
        epsg = int(np.array(f['science/LSAR/GUNW/metadata/radarGrid/epsg']))

        # Wavelenth in meters
        wavelength = isce3.core.speed_of_light / \
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
                        inc_angle_datacube = np.zeros(x_2d_radar.shape)
                    else:
                        inc_angle_datacube = inc_angle_cube[index, :, :]

                    dem_datacube = np.full(inc_angle_datacube.shape, hgt)

                    # Delay for the reference image
                    ref_aps_estimator = pa.PyAPS(reference_weather_model_file,
                                                 dem=dem_datacube,
                                                 inc=inc_angle_datacube,
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
                                                    inc=inc_angle_datacube,
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
                err_str = 'raider package is under development'
                error_channel.log(err_str)
                raise ValueError(err_str)

            # Save to the dictionary in memory
            tropo_delay_product = f'tropoDelay_{tropo_package}_{tropo_delay_direction}_{delay_product}'
            troposphere_delay_datacube[tropo_delay_product]  = tropo_delay_datacube

        f.close()

    return troposphere_delay_datacube


def write_to_GUNW_product(tropo_delay_datacube: dict, gunw_hdf5: str):
    '''
    Write the troposphere delay datacube to GUNW product

    Parameters
     ----------
     tropo_delay_datacube: dict
        troposphere delay datacube dictionary
      gunw_hdf5: str
         gunw hdf5 file
 
    Returns
     -------
       None
    '''
    
    with h5py.File(gunw_hdf5, 'a', libver='latest', swmr=True) as f:

        for product in tropo_delay_datacube.keys():

             radar_grid = f.get('science/LSAR/GUNW/metadata/radarGrid')
             
             # Troposphere delay product information
             products = product.split('_')
             package = products[1]
             delay_product = products[-1]
             delay_direction = products[2:-1]
            
             # Delay product
             delay_product = delay_product.lower()
    
             if delay_product == 'comb':
                 delay_product = 'combined'
                 
             # Delay direction
             delay_direction = '_'.join(delay_direction).lower()

             if delay_direction == 'line_of_sight_mapping':
                 delay_direction = 'Lineofsight'
             elif delay_direction == 'line_of_sight_raytracing':
                 delay_direction = 'Raytracing'
             else:
                 delay_direction = 'Zenith'
            
             # Troposphere delay Package
             tropo_pkg = package.lower()
             
             # pyAPS
             if tropo_pkg == 'pyaps':
                 tropo_pkg = 'pyAPS'
             # RAiDER
             if tropo_pkg == 'raider':
                 tropo_pkg = 'RAiDER'

             # Dataset description
             descr = f"{delay_product.capitalize()} {delay_direction.capitalize()}" + \
                     " Troposphere Delay Datacube Generated by {tropo_pkg}"

             # Product name
             product_name = f'{delay_product}{delay_direction}TroposphereDelay'

             # If there is no troposphere delay product, then createa new one
             if product_name not in radar_grid:
                 h5_prep._create_datasets(radar_grid, [0], np.float64,
                                          product_name, descr = descr,
                                          units='radians',
                                          data=tropo_delay_datacube[product].astype(np.float64))

             # If there exists the product, overwrite the old one
             else:
                 tropo_delay = radar_grid[product_name]
                 tropo_delay[:] = tropo_delay_datacube[product].astype(np.float64)

        f.close()

def run(cfg: dict, gunw_hdf5: str):
    '''
    compute the troposphere delay and write to GUNW product

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

    t_all = time.time()
    
    # Compute the troposphere delay datacube
    tropo_delay_datacube = compute_troposphere_delay(cfg, gunw_hdf5)
    
    # Write to GUNW product
    write_to_GUNW_product(tropo_delay_datacube, gunw_hdf5)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran troposphere delay in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":

    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    tropo_runcfg = InsarTroposphereRunConfig(args)
    _, out_paths = h5_prep.get_products_and_paths(tropo_runcfg.cfg)
    run(tropo_runcfg.cfg, gunw_hdf5 = out_paths['GUNW'])

