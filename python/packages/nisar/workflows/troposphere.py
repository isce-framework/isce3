#!/usr/bin/env python3
from datetime import datetime
import h5py
import journal
import numpy as np
import os
from osgeo import gdal, osr
from scipy.interpolate import RegularGridInterpolator
import time

import pyaps3 as pa

import RAiDER
from RAiDER.llreader import BoundingBox
from RAiDER.losreader import Zenith, Conventional, Raytracing
from RAiDER.delay import tropo_delay as raider_tropo_delay

import isce3
from isce3.core import transform_xy_to_latlon
from nisar.workflows import h5_prep
from nisar.workflows.troposphere_runconfig import InsarTroposphereRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse


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

    error_channel = journal.error('troposphere.compute_troposphere_delay')

    # Fetch the configurations
    tropo_weather_model_cfg = cfg['dynamic_ancillary_file_group']['troposphere_weather_model']
    tropo_cfg = cfg['processing']['troposphere_delay']

    weather_model_type = tropo_cfg['weather_model_type'].upper()
    reference_weather_model_file = tropo_weather_model_cfg['reference_troposphere_file']
    secondary_weather_model_file = tropo_weather_model_cfg['secondary_troposphere_file']

    tropo_package = tropo_cfg['package'].lower()
    tropo_delay_direction = tropo_cfg['delay_direction'].lower()

    tropo_delay_products = []
    # comb is short for the summation of wet and dry components
    for delay_type in ['wet', 'dry', 'comb']:
        if tropo_cfg[f'enable_{delay_type}_product']:
            if (delay_type == 'dry') and \
                    (tropo_package == 'raider'):
                delay_type = 'hydro'
        tropo_delay_products.append(delay_type)

    # Troposphere delay datacube
    troposphere_delay_datacube = dict()

    with h5py.File(gunw_hdf5, 'r', libver='latest', swmr=True) as h5_obj:

        # Fetch the GUWN Incidence Angle Datacube
        rdr_grid_path = 'science/LSAR/GUNW/metadata/radarGrid'

        inc_angle_cube = h5_obj[f'{rdr_grid_path}/incidenceAngle'][()]
        xcoord_radar_grid = h5_obj[f'{rdr_grid_path}/xCoordinates'][()]
        ycoord_radar_grid = h5_obj[f'{rdr_grid_path}/yCoordinates'][()]
        height_radar_grid = h5_obj[f'{rdr_grid_path}/heightAboveEllipsoid'][()]

        # EPSG code
        epsg = int(h5_obj['science/LSAR/GUNW/metadata/radarGrid/epsg'][()])

        # Wavelenth in meters
        wavelength = isce3.core.speed_of_light / \
                h5_obj['/science/LSAR/GUNW/grids/frequencyA/centerFrequency'][()]

        # X and y for the entire datacube
        y_2d_radar = np.tile(ycoord_radar_grid, (len(xcoord_radar_grid), 1)).T
        x_2d_radar = np.tile(xcoord_radar_grid, (len(ycoord_radar_grid), 1))

        # Lat/lon coordinates
        lat_datacube, lon_datacube, _ = transform_xy_to_latlon(
            epsg, x_2d_radar, y_2d_radar)

        # pyaps package
        if tropo_package == 'pyaps':

            for tropo_delay_product in tropo_delay_products:

                tropo_delay_datacube_list = []
                for index, hgt in enumerate(height_radar_grid):

                    dem_datacube = np.full(lat_datacube.shape, hgt)
                    # Delay for the reference image
                    ref_aps_estimator = pa.PyAPS(reference_weather_model_file,
                                                 dem=dem_datacube,
                                                 inc=0.0,
                                                 lat=lat_datacube,
                                                 lon=lon_datacube,
                                                 grib=weather_model_type,
                                                 humidity='Q',
                                                 model=weather_model_type,
                                                 verb=False,
                                                 Del=tropo_delay_product)

                    phs_ref = ref_aps_estimator.getdelay()

                    # Delay for the secondary image
                    second_aps_estimator = pa.PyAPS(secondary_weather_model_file,
                                                    dem=dem_datacube,
                                                    inc=0.0,
                                                    lat=lat_datacube,
                                                    lon=lon_datacube,
                                                    grib=weather_model_type,
                                                    humidity='Q',
                                                    model=weather_model_type,
                                                    verb=False,
                                                    Del=tropo_delay_product)

                    phs_second = second_aps_estimator.getdelay()

                    # Convert the delay in meters to radians
                    tropo_delay_datacube_list.append(
                            -(phs_ref - phs_second) *4.0*np.pi/wavelength)

                # Tropo delay datacube
                tropo_delay_datacube = np.stack(tropo_delay_datacube_list)
                tropo_delay_datacube_list = None

                if tropo_delay_direction == 'line_of_sight_mapping':
                    tropo_delay_datacube /= np.cos(np.deg2rad(inc_angle_cube))

                # Save to the dictionary in memory
                tropo_delay_product_name = f'tropoDelay_{tropo_package}_{tropo_delay_direction}_{tropo_delay_product}'
                troposphere_delay_datacube[tropo_delay_product_name]  = tropo_delay_datacube

        # raider package
        else:

            # Acquisition time for reference and secondary images
            acquisition_time_ref = h5_obj['science/LSAR/identification/referenceZeroDopplerStartTime'][()]\
                    .astype('datetime64[s]').astype(datetime)
            acquisition_time_second = h5_obj['science/LSAR/identification/secondaryZeroDopplerStartTime'][()]\
                    .astype('datetime64[s]').astype(datetime)

            # AOI bounding box
            margin = 0.1
            min_lat = np.min(lat_datacube)
            max_lat = np.max(lat_datacube)
            min_lon = np.min(lon_datacube)
            max_lon = np.max(lon_datacube)

            aoi = BoundingBox([min_lat - margin,
                               max_lat + margin,
                               min_lon - margin,
                               max_lon + margin])

            # Zenith
            delay_direction_obj = Zenith()

            if tropo_delay_direction == 'line_of_sight_raytracing':
                delay_direction_obj = Raytracing()

            # Height levels
            height_levels = list(height_radar_grid)

            # Tropodelay computation
            tropo_delay_reference, _ = raider_tropo_delay(dt=acquisition_time_ref,
                                                          weather_model_file=reference_weather_model_file,
                                                          aoi=aoi,
                                                          los=delay_direction_obj,
                                                          height_levels=height_levels,
                                                          out_proj=epsg)

            tropo_delay_secondary, _ = raider_tropo_delay(dt=acquisition_time_second,
                                                          weather_model_file=secondary_weather_model_file,
                                                          aoi=aoi,
                                                          los=delay_direction_obj,
                                                          height_levels=height_levels,
                                                          out_proj=epsg)

            for tropo_delay_product in tropo_delay_products:

                # Compute troposphere delay with raider package
                # comb is the summation of wet and hydro components
                if tropo_delay_product == 'comb':
                    tropo_delay = tropo_delay_reference['wet'] + tropo_delay_reference['hydro'] - \
                            tropo_delay_secondary['wet'] - tropo_delay_secondary['hydro']
                else:
                    tropo_delay = tropo_delay_reference[tropo_delay_product] - \
                            tropo_delay_secondary[tropo_delay_product]

                # Convert it to radians units
                tropo_delay_datacube = -tropo_delay * 4.0 * np.pi/wavelength

                # Interpolate to radar grid to keep its dimension consistent with other datacubes
                tropo_delay_interpolator = RegularGridInterpolator((tropo_delay_reference.z,
                                                                    tropo_delay_reference.y,
                                                                    tropo_delay_reference.x),
                                                                   tropo_delay_datacube,
                                                                   method='linear')

                # Interpolate the troposphere delay
                hv, yv, xv = np.meshgrid(height_radar_grid,
                                         ycoord_radar_grid,
                                         xcoord_radar_grid,
                                         indexing='ij')

                pnts = np.stack(
                        (hv.flatten(), yv.flatten(), xv.flatten()), axis=-1)

                # Interpolate
                tropo_delay_datacube = tropo_delay_interpolator(
                        pnts).reshape(inc_angle_cube.shape)

                # Line of sight mapping
                if tropo_delay_direction == 'line_of_sight_mapping':
                    tropo_delay_datacube /= np.cos(np.deg2rad(inc_angle_cube))

                # Save to the dictionary in memory
                tropo_delay_product_name = f'tropoDelay_{tropo_package}_{tropo_delay_direction}_{tropo_delay_product}'
                troposphere_delay_datacube[tropo_delay_product_name]  = tropo_delay_datacube


    return troposphere_delay_datacube


def write_to_GUNW_product(tropo_delay_datacubes: dict, gunw_hdf5: str):
    '''
    Write the troposphere delay datacubes to GUNW product

    Parameters
     ----------
     tropo_delay_datacubes: dict
        troposphere delay datacube dictionary
      gunw_hdf5: str
         gunw hdf5 file

    Returns
     -------
       None
    '''

    with h5py.File(gunw_hdf5, 'a', libver='latest', swmr=True) as f:

        for product_name, product_cube in tropo_delay_datacubes.items():

             radar_grid = f.get('science/LSAR/GUNW/metadata/radarGrid')

             # Troposphere delay product information
             products = product_name.split('_')
             package = products[1]
             delay_product = products[-1]
             delay_direction = products[2:-1]

             # Delay product
             delay_product = delay_product.lower()

             if delay_product == 'comb':
                 delay_product = 'combined'

             # The 'hydro' term is used by radier package for the dry component,
             # and the delay product name is changed to 'dry' to be same with the SET name
             if delay_product == 'hydro':
                 delay_product = 'dry'

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
             output_product_name = f'{delay_product}TroposphericPhaseScreen'

             # If there is no troposphere delay product, then createa new one
             if output_product_name not in radar_grid:
                 h5_prep._create_datasets(radar_grid, [0], np.float64,
                                          output_product_name, descr = descr,
                                          units='radians',
                                          data=product_cube.astype(np.float64))

             # If there exists the product, overwrite the old one
             else:
                 tropo_delay = radar_grid[output_product_name]
                 tropo_delay[:] = product_cube.astype(np.float64)

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

