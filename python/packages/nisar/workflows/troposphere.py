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

    if epsg != 4326:
        # Transform the xy to lat/lon
        transformer_xy_to_latlon = osr.CoordinateTransformation(srs_src, srs_wgs84)

        # Stack the x and y
        x_y_pnts_radar = np.stack((x.flatten(), y.flatten()), axis=-1)

        # Transform to lat/lon
        lat_lon_radar = np.array(
            transformer_xy_to_latlon.TransformPoints(x_y_pnts_radar))

        # Lat lon of data cube
        lat_datacube = lat_lon_radar[:, 0].reshape(x.shape)
        lon_datacube = lat_lon_radar[:, 1].reshape(x.shape)
    else:
        lat_datacube = y.copy()
        lon_datacube = x.copy()

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

    error_channel = journal.error('troposphere.compute_troposphere_delay')

    # Fetch the configurations
    tropo_weather_model_cfg = cfg['dynamic_ancillary_file_group']['troposphere_weather_model']
    tropo_cfg = cfg['processing']['troposphere_delay']

    weather_model_type = tropo_cfg['weather_model_type'].upper()
    reference_weather_model_file = tropo_weather_model_cfg['reference_troposphere_file']
    secondary_weather_model_file = tropo_weather_model_cfg['secondary_troposphere_file']

    tropo_package = tropo_cfg['package'].lower()
    tropo_delay_direction = tropo_cfg['delay_direction'].lower()
    tropo_delay_products = tropo_cfg['delay_product']

    # Troposphere delay datacube
    troposphere_delay_datacube = dict()

    with h5py.File(gunw_hdf5, 'r', libver='latest', swmr=True) as f:

        # Fetch the GUWN Incidence Angle Datacube
        rdr_grid_path = 'science/LSAR/GUNW/metadata/radarGrid'

        inc_angle_cube = f[f'{rdr_grid_path}/incidenceAngle'][()]
        xcoord_radar_grid = f[f'{rdr_grid_path}/xCoordinates'][()]
        ycoord_radar_grid = f[f'{rdr_grid_path}/yCoordinates'][()]
        height_radar_grid = f[f'{rdr_grid_path}/heightAboveEllipsoid'][()]

        # EPSG code
        epsg = int(f['science/LSAR/GUNW/metadata/radarGrid/epsg'][()])

        # Wavelenth in meters
        wavelength = isce3.core.speed_of_light / \
                f['/science/LSAR/GUNW/grids/frequencyA/centerFrequency'][()]

        # X and y for the entire datacube
        y_2d_radar = np.tile(ycoord_radar_grid, (len(xcoord_radar_grid), 1)).T
        x_2d_radar = np.tile(xcoord_radar_grid, (len(ycoord_radar_grid), 1))

        # Lat/lon coordinates
        lat_datacube, lon_datacube, _ = transform_xy_to_latlon(
            epsg, x_2d_radar, y_2d_radar)

        # pyaps package
        if tropo_package == 'pyaps':

            for tropo_delay_product in tropo_delay_products:

                delay_type = 'dry' if tropo_delay_product == 'hydro' else tropo_delay_product

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
                                                 model=weather_model_type,
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
                                                    model=weather_model_type,
                                                    verb=False,
                                                    Del=delay_type)

                    phs_second = second_aps_estimator.getdelay()

                    # Convert the delay in meters to radians
                    tropo_delay_datacube_list.append(
                        (phs_ref - phs_second) *4.0*np.pi/wavelength)

                # Tropo delay datacube
                tropo_delay_datacube = np.stack(tropo_delay_datacube_list)
                tropo_delay_datacube_list = None

                # Save to the dictionary in memory
                tropo_delay_product_name = f'tropoDelay_{tropo_package}_{tropo_delay_direction}_{delay_type}'
                troposphere_delay_datacube[tropo_delay_product_name]  = tropo_delay_datacube

        # raider package
        else:

            # Acquisition time for reference and secondary images
            acquisition_time_ref = f['science/LSAR/identification/referenceZeroDopplerStartTime'][()]\
                    .astype('datetime64[s]').astype(datetime)
            acquisition_time_second = f['science/LSAR/identification/secondaryZeroDopplerStartTime'][()]\
                    .astype('datetime64[s]').astype(datetime)

            x_cube_spacing = (
                max(xcoord_radar_grid) - min(xcoord_radar_grid))/(len(xcoord_radar_grid)-1)
            y_cube_spacing = (
                max(ycoord_radar_grid) - min(ycoord_radar_grid))/(len(ycoord_radar_grid)-1)

            # Cube spacing using the minimum spacing
            cube_spacing = min(x_cube_spacing, y_cube_spacing)

            # To speed up the interpolation, the cube spacing is set >= 5km
            if epsg != 4326:
                cube_spacing = 5000 if cube_spacing < 5000 else cube_spacing
            else:
                cube_spacing = 0.05 if cube_spacing < 0.05 else cube_spacing

            # AOI bounding box
            min_lat = np.min(lat_datacube)
            max_lat = np.max(lat_datacube)
            min_lon = np.min(lon_datacube)
            max_lon = np.max(lon_datacube)

            aoi = BoundingBox([min_lat, max_lat, min_lon, max_lon])

            # Default of line of sight is zenith
            los = Zenith()

            if tropo_delay_direction == 'line_of_sight_raytracing':
                los = Raytracing()

            # Height levels
            height_levels = list(height_radar_grid)

            # Tropodelay computation
            tropo_delay_reference, _ = raider_tropo_delay(dt=acquisition_time_ref,
                                                          weather_model_file=reference_weather_model_file,
                                                          aoi=aoi,
                                                          los=los,
                                                          height_levels=height_levels,
                                                          out_proj=epsg,
                                                          cube_spacing_m=cube_spacing)

            tropo_delay_secondary, _ = raider_tropo_delay(dt=acquisition_time_second,
                                                          weather_model_file=secondary_weather_model_file,
                                                          aoi=aoi,
                                                          los=los,
                                                          height_levels=height_levels,
                                                          out_proj=epsg,
                                                          cube_spacing_m=cube_spacing)


            for tropo_delay_product in tropo_delay_products:

                # Troposphere delay by raider package
                if tropo_delay_product == 'comb':
                    tropo_delay = tropo_delay_reference['wet'] + tropo_delay_reference['hydro'] - \
                            tropo_delay_secondary['wet'] - tropo_delay_secondary['hydro']
                else:
                    tropo_delay = tropo_delay_reference[tropo_delay_product] - \
                            tropo_delay_secondary[tropo_delay_product]

                # Convert it to radians units
                tropo_delay_datacube = tropo_delay*4.0*np.pi/wavelength

                # Interpolate to radar grid to keep its dimension consistent with other datacubes
                tropo_delay_interpolator = RegularGridInterpolator((tropo_delay_reference.z,
                                                                    tropo_delay_reference.y,
                                                                    tropo_delay_reference.x),
                                                                   tropo_delay_datacube,
                                                                   method='linear')

                delay_type = 'dry' if tropo_delay_product == 'hydro' else tropo_delay_product

                # Interoplate the troposhphere delay
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
                tropo_delay_product_name = f'tropoDelay_{tropo_package}_{tropo_delay_direction}_{delay_type}'
                troposphere_delay_datacube[tropo_delay_product_name]  = tropo_delay_datacube


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
             product_name = f'{delay_product}TroposphericPhaseScreen'

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

