#!/usr/bin/env python3
from datetime import datetime
import journal
import numpy as np
import os
import time

import pyaps3 as pa

import isce3
from isce3.core import transform_xy_to_latlon
from isce3.io import HDF5OptimizedReader
from nisar.workflows import h5_prep
from nisar.workflows.troposphere_runconfig import InsarTroposphereRunConfig
from nisar.products.insar.product_paths import GUNWGroupsPaths
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

    # Instantiate GUNW product object to avoid hard-coded paths to GUNW datasets
    gunw_obj = GUNWGroupsPaths()

    # Fetch the configurations
    tropo_weather_model_cfg = cfg['dynamic_ancillary_file_group']\
        ['troposphere_weather_model_files']
    tropo_cfg = cfg['processing']['troposphere_delay']

    scratch_path = cfg['product_path_group']['scratch_path']

    weather_model_type = tropo_cfg['weather_model_type'].upper()
    reference_weather_model_file = \
        tropo_weather_model_cfg['reference_troposphere_file']
    secondary_weather_model_file = \
        tropo_weather_model_cfg['secondary_troposphere_file']

    tropo_package = tropo_cfg['package'].lower()
    tropo_delay_direction = tropo_cfg['delay_direction'].lower()

    tropo_delay_products = []
    # comb is short for the summation of wet and dry components
    for delay_type in ['wet', 'hydrostatic', 'comb']:
        if tropo_cfg[f'enable_{delay_type}_product']:
            if (delay_type == 'hydrostatic') and \
                    (tropo_package == 'raider'):
                delay_type = 'hydro'
            if (delay_type == 'hydrostatic') and \
                    (tropo_package == 'pyaps'):
                delay_type = 'dry'

            tropo_delay_products.append(delay_type)

    # Troposphere delay datacube
    troposphere_delay_datacube = dict()

    with HDF5OptimizedReader(name=gunw_hdf5, mode='r', libver='latest', swmr=True) as h5_obj:

        # Fetch the GUWN Incidence Angle Datacube
        rdr_grid_path = gunw_obj.RadarGridPath

        inc_angle_cube = h5_obj[f'{rdr_grid_path}/incidenceAngle'][()]
        xcoord_radar_grid = h5_obj[f'{rdr_grid_path}/xCoordinates'][()]
        ycoord_radar_grid = h5_obj[f'{rdr_grid_path}/yCoordinates'][()]
        height_radar_grid = h5_obj[f'{rdr_grid_path}/heightAboveEllipsoid'][()]

        # EPSG code
        epsg = int(h5_obj[f'{rdr_grid_path}/projection'].attrs['epsg_code'])

        # Wavelenth in meters
        wavelength = isce3.core.speed_of_light / \
                h5_obj[f'{gunw_obj.GridsPath}/frequencyA/centerFrequency'][()]

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
                            -(phs_ref - phs_second) * 4.0 * np.pi / wavelength)

                # Tropo delay datacube
                tropo_delay_datacube = np.stack(tropo_delay_datacube_list)
                tropo_delay_datacube_list = None

                if tropo_delay_direction == 'line_of_sight_mapping':
                    tropo_delay_datacube /= np.cos(np.deg2rad(inc_angle_cube))

                # Save to the dictionary in memory
                tropo_delay_product_name = \
                    f'tropoDelay_{tropo_package}_{tropo_delay_direction}_{tropo_delay_product}'
                troposphere_delay_datacube[tropo_delay_product_name]  = tropo_delay_datacube

        # raider package
        else:
            import xarray as xr
            from RAiDER.llreader import BoundingBox
            from RAiDER.losreader import Zenith, Raytracing
            from RAiDER.delay import tropo_delay as raider_tropo_delay
            from RAiDER.models.hres import HRES

            def _convert_HRES_to_raider_NetCDF(weather_model_file,
                                              lat_lon_bounds,
                                              weather_model_output_dir):
                '''
                Internal convenience function to convert the ECMWF NetCDF to RAiDER NetCDF

                Parameters
                ----------
                 weather_model_file: str
                    HRES NetCDF weather model file
                 lat_lon_bounds: list
                     bounding box of the RSLC
                 weather_model_output_dir: str
                     the output directory of the RAiDER internal NetCDF file
                Returns
                -------
                     the path of the RAiDER internal NetCDF file
                 '''

                os.makedirs(weather_model_output_dir, exist_ok=True)
                ds = xr.open_dataset(weather_model_file)

                # Get the datetime of the weather model file
                weather_model_time = \
                    ds.time.values.astype('datetime64[s]').astype(datetime)[0]
                hres = HRES()
                # Set up the time, Lat/Lon, and working location, where
                # the lat/lon bounds are applied to clip the global
                # weather model to minimize the data processing
                hres.setTime(weather_model_time)
                hres.set_latlon_bounds(ll_bounds = lat_lon_bounds)
                hres.set_wmLoc(weather_model_output_dir)

                # Load the ECMWF NetCDF weather model
                hres.load_weather(weather_model_file)

                # Process the weather model data
                hres._find_e()
                hres._uniform_in_z(_zlevels=None)

                # This function implemented in the RAiDER
                # fills the NaNs with 0
                hres._checkForNans()

                hres._get_wet_refractivity()
                hres._get_hydro_refractivity()
                hres._adjust_grid(hres.get_latlon_bounds())

                # Compute Zenith delays at the weather model grid nodes
                hres._getZTD()

                output_file = hres.out_file(weather_model_output_dir)
                hres._out_name =  output_file

                # Return the ouput file if it exists
                if os.path.exists(output_file):
                    return output_file
                else:
                    # Write to hard drive
                    return hres.write()

            # ouput location
            weather_model_output_dir = \
                os.path.join(scratch_path, 'weather_model_files')

            # Acquisition time for reference and secondary images

            acquisition_time_ref = h5_obj[f'{gunw_obj.IdentificationPath}/referenceZeroDopplerStartTime'][()]\
                    .astype('datetime64[s]').astype(datetime)
            acquisition_time_second = h5_obj[f'{gunw_obj.IdentificationPath}/secondaryZeroDopplerStartTime'][()]\
                    .astype('datetime64[s]').astype(datetime)

            # AOI bounding box
            margin = 0.1
            min_lat = np.min(lat_datacube)
            max_lat = np.max(lat_datacube)
            min_lon = np.min(lon_datacube)
            max_lon = np.max(lon_datacube)

            lat_lon_bounds = [min_lat - margin,
                              max_lat + margin,
                              min_lon - margin,
                              max_lon + margin]

            aoi = BoundingBox(lat_lon_bounds)
            aoi.xpts = xcoord_radar_grid
            aoi.ypts = ycoord_radar_grid

            # Zenith
            delay_direction_obj = Zenith()

            if tropo_delay_direction == 'line_of_sight_raytracing':
                delay_direction_obj = Raytracing()

            # Height levels
            height_levels = list(height_radar_grid)

            # If the input weather model is HRES,
            # convert it to the RAiDER internal NetCDF
            if weather_model_type == 'HRES':
                reference_weather_model_file = \
                    _convert_HRES_to_raider_NetCDF(reference_weather_model_file,
                                                   lat_lon_bounds, weather_model_output_dir)

                secondary_weather_model_file = \
                    _convert_HRES_to_raider_NetCDF(secondary_weather_model_file,
                                                   lat_lon_bounds, weather_model_output_dir)

            # Troposphere delay datacube computation
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
                    tropo_delay = \
                        tropo_delay_reference['wet'] + tropo_delay_reference['hydro'] - \
                            tropo_delay_secondary['wet'] - tropo_delay_secondary['hydro']
                else:
                    tropo_delay = tropo_delay_reference[tropo_delay_product] - \
                            tropo_delay_secondary[tropo_delay_product]

                # Convert it to radians units
                tropo_delay_datacube = -tropo_delay * 4.0 * np.pi / wavelength

                # Line of sight mapping
                if tropo_delay_direction == 'line_of_sight_mapping':
                    tropo_delay_datacube /= np.cos(np.deg2rad(inc_angle_cube))

                # Save to the dictionary in memory
                tropo_delay_product_name = \
                    f'tropoDelay_{tropo_package}_{tropo_delay_direction}_{tropo_delay_product}'
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
    # Instantiate GUNW object to avoid hard-coded path to GUNW datasets
    gunw_obj = GUNWGroupsPaths()
    with HDF5OptimizedReader(name=gunw_hdf5, mode='a', libver='latest', swmr=True) as f:

        for product_name, product_cube in tropo_delay_datacubes.items():

             radar_grid = f.get(gunw_obj.RadarGridPath)

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
             # The 'dry' term is used by the pyaps package for the dry comopnent
             # NISAR uses 'hydrostatic' to describe the dry component
             if delay_product in ['hydro', 'dry']:
                 delay_product = 'hydrostatic'

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
             descr = f"{delay_product.capitalize()} component of the troposphere phase screen"

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
