#!/usr/bin/env python3
import datetime
import multiprocessing
import re
import time

import isce3
import journal
import numpy as np
import pandas as pd
from isce3.core import transform_xy_to_latlon
from isce3.io import HDF5OptimizedReader
from nisar.products.insar.product_paths import GUNWGroupsPaths
from nisar.workflows.h5_prep import get_products_and_paths
from nisar.workflows.solid_earth_tides_runconfig import \
    InsarSolidEarthTidesRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse
from pysolid.solid import solid_grid


def solid_grid_pixel_rounded_to_nearest_sec(timestamp:  np.datetime64,
                                            lon: float,
                                            lat: float):
    """
    Calcaute the nearest the solid grid components

    Parameters
    ----------
    timestamp :  np.datetime64,
        datacube datetime
    lon : float
        longitude in degrees
    lat : float
        latitude in degrees

    Returns
    -------
    np.ndarray
        the interpolated solid earth tides in meters
        along the east, north, and up directions
    """

    tide_e, tide_n, tide_u = solid_grid(timestamp.year,
                                        timestamp.month,
                                        timestamp.day,
                                        timestamp.hour,
                                        timestamp.minute,
                                        timestamp.second,
                                        lat, -0.1, 1,
                                        lon, 0.1, 1)

    # The output is a 1 x 1 matrix and we want the floats within them
    return np.array([tide_e[0, 0], tide_n[0, 0], tide_u[0, 0]])

def interpolate_solid_grid_pixel(ref_epoch: np.datetime64,
                                 az_time: float,
                                 slant_range : float,
                                 lon: float,
                                 lat: float):
    """
    Interpolate the solid grid to account for the decimal seconds

    Parameters
    ----------
    ref_epoch : np.datetime64
        reference epoch
    az_time : float
        azimuth zero doppler time in seconds
    slant_range : float
        slant range in meters
    lon : float
        longitude in degrees
    lat : float
        latitude in degrees

    Returns
    -------
    interpolated_se_tides : np.ndarray
        the interpolated solid earth tides in meters
        along the east, north, and up directions respetively
    """

    if np.any(np.isnan(np.array([az_time,
                                 slant_range, lon, lat]))):
        return np.array([np.nan] * 3)

    # The one-way slant range time is added
    dt = ref_epoch + datetime.timedelta(
        seconds=az_time + slant_range / isce3.core.speed_of_light)

    dt_sec_floor = dt.floor('S')
    dt_sec_ceil = dt.ceil('S')

    seconds_diff_low = (dt - dt_sec_floor).total_seconds()
    seconds_diff_high = (dt_sec_ceil - dt).total_seconds()

    for diff_time in [seconds_diff_low, seconds_diff_high]:
        if (diff_time < 0) or (diff_time >= 1):
            raise ValueError('The truncated times were invalid; should be in [0, 1)')

    se_tides_low = \
        solid_grid_pixel_rounded_to_nearest_sec(dt_sec_floor, lon, lat)
    if abs(seconds_diff_low) < 1e-9:
        # the case when the ceiling and floor have the same result,
        # for example, the ceiling and floor of the 2018-09-28T04:04:44
        # are the same, and the interpolation result will be zero.
        interpolated_se_tides = se_tides_low
    else:
        se_tides_high = \
            solid_grid_pixel_rounded_to_nearest_sec(dt_sec_ceil, lon, lat)
        interpolated_se_tides = \
            se_tides_low * seconds_diff_high + se_tides_high * seconds_diff_low

    return interpolated_se_tides

def add_solid_earth_to_gunw_hdf5(los_solid_earth_tides,
                                 gunw_hdf5):
    '''
    Add the solid earth phase datacube to GUNW product

    Parameters
    ----------
    los_solid_earth_tides: np.ndarray
        solid earth tides along los direction
    gunw_hdf5: str
         GUNW HDF5 file where SET will be written
    '''

    with HDF5OptimizedReader(name=gunw_hdf5, mode='a', libver='latest', swmr=True) as hdf:
        radar_grid = hdf.get(GUNWGroupsPaths().RadarGridPath)
        product_names = ['slantRangeSolidEarthTidesPhase']

        for product_name, solid_earth_tides_product in zip(product_names,
                                                           los_solid_earth_tides):
            radar_grid[product_name][...] = solid_earth_tides_product


def solid_grid_pixel_parallel_task(args):
    """
    Parallel procssing task wrapper for the solid earth tides computation

    Parameters
    ----------
    args : tuple
        the zipped parameters incuding reference epoch,
        azimuth time, slant range, longitude, and latitude

    Returns
    -------
    interpolated_se_tides : np.ndarray
        the interpolated solid earth tides in meters
        along the east, north, and up directions respetively
    """

    # unzip the parameters
    ref_epoch, azi_time, slant_range, lon, lat = args

    return interpolate_solid_grid_pixel(ref_epoch,
                                        azi_time,
                                        slant_range,
                                        lon,
                                        lat)

def calculate_solid_earth_tides(ref_epoch : np.datetime64,
                                azimuth_time_datacube : np.ndarray,
                                slant_range_datacube : np.ndarray,
                                height_datacube : np.ndarray,
                                latitude_datacube : np.ndarray,
                                longitude_datacube :np.ndarray):

    """
    Calculate the solid earth tides components in east, north, and up directions

    Parameters
    ----------
    ref_epoch : np.datetime64
        reference epoch
    azimuth_time_datacube : numpy.ndarray
        azimuth time datacube
    slant_range_datacube : numpy.ndarray
        slant range datacube
    height_datacube: numpy.ndarray
        heights datacube
    latitude_datacube: numpy.ndarray
        latitude datacube
    longitude_datacube : numpy.ndarray
        longitude datacube

    Returns
    -------
    tide_e: np.ndarray
        solid earth tides datacube in meters along the east
    tide_n: np.ndarray
        solid earth tides datacube in meters along the north
    tide_u: np.ndarray
        solid earth tides datacube in meters along the up
    """

    # Zip the parameters
    input_data = zip(np.array([ref_epoch] * azimuth_time_datacube.size),
                     azimuth_time_datacube.ravel(),
                     slant_range_datacube.ravel(),
                     longitude_datacube.ravel(),
                     latitude_datacube.ravel())

    # Using the multiprocessing to speed up the process
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(pool.map(solid_grid_pixel_parallel_task,
                                input_data))

    datacube_shape = height_datacube.shape
    tide_e, tide_n, tide_u = (np.array(arr).reshape(datacube_shape)
                              for arr in zip(*results))

    return tide_e, tide_n, tide_u


def _extract_params_from_gunw_hdf5(gunw_hdf5_path: str):

    err_channel = journal.error(
        "solid_earth_tides._extract_params_from_gunw_hdf5")

    # Instantiate GUNW object to avoid hard-coded paths to GUNW datasets
    gunw_obj = GUNWGroupsPaths()
    with HDF5OptimizedReader(name=gunw_hdf5_path, mode='r', libver='latest', swmr=True) as h5_obj:
        # Fetch the GUWN Incidence Angle Datacube
        rdr_grid_path = gunw_obj.RadarGridPath
        [inc_angle_cube,
         los_unit_vector_x_cube,
         los_unit_vector_y_cube,
         xcoord_radar_grid,
         ycoord_radar_grid,
         height_radar_grid,
         ref_zero_doppler_azimuth_time,
         ref_slant_range,
         sec_zero_doppler_azimuth_time,
         sec_slant_range] =[h5_obj[f'{rdr_grid_path}/{item}'][()]
                                   for item in ['incidenceAngle',
                                                'losUnitVectorX', 'losUnitVectorY',
                                                'xCoordinates', 'yCoordinates',
                                                'heightAboveEllipsoid',
                                                'referenceZeroDopplerAzimuthTime',
                                                'referenceSlantRange',
                                                'secondaryZeroDopplerAzimuthTime',
                                                'secondarySlantRange']]
        # Extract the reference epochs
        def _extract_ref_epoch(ds_name: str):
            pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d+'
            input_string = str(h5_obj[f'{rdr_grid_path}/{ds_name}'].attrs['units'])
            match = re.search(pattern, input_string)

            # If match is None, search using a different pattern
            if match is None:
                pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
                match = re.search(pattern, input_string)
            if match is None:
                err_msg = 'No reference epoch is found'
                err_channel.log(err_msg)
                raise RuntimeError(err_msg)

            return pd.to_datetime(match.group(0))

        ref_ref_epoch = _extract_ref_epoch('referenceZeroDopplerAzimuthTime')
        sec_ref_repch = _extract_ref_epoch('secondaryZeroDopplerAzimuthTime')

        projection_dataset = h5_obj[f'{rdr_grid_path}/projection']
        epsg = projection_dataset.attrs['epsg_code']

         # Wavelength in meters
        wavelength = isce3.core.speed_of_light / \
                h5_obj[f'{gunw_obj.GridsPath}/frequencyA/centerFrequency'][()]

        return (inc_angle_cube,
                los_unit_vector_x_cube,
                los_unit_vector_y_cube,
                xcoord_radar_grid,
                ycoord_radar_grid,
                height_radar_grid,
                ref_ref_epoch,
                ref_zero_doppler_azimuth_time,
                ref_slant_range,
                sec_ref_repch,
                sec_zero_doppler_azimuth_time,
                sec_slant_range,
                epsg,
                wavelength)

def compute_solid_earth_tides(gunw_hdf5_path: str):
    '''
    Compute the LOS solid earth tides datacube between the reference
    and secondary RSLC

    Parameters
    ----------
    gunw_hdf5_path: str
        path to NISAR GUNW hdf5 file

    Returns
    ----------
    solid_earth_tides: np.ndarray
        solid earth tides along the LOS direction
    '''
    # Extract the HDF5 parameters
    (inc_angle_cube,
     los_unit_vector_x_cube,
     los_unit_vector_y_cube,
     xcoord_radar_grid,
     ycoord_radar_grid,
     height_radar_grid,
     ref_ref_epoch,
     ref_zero_doppler_azimuth_time,
     ref_slant_range,
     sec_ref_epoch,
     sec_zero_doppler_azimuth_time,
     sec_slant_range,
     epsg,
     wavelength) = _extract_params_from_gunw_hdf5(gunw_hdf5_path)

    x_2d, y_2d = \
        np.meshgrid(xcoord_radar_grid,
                    ycoord_radar_grid,
                    indexing='xy')

    # Lat/lon coordinates in 2D dimension
    lat_2d, lon_2d, _ = transform_xy_to_latlon(
        int(epsg), x_2d, y_2d)

    # tile LLH to match radar grid
    tile_dims = (len(height_radar_grid), 1, 1)
    latitude_mesh_arr = np.tile(lat_2d, tile_dims)
    longitude_mesh_arr = np.tile(lon_2d, tile_dims)
    height_mesh_arr = height_radar_grid[:, None, None] * \
        np.tile(np.ones(lon_2d.shape), tile_dims)

    # Caculate the solid earth tides for the reference RSLC
    ref_tide_e, ref_tide_n, ref_tide_u = \
        calculate_solid_earth_tides(ref_ref_epoch,
                                    ref_zero_doppler_azimuth_time,
                                    ref_slant_range,
                                    height_mesh_arr,
                                    latitude_mesh_arr,
                                    longitude_mesh_arr)

    # Caculate the solid earth tides for the secondary RSLC
    sec_tide_e, sec_tide_n, sec_tide_u = \
        calculate_solid_earth_tides(sec_ref_epoch,
                                    sec_zero_doppler_azimuth_time,
                                    sec_slant_range,
                                    height_mesh_arr,
                                    latitude_mesh_arr,
                                    longitude_mesh_arr)

    # Azimuth angle, the minus sign is because of the anti-clockwise positive definition
    azimuth_angle = -np.arctan2(los_unit_vector_x_cube, los_unit_vector_y_cube)

    # Incidence angle in radians
    inc_angle = np.deg2rad(inc_angle_cube)

    # Differences between the reference and secondary RSLC
    tide_e = ref_tide_e - sec_tide_e
    tide_n = ref_tide_n - sec_tide_n
    tide_u = ref_tide_u - sec_tide_u

    # Solidearth tides datacube along the LOS in meters
    los_solid_earth_tides_datacube =(-tide_e * np.sin(inc_angle) * np.sin(azimuth_angle)
                                     + tide_n * np.sin(inc_angle) * np.cos(azimuth_angle)
                                     + tide_u  * np.cos(inc_angle))

    # Convert to phase screen in radians
    los_solid_earth_tides_datacube *= -4.0 * np.pi / wavelength

    return los_solid_earth_tides_datacube


def run(cfg: dict, gunw_hdf5_path: str):
    '''
    compute the solid earth tides and write to GUNW product

    Parameters
    ----------
    cfg: dict
        runconfig dictionary
    gunw_hdf5_path: str
        path to GUNW HDF5 file
    '''

    # Create info channels
    info_channel = journal.info("solid_earth_tides.run")
    info_channel.log("starting solid earth tides computation")

    t_all = time.time()

    # Compute the solid earth tides along los direction
    los_solid_earth_tides = compute_solid_earth_tides(gunw_hdf5_path)

    # Write the solid earth tides to GUNW product
    add_solid_earth_to_gunw_hdf5(los_solid_earth_tides,
                                 gunw_hdf5_path)

    t_all_elapsed = time.time() - t_all
    info_channel.log(
        f"successfully ran solid earth tides in {t_all_elapsed:.3f} seconds")

if __name__ == "__main__":

    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    solidearth_tides_runcfg = InsarSolidEarthTidesRunConfig(args)
    _, out_paths = get_products_and_paths(solidearth_tides_runcfg.cfg)
    run(solidearth_tides_runcfg.cfg, gunw_hdf5_path=out_paths['GUNW'])
