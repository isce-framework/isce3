#!/usr/bin/env python3
from datetime import datetime
import h5py
import journal
import numpy as np
import os
from osgeo import gdal, osr
from scipy.interpolate import RegularGridInterpolator

import time
import pysolid

import isce3
from nisar.workflows.h5_prep import add_solid_earth_to_gunw_hdf5
from nisar.workflows.h5_prep import get_products_and_paths
from nisar.workflows.solid_earth_tides_runconfig import InsarSolidEarthTidesRunConfig
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
    srs_src.ImportFromEPSG(int(epsg))

    srs_wgs84 = osr.SpatialReference()
    srs_wgs84.ImportFromEPSG(4326)

    # Transform to EPSG:4326 lat/lon
    if int(epsg) != 4326:

        # Transformer
        xy_to_latlon_transform_obj = osr.CoordinateTransformation(srs_src, srs_wgs84)

        # Stack the x and y
        x_y_pnts_radar = np.stack((x.flatten(), y.flatten()), axis=-1)

        # Transform to lat/lon
        lat_lon_radar = np.array(
                xy_to_latlon_transform_obj.TransformPoints(x_y_pnts_radar))

        # Lat lon of data cube
        lat_datacube = lat_lon_radar[:, 0].reshape(x.shape)
        lon_datacube = lat_lon_radar[:, 1].reshape(x.shape)
    else:
        lat_datacube = y.copy()
        lon_datacube = x.copy()

    # 0.1 degrees is added  to make sure the weather model cover the entire image
    margin = 0.1

    # Extent of the data cube
    cube_extent = (np.nanmin(lat_datacube) - margin, np.nanmax(lat_datacube) + margin,
                   np.nanmin(lon_datacube) - margin, np.nanmax(lon_datacube) + margin)

    return lat_datacube, lon_datacube, cube_extent


def cal_solid_earth_tides(inc_angle_datacube,
                          los_unit_vector_x_datacube,
                          los_unit_vector_y_datacube,
                          xcoord_of_datacube,
                          ycoord_of_datacube,
                          epsg,
                          wavelength,
                          reference_start_time,
                          secondary_start_time):

    '''
    calculate the solid earth tides components along LOS, east, north, and up directions

    Parameters
     ----------
     inc_angle_datacube: numpy.ndarray
        incidence angle datacube in degrees
     los_unit_vector_x_datacube: numpy.ndarray
        unit vector X datacube in ENU projection
     los_unit_vector_y_datacube: numpy.ndarray
        unit vector y datacube in ENU projection
     xcoord_of_datacube: numpy.ndarray
        xcoordinates of datacube
     ycoord_of_datacube: numpy.ndarray
        ycoordinates of datacube
     epsg: int
     EPSG code of the datacube
     wavelength: float
        radar wavelength in meters
     reference_start_time: datetime.datetime
       start time of the reference image
     secondary_start_time: datetime.datetime
       start time of the secondary image

    Returns
     -------
     solid_earth_tides: tuple
        solid earth tides along the los, easth, north, and up directions
    '''

    # X and y for the entire datacube
    y_2d_radar = np.tile(ycoord_of_datacube, (len(xcoord_of_datacube), 1)).T
    x_2d_radar = np.tile(xcoord_of_datacube, (len(ycoord_of_datacube), 1))

    # Lat/lon coordinates
    lat_datacube, lon_datacube, cube_extents = transform_xy_to_latlon(
        epsg, x_2d_radar, y_2d_radar)

    # Datacube size
    cube_y_size, cube_x_size = lat_datacube.shape

    # Configurations for pySolid
    y_end, y_first, x_first, x_end = cube_extents

    y_step = np.max(
        lat_datacube[0, :] - lat_datacube[cube_y_size-1, :]) / (cube_y_size - 1)
    x_step = np.max(lon_datacube[:, cube_x_size-1] -
                    lon_datacube[:, 0]) / (cube_x_size - 1)

    # Fix the step size around 10km if the spacing of the datacube is less than 10km
    x_step = 0.1 if x_step < 0.1 else x_step
    y_step = 0.1 if y_step < 0.1 else y_step

    # Get dimensions of earth tides grid
    width = int((y_first - y_end) / y_step + 1)
    length = int((x_end - x_first) / x_step + 1)

    # Recalculate the steps
    x_samples, x_step = np.linspace(x_first, x_end, num=length, retstep=True)
    y_samples, y_step = np.linspace(y_first, y_end, num=width,  retstep=True)

    # Parameters for pySolid
    params = {'LENGTH': length,
              'WIDTH': width,
              'X_FIRST': x_first,
              'Y_FIRST': y_first,
              'X_STEP': x_step,
              'Y_STEP': y_step}

    # Solid earth tides for both reference and secondary dates
    ref_tide_e, ref_tide_n, ref_tide_u = pysolid.calc_solid_earth_tides_grid(reference_start_time,
                                                                             params,
                                                                             step_size = 1000,
                                                                             display=False,
                                                                             verbose=True)

    sec_tide_e, sec_tide_n, sec_tide_u = pysolid.calc_solid_earth_tides_grid(secondary_start_time,
                                                                             params,
                                                                             step_size = 1000,
                                                                             display=False,
                                                                             verbose=True)


    # Points
    pnts = np.stack(
        (lon_datacube.flatten(), lat_datacube.flatten()), axis=-1)

    y_samples = np.flip(y_samples)
    xy_samples = (x_samples, y_samples)
    cube_shape = lat_datacube.shape

    # Interpolation, the flip function applied here is to fit the scipy==1.8
    # which requires strict ascending or descending
    # (ref - sec) tide east
    ref_sec_tide_e_interp = RegularGridInterpolator(xy_samples,
                                                    np.flip(ref_tide_e - sec_tide_e,
                                                            axis=0))
    ref_sec_tide_e = ref_sec_tide_e_interp(pnts).reshape(cube_shape)

    # (ref - sec) tide north
    ref_sec_tide_n_interp = RegularGridInterpolator(xy_samples,
                                                    np.flip(ref_tide_n - sec_tide_n,
                                                            axis=0))
    ref_sec_tide_n = ref_sec_tide_n_interp(pnts).reshape(cube_shape)

    # (ref - sec) tide up
    ref_sec_tide_u_interp = RegularGridInterpolator(xy_samples,
                                                    np.flip(ref_tide_u - sec_tide_u,
                                                            axis=0))
    ref_sec_tide_u = ref_sec_tide_u_interp(pnts).reshape(cube_shape)


    # Azimuth angle, the minus sign is because of the anti-clockwise positive definition
    azimuth_angle = -np.arctan2(los_unit_vector_x_datacube, los_unit_vector_y_datacube)

    # Incidence angle in radians
    inc_angle = np.deg2rad(inc_angle_datacube)

    # Solidearth tides datacube along the LOS in meters
    los_solid_earth_tides_datacube =(-ref_sec_tide_e * np.sin(inc_angle) * np.sin(azimuth_angle)
                                     + ref_sec_tide_n * np.sin(inc_angle) * np.cos(azimuth_angle)
                                     + ref_sec_tide_u  * np.cos(inc_angle))

    # Convert to phase screen
    los_solid_earth_tides_datacube *= -4.0 * np.pi / wavelength

    return (los_solid_earth_tides_datacube,
            ref_sec_tide_e,
            ref_sec_tide_n,
            ref_sec_tide_u)


def compute_solid_earth_tides(gunw_hdf5: str):
    '''
    Compute the solid earth tides datacube along LOS

    Parameters
     ----------
     cfg: dict
        runconfig dictionary
     gunw_hdf5: str
        NISAR GUNW hdf5 file

    Returns
     -------
     solid_earth_tides: tuple
        solid earth tides along the los, easth, north, and up directions
    '''

    with h5py.File(gunw_hdf5, 'r', libver='latest', swmr=True) as f:

        # Fetch the GUWN Incidence Angle Datacube
        rdr_grid_path = 'science/LSAR/GUNW/metadata/radarGrid'

        inc_angle_cube = f[f'{rdr_grid_path}/incidenceAngle'][()]
        los_unit_vector_x_cube = f[f'{rdr_grid_path}/losUnitVectorX'][()]
        los_unit_vector_y_cube = f[f'{rdr_grid_path}/losUnitVectorY'][()]

        xcoord_radar_grid = f[f'{rdr_grid_path}/xCoordinates'][()]
        ycoord_radar_grid = f[f'{rdr_grid_path}/yCoordinates'][()]

        # EPSG code
        epsg = f['science/LSAR/GUNW/metadata/radarGrid/epsg'][()]

        # Wavelenth in meters
        wavelength = isce3.core.speed_of_light / \
                f['/science/LSAR/GUNW/grids/frequencyA/centerFrequency'][()]

        # Start time of the reference and secondary image
        ref_start_time = f['science/LSAR/identification/referenceZeroDopplerStartTime'][()]\
                .astype('datetime64[s]').astype(datetime)
        sec_start_time = f['science/LSAR/identification/secondaryZeroDopplerStartTime'][()]\
                .astype('datetime64[s]').astype(datetime)

        # Caculate the solid earth tides
        solid_earth_tides = cal_solid_earth_tides(inc_angle_cube,
                                                  los_unit_vector_x_cube,
                                                  los_unit_vector_y_cube,
                                                  xcoord_radar_grid,
                                                  ycoord_radar_grid,
                                                  int(epsg),
                                                  wavelength,
                                                  ref_start_time,
                                                  sec_start_time)

        f.close()

    return solid_earth_tides


def run(cfg: dict, gunw_hdf5: str):
    '''
    compute the solid earth tides and write to GUNW product

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
    info_channel = journal.info("solid_earth_tides.run")
    info_channel.log("starting solid earth tides computation")

    t_all = time.time()

    # Compute the solid earth tides along los, east, north, and up directions
    solid_earth_tides = compute_solid_earth_tides(gunw_hdf5)

    # Write the Solid Earth tides to GUNW product, where the los is a datacube
    # and east, north, and up_solid_earth_tides are in 2D grid
    add_solid_earth_to_gunw_hdf5(solid_earth_tides,
                                 gunw_hdf5)

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
    run(solidearth_tides_runcfg.cfg, gunw_hdf5=out_paths['GUNW'])
