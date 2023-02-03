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
from nisar.workflows import h5_prep
from nisar.workflows.solidearth_tides_runconfig import InsarSolidEarthTidesRunConfig
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


def compute_solidearth_tides(cfg: dict, gunw_hdf5: str):
    '''
    Compute the solidearth tides datacube along LOS

    Parameters
     ----------
     cfg: dict
        runconfig dictionary
     gunw_hdf5: str
        NISAR GUNW hdf5 file

    Returns
     -------
     solidearth_tides: np.ndarray
        solidearth tides datacube
    '''

    error_channel = journal.error('solidearth_tides.compute_solidearth_tides')

    # Fetch the configurations
    solidearth_tides_cfg = cfg['processing']['solidearth_tides']

   # Step size to speed up the calcuation
    step_size = solidearth_tides_cfg['step_size']

    with h5py.File(gunw_hdf5, 'r', libver='latest', swmr=True) as f:

        # Fetch the GUWN Incidence Angle Datacube
        rdr_grid_path = 'science/LSAR/GUNW/metadata/radarGrid'

        inc_angle_cube = np.array(f[f'{rdr_grid_path}/incidenceAngle'])
        los_unit_vector_x_cube = np.array(f[f'{rdr_grid_path}/losUnitVectorX'])
        los_unit_vector_y_cube = np.array(f[f'{rdr_grid_path}/losUnitVectorY'])

        xcoord_radar_grid = np.array(f[f'{rdr_grid_path}/xCoordinates'])
        ycoord_radar_grid = np.array(f[f'{rdr_grid_path}/yCoordinates'])
        height_radar_grid = np.array(
            f[f'{rdr_grid_path}/heightAboveEllipsoid'])

        # EPSG code
        epsg = int(np.array(f['science/LSAR/GUNW/metadata/radarGrid/epsg']))

        # Wavelenth in meters
        wavelength = isce3.core.speed_of_light / \
            float(
                np.array(f['/science/LSAR/GUNW/grids/frequencyA/centerFrequency']))

        # Start time of the reference and secondary image
        ref_start_time = str(np.array(
            f['science/LSAR/identification/referenceZeroDopplerStartTime']).astype(str))
        sec_start_time = str(np.array(
            f['science/LSAR/identification/secondaryZeroDopplerStartTime']).astype(str))

        # Make sure that timestamp format is 'YYYY-MM-DDTH:M:S' where the second has no decmials
        ref_start_time = ref_start_time[:19]
        sec_start_time = sec_start_time[:19]

        # reference and secondary start time
        ref_date = datetime.strptime(ref_start_time, '%Y-%m-%dT%H:%M:%S')
        sec_date = datetime.strptime(sec_start_time, '%Y-%m-%dT%H:%M:%S')

        # X and y for the entire datacube
        y_2d_radar = np.tile(ycoord_radar_grid, (len(xcoord_radar_grid), 1)).T
        x_2d_radar = np.tile(xcoord_radar_grid, (len(ycoord_radar_grid), 1))

        # Lat/lon coordinates
        lat_datacube, lon_datacube, _ = transform_xy_to_latlon(
            epsg, x_2d_radar, y_2d_radar)

        # Datacube size
        cube_y_size, cube_x_size = lat_datacube.shape

        # Configurations for pySolid
        # 0.1 degrees margin to make sure the datacube is entirely covered
        margin = 0.1

        x_first = np.min(lon_datacube) - margin
        y_first = np.max(lat_datacube) + margin

        x_end = np.max(lon_datacube) + margin
        y_end = np.min(lat_datacube) - margin

        y_step = np.max(
            lat_datacube[0, :] - lat_datacube[cube_y_size-1, :])/(cube_y_size - 1)
        x_step = np.max(lon_datacube[:, cube_x_size-1] -
                        lon_datacube[:, 0])/(cube_x_size - 1)

        width = int((y_first - y_end)/y_step + 1)
        length = int((x_end - x_first)/x_step + 1)

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
        ref_tide_e, ref_tide_n, ref_tide_u = pysolid.calc_solid_earth_tides_grid(ref_date,
                                                                                 params,
                                                                                 step_size=step_size,
                                                                                 display=False,
                                                                                 verbose=True)

        sec_tide_e, sec_tide_n, sec_tide_u = pysolid.calc_solid_earth_tides_grid(sec_date,
                                                                                 params,
                                                                                 step_size=step_size,
                                                                                 display=False,
                                                                                 verbose=True)

        # Points
        pnts = np.stack(
            (lon_datacube.flatten(), lat_datacube.flatten()), axis=-1)

        # Interpolation, the flip function is applied is to fit the scipy==1.8
        # which requires strict ascending or descending
        # ref tide east
        ref_tide_e_interp = RegularGridInterpolator((x_samples, np.flip(y_samples)),
                                                    np.flip(ref_tide_e, axis=0))
        ref_tide_e = ref_tide_e_interp(pnts).reshape(lat_datacube.shape)

        # ref tide north
        ref_tide_n_interp = RegularGridInterpolator((x_samples, np.flip(y_samples)),
                                                    np.flip(ref_tide_n, axis=0))
        ref_tide_n = ref_tide_n_interp(pnts).reshape(lat_datacube.shape)

        # ref tide up
        ref_tide_u_interp = RegularGridInterpolator((x_samples, np.flip(y_samples)),
                                                    np.flip(ref_tide_u, axis=0))
        ref_tide_u = ref_tide_u_interp(pnts).reshape(lat_datacube.shape)

        # sec tide east
        sec_tide_e_interp = RegularGridInterpolator((x_samples, np.flip(y_samples)),
                                                    np.flip(sec_tide_e, axis=0))
        sec_tide_e = sec_tide_e_interp(pnts).reshape(lat_datacube.shape)

        # sec tide north
        sec_tide_n_interp = RegularGridInterpolator((x_samples, np.flip(y_samples)),
                                                    np.flip(sec_tide_n, axis=0))
        sec_tide_n = sec_tide_n_interp(pnts).reshape(lat_datacube.shape)

        # sec tide up
        sec_tide_u_interp = RegularGridInterpolator((x_samples, np.flip(y_samples)),
                                                    np.flip(sec_tide_u, axis=0))
        sec_tide_u = sec_tide_u_interp(pnts).reshape(lat_datacube.shape)

        # Azimuth angle, the minus sign is because of the anti-clockwise positive defination
        azimuth_angle = -np.arctan2(los_unit_vector_x_cube, los_unit_vector_y_cube)

        inc_angle = np.deg2rad(inc_angle_cube)

        # Solidearth tides datacube along the LOS inn meters
        solidearth_tides_datacube =(-(ref_tide_e - sec_tide_e) * np.sin(inc_angle) * np.sin(azimuth_angle)
                                    + (ref_tide_n - sec_tide_n) * np.sin(inc_angle) * np.cos(azimuth_angle)
                                    + (ref_tide_u - sec_tide_u) * np.cos(inc_angle))

        # Convert to radians
        solidearth_tides_datacube *= 4.0*np.pi/wavelength

        f.close()

    return solidearth_tides_datacube


def write_to_GUNW_product(solidearth_tides_datacube: np.ndarray, gunw_hdf5: str):
    '''
    Write the solid earth tides datacube to GUNW product

    Parameters
     ----------
     solidearth_tides_datacube: np.ndarray
        solid earth tides datacube
      gunw_hdf5: str
         gunw hdf5 file

    Returns
     -------
       None
    '''

    with h5py.File(gunw_hdf5, 'a', libver='latest', swmr=True) as f:

        radar_grid = f.get('science/LSAR/GUNW/metadata/radarGrid')

        # Dataset description
        descr = f"Solid earth tides phase screen along line of sight"

        # Product name
        product_name = f'losSolidEarthTidesPhaseScreen'

        # If there is no troposphere delay product, then createa new one
        if product_name not in radar_grid:
            h5_prep._create_datasets(radar_grid, [0], np.float64,
                                     product_name, descr=descr,
                                     units='radians',
                                     data=solidearth_tides_datacube.astype(np.float64))

        # If there exists the product, overwrite the old one
        else:
            solidearth_tides = radar_grid[product_name]
            solidearth_tides[:] = solidearth_tides_datacube.astype(
                np.float64)

        f.close()


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
    info_channel = journal.info("solidearth_tides.run")
    info_channel.log("starting solid earth tides computation")

    t_all = time.time()

    # Compute the solid earth tides  datacube
    solidearth_tides_datacube = compute_solidearth_tides(cfg, gunw_hdf5)

    # Write to GUNW product
    write_to_GUNW_product(solidearth_tides_datacube, gunw_hdf5)

    t_all_elapsed = time.time() - t_all
    info_channel.log(
        f"successfully ran solid earth tides in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":

    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    solidearth_tides_runcfg = InsarSolidEarthTidesRunConfig(args)
    _, out_paths = h5_prep.get_products_and_paths(solidearth_tides_runcfg.cfg)
    run(solidearth_tides_runcfg.cfg, gunw_hdf5=out_paths['GUNW'])
