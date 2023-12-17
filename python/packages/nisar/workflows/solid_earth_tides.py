#!/usr/bin/env python3
import time
from datetime import datetime

import h5py
import isce3
import journal
import numpy as np
import pysolid
from isce3.core import transform_xy_to_latlon
from nisar.products.insar.product_paths import GUNWGroupsPaths
from nisar.workflows.h5_prep import get_products_and_paths
from nisar.workflows.solid_earth_tides_runconfig import \
    InsarSolidEarthTidesRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse
from scipy.interpolate import RegularGridInterpolator


def add_solid_earth_to_gunw_hdf5(solid_earth_tides,
                                 gunw_hdf5):
    '''
    Add the solid earth phase datacube to GUNW product

    Parameters
    ----------
    solid_earth_tides: tuple
        solid earth tides along the slant range and along-track directions
    gunw_hdf5: str
         GUNW HDF5 file where SET will be written
    '''

    with h5py.File(gunw_hdf5, 'a', libver='latest', swmr=True) as hdf:
        radar_grid = hdf.get(GUNWGroupsPaths().RadarGridPath)
        product_names = ['slantRangeSolidEarthTidesPhase', 'alongTrackSolidEarthTidesPhase']

        for  product_name, solid_earth_tides_product in zip(product_names,
                                                            solid_earth_tides):
            radar_grid[product_name][...] = solid_earth_tides_product

def calculate_solid_earth_tides(inc_angle_datacube,
                                los_unit_vector_x_datacube,
                                los_unit_vector_y_datacube,
                                xcoord_of_datacube,
                                ycoord_of_datacube,
                                epsg,
                                wavelength,
                                reference_start_time,
                                secondary_start_time):

    '''
    calculate the solid earth tides components along LOS and azimuth directions

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
        solid earth tides along the los and azimuth directions
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

    # Fix the step size around 10km if the spacing of the datacube is less than 10km
    # 0.1 is degrees approx. 10km
    y_step = max(0.1,
                 np.max(lat_datacube[0, :] - lat_datacube[cube_y_size-1, :]) /
                 (cube_y_size - 1))

    x_step = max(0.1,
                 np.max(lon_datacube[:, cube_x_size-1] - lon_datacube[:, 0]) /
                 (cube_x_size - 1))

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

    # Points
    pnts = np.stack(
        (lon_datacube.flatten(), lat_datacube.flatten()), axis=-1)

    y_samples = np.flip(y_samples)
    xy_samples = (x_samples, y_samples)
    cube_shape = lat_datacube.shape

    # Solid earth tides for both reference and secondary dates
    ref_tides, sec_tides = [
        pysolid.calc_solid_earth_tides_grid(start_time, params,
                                            step_size = 1000,
                                            display=False,
                                            verbose=True)
        for start_time in [reference_start_time, secondary_start_time]]

    # Interpolation, the flip function applied here is to fit the scipy==1.8
    # which requires strict ascending or descending
    # (ref - sec) tide
    def intepolate_tide(ref_tide, sec_tide, xy_samples, pnts, cube_shape):
        tide_interp = RegularGridInterpolator(xy_samples,
                                              np.flip(ref_tide - sec_tide,
                                              axis=0))
        return tide_interp(pnts).reshape(cube_shape)

    ref_sec_tide_e, ref_sec_tide_n, ref_sec_tide_u = [
        intepolate_tide(ref_tide, sec_tide, xy_samples, pnts, cube_shape)
        for ref_tide, sec_tide in zip(ref_tides, sec_tides)]

    # Azimuth angle, the minus sign is because of the anti-clockwise positive definition
    azimuth_angle = -np.arctan2(los_unit_vector_x_datacube, los_unit_vector_y_datacube)

    # Incidence angle in radians
    inc_angle = np.deg2rad(inc_angle_datacube)

    # Solid earth tides along the azimith direction
    azimuth_solid_earth_tides_datacube = (-ref_sec_tide_e * np.sin(azimuth_angle) +
                                          + ref_sec_tide_n * np.cos(azimuth_angle))

    # Solidearth tides datacube along the LOS in meters
    los_solid_earth_tides_datacube =(-ref_sec_tide_e * np.sin(inc_angle) * np.sin(azimuth_angle)
                                     + ref_sec_tide_n * np.sin(inc_angle) * np.cos(azimuth_angle)
                                     + ref_sec_tide_u  * np.cos(inc_angle))

    # Convert to phase screen
    los_solid_earth_tides_datacube *= -4.0 * np.pi / wavelength
    azimuth_solid_earth_tides_datacube *= -4.0 * np.pi / wavelength

    return (los_solid_earth_tides_datacube,
            azimuth_solid_earth_tides_datacube)



def _extract_params_from_gunw_hdf5(gunw_hdf5_path: str):

    # Instantiate GUNW object to avoid hard-coded paths to GUNW datasets
    gunw_obj = GUNWGroupsPaths()
    with h5py.File(gunw_hdf5_path, 'r', libver='latest', swmr=True) as h5_obj:

        # Fetch the GUWN Incidence Angle Datacube
        rdr_grid_path = gunw_obj.RadarGridPath
        id_path = gunw_obj.IdentificationPath
        [inc_angle_cube,
         los_unit_vector_x_cube,
         los_unit_vector_y_cube,
         xcoord_radar_grid,
         ycoord_radar_grid,
         height_radar_grid] =[h5_obj[f'{rdr_grid_path}/{item}'][()]
                                   for item in ['incidenceAngle',
                                                'losUnitVectorX', 'losUnitVectorY',
                                                'xCoordinates', 'yCoordinates',
                                                'heightAboveEllipsoid']]
        projection_dataset = h5_obj[f'{rdr_grid_path}/projection']
        epsg = projection_dataset.attrs['epsg_code']

         # Wavelenth in meters
        wavelength = isce3.core.speed_of_light / \
                h5_obj[f'{gunw_obj.GridsPath}/frequencyA/centerFrequency'][()]

        # Start time of the reference and secondary image
        ref_start_time, sec_start_time = [h5_obj[f'{id_path}/{x}ZeroDopplerStartTime'][()]\
                .astype('datetime64[s]').astype(datetime) for x in ['reference', 'secondary']]

        return (inc_angle_cube,
                los_unit_vector_x_cube,
                los_unit_vector_y_cube,
                xcoord_radar_grid,
                ycoord_radar_grid,
                height_radar_grid,
                epsg,
                wavelength,
                ref_start_time,
                sec_start_time)


def compute_solid_earth_tides(gunw_hdf5_path: str):
    '''
    Compute the solid earth tides datacube along LOS

    Parameters
    ----------
    gunw_hdf5_path: str
        path to NISAR GUNW hdf5 file

    Returns
    ----------
    solid_earth_tides: tuple
        solid earth tides along the los and azimuth directions
    '''

    # Extract the HDF5 parameters
    inc_angle_cube,\
    los_unit_vector_x_cube,\
    los_unit_vector_y_cube,\
    xcoord_radar_grid,\
    ycoord_radar_grid,\
    height_radar_grid,\
    epsg,\
    wavelength,\
    ref_start_time,\
    sec_start_time = _extract_params_from_gunw_hdf5(gunw_hdf5_path)

    # Caculate the solid earth tides
    solid_earth_tides = calculate_solid_earth_tides(inc_angle_cube,
                                                    los_unit_vector_x_cube,
                                                    los_unit_vector_y_cube,
                                                    xcoord_radar_grid,
                                                    ycoord_radar_grid,
                                                    int(epsg),
                                                    wavelength,
                                                    ref_start_time,
                                                    sec_start_time)

    return solid_earth_tides


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

    # Compute the solid earth tides along slant range and along-track directions
    solid_earth_tides = compute_solid_earth_tides(gunw_hdf5_path)

    # Write the solid earth tides to GUNW product
    add_solid_earth_to_gunw_hdf5(solid_earth_tides,
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
