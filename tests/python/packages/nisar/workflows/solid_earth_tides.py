import os
from osgeo import gdal

from datetime import datetime
import h5py
import isce3
import iscetest
import numpy as np
from nisar.workflows import solid_earth_tides
from nisar.workflows.solid_earth_tides import extract_params_from_gunw_hdf5
import pysolid
from scipy.interpolate import RegularGridInterpolator
import yaml

def get_dem_info(dem_file_path: str):
    '''
    get dem information

    Parameters
    ----------
    dem_file_path: str
      dem file

    Returns
    -------
    (heights, y_2d, x_2d): tuple
    '''

    src_ds_dem = gdal.Open(dem_file_path)

    # GeoTransform information
    ulx, xres, xskew, uly, yskew, yres = src_ds_dem.GetGeoTransform()

    # Heights
    heights = src_ds_dem.GetRasterBand(1).ReadAsArray()

    ysize, xsize = heights.shape

    # X and Y
    x = [ulx + i*xres for i in range(xsize)]
    y = [uly + i*yres for i in range(ysize)]

    # X and Y in 2D
    y_2d = np.tile(np.array(y), (xsize, 1)).T
    x_2d = np.tile(np.array(x), (ysize, 1))

    return (heights, y_2d, x_2d)


def test_solid_earth_tides_run():
    '''
    test the solid earth tides  by pySolid package
    '''

    # Load yaml file
    test_yaml = os.path.join(
        iscetest.data, 'solid_earth_tides/solid_earth_tides_runconfig.yaml')

    with open(test_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = cfg['runconfig']['groups']

        dem_file = os.path.join(
            iscetest.data, cfg['dynamic_ancillary_file_group']['dem_file'])
        gunw_hdf5 = os.path.join(iscetest.data, 'solid_earth_tides/GUNW_product.h5')

        # Compute the solid earth tides datacube in radians
        solid_earth_tides_components = solid_earth_tides.compute_solid_earth_tides(gunw_hdf5)
        # Use the LOS component for the unit test
        solid_earth_tides_datacube = solid_earth_tides_components[0]

        # Compute the solid earth tides by test dem using datacube
        heights, y_2d, x_2d = get_dem_info(dem_file)

        dem_pnts = np.stack(
                (heights.flatten(), y_2d.flatten(), x_2d.flatten()), axis=-1)

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
        sec_start_time = extract_params_from_gunw_hdf5(gunw_hdf5)

        # solid earth tides in centimeters
        solid_earth_tides_datacube = wavelength * \
            solid_earth_tides_datacube / (np.pi * 4.0) * 100.0

        # Make the Y coordinates ascending
        ycoord_radar_grid = np.flip(ycoord_radar_grid)
        solid_earth_tides_datacube = np.flip(solid_earth_tides_datacube, axis=1)

        # Solid earth tides  interpolator
        solid_earth_interpolator = RegularGridInterpolator((height_radar_grid,
                                                            ycoord_radar_grid,
                                                            xcoord_radar_grid),
                                                           solid_earth_tides_datacube,
                                                           method='linear')

        # Solid earth tides from datacube via interpolation
        solid_earth_tides_from_datacube = solid_earth_interpolator(
                dem_pnts).reshape(heights.shape)

        # Test if there is any NaN value
        assert not np.isnan(solid_earth_tides_from_datacube).any()

        def intepolate_cube(data_cube,
                            height_radar_grid,
                            ycoord_radar_grid,
                            xcoord_radar_grid,
                            pnts,
                            cube_shape):
            cube_interp = RegularGridInterpolator((height_radar_grid,
                                                  ycoord_radar_grid,
                                                  xcoord_radar_grid),
                                                  np.flip(data_cube,axis=1),
                                                  method = 'linear')

            return cube_interp(pnts).reshape(cube_shape)


        # los unit vector y, x, and incidence angle  from datacube via interpolation
        inc_angle_from_datacube,\
        los_unit_vector_x_from_datacube,\
        los_unit_vector_y_from_datacube = [intepolate_cube(data_cube,
                                                           height_radar_grid,
                                                           ycoord_radar_grid,
                                                           xcoord_radar_grid,
                                                           dem_pnts,
                                                           heights.shape) \
                                            for data_cube in [inc_angle_cube,
                                                              los_unit_vector_x_cube,
                                                              los_unit_vector_y_cube]]

        # The following steps are to compute the high resolution solid earth tides
        # directly from the pySolid package with small step size = 1

        # Convert the X/Y to lat/lon
        lat, lon, extents= solid_earth_tides.transform_xy_to_latlon(int(epsg), x_2d, y_2d)

        # Size of dem
        y_size, x_size = lat.shape

        # Configurations for pySolid
        y_end, y_first, x_first, x_end = extents

        y_step = np.max(lat[0, :] - lat[y_size-1, :]) / (y_size - 1)
        x_step = np.max(lon[:, x_size-1] - lon[:, 0]) / (x_size - 1)

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


        # Points from the Lat/Lon fed into pySolid
        pnts = np.stack((lon.flatten(), lat.flatten()), axis=-1)

        y_samples = np.flip(y_samples)
        xy_samples = (x_samples, y_samples)
        shape = lat.shape

        # Solid earth tides for both reference and secondary dates
        ref_tides, sec_tides = [
            pysolid.calc_solid_earth_tides_grid(start_time, params,
                                                step_size = 1,
                                                display=False,
                                                verbose=True)
            for start_time in [ref_start_time, sec_start_time]]

        # Interpolation, the flip function applied here is to fit the scipy==1.8
        # which requires strict ascending or descending
        # (ref - sec) tide
        def intepolate_tide(ref_tide, sec_tide, xy_samples, pnts, cube_shape):
            tide_interp = RegularGridInterpolator(xy_samples,
                                                  np.flip(ref_tide - sec_tide,
                                                  axis=0))
            return tide_interp(pnts).reshape(cube_shape)

        ref_sec_tide_e, ref_sec_tide_n, ref_sec_tide_u = [
            intepolate_tide(ref_tide, sec_tide, xy_samples, pnts, lat.shape)
            for ref_tide, sec_tide in zip(ref_tides, sec_tides)]


        # Azimuth angle, the minus sign is because of the anti-clockwise positive definition
        azimuth_angle = -np.arctan2(los_unit_vector_x_from_datacube,
                                    los_unit_vector_y_from_datacube)

        # Incidence angle in radians
        inc_angle = np.deg2rad(inc_angle_from_datacube)

        # Solidearth tides datacube along the LOS in meters
        los_solid_earth_tides_datacube =(-ref_sec_tide_e * np.sin(inc_angle) * np.sin(azimuth_angle)
                                         + ref_sec_tide_n * np.sin(inc_angle) * np.cos(azimuth_angle)
                                         + ref_sec_tide_u  * np.cos(inc_angle))

        # Convert to centimeters
        high_resolution_solid_earth_tides  = -100.0 * los_solid_earth_tides_datacube


        f.close()

    # Compare solid earth tides interpolated from low resolution data cube with the
    # one computed at high resolution. An absolute tolerance of 1 centimeter
    # is considered for the comparison.
    np.testing.assert_allclose(
        high_resolution_solid_earth_tides,
        solid_earth_tides_from_datacube, atol=1.0)

if __name__ == '__main__':
    test_solid_earth_tides_run()
