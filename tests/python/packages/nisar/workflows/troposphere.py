import os
from osgeo import gdal

import h5py
import isce3
import iscetest
import numpy as np
from nisar.workflows import troposphere
import pyaps3 as pa
from nisar.products.insar.product_paths import GUNWGroupsPaths
from scipy.interpolate import RegularGridInterpolator
import yaml

def get_dem_info(dem_file: str):
    '''
    get dem information

    Parameters
     ----------
     dem_file: str
        dem file

    Returns
     -------
     (heights, y_2d, x_2d): tuple
    '''

    src_ds_dem = gdal.Open(dem_file)

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


def test_troposphere_aps_run():
    '''
    test the troposphere delay by pyAPS package
    '''

    # Load yaml file
    test_yaml = os.path.join(
        iscetest.data, 'troposphere/troposphere_aps_test.yaml')

    with open(test_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = cfg['runconfig']['groups']

        # Load the test weather files
        tropo_weather_model_cfg = cfg['dynamic_ancillary_file_group']['troposphere_weather_model_files']

        weather_reference_file = \
            os.path.join(
                iscetest.data, tropo_weather_model_cfg['reference_troposphere_file'])
        weather_secondary_file = \
            os.path.join(
                iscetest.data, tropo_weather_model_cfg['secondary_troposphere_file'])

        cfg['dynamic_ancillary_file_group']['troposphere_weather_model_files']\
            ['reference_troposphere_file'] = weather_reference_file
        cfg['dynamic_ancillary_file_group']['troposphere_weather_model_files']\
            ['secondary_troposphere_file'] = weather_secondary_file

        dem_file = os.path.join(
            iscetest.data, cfg['dynamic_ancillary_file_group']['dem_file'])
        gunw_hdf5 = os.path.join(iscetest.data, 'troposphere/GUNW_product.h5')

        # Compute the troposphere delay datacube
        tropo_delay_datacube = troposphere.compute_troposphere_delay(
            cfg, gunw_hdf5)

        # Compute the troposhere delay by test dem using datacube
        heights, y_2d, x_2d = get_dem_info(dem_file)

        # Zenith
        los = np.zeros(heights.shape)

        pnts = np.stack(
            (heights.flatten(), y_2d.flatten(), x_2d.flatten()), axis=-1)

        # Create GUNW object to avoid hard-coded paths to GUNW datasets
        gunw_obj = GUNWGroupsPaths()

        with h5py.File(gunw_hdf5, 'r') as hdf:

            # EPSG Code
            epsg = \
                int(hdf[f'{gunw_obj.RadarGridPath}/projection'].attrs['epsg_code'])

            # Incidence Angle Datacube
            inc_angle_datacube = \
                    hdf[f'{gunw_obj.RadarGridPath}/incidenceAngle'][()]

            # Coordinates X
            xcoord_radar_grid = \
                    hdf[f'{gunw_obj.RadarGridPath}/xCoordinates'][()]

            # Coordinate Y
            ycoord_radar_grid = \
                    hdf[f'{gunw_obj.RadarGridPath}/yCoordinates'][()]

            # Heights
            height_radar_grid = \
                    hdf[f'{gunw_obj.RadarGridPath}/heightAboveEllipsoid'][()]

            # Wavelength
            wavelength = isce3.core.speed_of_light / \
                float(hdf[f'{gunw_obj.GridsPath}/frequencyA/centerFrequency'][()])

        # Troposphere product parameters
        tropo_package = cfg['processing']['troposphere_delay']['package']
        tropo_weather_model_type = cfg['processing']['troposphere_delay']['weather_model_type']
        tropo_delay_direction = cfg['processing']['troposphere_delay']['delay_direction']

        for delay_type in ['wet', 'hydrostatic', 'comb']:
            if cfg['processing']['troposphere_delay'][f'enable_{delay_type}_product']:
                if (tropo_package.lower() == 'pyaps') and \
                    (delay_type == 'hydrostatic'):
                    tropo_delay_product = 'dry'
                else:
                    tropo_delay_product = delay_type

        # Dictionary key
        delay_product = f'tropoDelay_{tropo_package}_{tropo_delay_direction}_{tropo_delay_product}'

        # Test if there is any NaN value in the datacube
        assert (not np.isnan(tropo_delay_datacube[delay_product]).any())

        # Troposphere delay in centimeters
        delay_datacube = wavelength * \
            tropo_delay_datacube[delay_product] / (np.pi * 4.0) * 100.0

        # Make the Y coordinates ascending
        ycoord_radar_grid = np.flip(ycoord_radar_grid)
        delay_datacube = np.flip(delay_datacube, axis=1)

        # Troposphere delay interpolator
        tropo_delay_interpolator = RegularGridInterpolator((height_radar_grid,
                                                            ycoord_radar_grid,
                                                            xcoord_radar_grid),
                                                           delay_datacube,
                                                           method='linear')

        # Troposphere delay from datacube via interpolation
        tropo_delay_from_datacube = tropo_delay_interpolator(
            pnts).reshape(heights.shape)

        # Convert the X/Y to lat/lon
        lat, lon, _ = troposphere.transform_xy_to_latlon(epsg, x_2d, y_2d)

        # Compute the troposhere delay by test dem using the package
        reference_obj = pa.PyAPS(weather_reference_file,
                                 dem=heights,
                                 inc=los,
                                 lat=lat,
                                 lon=lon,
                                 grib=tropo_weather_model_type,
                                 humidity='Q',
                                 model=tropo_weather_model_type,
                                 verb=False,
                                 Del=tropo_delay_product)

        secondary_obj = pa.PyAPS(weather_secondary_file,
                                 dem=heights,
                                 inc=los,
                                 lat=lat,
                                 lon=lon,
                                 grib=tropo_weather_model_type,
                                 humidity='Q',
                                 model=tropo_weather_model_type,
                                 verb=False,
                                 Del=tropo_delay_product)

        reference_delay = reference_obj.getdelay()
        secondary_delay = secondary_obj.getdelay()

        # Troposphere delay computed at high resolution (i.e., product spacing)
        high_resolution_tropo_delay = -(reference_delay - secondary_delay) * 100.0

        f.close()

    # Compare tropospheric delay interpolated from low resolution data cube with the
    # delay computed at high resolution. An absolute tolerance of 1 centimeter
    # is considered for the comparison.
    np.testing.assert_allclose(
        high_resolution_tropo_delay, tropo_delay_from_datacube, atol=1.0)

if __name__ == '__main__':
    test_troposphere_aps_run()
