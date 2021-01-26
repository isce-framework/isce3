#!/usr/bin/env python3
import os
import numpy as np
from osgeo import gdal
import iscetest
import pybind_isce3 as isce
from pybind_nisar.products.readers import SLC

geocode_modes = {'interp':isce.geocode.GeocodeOutputMode.INTERP,
        'area':isce.geocode.GeocodeOutputMode.AREA_PROJECTION}
input_axis = ['x', 'y']


# run tests
def test_run():
    # load parameters shared across all test runs
    # init geocode object and populate members
    rslc = SLC(hdf5file=os.path.join(iscetest.data, "envisat.h5"))
    geo_obj = isce.geocode.GeocodeFloat64()
    geo_obj.orbit = rslc.getOrbit()
    geo_obj.doppler = rslc.getDopplerCentroid()
    geo_obj.ellipsoid = isce.core.Ellipsoid()
    geo_obj.threshold_geo2rdr = 1e-9
    geo_obj.numiter_geo2rdr = 25
    geo_obj.lines_per_block = 1000
    geo_obj.dem_block_margin = 1e-1
    geo_obj.radar_block_margin = 10
    geo_obj.interpolator = 'biquintic'

    # prepare geogrid
    geogrid_start_x = -115.6
    geogrid_start_y = 34.832
    reduction_factor = 10
    geogrid_spacingX = reduction_factor * 0.0002
    geogrid_spacingY = reduction_factor * -8.0e-5
    geo_grid_length = int(380 / reduction_factor)
    geo_grid_width = int(400 / reduction_factor)
    epsgcode = 4326
    geo_obj.geogrid(geogrid_start_x, geogrid_start_y, geogrid_spacingX,
                   geogrid_spacingY, geo_grid_width, geo_grid_length, epsgcode)

    # get radar grid from HDF5
    radar_grid = isce.product.RadarGridParameters(os.path.join(iscetest.data, "envisat.h5"))

    # load test DEM
    dem_raster = isce.io.Raster(os.path.join(iscetest.data, "geocode/zeroHeightDEM.geo"))

    # iterate thru axis
    for axis in input_axis:
        # load axis input raster
        input_raster = isce.io.Raster(os.path.join(iscetest.data, f"geocode/{axis}.rdr"))

        #  iterate thru geocode modes
        for key, value in geocode_modes.items():
            # prepare output raster
            output_path = f"{axis}_{key}.geo"
            output_raster = isce.io.Raster(output_path,
                    geo_grid_width, geo_grid_length, 1,
                    gdal.GDT_Float64, "ENVI")

            # geocode based on axis and mode
            geo_obj.geocode(radar_grid,
                    input_raster,
                    output_raster,
                    dem_raster,
                    value)


def test_validate():
    # validate generated data

    # iterate thru axis
    for axis in input_axis:
        #  iterate thru geocode modes
        for key, value in geocode_modes.items():
            test_raster = f"{axis}_{key}.geo"
            ds = gdal.Open(test_raster, gdal.GA_ReadOnly)
            geo_arr = ds.GetRasterBand(1).ReadAsArray()
            geo_arr = np.ma.masked_array(geo_arr, mask=np.isnan(geo_arr))
            ds = None

            # get transform and meshgrids once for common geogrid
            if "x_interp.geo" == test_raster:
                geo_trans = isce.io.Raster("x_area.geo").get_geotransform()
                x0 = geo_trans[0] + geo_trans[1] / 2.0
                dx = geo_trans[1]
                y0 = geo_trans[3] + geo_trans[5] / 2.0
                dy = geo_trans[5]

                pixels, lines = geo_arr.shape
                meshx, meshy = np.meshgrid(np.arange(lines), np.arange(pixels))
                grid_lon = x0 + meshx * dx
                grid_lat = y0 + meshy * dy

            # calculate error
            if axis == 'x':
                err = geo_arr - grid_lon
            else:
                err = geo_arr - grid_lat

            # calculate avg square difference error
            rmse = np.sqrt(np.sum(err**2) / np.count_nonzero(~geo_arr.mask))

            if key == 'interp':
                # get max err
                max_err = np.nanmax(err)

                assert( max_err < 1.0e-8 ), f'{test_raster} max error fail'

            if axis == 'x':
                rmse_err_threshold = 0.5 * dx
            else:
                rmse_err_threshold = 0.5 * abs(dy)
            assert( rmse  < rmse_err_threshold ), f'{test_raster} RMSE fail'


if __name__ == "__main__":
    test_run()
    test_validate()

# end of file
