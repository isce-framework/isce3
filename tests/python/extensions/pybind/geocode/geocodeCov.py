#!/usr/bin/env python3
import os
import numpy as np
from osgeo import gdal
import iscetest
import isce3.ext.isce3 as isce
import isce3
from nisar.products.readers import SLC

geocode_modes = {'interp': isce.geocode.GeocodeOutputMode.INTERP,
                 'area': isce.geocode.GeocodeOutputMode.AREA_PROJECTION}
input_axis = ['x', 'y']


# run tests
def test_geocode_cov():
    # load parameters shared across all test runs
    # init geocode object and populate members
    rslc = SLC(hdf5file=os.path.join(iscetest.data, "envisat.h5"))
    geo_obj = isce.geocode.GeocodeFloat64()
    geo_obj.orbit = rslc.getOrbit()
    geo_obj.doppler = rslc.getDopplerCentroid()
    geo_obj.ellipsoid = isce.core.Ellipsoid()
    geo_obj.threshold_geo2rdr = 1e-9
    geo_obj.numiter_geo2rdr = 25
    geo_obj.radar_block_margin = 10
    geo_obj.data_interpolator = 'biquintic'

    # prepare geogrid
    geogrid_start_x = -115.65
    geogrid_start_y = 34.85
    geogrid_end_x = -115.5
    geogrid_end_y = 34.78
    geogrid_spacing_x = 0.002
    geogrid_spacing_y = -0.0008

    geo_grid_length = int((geogrid_end_y - geogrid_start_y) /
                          geogrid_spacing_y)
    geo_grid_width = int((geogrid_end_x - geogrid_start_x) / geogrid_spacing_x)
    epsgcode = 4326
    geo_obj.geogrid(geogrid_start_x, geogrid_start_y, geogrid_spacing_x,
                    geogrid_spacing_y, geo_grid_width, geo_grid_length,
                    epsgcode)

    # get radar grid from HDF5
    radar_grid = isce.product.RadarGridParameters(os.path.join(iscetest.data,
                                                               "envisat.h5"))

    # load test DEM
    dem_raster = isce.io.Raster(os.path.join(iscetest.data,
                                             "geocode/zeroHeightDEM.geo"))

    # iterate thru axis
    for axis in input_axis:
        # load axis input raster
        xy_filename = os.path.join(iscetest.data, f"geocode/{axis}.rdr")
        print(f'testing file {xy_filename}')

        # create sub_swath object
        n_sub_swaths = 1
        gdal_ds = gdal.Open(xy_filename)
        xy_array = gdal_ds.GetRasterBand(1).ReadAsArray()
        quantile_10_value, quantile_90_value = np.percentile(xy_array,
                                                             [10, 90])
        print('    data quantiles for evaluating the masking of'
              ' valid-samples sub-swath:')
        print(f'        quantile 10: {quantile_10_value}')
        print(f'        quantile 90: {quantile_90_value}')
        sub_swath_array = np.zeros((radar_grid.length, 2), np.int32)
        valid_values = np.logical_and(xy_array > quantile_10_value,
                                      xy_array < quantile_90_value)
        if axis == 'x':
            x_quantile_10_value = quantile_10_value
            x_quantile_90_value = quantile_90_value
        else:
            y_quantile_10_value = quantile_10_value
            y_quantile_90_value = quantile_90_value

        for i in range(radar_grid.length):
            for j in range(radar_grid.width):
                if valid_values[i, j]:
                    sub_swath_array[i, 0] = j
                    break
            else:
                # The line is invalid
                sub_swath_array[i, 0] = -1
                sub_swath_array[i, 1] = -1
                continue
            for j in reversed(range(radar_grid.width)):
                if valid_values[i, j]:
                    sub_swath_array[i, 1] = j
                    break

        sub_swath = isce3.product.SubSwaths(radar_grid.length,
                                            radar_grid.width,
                                            n_sub_swaths)
        sub_swath.set_valid_samples_array(1, sub_swath_array)

        input_raster = isce.io.Raster(xy_filename)

        #  iterate thru geocode modes
        for key, value in geocode_modes.items():
            for apply_sub_swath_mask in [False, True]:

                # prepare output raster
                sub_swath_kwargs = {}
                if apply_sub_swath_mask:
                    sub_swath_str = '_sub_swath_masked'
                    sub_swath_mask_path = f"{axis}_{key}_sub_swath_mask.geo"
                    out_mask = isce.io.Raster(
                        sub_swath_mask_path,
                        geo_grid_width, geo_grid_length, 1,
                        gdal.GDT_Byte, "ENVI")

                    sub_swath_kwargs['sub_swaths'] = sub_swath
                    sub_swath_kwargs['out_mask'] = \
                        out_mask

                else:
                    sub_swath_str = ''

                output_path = f"{axis}_{key}{sub_swath_str}.geo"
                print(f'   output file: {output_path}')
                output_raster = isce.io.Raster(
                    output_path,
                    geo_grid_width, geo_grid_length, 1,
                    gdal.GDT_Float64, "ENVI")

                # geocode based on axis and mode
                geo_obj.geocode(radar_grid,
                                input_raster,
                                output_raster,
                                dem_raster,
                                value, **sub_swath_kwargs)

    # flush output layers
    output_raster.close_dataset()
    del geo_obj

    # validate generated data

    grid_x = None

    # iterate thru axis
    for axis in input_axis:

        #  iterate thru geocode modes
        for key, value in geocode_modes.items():

            for apply_sub_swath_mask in [False, True]:

                # prepare output raster
                if apply_sub_swath_mask:
                    sub_swath_str = '_sub_swath_masked'
                else:
                    sub_swath_str = ''

                test_raster = f"{axis}_{key}{sub_swath_str}.geo"
                print(f'   verifying file: {test_raster}')
                ds = gdal.Open(test_raster, gdal.GA_ReadOnly)
                geo_arr = ds.GetRasterBand(1).ReadAsArray()
                geo_arr = np.ma.masked_array(geo_arr, mask=np.isnan(geo_arr))
                ds = None

                # get transform and meshgrids once for common geogrid
                if grid_x is None:
                    geo_trans = isce.io.Raster(test_raster).get_geotransform()
                    x0 = geo_trans[0] + geo_trans[1] / 2.0
                    dx = geo_trans[1]
                    y0 = geo_trans[3] + geo_trans[5] / 2.0
                    dy = geo_trans[5]

                    pixels, lines = geo_arr.shape
                    meshx, meshy = np.meshgrid(np.arange(lines),
                                               np.arange(pixels))
                    grid_x = x0 + meshx * dx
                    grid_y = y0 + meshy * dy

                    # creates mask of pixels expected to be
                    # within 10-90 quantile range
                    sub_swath_expected_exp_within_quantile_lon = \
                        np.logical_and(grid_x > x_quantile_10_value,
                                       grid_x < x_quantile_90_value)
                    sub_swath_expected_exp_within_quantile_lat = \
                        np.logical_and(grid_y > y_quantile_10_value,
                                       grid_y < y_quantile_90_value)

                # calculate error
                if axis == 'x':
                    err = geo_arr - grid_x
                else:
                    err = geo_arr - grid_y

                # calculate avg square difference error
                rmse = np.sqrt(np.sum(err**2) / np.count_nonzero(~geo_arr.mask))

                if key == 'interp':
                    # get max err
                    max_err = np.nanmax(err)

                    assert (max_err < 1.0e-8), f'{test_raster} max error fail'

                if axis == 'x':
                    rmse_err_threshold = 0.5 * dx
                else:
                    rmse_err_threshold = 0.5 * abs(dy)
                assert (rmse < rmse_err_threshold), f'{test_raster} RMSE fail'

                if not apply_sub_swath_mask:
                    continue

                # select mask of pixels within 10-90 quantile range
                if axis == 'x':
                    sub_swath_expected_exp_within_quantile = \
                        sub_swath_expected_exp_within_quantile_lon
                else:
                    sub_swath_expected_exp_within_quantile = \
                        sub_swath_expected_exp_within_quantile_lat

                # test valid-samples sub-swath mask (output of GeocodeCov)
                sub_swath_mask_path = f"{axis}_{key}_sub_swath_mask.geo"
                gdal_ds = gdal.Open(sub_swath_mask_path)
                sub_swath_mask_array = gdal_ds.GetRasterBand(1).ReadAsArray()

                # the outside area represented by
                # `~ sub_swath_expected_exp_within_quantile` should include
                # `sub_swath_mask_array``.

                # first, check the mask and assert that there are no masked
                # points outside of the selected area.
                assert np.sum((sub_swath_mask_array) &
                              (~sub_swath_expected_exp_within_quantile)) == 0

                # then, we check the geocoded array and assert that there are
                # valid points inside the selected area
                assert np.sum((np.isfinite(geo_arr)) &
                              (sub_swath_expected_exp_within_quantile)) > 0

                # finally, we check the geocoded array again and assert
                # that there are no valid points outside of the selected area
                assert np.sum((np.isfinite(geo_arr)) &
                              (~sub_swath_expected_exp_within_quantile)) == 0


if __name__ == "__main__":
    test_geocode_cov()

# end of file
