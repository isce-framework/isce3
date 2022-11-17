#!/usr/bin/env python3
import os
from pathlib import Path
import pytest
import types

import numpy as np
from osgeo import gdal

import iscetest
import isce3.ext.isce3 as isce
from nisar.products.readers import SLC


@pytest.fixture(scope='session')
def unit_test_params():
    '''
    test parameters shared by all geocode_slc tests
    '''
    # load h5 for doppler and orbit
    params = types.SimpleNamespace()

    # define geogrid
    geogrid = isce.product.GeoGridParameters(start_x=-115.65,
                                             start_y=34.84,
                                             spacing_x=0.0002,
                                             spacing_y=-8.0e-5,
                                             width=500,
                                             length=500,
                                             epsg=4326)

    params.geogrid = geogrid

    # define geotransform
    params.geotrans = [geogrid.start_x, geogrid.spacing_x, 0.0,
                       geogrid.start_y, 0.0, geogrid.spacing_y]

    input_h5_path = os.path.join(iscetest.data, "envisat.h5")

    params.radargrid = isce.product.RadarGridParameters(input_h5_path)

    # init SLC object and extract necessary test params from it
    rslc = SLC(hdf5file=input_h5_path)

    params.orbit = rslc.getOrbit()

    img_doppler = rslc.getDopplerCentroid()
    params.img_doppler = img_doppler

    params.native_doppler = isce.core.LUT2d(img_doppler.x_start,
            img_doppler.y_start, img_doppler.x_spacing,
            img_doppler.y_spacing, np.zeros((geogrid.length,geogrid.width)))

    # create DEM raster object
    params.dem_raster = isce.io.Raster(os.path.join(iscetest.data,
                                       "geocode/zeroHeightDEM.geo"))

    return params


def geocode_slc_test_cases(radargrid):
    '''
    Generator for geocodeSlc test cases

    Given a radar grid, generate correction LUT2ds in range and azimuth
    directions. Returns axis, offset mode name, range and azimuth correction
    LUT2ds and offset corrected radar grid.
    '''
    # multiplicative factor applied to range pixel spacing and azimuth time
    # interval to be added to starting range and azimuth time of radar grid
    offset_factor = 10
    rg_pxl_spacing = radargrid.range_pixel_spacing
    range_offset = offset_factor * rg_pxl_spacing
    az_time_interval = 1 / radargrid.prf
    azimuth_offset = offset_factor * az_time_interval

    # despite uniform value LUT2d set to interp mode nearest just in case
    method = isce.core.DataInterpMethod.NEAREST

    # array of ones to be multiplied by respective offset value
    # shape unchanging; no noeed to be in loop as only starting values change
    ones = np.ones(radargrid.shape)

    for axis in 'xy':
        for offset_mode in ['', 'rg', 'az', 'rg_az']:
            # create radar and apply positive offsets in range and azimuth
            offset_radargrid = radargrid.copy()

            # apply offsets as required by mode
            if 'rg' in offset_mode:
                offset_radargrid.starting_range += range_offset
            if 'az' in offset_mode:
                offset_radargrid.sensing_start += azimuth_offset

            # slant range vector for LUT2d
            srange_vec = np.linspace(offset_radargrid.starting_range,
                                     offset_radargrid.end_range,
                                     radargrid.width)

            # azimuth vector for LUT2d
            az_time_vec = np.linspace(offset_radargrid.sensing_start,
                                      offset_radargrid.sensing_stop,
                                      radargrid.length)

            # corrections LUT2ds will use the negative offsets
            # should cancel positive offset applied to radar grid
            srange_correction = isce.core.LUT2d()
            if 'rg' in offset_mode:
                srange_correction = isce.core.LUT2d(srange_vec, az_time_vec,
                                                    range_offset * ones,
                                                    method)

            az_time_correction = isce.core.LUT2d()
            if 'az' in offset_mode:
                az_time_correction = isce.core.LUT2d(srange_vec, az_time_vec,
                                                     azimuth_offset * ones,
                                                     method)

            yield (axis, offset_mode, srange_correction, az_time_correction,
                   offset_radargrid)


def run_geocode_slc_raster(test_case, unit_test_params):
    '''
    wrapper for geocode_slc raster mode
    '''
    # extract test specific params
    (axis, correction_mode, srange_correction, az_time_correction,
     test_rdrgrid) = test_case

    in_raster = isce.io.Raster(os.path.join(iscetest.data,
                                            f"geocodeslc/{axis}.slc"))

    Path(f"{axis}{correction_mode}_raster.geo").touch()
    out_raster = isce.io.Raster(f"{axis}_{correction_mode}_raster.geo",
                                 unit_test_params.geogrid.width,
                                 unit_test_params.geogrid.length, 1,
                                 gdal.GDT_CFloat32, "ENVI")

    isce.geocode.geocode_slc(output_raster=out_raster,
        input_raster=in_raster,
        dem_raster=unit_test_params.dem_raster,
        radargrid=test_rdrgrid,
        geogrid=unit_test_params.geogrid,
        orbit=unit_test_params.orbit,
        native_doppler=unit_test_params.native_doppler,
        image_grid_doppler=unit_test_params.img_doppler,
        ellipsoid=isce.core.Ellipsoid(),
        threshold_geo2rdr=1.0e-9,
        numiter_geo2rdr=25,
        lines_per_block=1000,
        flatten=False,
        az_time_correction=az_time_correction,
        srange_correction=srange_correction)

    # set geotransform
    out_raster.set_geotransform(unit_test_params.geotrans)


def run_geocode_slc_array(test_case, unit_test_params):
    '''
    wrapper for geocode_slc array mode
    '''
    # extract test specific params
    (axis, correction_mode, srange_correction, az_time_correction,
     test_rdrgrid) = test_case

    out_shape = (unit_test_params.geogrid.width,
                 unit_test_params.geogrid.length)

    # load input as array
    in_path = os.path.join(iscetest.data, f"geocodeslc/{axis}.slc")
    ds = gdal.Open(in_path, gdal.GA_ReadOnly)
    in_data = ds.GetRasterBand(1).ReadAsArray()

    # output file name for geocodeSlc array mode
    out_path = f"{axis}_{correction_mode}_array.geo"
    Path(out_path).touch()

    # empty array to be written to by geocode_slc array mode
    out_data = np.zeros(out_shape, dtype=np.complex64)

    isce.geocode.geocode_slc(
        geo_data_block=out_data,
        rdr_data_block=in_data,
        dem_raster=unit_test_params.dem_raster,
        radargrid=test_rdrgrid,
        geogrid=unit_test_params.geogrid,
        orbit=unit_test_params.orbit,
        native_doppler= unit_test_params.native_doppler,
        image_grid_doppler=unit_test_params.img_doppler,
        ellipsoid=isce.core.Ellipsoid(),
        threshold_geo2rdr=1.0e-9,
        numiter_geo2rdr=25,
        azimuth_first_line=0,
        range_first_pixel=0,
        flatten=False,
        az_time_correction=az_time_correction,
        srange_correction=srange_correction)

    # set geotransform in output raster
    out_raster = isce.io.Raster(out_path, unit_test_params.geogrid.width,
                                unit_test_params.geogrid.length, 1,
                                gdal.GDT_CFloat32,  "ENVI")
    out_raster.set_geotransform(unit_test_params.geotrans)
    del out_raster

    # write output to raster
    ds = gdal.Open(out_path, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(out_data)


def test_run(unit_test_params):
    '''
    run geocodeSlc bindings with same parameters as C++ test to make sure it
    does not crash for both raster and array modes
    '''
    # run raster mode for all test cases
    for test_case in geocode_slc_test_cases(unit_test_params.radargrid):
        run_geocode_slc_raster(test_case, unit_test_params)

    # run array mode for all test cases in seperate loop to avoid
    # isce3.io.raster-related? glitches
    for test_case in geocode_slc_test_cases(unit_test_params.radargrid):
        run_geocode_slc_array(test_case, unit_test_params)


def test_validate(unit_test_params):
    '''
    validate test outputs
    '''
    # get geotransform from test data
    x0 = np.radians(unit_test_params.geotrans[0] + unit_test_params.geotrans[1] / 2.0)
    dx = np.radians(unit_test_params.geotrans[1])
    y0 = np.radians(unit_test_params.geotrans[3] + unit_test_params.geotrans[5] / 2.0)
    dy = np.radians(unit_test_params.geotrans[5])

    # check values of geocoded outputs
    for axis, correction_mode, *_, \
        in geocode_slc_test_cases(unit_test_params.radargrid):

        for mode in ['array', 'raster']:
            # get phase of complex test data and mask NaN (default invalid val)
            test_raster = f"{axis}_{correction_mode}_{mode}.geo"
            ds = gdal.Open(test_raster, gdal.GA_ReadOnly)
            test_arr = np.angle(ds.GetRasterBand(1).ReadAsArray())
            # mask with NaN since NaN is used to mark invalid pixels
            test_mask = np.isnan(test_arr)
            test_arr = np.ma.masked_array(test_arr, mask=test_mask)
            ds = None

            # use geotransform to make lat/lon mesh
            pixels, lines = test_arr.shape
            meshx, meshy = np.meshgrid(np.arange(lines), np.arange(pixels))

            # calculate and check error within bounds
            if axis == 'x':
                grid_lon = np.ma.masked_array(x0 + meshx * dx, mask=test_mask)

                err = np.nanmax(np.abs(test_arr - grid_lon))
            else:
                grid_lat = np.ma.masked_array(y0 + meshy * dy, mask=test_mask)

                err = np.nanmax(np.abs(test_arr - grid_lat))

            # check max diff of masked arrays
            assert(err < 1.0e-6), f'{test_raster} max error fail'
