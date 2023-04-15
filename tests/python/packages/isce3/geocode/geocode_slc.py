#!/usr/bin/env python3
'''
test isce3.geocode.geocode_slc array and raster modes
'''
import json
import os
from pathlib import Path
import pytest
import types

import journal
import numpy as np
from osgeo import gdal
from scipy import interpolate

import iscetest
import isce3
from isce3.atmosphere.tec_product import tec_lut2d_from_json
from isce3.ext.isce3.geocode import geocode_slc as geocode_slc_raster
from isce3.geometry import compute_incidence_angle
from nisar.products.readers import SLC

def make_tec_file(unit_test_params):
    '''
    create TEC file using radar grid from envisat.h5 that yields a uniform
    slant range offset when processed with tec_lut2d_from_json()
    We ignore topside TEC and simulate total TEC at near and far ranges such
    that the slant range delay at near and far ranges are the same.
    solve for sub_orbital_tec from:
    delta_r = K * sub_orbital_tec * TECU / center_freq**2 / np.cos(incidence)

    yields:
    sub_orbital_tec = delta_r * np.cos(incidence) * center_freq**2 / (TECU * K)
    '''
    radar_grid = unit_test_params.radargrid

    # create linspace for radar grid sensing time
    t_rdr_grid = np.linspace(radar_grid.sensing_start,
                             radar_grid.sensing_stop + 1.0 / radar_grid.prf,
                             radar_grid.length)

    # TEC coefficients
    K = 40.31 # its a constant in m3/s2
    TECU = 1e16 # its a constant to convert the TEC product to electrons / m2

    # set delta_r to value used to test slant range offset correction in
    # geocode_slc_test_cases()
    offset_factor = 10.0
    delta_r = offset_factor * radar_grid.range_pixel_spacing

    # compute common TEC coefficient used for both near and far TEC
    common_tec_coeff = delta_r * unit_test_params.center_freq**2 / (K * TECU)

    # get TEC times in ISO format
    # +/- 50 sec from stop/start of radar grid
    margin = 50.
    # 10 sec increments - also snap to multiples of 10 sec
    snap = 10.
    start = np.floor(radar_grid.sensing_start / snap) * snap - margin
    stop = np.ceil(radar_grid.sensing_stop / snap) * snap + margin
    t_tec = np.arange(start, stop + 1.0, snap)
    t_tec_iso_fmt = [(radar_grid.ref_epoch + isce3.core.TimeDelta(t)).isoformat()[:-3]
                     for t in t_tec]

    # compute total TEC
    total_tec = []
    for rdr_grid_range in [radar_grid.starting_range,
                           radar_grid.end_range]:
        inc_angs = [compute_incidence_angle(t, rdr_grid_range,
                                            unit_test_params.orbit,
                                            isce3.core.LUT2d(),
                                            radar_grid,
                                            isce3.geometry.DEMInterpolator(),
                                            isce3.core.Ellipsoid())
                    for t in t_rdr_grid]
        total_tec_rdr_grid = common_tec_coeff * np.cos(inc_angs)

        # near and far top TEC = 0 to allow sub orbital TEC = total TEC
        # create extraplotor/interpolators for near and far
        total_tec_interp = interpolate.interp1d(t_rdr_grid, total_tec_rdr_grid,
                                                'linear',
                                                fill_value="extrapolate")

        # compute near and far total TEC
        total_tec.append(total_tec_interp(t_tec))
    total_tec_near, total_tec_far = total_tec

    # load relevant TEC into dict and write to JSON
    # top TEC = 0 to allow sub orbital TEC = total TEC
    tec_zeros = list(np.zeros(total_tec_near.shape))
    tec_dict ={}
    tec_dict['utc'] = t_tec_iso_fmt
    tec_dict['totTecNr'] = list(total_tec_near)
    tec_dict['topTecNr'] = tec_zeros
    tec_dict['totTecFr'] = list(total_tec_far)
    tec_dict['topTecFr'] = tec_zeros
    with open(unit_test_params.tec_json_path, 'w') as fp:
        json.dump(tec_dict, fp)


@pytest.fixture(scope='session')
def unit_test_params():
    '''
    test parameters shared by all geocode_slc tests
    '''
    # load h5 for doppler and orbit
    params = types.SimpleNamespace()

    # define geogrid
    geogrid = isce3.product.GeoGridParameters(start_x=-115.65,
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

    params.radargrid = isce3.product.RadarGridParameters(input_h5_path)

    # init SLC object and extract necessary test params from it
    rslc = SLC(hdf5file=input_h5_path)

    params.orbit = rslc.getOrbit()

    img_doppler = rslc.getDopplerCentroid()
    params.img_doppler = img_doppler

    params.center_freq = rslc.getSwathMetadata().processed_center_frequency

    params.native_doppler = isce3.core.LUT2d(img_doppler.x_start,
            img_doppler.y_start, img_doppler.x_spacing,
            img_doppler.y_spacing, np.zeros((geogrid.length,geogrid.width)))

    # create DEM raster object
    params.dem_path = os.path.join(iscetest.data, "geocode/zeroHeightDEM.geo")
    params.dem_raster = isce3.io.Raster(params.dem_path)

    # half pixel offset and grid size in radians for validataion
    params.x0 = np.radians(params.geotrans[0] + params.geotrans[1] / 2.0)
    params.dx = np.radians(params.geotrans[1])
    params.y0 = np.radians(params.geotrans[3] + params.geotrans[5] / 2.0)
    params.dy = np.radians(params.geotrans[5])

    # multiplicative factor applied to range pixel spacing and azimuth time
    # interval to be added to starting range and azimuth time of radar grid
    params.offset_factor = 10.0

    # TEC JSON containing TEC values that generate range offsets that match the
    # fixed range offset used to test range correction
    params.tec_json_path = 'test_tec.json'
    make_tec_file(params)

    return params


def geocode_slc_test_cases(unit_test_params):
    '''
    Generator for geocodeSlc test cases

    Given a radar grid, generate correction LUT2ds in range and azimuth
    directions. Returns axis, offset mode name, range and azimuth correction
    LUT2ds and offset corrected radar grid.
    '''
    radargrid = unit_test_params.radargrid
    offset_factor = unit_test_params.offset_factor

    rg_pxl_spacing = radargrid.range_pixel_spacing
    range_offset = offset_factor * rg_pxl_spacing
    az_time_interval = 1 / radargrid.prf
    azimuth_offset = offset_factor * az_time_interval

    # despite uniform value LUT2d set to interp mode nearest just in case
    method = isce3.core.DataInterpMethod.NEAREST

    # array of ones to be multiplied by respective offset value
    # shape unchanging; no noeed to be in loop as only starting values change
    ones = np.ones(radargrid.shape)

    for axis in 'xy':
        for offset_mode in ['', 'rg', 'az', 'rg_az', 'tec']:
            # create radar and apply positive offsets in range and azimuth
            offset_radargrid = radargrid.copy()

            # apply offsets as required by mode
            if 'rg' in offset_mode or 'tec' == offset_mode:
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
            srange_correction = isce3.core.LUT2d()
            if 'rg' in offset_mode:
                srange_correction = isce3.core.LUT2d(srange_vec, az_time_vec,
                                                    range_offset * ones,
                                                    method)
            elif 'tec' == offset_mode:
                srange_correction = \
                    tec_lut2d_from_json(unit_test_params.tec_json_path,
                                        unit_test_params.center_freq,
                                        unit_test_params.orbit,
                                        offset_radargrid,
                                        isce3.core.LUT2d(),
                                        unit_test_params.dem_path)

            az_time_correction = isce3.core.LUT2d()
            if 'az' in offset_mode:
                az_time_correction = isce3.core.LUT2d(srange_vec, az_time_vec,
                                                     azimuth_offset * ones,
                                                     method)

            yield (axis, offset_mode, srange_correction, az_time_correction,
                   offset_radargrid)


def run_geocode_slc_arrays(test_case, unit_test_params, extra_input=False,
                           non_matching_shape=False):
    '''
    wrapper for geocode_slc array mode
    '''
    # extract test specific params
    (axis, correction_mode, srange_correction, az_time_correction,
     test_rdrgrid) = test_case

    out_shape = (unit_test_params.geogrid.width,
                 unit_test_params.geogrid.length)

    # load input as list of arrays
    in_path = os.path.join(iscetest.data, f"geocodeslc/{axis}.slc")
    ds = gdal.Open(in_path, gdal.GA_ReadOnly)
    arr = ds.GetRasterBand(1).ReadAsArray()
    in_list = [arr, arr]
    # if extra input enabled, append extra array to input list
    if extra_input:
        in_list.append(arr)

    # output file name for geocodeSlc array mode
    out_path = f"{axis}_{correction_mode}_arrays.geo"
    # if forcing error, change output file name to not break outpu validation
    if extra_input or non_matching_shape:
        out_path += '_broken'
    Path(out_path).touch()

    # list of empty array to be written to by geocode_slc array mode
    out_zeros = np.zeros(out_shape, dtype=np.complex64)
    # if non matching shape enabled, ensure output array shapes do not match
    if non_matching_shape:
        wrong_shape = (out_shape[0], out_shape[1] + 1)
        out_list = [out_zeros, np.zeros(wrong_shape, dtype=np.complex64)]
    else:
        out_list = [out_zeros, out_zeros.copy()]

    isce3.geocode.geocode_slc(
        geo_data_blocks=out_list,
        rdr_data_blocks=in_list,
        dem_raster=unit_test_params.dem_raster,
        radargrid=test_rdrgrid,
        geogrid=unit_test_params.geogrid,
        orbit=unit_test_params.orbit,
        native_doppler= unit_test_params.native_doppler,
        image_grid_doppler=unit_test_params.img_doppler,
        ellipsoid=isce3.core.Ellipsoid(),
        threshold_geo2rdr=1.0e-9,
        num_iter_geo2rdr=25,
        first_azimuth_line=0,
        first_range_sample=0,
        flatten=False,
        az_time_correction=az_time_correction,
        srange_correction=srange_correction)

    # set geotransform in output raster
    out_raster = isce3.io.Raster(out_path, unit_test_params.geogrid.width,
                                unit_test_params.geogrid.length, 2,
                                gdal.GDT_CFloat32,  "ENVI")
    out_raster.set_geotransform(unit_test_params.geotrans)
    out_raster.close_dataset()

    # write output to raster
    ds = gdal.Open(out_path, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(out_list[0])
    ds.GetRasterBand(2).WriteArray(out_list[1])


def run_geocode_slc_array(test_case, unit_test_params):
    '''
    wrapper for geocode_slc array mode
    '''
    # extract test specific params
    (axis, correction_mode, srange_correction, az_time_correction,
     test_rdrgrid) = test_case

    out_shape = (unit_test_params.geogrid.width,
                 unit_test_params.geogrid.length)

    # load input as list of arrays
    in_path = os.path.join(iscetest.data, f"geocodeslc/{axis}.slc")
    ds = gdal.Open(in_path, gdal.GA_ReadOnly)
    in_data = ds.GetRasterBand(1).ReadAsArray()

    # list of empty array to be written to by geocode_slc array mode
    out_data = np.zeros(out_shape, dtype=np.complex64)

    isce3.geocode.geocode_slc(
        geo_data_blocks=out_data,
        rdr_data_blocks=in_data,
        dem_raster=unit_test_params.dem_raster,
        radargrid=test_rdrgrid,
        geogrid=unit_test_params.geogrid,
        orbit=unit_test_params.orbit,
        native_doppler= unit_test_params.native_doppler,
        image_grid_doppler=unit_test_params.img_doppler,
        ellipsoid=isce3.core.Ellipsoid(),
        threshold_geo2rdr=1.0e-9,
        num_iter_geo2rdr=25,
        first_azimuth_line=0,
        first_range_sample=0,
        flatten=False,
        az_time_correction=az_time_correction,
        srange_correction=srange_correction)

    # output file name for geocodeSlc array mode
    out_path = f"{axis}_{correction_mode}_array.geo"
    Path(out_path).touch()

    # set geotransform in output raster
    out_raster = isce3.io.Raster(out_path, unit_test_params.geogrid.width,
                                unit_test_params.geogrid.length, 1,
                                gdal.GDT_CFloat32,  "ENVI")
    out_raster.set_geotransform(unit_test_params.geotrans)
    del out_raster

    # write output to raster
    ds = gdal.Open(out_path, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(out_data)


def test_run_array_mode(unit_test_params):
    '''
    run geocodeSlc array bindings with same parameters as C++ test to make sure
    it does not crash
    '''
    # run array mode for all test cases
    for test_case in geocode_slc_test_cases(unit_test_params):
        run_geocode_slc_array(test_case, unit_test_params)


def test_run_arrays_mode(unit_test_params):
    '''
    run geocodeSlc list of array bindings with same parameters as C++ test to
    make sure it does not crash
    '''
    # run array mode for all test cases
    for test_case in geocode_slc_test_cases(unit_test_params):
        run_geocode_slc_arrays(test_case, unit_test_params)


def test_run_arrays_exceptions(unit_test_params):
    '''
    run geocodeSlc list of array bindings with erroneous parameters to test
    input checking
    '''
    # run array mode for all test cases with forced erroneous inputs to ensure
    # correct exceptions are raised
    for test_case in geocode_slc_test_cases(unit_test_params):
        with np.testing.assert_raises(journal.ext.journal.ApplicationError):
            run_geocode_slc_arrays(test_case, unit_test_params,
                                   extra_input=True)

        with np.testing.assert_raises(journal.ext.journal.ApplicationError):
            run_geocode_slc_arrays(test_case, unit_test_params,
                                   non_matching_shape=True)

        # break out of loop - no need to repeat assert tests
        break


def run_geocode_slc_raster(test_case, unit_test_params):
    '''
    wrapper for geocode_slc raster mode
    '''
    # extract test specific params
    (axis, correction_mode, srange_correction, az_time_correction,
     test_rdrgrid) = test_case

    in_raster = isce3.io.Raster(os.path.join(iscetest.data,
                                            f"geocodeslc/{axis}.slc"))

    Path(f"{axis}{correction_mode}_raster.geo").touch()
    out_raster = isce3.io.Raster(f"{axis}_{correction_mode}_raster.geo",
                                 unit_test_params.geogrid.width,
                                 unit_test_params.geogrid.length, 1,
                                 gdal.GDT_CFloat32, "ENVI")

    geocode_slc_raster(output_raster=out_raster,
        input_raster=in_raster,
        dem_raster=unit_test_params.dem_raster,
        radargrid=test_rdrgrid,
        geogrid=unit_test_params.geogrid,
        orbit=unit_test_params.orbit,
        native_doppler=unit_test_params.native_doppler,
        image_grid_doppler=unit_test_params.img_doppler,
        ellipsoid=isce3.core.Ellipsoid(),
        threshold_geo2rdr=1.0e-9,
        numiter_geo2rdr=25,
        lines_per_block=1000,
        flatten=False,
        az_time_correction=az_time_correction,
        srange_correction=srange_correction)

    # set geotransform
    out_raster.set_geotransform(unit_test_params.geotrans)


def test_run_raster_mode(unit_test_params):
    '''
    run geocodeSlc raster bindings with same parameters as C++ test to make
    sure it does not crash
    '''
    # run raster mode for all test cases
    for test_case in geocode_slc_test_cases(unit_test_params):
        run_geocode_slc_raster(test_case, unit_test_params)


def validate_raster(unit_test_params, mode, raster_layer=1):
    '''
    validate test outputs
    '''
    # check values of geocoded outputs
    for axis, correction_mode, *_, \
        in geocode_slc_test_cases(unit_test_params):

        # get phase of complex test data
        test_raster = f"{axis}_{correction_mode}_{mode}.geo"
        ds = gdal.Open(test_raster, gdal.GA_ReadOnly)
        test_arr = np.angle(ds.GetRasterBand(raster_layer).ReadAsArray())
        # mask with NaN since NaN is used to mark invalid pixels
        test_mask = np.isnan(test_arr)
        test_arr = np.ma.masked_array(test_arr, mask=test_mask)

        # use geotransform to make lat/lon mesh
        ny, nx = test_arr.shape
        meshx, meshy = np.meshgrid(np.arange(nx), np.arange(ny))

        # calculate and check error within bounds
        if axis == 'x':
            grid_lon = np.ma.masked_array(unit_test_params.x0 +
                                          meshx * unit_test_params.dx,
                                          mask=test_mask)

            err = np.nanmax(np.abs(test_arr - grid_lon))
        else:
            grid_lat = np.ma.masked_array(unit_test_params.y0 +
                                          meshy * unit_test_params.dy,
                                          mask=test_mask)

            err = np.nanmax(np.abs(test_arr - grid_lat))

        # check max diff of masked arrays
        assert(err < 1.0e-6), f'{test_raster} max error fail'


def test_array_mode(unit_test_params):
    validate_raster(unit_test_params, 'array')


def test_arrays_mode(unit_test_params):
    validate_raster(unit_test_params, 'arrays', 1)
    validate_raster(unit_test_params, 'arrays', 2)


def test_raster_mode(unit_test_params):
    validate_raster(unit_test_params, 'raster')
