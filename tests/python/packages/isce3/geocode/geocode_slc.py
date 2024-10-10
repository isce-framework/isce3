#!/usr/bin/env python3
'''
test isce3.geocode.geocode_slc array and raster modes
'''
import itertools
import json
import os
from pathlib import Path
import types

import journal
import numpy as np
from osgeo import gdal
import pytest
from scipy import interpolate

import iscetest
import isce3
from isce3.atmosphere.tec_product import tec_lut2d_from_json_srg
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
        # create extrapolator/interpolators for near and far
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


def make_subswaths():
    """
    A helper function that creates a SubSwaths object based on axis, where
    subswath start and stop indices per range line defined by the 30th
    and 70th percentile of the angle of the SLC array values.

    Returns
    ----------
    isce3.product.SubSwaths
        The generated SubSwaths object.
    """

    # dict for subswath per axis, x or y
    subswaths = {}
    # dicts for lo and hi angle of complex SLC after masking of by 30th and
    # 70th percentiles
    val_lo = {}
    val_hi = {}

    # iterate over axis
    for xy in "xy":
        # load SLC as angle
        axis_input_path = f"{iscetest.data}/geocodeslc/{xy}.slc"
        ds = gdal.Open(axis_input_path, gdal.GA_ReadOnly)
        arr = np.angle(ds.GetRasterBand(1).ReadAsArray())

        # get 30th/lo to 70th/hi percentile of angle of current input
        val_lo[xy], val_hi[xy] = np.percentile(arr, [30, 70])

        # mask for anything below 30th percentile or above 70th percentile
        mask = np.logical_or(arr < val_lo[xy], arr > val_hi[xy])

        # populate array of start and stop at each range line
        length, width = mask.shape
        subswath_array = np.zeros((length, 2), np.int32)
        for j in range(length):
            # get unmasked indices
            inds_unmasked = np.nonzero(mask[j, :] == False)[0]

            # from front of row find first unmasked element
            subswath_array[j, 0] = inds_unmasked[0]

            # from back of row find last unmasked element
            subswath_array[j, 1] = inds_unmasked[-1] + 1

        # create subswath object for current axis
        n_subswaths = 1
        subswaths[xy] = isce3.product.SubSwaths(length,
                                                width,
                                                n_subswaths)
        subswaths[xy].set_valid_samples_array(1, subswath_array)

    return subswaths, val_lo, val_hi


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

    # half pixel offset and grid size in radians for validation
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

    # prepare subswath that masks radar grid so nothing is geocoded
    params.subswaths, params.subswaths_lo_val, params.subswaths_hi_val = \
        make_subswaths()

    return params


def geocode_slc_test_cases(unit_test_params):
    '''
    Generator for geocodeSlc test cases

    Given a radar grid, generate correction LUT2ds in range and azimuth
    directions. Returns axis, offset mode name, range and azimuth correction
    LUT2ds and offset corrected radar grid.
    '''
    test_case = types.SimpleNamespace()

    radargrid = unit_test_params.radargrid
    offset_factor = unit_test_params.offset_factor

    rg_pxl_spacing = radargrid.range_pixel_spacing
    range_offset = offset_factor * rg_pxl_spacing
    az_time_interval = 1 / radargrid.prf
    azimuth_offset = offset_factor * az_time_interval

    # despite uniform value LUT2d set to interp mode nearest just in case
    method = isce3.core.DataInterpMethod.NEAREST

    # array of ones to be multiplied by respective offset value
    # shape unchanging; no need to be in loop as only starting values change
    ones = np.ones(radargrid.shape)

    for axis, flatten_enabled in itertools.product('xy', [True, False]):
        test_case.axis = axis
        test_case.flatten_enabled = flatten_enabled
        for test_mode in ['', 'rg', 'az', 'rg_az', 'tec', 'subswath']:
            test_case.test_mode = test_mode

            # create radar and apply positive offsets in range and azimuth
            test_case.radargrid = radargrid.copy()

            # apply offsets as required by mode
            if 'rg' in test_mode or 'tec' == test_mode:
                test_case.radargrid.starting_range += range_offset
            if 'az' in test_mode:
                test_case.radargrid.sensing_start += azimuth_offset

            # slant range vector for LUT2d
            srange_vec = np.linspace(test_case.radargrid.starting_range,
                                     test_case.radargrid.end_range,
                                     radargrid.width)

            # azimuth vector for LUT2d
            az_time_vec = np.linspace(test_case.radargrid.sensing_start,
                                      test_case.radargrid.sensing_stop,
                                      radargrid.length)

            # corrections LUT2ds will use the negative offsets
            # should cancel positive offset applied to radar grid
            srange_correction = isce3.core.LUT2d()
            if 'rg' in test_mode:
                srange_correction = isce3.core.LUT2d(srange_vec,
                                                     az_time_vec,
                                                     range_offset * ones,
                                                     method)
            elif 'tec' == test_mode:
                srange_correction = \
                    tec_lut2d_from_json_srg(unit_test_params.tec_json_path,
                                        unit_test_params.center_freq,
                                        unit_test_params.orbit,
                                        test_case.radargrid,
                                        isce3.core.LUT2d(),
                                        unit_test_params.dem_path)
            test_case.srange_correction = srange_correction

            az_time_correction = isce3.core.LUT2d()
            if 'az' in test_mode:
                az_time_correction = isce3.core.LUT2d(srange_vec,
                                                      az_time_vec,
                                                      azimuth_offset * ones,
                                                      method)
            test_case.az_time_correction = az_time_correction

            test_case.need_flatten_phase_raster = \
                axis == 'x' and not flatten_enabled and not test_mode

            # allow flatten for special case to return flatten true if
            # x-axis and no offsets
            if flatten_enabled and ((axis == 'x' and test_mode != '')
                                    or axis == 'y'):
                continue

            # redundant to test subswath with flattening disabled and enabled
            test_case.subswath_enabled = \
                test_mode == 'subswath' and not flatten_enabled

            # prepare input and output paths
            test_case.input_path = \
                os.path.join(iscetest.data, f"geocodeslc/{axis}.slc")

            flat_str = 'flattened' if test_case.flatten_enabled else 'unflattened'
            common_output_prefix = \
                f'{axis}_{test_case.test_mode}_geocode_slc_mode_{flat_str}'
            test_case.output_path = f'{common_output_prefix}.geo'
            test_case.flatten_phase_path = f'{common_output_prefix}_phase.bin'

            yield test_case


def run_geocode_slc_arrays(test_case, unit_test_params, extra_input=False,
                           non_matching_shape=False):
    '''
    wrapper for geocode_slc array mode
    '''
    out_shape = (unit_test_params.geogrid.width,
                 unit_test_params.geogrid.length)

    # load input as list of arrays
    ds = gdal.Open(test_case.input_path, gdal.GA_ReadOnly)
    arr = ds.GetRasterBand(1).ReadAsArray()
    in_list = [arr, arr]
    # if extra input enabled, append extra array to input list
    if extra_input:
        in_list.append(arr)

    # if forcing error, change output file name to not break output validation
    output_path = test_case.output_path.replace('geocode_slc_mode', 'arrays')
    if extra_input or non_matching_shape:
        output_path += '_broken'
    Path(output_path).touch()

    # list of empty array to be written to by geocode_slc array mode
    out_zeros = np.zeros(out_shape, dtype=np.complex64)
    # if non matching shape enabled, ensure output array shapes do not match
    if non_matching_shape:
        wrong_shape = (out_shape[0], out_shape[1] + 1)
        out_list = [out_zeros, np.zeros(wrong_shape, dtype=np.complex64)]
    else:
        out_list = [out_zeros, out_zeros.copy()]
    mask_data = np.zeros(out_shape, dtype=np.ubyte)

    isce3.geocode.geocode_slc(
        geo_data_blocks=out_list,
        mask_block=mask_data,
        rdr_data_blocks=in_list,
        dem_raster=unit_test_params.dem_raster,
        radargrid=test_case.radargrid,
        geogrid=unit_test_params.geogrid,
        orbit=unit_test_params.orbit,
        native_doppler= unit_test_params.native_doppler,
        image_grid_doppler=unit_test_params.img_doppler,
        ellipsoid=isce3.core.Ellipsoid(),
        threshold_geo2rdr=1.0e-9,
        num_iter_geo2rdr=25,
        first_azimuth_line=0,
        first_range_sample=0,
        flatten=test_case.flatten_enabled,
        az_time_correction=test_case.az_time_correction,
        srange_correction=test_case.srange_correction)

    # set geotransform in output raster
    out_raster = isce3.io.Raster(output_path, unit_test_params.geogrid.width,
                                 unit_test_params.geogrid.length, 2,
                                 gdal.GDT_CFloat32,  "ENVI")
    out_raster.set_geotransform(unit_test_params.geotrans)
    out_raster.close_dataset()

    # write output to raster
    ds = gdal.Open(output_path, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(out_list[0])
    ds.GetRasterBand(2).WriteArray(out_list[1])


def run_geocode_slc_array_subswath(test_case, unit_test_params):
    '''
    wrapper for geocode_slc array mode
    '''
    # extract test specific params
    out_shape = (unit_test_params.geogrid.width,
                 unit_test_params.geogrid.length)

    # load input as list of arrays
    ds = gdal.Open(test_case.input_path, gdal.GA_ReadOnly)
    in_data = ds.GetRasterBand(1).ReadAsArray()

    # list of empty array to be written to by geocode_slc array mode
    out_data = np.zeros(out_shape, dtype=np.complex64)
    mask_data = np.zeros(out_shape, dtype=np.ubyte)
    
    # Populate geocode_slc kwargs as needed
    kwargs = {}
    if test_case.need_flatten_phase_raster:
        flatten_phase_data = np.nan * np.zeros(out_shape,dtype=np.float64)
        kwargs['flatten_phase_block'] = flatten_phase_data

    if test_case.subswath_enabled:
        kwargs['subswaths'] = unit_test_params.subswaths[test_case.axis]
    
    isce3.geocode.geocode_slc(
        geo_data_blocks=out_data,
        rdr_data_blocks=in_data,
        dem_raster=unit_test_params.dem_raster,
        radargrid=test_case.radargrid,
        geogrid=unit_test_params.geogrid,
        orbit=unit_test_params.orbit,
        native_doppler=unit_test_params.native_doppler,
        image_grid_doppler=unit_test_params.img_doppler,
        ellipsoid=isce3.core.Ellipsoid(),
        threshold_geo2rdr=1.0e-9,
        num_iter_geo2rdr=25,
        mask_block=mask_data,
        first_azimuth_line=0,
        first_range_sample=0,
        flatten=test_case.flatten_enabled,
        az_time_correction=test_case.az_time_correction,
        srange_correction=test_case.srange_correction,
        **kwargs)

    # output file name for geocodeSlc array mode
    output_path = test_case.output_path.replace('geocode_slc_mode', 'array')
    Path(output_path).touch()

    # set geotransform in output raster
    out_raster = isce3.io.Raster(output_path, unit_test_params.geogrid.width,
                                 unit_test_params.geogrid.length, 1,
                                 gdal.GDT_CFloat32,  "ENVI")
    out_raster.set_geotransform(unit_test_params.geotrans)
    del out_raster

    # write output to raster
    ds = gdal.Open(output_path, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(out_data)

    # create flatten phase raster if not geocoding with flattening enabled
    if test_case.need_flatten_phase_raster:
        flatten_phase_path = \
            test_case.flatten_phase_path.replace('geocode_slc_mode', 'array')
        # flatten phase output file name for geocodeSlc array mode
        Path(flatten_phase_path).touch()

        # set geotransform in flatten phase output raster
        flatten_raster = isce3.io.Raster(flatten_phase_path,
                                         unit_test_params.geogrid.width,
                                         unit_test_params.geogrid.length, 1,
                                         gdal.GDT_Float64,  "ENVI")
        del flatten_raster

        # write output to raster
        ds = gdal.Open(flatten_phase_path, gdal.GA_Update)
        ds.GetRasterBand(1).WriteArray(flatten_phase_data)

    ds = None


def run_geocode_slc_array(test_case, unit_test_params):
    '''
    wrapper for geocode_slc array mode without subswath mask
    '''
    # extract test specific params
    out_shape = (unit_test_params.geogrid.width,
                 unit_test_params.geogrid.length)

    # load input as list of arrays
    ds = gdal.Open(test_case.input_path, gdal.GA_ReadOnly)
    in_data = ds.GetRasterBand(1).ReadAsArray()

    # list of empty array to be written to by geocode_slc array mode
    out_data = np.zeros(out_shape, dtype=np.complex64)
     
    # Populate geocode_slc kwargs as needed
    kwargs = {}
    if test_case.need_flatten_phase_raster:
        flatten_phase_data = np.nan * np.zeros(out_shape,dtype=np.float64)
        kwargs['flatten_phase_block'] = flatten_phase_data

    isce3.geocode.geocode_slc(
        geo_data_blocks=out_data,
        rdr_data_blocks=in_data,
        dem_raster=unit_test_params.dem_raster,
        radargrid=test_case.radargrid,
        geogrid=unit_test_params.geogrid,
        orbit=unit_test_params.orbit,
        native_doppler= unit_test_params.native_doppler,
        image_grid_doppler=unit_test_params.img_doppler,
        ellipsoid=isce3.core.Ellipsoid(),
        threshold_geo2rdr=1.0e-9,
        num_iter_geo2rdr=25,
        first_azimuth_line=0,
        first_range_sample=0,
        flatten=test_case.flatten_enabled,
        az_time_correction=test_case.az_time_correction,
        srange_correction=test_case.srange_correction,
        **kwargs)

    # output file name for geocodeSlc array mode
    output_path = test_case.output_path.replace('geocode_slc_mode', 'array')
    Path(output_path).touch()

    # set geotransform in output raster
    out_raster = isce3.io.Raster(output_path, unit_test_params.geogrid.width,
                                 unit_test_params.geogrid.length, 1,
                                 gdal.GDT_CFloat32,  "ENVI")
    out_raster.set_geotransform(unit_test_params.geotrans)
    del out_raster

    # write output to raster
    ds = gdal.Open(output_path, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(out_data)

    # create flatten phase raster if not geocoding with flattening enabled
    if test_case.need_flatten_phase_raster:
        flatten_phase_path = \
            test_case.flatten_phase_path.replace('geocode_slc_mode', 'array')
        # flatten phase output file name for geocodeSlc array mode
        Path(flatten_phase_path).touch()

        # set geotransform in flatten phase output raster
        flatten_raster = isce3.io.Raster(flatten_phase_path,
                                         unit_test_params.geogrid.width,
                                         unit_test_params.geogrid.length, 1,
                                         gdal.GDT_Float64,  "ENVI")
        del flatten_raster

        # write output to raster
        ds = gdal.Open(flatten_phase_path, gdal.GA_Update)
        ds.GetRasterBand(1).WriteArray(flatten_phase_data)

    ds = None


def test_run_array_mode(unit_test_params):
    '''
    run geocodeSlc array bindings with same parameters as C++ test to make sure
    it does not crash
    '''
    # run array mode for all test cases
    for test_case in geocode_slc_test_cases(unit_test_params):
        print(test_case)
        if test_case.test_mode == 'subswath':
            run_geocode_slc_array_subswath(test_case, unit_test_params)
        else:
            run_geocode_slc_array(test_case, unit_test_params)
    print("ALL array mode run completed")

def test_run_arrays_mode(unit_test_params):
    '''
    run geocodeSlc list of array bindings with same parameters as C++ test to
    make sure it does not crash
    '''
    # run array mode for all test cases
    for test_case in geocode_slc_test_cases(unit_test_params):
        # skip flattening for multiple arrays
        if test_case.flatten_enabled:
            continue
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
    # prepare input and output(s)
    in_raster = isce3.io.Raster(test_case.input_path)

    output_path = test_case.output_path.replace('geocode_slc_mode', 'raster')
    Path(output_path).touch()
    out_raster = isce3.io.Raster(output_path, unit_test_params.geogrid.width,
                                 unit_test_params.geogrid.length, 1,
                                 gdal.GDT_CFloat32, "ENVI")

    # prepare flattening phase raster if necessary
    flatten_kwargs = {}
    if test_case.need_flatten_phase_raster:
        flatten_phase_path = \
            test_case.flatten_phase_path.replace('geocode_slc_mode', 'raster')
        flatten_phase_raster = isce3.io.Raster(flatten_phase_path,
                                               unit_test_params.geogrid.width,
                                               unit_test_params.geogrid.length,
                                               1, gdal.GDT_Float64, "ENVI")
        flatten_kwargs['flatten_phase_raster'] = flatten_phase_raster

    geocode_slc_raster(output_raster=out_raster,
        input_raster=in_raster,
        dem_raster=unit_test_params.dem_raster,
        radargrid=test_case.radargrid,
        geogrid=unit_test_params.geogrid,
        orbit=unit_test_params.orbit,
        native_doppler=unit_test_params.native_doppler,
        image_grid_doppler=unit_test_params.img_doppler,
        ellipsoid=isce3.core.Ellipsoid(),
        threshold_geo2rdr=1.0e-9,
        numiter_geo2rdr=25,
        lines_per_block=1000,
        flatten=test_case.flatten_enabled,
        az_time_correction=test_case.az_time_correction,
        srange_correction=test_case.srange_correction,
        **flatten_kwargs)

    # set geotransform
    out_raster.set_geotransform(unit_test_params.geotrans)


def test_run_raster_mode(unit_test_params):
    '''
    run geocodeSlc raster bindings with same parameters as C++ test to make
    sure it does not crash
    '''
    # run raster mode for all test cases
    for test_case in geocode_slc_test_cases(unit_test_params):
        # skip subswath masking for deprecated raster mode - not supported
        if test_case.subswath_enabled:
            continue
        run_geocode_slc_raster(test_case, unit_test_params)


def _get_masked_raster_array(raster_path, raster_layer=1, array_op=None):
    # open raster as dataset and convert to angle as needed
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    test_arr = ds.GetRasterBand(raster_layer).ReadAsArray()
    if array_op is not None:
        test_arr = array_op(test_arr)

    # mask with NaN since NaN is used to mark invalid pixels
    test_mask = np.isnan(test_arr)
    test_arr = np.ma.masked_array(test_arr, mask=test_mask)

    return test_arr


def check_raster_path(path, params):
    if not os.path.exists(path):
        print(path)
        print(params)


def validate_slc_raster(unit_test_params, mode, raster_layer=1):
    '''
    validate test outputs
    '''
    # check values of geocoded outputs
    for test_case in geocode_slc_test_cases(unit_test_params):

        # skip flattening cases - flattened geocoded SLC only for comparison
        # against unflattened geocoded SLC with flattening applied
        # skip raster mode when subswath enabled - raster mode does not support
        # subswath masking
        raster_susbswath = mode == 'raster' and test_case.subswath_enabled
        if test_case.flatten_enabled or raster_susbswath:
            continue

        # get phase of complex test data
        output_path = test_case.output_path.replace('geocode_slc_mode', mode)
        test_phase_arr = \
            _get_masked_raster_array(output_path, raster_layer, np.angle)
        test_mask = test_phase_arr.mask

        # use geotransform to make lat/lon mesh
        ny, nx = test_phase_arr.shape
        meshx, meshy = np.meshgrid(np.arange(nx), np.arange(ny))

        # calculate and check error within bounds
        if test_case.axis == 'x':
            grid_lon = np.ma.masked_array(unit_test_params.x0 +
                                          meshx * unit_test_params.dx,
                                          mask=test_mask)

            err = np.nanmax(np.abs(test_phase_arr - grid_lon))
        else:
            grid_lat = np.ma.masked_array(unit_test_params.y0 +
                                          meshy * unit_test_params.dy,
                                          mask=test_mask)

            err = np.nanmax(np.abs(test_phase_arr - grid_lat))

        # check max diff of masked arrays
        assert(err < 1.0e-6), f'{test_case.output_path} max error fail'


def test_validate_array_mode(unit_test_params):
    validate_slc_raster(unit_test_params, 'array')


def test_validate_arrays_mode(unit_test_params):
    validate_slc_raster(unit_test_params, 'arrays', 1)
    validate_slc_raster(unit_test_params, 'arrays', 2)


def test_validate_raster_mode(unit_test_params):
    validate_slc_raster(unit_test_params, 'raster')


def test_flatten_application():
    for run_mode in ['array', 'raster']:
        # load SLCs geocoded along x-axis with flattening and without offset
        # applied as phase
        x_phase_flattened_cpp = \
            _get_masked_raster_array(f'x__{run_mode}_flattened.geo', 1,
                                     np.angle)

        # load SLCs geocoded along x-axis with flattening and without offset
        # applied as complex number
        x_slc_unflattened = \
            _get_masked_raster_array(f'x__{run_mode}_unflattened.geo')

        # load corresponding geocoded flattening phase and
        flattening_phase = _get_masked_raster_array(
            f'x__{run_mode}_unflattened_phase.bin', 1,
            lambda x: np.exp(1j * x))
        x_phase_flattened_py = np.angle(x_slc_unflattened * flattening_phase)

        # compare 2 SLCs
        assert(np.nanmax(x_phase_flattened_cpp - x_phase_flattened_py) < 1e-6)


def test_subswath_masking(unit_test_params):
    for test_case in geocode_slc_test_cases(unit_test_params):
        # only test if subswath mode is enabled
        if not test_case.subswath_enabled:
            continue

        # array modes and corresponding raster band indices
        modes_bands = [('array', 1), ('arrays', 1), ('arrays', 2)]
        for arr_mode, i_layer in modes_bands:
            # load subswath masked GSLC (only for array mode)
            output_path = test_case.output_path.replace('geocode_slc_mode',
                                                        arr_mode)
            subswath_masked_arr = _get_masked_raster_array(output_path,
                                                           i_layer,
                                                           array_op=np.angle)

            # create mask of output where values between 30th and 70th percentile
            # are True and everything else is False
            in_subswath_mask_bounds = \
                np.logical_and(subswath_masked_arr >= unit_test_params.subswaths_lo_val[test_case.axis],
                               subswath_masked_arr <= unit_test_params.subswaths_hi_val[test_case.axis])

            # check that angles of all geocoded values are between 30th and 70th
            # percentile
            assert(np.all(in_subswath_mask_bounds),
                   f"{test_case.output_path} with not all NaN")
