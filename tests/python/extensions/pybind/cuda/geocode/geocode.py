#/usr/bin/env python3

import itertools
import json
import os
import types

import numpy as np
from osgeo import gdal

import isce3
from isce3.atmosphere.tec_product import tec_lut2d_from_json_srg
from isce3.geometry import compute_incidence_angle
import iscetest
from nisar.products.readers import SLC
import pytest
from scipy import interpolate

def make_subswaths():
    # helper funtion to create subswath object based on axis where subswath
    # start and stop indices per range line defined by the 30th and 70th
    # percentile of the angle of the SLC array values

    # dict for subswath per axis, x or y
    subswaths = {}
    # dicts for lo and hi angle of complex SLC after masking of by 30th and
    # 70th percentiles
    val_lo = {}
    val_hi = {}

    # iterate over axis
    for i, xy in enumerate("xy"):
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


@pytest.fixture(scope='module')
def unit_test_params(tmp_path_factory):
    '''
    test parameters shared by all geocode tests
    '''
    # load h5 for doppler and orbit
    params = types.SimpleNamespace()

    # create temporary output directory for test output
    params.out_dir = tmp_path_factory.mktemp('test_output')
    # for debugging outputs in case of crash comment above and uncomment below
    # import pathlib
    # params.out_dir = pathlib.Path('test_output').mkdir(parents=True, exist_ok=True)
    # params.out_dir = pathlib.Path('test_output')

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

    # XXX This attribute of NISAR RSLCs is supposed to be the image grid Doppler
    # but for the envisat.h5 file, it's commonly used in isce3 unit tests to
    # represent native Doppler.
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
    params.x0 = np.radians(geogrid.start_x + geogrid.spacing_x / 2.0)
    params.dx = np.radians(geogrid.spacing_x)
    params.y0 = np.radians(geogrid.start_y + geogrid.spacing_y / 2.0)
    params.dy = np.radians(geogrid.spacing_y)

    # multiplicative factor applied to range pixel spacing and azimuth time
    # interval to be added to starting range and azimuth time of radar grid
    params.offset_factor = 10.0

    # TEC JSON containing TEC values that generate range offsets that match the
    # fixed range offset used to test range correction
    params.tec_json_path = params.out_dir / 'test_tec.json'
    make_tec_file(params)

    # compute unmasked validation arrays based on the geogrid being geocoded to
    # params.grid_lon - array where each column is the longitude of the geogrid
    # column
    # params.grid_lat - array where each row is the latitude of the geogrid row
    nx = geogrid.width
    ny = geogrid.length
    meshx, meshy = np.meshgrid(np.arange(nx), np.arange(ny))
    params.grid_lon = params.x0 + meshx * params.dx
    params.grid_lat = params.y0 + meshy * params.dy

    # prepare subswath that masks radar grid so nothing is geocoded
    params.subswaths, params.subswaths_lo_val, params.subswaths_hi_val = \
        make_subswaths()

    return params


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
    # geocode_test_cases()
    offset_factor = unit_test_params.offset_factor
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
    with open(str(unit_test_params.tec_json_path), 'w') as fp:
        json.dump(tec_dict, fp)


def geocode_test_cases(unit_test_params):
    '''
    Generator for CUDA geocode test cases

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

    axes = 'xy'
    test_modes = ['', 'rg', 'az', 'rg_az', 'tec', 'blocked', 'subswath']
    for axis, test_mode in itertools.product(axes, test_modes):
        is_block_mode = test_mode == 'blocked'

        # skip blocked if for y-axis. x-axis testing sufficient
        if axis == 'y' and is_block_mode:
            continue

        # if testing block mode, lines_per_block is chosen to be less than the
        # total number of lines in the dataset
        test_case.lines_per_block = 126 if is_block_mode else 1000
        test_case.axis = axis
        test_case.test_mode = test_mode

        # create radar and apply positive offsets in range and azimuth
        test_case.radargrid = radargrid.copy()

        # apply offsets as required by mode
        if 'rg' in test_mode or 'tec' == test_mode:
            test_case.radargrid.starting_range += range_offset
        if 'az' in test_mode:
            test_case.radargrid.sensing_start += azimuth_offset

        test_case.rdr_geometry = \
            isce3.container.RadarGeometry(test_case.radargrid,
                                          unit_test_params.orbit,
                                          unit_test_params.img_doppler)

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

        # prepare input and output paths
        test_case.input_path = \
            os.path.join(iscetest.data, f"geocodeslc/{axis}.slc")

        common_output_prefix = f'{axis}_{test_case.test_mode}'
        test_case.output_path = \
            unit_test_params.out_dir / f'{common_output_prefix}.geo'

        test_case.subswath_enabled = test_mode == 'subswath'

        # assign validation array based on axis
        if axis == 'x':
            test_case.validation_arr = unit_test_params.grid_lon
        else:
            test_case.validation_arr = unit_test_params.grid_lat

        yield test_case


def run_cuda_geocode(test_case, unit_test_params):
    # Set interp method
    interp_method = [isce3.core.DataInterpMethod.SINC]
    raster_dtype = [isce3.io.gdal.GDT_CFloat32]
    invalid_value = [np.nan]

    # init CUDA geocode obj for given test case
    cu_geocode = \
        isce3.cuda.geocode.Geocode(unit_test_params.geogrid,
                                   test_case.rdr_geometry,
                                   lines_per_block=test_case.lines_per_block)

    # output and input as arrays to match binding signature
    output_raster = [isce3.io.Raster(str(test_case.output_path),
                                     unit_test_params.geogrid.width,
                                     unit_test_params.geogrid.length, 1,
                                     gdal.GDT_CFloat32, "ENVI")]
    input_raster = [isce3.io.Raster(test_case.input_path)]

    # Populate geocode_slc kwargs as needed
    kwargs = {}
    if test_case.subswath_enabled:
        kwargs['subswaths'] = unit_test_params.subswaths[test_case.axis]

    # geocode raster for given test case
    cu_geocode.geocode_rasters(output_raster,
                               input_raster,
                               interp_method,
                               raster_dtype,
                               invalid_value,
                               unit_test_params.dem_raster,
                               native_doppler=unit_test_params.native_doppler,
                               az_time_correction=test_case.az_time_correction,
                               srange_correction=test_case.srange_correction,
                               **kwargs)

    output_raster[0].set_geotransform(unit_test_params.geotrans)


def test_cuda_geocode(unit_test_params):
    '''
    run CUDA geocode to make sure it does not crash
    '''
    # run array mode for all test cases
    for test_case in geocode_test_cases(unit_test_params):
        run_cuda_geocode(test_case, unit_test_params)


def _get_raster_array_masked(raster_path, array_op=None):
    # open raster as dataset, convert to angle as needed, and mask invalids
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    test_arr = ds.GetRasterBand(1).ReadAsArray()
    if array_op is not None:
        test_arr = array_op(test_arr)

    # mask with NaN since NaN is used to mark invalid pixels
    test_arr = np.ma.masked_invalid(test_arr)

    return test_arr


def test_validate(unit_test_params):
    '''
    validate test outputs
    '''
    # check values of geocoded rasters
    for test_case in geocode_test_cases(unit_test_params):
        # get masked array phase of complex test data and corresponding mask of
        # pixels outside geocoded radar grid
        test_phase_arr = \
            _get_raster_array_masked(str(test_case.output_path), np.angle)

        # compute maximum error test phase and validate
        err_arr = np.abs(test_phase_arr - test_case.validation_arr)

        err = np.nanmax(err_arr)

        assert(err < 5.0e-7), f'{str(test_case.output_path)} max error fail'

        # count number of invalids (NaN) and validate
        percent_invalid = \
            np.count_nonzero(np.ma.getmask(test_phase_arr)) \
            / test_phase_arr.size

        # skip percent invalid check for subswath enabled
        # percent valid pixels only computed for subswath less rasters
        if not test_case.subswath_enabled:
            assert(percent_invalid < 0.6), \
                f'{str(test_case.output_path)} percent invalid fail'


def test_subswath_masking(unit_test_params):
    for test_case in geocode_test_cases(unit_test_params):
        # only test if subswath mode is enabled
        if not test_case.subswath_enabled:
            continue

        # load angle of subswath masked output
        subswath_masked_arr = \
            _get_raster_array_masked(str(test_case.output_path),
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
