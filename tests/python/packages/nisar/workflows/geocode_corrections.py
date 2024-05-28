import argparse
import itertools
import json
import os
import types

import numpy as np
import pytest

import isce3
from isce3.atmosphere.tec_product import (tec_lut2d_from_json_az,
                                          tec_lut2d_from_json_srg)
import iscetest
from nisar.products.readers import SLC
from nisar.workflows.geocode_corrections import get_az_srg_corrections
from nisar.workflows.gslc_runconfig import GSLCRunConfig


def make_zero_tec_file(unit_test_params):
    '''
    Create TEC file with all zero TEC values and a slightly larger time domain
    than that of the radar grid or orbit (time domain of whichever is larger is
    used) to be used to generate azimuth and slant range LUT2d's. Resulting TEC
    file will not yield any meaningful TEC LUT2d's and is only needed to test
    if time bounds of resulting TEC LUT2d's are in bounds.
    '''
    radargrid = unit_test_params.radargrid
    orbit = unit_test_params.orbit

    # Determine if radar grid or orbit has the larger time domain.
    # Unit test RSLC radar grid and orbit have the same reference epoch so
    # determining start/stop w.r.t time since reference epoch is sufficient.
    tec_start = radargrid.sensing_datetime[0] \
        if radargrid.sensing_start < orbit.start_time else \
        orbit.start_datetime
    tec_stop = radargrid.sensing_datetime(radargrid.length - 1) \
        if radargrid.sensing_start > orbit.start_time else \
        orbit.end_datetime

    # Snap seconds to multiple of 10 to be consistent with actual TEC product.
    # Floor start and ceil end to get larger time domain.
    snap = 10
    tec_start.second = int(np.floor(tec_start.second / snap) * snap)
    s = int(np.ceil(tec_stop.second / snap) * snap) - tec_stop.second
    tec_stop += isce3.core.TimeDelta(s)

    # Get list of TEC times as UTC strings.
    def _tec_time_generator():
        # Convenience function to aid following list comprehension.
        t = tec_start
        while t <= tec_stop:
            yield t.isoformat()
            # Increment by 10 seconds.
            t += isce3.core.TimeDelta(10.0)
    utc_times = [t for t in _tec_time_generator()]

    # Construct TEC JSON dict with zeros for all TEC values.
    tec_dict = {"utc":utc_times}
    tec_keys = ["totTecNr", "topTecNr", "totTecFr", "topTecFr"]
    tec_zeros = list(np.zeros(len(utc_times)))
    for tec_key in tec_keys:
        tec_dict[tec_key] = tec_zeros

    # Write to disk.
    with open(str(unit_test_params.tec_json_path), 'w') as fp:
        json.dump(tec_dict, fp)


@pytest.fixture(scope='module')
def unit_test_params(tmp_path_factory):
    '''
    test parameters shared by all tec_product tests
    '''
    # load h5 for doppler and orbit
    params = types.SimpleNamespace()

    # create temporary output directory for test output
    params.out_dir = tmp_path_factory.mktemp('test_output')

    input_h5_path = os.path.join(iscetest.data, "envisat.h5")

    params.radargrid = isce3.product.RadarGridParameters(input_h5_path)

    # init SLC object and extract necessary test params from it
    params.rslc_obj = SLC(hdf5file=input_h5_path)

    params.orbit = params.rslc_obj.getOrbit()

    # TEC JSON containing TEC values that generate range offsets that match the
    # fixed range offset used to test range correction
    params.tec_json_path = params.out_dir / 'test_tec.json'
    make_zero_tec_file(params)

    return params


def _validate_lut(lut, orbit, radargrid, az_or_srg):
    """
    Ensure given LUT2d time domain lies within that of the orbit and and
    outside that of the radar grid. Unit test RSLC from envisat has a radar
    grid whose time domain lies entirely within that of the orbit.
    """
    srg_start_stop = [lut.x_start,
                      lut.x_start + lut.x_spacing * (lut.width - 1)]
    for srg, start_or_stop in zip(srg_start_stop, ['start', 'stop']):
        # Check if outside radar grid time domain.
        assert not(radargrid.starting_range < srg < radargrid.end_range), \
               f'{az_or_srg} LUT2d {start_or_stop} outside radar grid range domain'

    t_start_stop = [lut.y_start,
                    lut.y_start + lut.y_spacing * (lut.length - 1)]
    for t, start_or_stop in zip(t_start_stop, ['start', 'stop']):
        # Check if within orbit time domain.
        assert orbit.start_time < t < orbit.end_time, \
               f'{az_or_srg} LUT2d {start_or_stop} outside orbit time domain'
        # Check if outside radar grid time domain.
        assert not(radargrid.sensing_start < t < radargrid.sensing_stop), \
               f'{az_or_srg} LUT2d {start_or_stop} outside radar grid time domain'


def test_geocode_corrections(unit_test_params):
    """
    Generate slant range TEC to ensure function does not crash. Actual
    validation occurs in unit test for pybind.cuda.geocode where a synthetic
    TEC is created to offset a range offset added to the radar grid starting
    range.
    """
    test_yaml = os.path.join(iscetest.data, 'geocodeslc/test_gslc.yaml')

    # load text then substitute test directory paths since data dir is read only
    with open(test_yaml) as fh_test_yaml:
        test_yaml = fh_test_yaml.read(). \
            replace('@ISCETEST@', iscetest.data). \
            replace('@TEST_BLOCK_SZ_X@', '133'). \
            replace('@TEST_BLOCK_SZ_Y@', '1000')

    # create CLI input namespace with yaml text instead of file path
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # init runconfig object and set dict items common to all tests
    runconfig = GSLCRunConfig(args)
    cfg = runconfig.cfg
    cfg['product_path_group']['scratch_path'] = unit_test_params.out_dir

    # values to test with and without SET
    set_options = [True, False]
    # values to test with and without TEC
    tec_files = [None, unit_test_params.tec_json_path]
    for set_enabled, tec_file in itertools.product(set_options, tec_files):
        # modify config dict to enable SET and TEC
        cfg['processing']['correction_luts']['solid_earth_tides_enabled'] = set_enabled
        cfg["dynamic_ancillary_file_group"]['tec_file'] = tec_file
        tec_enabled = tec_file is not None

        az_corrections, srange_correction = \
                get_az_srg_corrections(cfg,
                                       unit_test_params.rslc_obj,
                                       'A',
                                       unit_test_params.orbit)

        for corr_lut2d, corr_type in zip([az_corrections, srange_correction],
                                         ['azimuth', 'slant range']):
            # if neither correction enabled no data is in either LUT2d
            if (not set_enabled and not tec_enabled) \
                    or (set_enabled and not tec_enabled and corr_type == "slant_range"):
                assert not corr_lut2d.have_data
            else:
                _validate_lut(srange_correction,
                              unit_test_params.orbit,
                              unit_test_params.radargrid,
                              corr_type)
