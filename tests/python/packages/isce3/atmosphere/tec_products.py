import json
import os
import types

import numpy as np

import isce3
from isce3.atmosphere.tec_product import (tec_lut2d_from_json_az,
                                          tec_lut2d_from_json_srg)
from isce3.geometry import compute_incidence_angle
import iscetest
from nisar.products.readers import SLC
import pytest
from scipy import interpolate


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
    rslc = SLC(hdf5file=input_h5_path)

    params.orbit = rslc.getOrbit()

    params.center_freq = rslc.getSwathMetadata().processed_center_frequency

    # create DEM raster object
    params.dem_path = os.path.join(iscetest.data, "geocode/zeroHeightDEM.geo")

    # multiplicative factor applied to range pixel spacing and azimuth time
    # interval to be added to starting range and azimuth time of radar grid
    params.offset_factor = 10.0

    # TEC JSON containing TEC values that generate range offsets that match the
    # fixed range offset used to test range correction
    params.tec_json_path = params.out_dir / 'test_tec.json'
    make_zero_tec_file(params)

    return params


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


def _validate_tec_lut(tec_lut, orbit, radargrid, az_or_srg):
    """
    Ensure given TEC LUT2d time domain lies within that of the orbit and and
    outside that of the radar grid. Unit test RSLC from envisat has a radar
    grid whose time domain lies entirely within that of the orbit.
    """
    t_start_stop = [tec_lut.y_start,
                    tec_lut.y_start + tec_lut.y_spacing * (tec_lut.length - 1)]
    for t, start_or_stop in zip(t_start_stop, ['start', 'stop']):
        # Check if within orbit time domain.
        assert orbit.start_time < t < orbit.end_time, \
               f'{az_or_srg} LUT2d {start_or_stop} outside orbit time domain'
        # Check if outside radar grid time domain.
        assert not(radargrid.sensing_start < t < radargrid.sensing_stop), \
               f'{az_or_srg} LUT2d {start_or_stop} outside radar grid time domain'


def test_srg_tec(unit_test_params):
    """
    Generate slant range TEC to ensure function does not crash. Actual
    validation occurs in unit test for pybind.cuda.geocode where a synthetic
    TEC is created to offset a range offset added to the radar grid starting
    range.
    """
    srange_correction = \
        tec_lut2d_from_json_srg(unit_test_params.tec_json_path,
                                unit_test_params.center_freq,
                                unit_test_params.orbit,
                                unit_test_params.radargrid,
                                isce3.core.LUT2d(),
                                unit_test_params.dem_path)
    _validate_tec_lut(srange_correction,
                      unit_test_params.orbit,
                      unit_test_params.radargrid,
                      'slant range')


def test_az_tec(unit_test_params):
    """
    Generate azimuth TEC to ensure function does not crash. Actual validation
    should occur in unit test for pybind.cuda.geocode where a synthetic TEC is
    created to offset a azimuth time offset added to the radar grid starting
    time.
    """
    az_correction = \
        tec_lut2d_from_json_az(unit_test_params.tec_json_path,
                               unit_test_params.center_freq,
                               unit_test_params.orbit,
                               unit_test_params.radargrid)
    _validate_tec_lut(az_correction,
                      unit_test_params.orbit,
                      unit_test_params.radargrid,
                      'azimuth')
