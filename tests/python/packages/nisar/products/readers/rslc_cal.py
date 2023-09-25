from datetime import datetime, timedelta
import iscetest
from nisar.products.readers.rslc_cal import (parse_rslc_calibration,
    check_cal_validity_dates)
import nisar.workflows.helpers as helpers
from numpy import deg2rad, exp, pi, iscomplexobj
import numpy.testing as npt
from pathlib import Path
import pytest
import yamale


def get_filename():
    return Path(iscetest.data) / "rslc_calibration.yaml"

def get_dummy_scale(x):
    return (1.0 + x) * exp(1j * deg2rad(x))


def test_parse():
    cal = parse_rslc_calibration(get_filename())

    npt.assert_equal(cal.common_delay, 0.05)

    npt.assert_equal(cal.hh.delay, 0.01)
    npt.assert_equal(cal.hh.scale, get_dummy_scale(0.01))
    npt.assert_equal(cal.hh.scale_slope, 0.01 * 180/pi)

    npt.assert_equal(cal.hv.delay, 0.02)
    npt.assert_equal(cal.hv.scale, get_dummy_scale(0.02))
    npt.assert_equal(cal.hv.scale_slope, 0.02 * 180/pi)

    npt.assert_equal(cal.vh.delay, 0.03)
    npt.assert_equal(cal.vh.scale, get_dummy_scale(0.03))
    npt.assert_equal(cal.vh.scale_slope, 0.03 * 180/pi)

    npt.assert_equal(cal.vv.delay, 0.04)
    npt.assert_equal(cal.vv.scale, get_dummy_scale(0.04))
    npt.assert_equal(cal.vv.scale_slope, 0.04 * 180/pi)

    generated = datetime(2023, 1, 19, 12, 27, 10)
    npt.assert_equal(cal.generated_date, generated)

    validity = datetime(2022, 1, 1, 12, 0, 0)
    npt.assert_equal(cal.valid_after_date, validity)

    npt.assert_equal(cal.valid_before_date, None)

    npt.assert_equal(cal.reference_range, 900e3)

    assert iscomplexobj(cal.hh.scale)


def test_bandwidth_specific():
    cal = parse_rslc_calibration(get_filename(), bandwidth=5e6)
    npt.assert_equal(cal.common_delay, 0.01)
    npt.assert_equal(cal.hh.delay, 0.01)
    npt.assert_equal(cal.vv.delay, 0.0)

    cal = parse_rslc_calibration(get_filename(), bandwidth=20e6)
    npt.assert_equal(cal.common_delay, 0.02)
    npt.assert_equal(cal.hh.delay, 0.01)
    npt.assert_equal(cal.vv.delay, 0.04)

    cal = parse_rslc_calibration(get_filename(), bandwidth=40e6)
    npt.assert_equal(cal.common_delay, 0.03)
    npt.assert_equal(cal.hh.delay, 0.01)
    npt.assert_equal(cal.vv.delay, 0.04)

    cal = parse_rslc_calibration(get_filename(), bandwidth=80e6)
    npt.assert_equal(cal.common_delay, 0.04)
    npt.assert_equal(cal.hh.delay, 0.01)
    npt.assert_equal(cal.vv.delay, 0.04)


def test_schema():
    schema = yamale.make_schema(
        f"{helpers.WORKFLOW_SCRIPTS_DIR}/schemas/rslc_calibration.yaml",
        parser="ruamel")
    data = yamale.make_data(get_filename(), parser="ruamel")
    yamale.validate(schema, data)


def test_dates():
    cal = parse_rslc_calibration(get_filename(), bandwidth=5e6)

    radar_time = cal.valid_after_date + timedelta(days=1)
    check_cal_validity_dates(cal, radar_time)

    radar_time = cal.valid_after_date - timedelta(days=1)
    with pytest.raises(Exception):
        check_cal_validity_dates(cal, radar_time)
