import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import iscetest
import numpy as np
import os
import numpy.testing as npt
from nisar.workflows.point_target_analysis import process_corner_reflector_csv, slc_pt_performance
import pytest


@pytest.mark.parametrize("kwargs", [
    dict(nov=16),
    dict(shift_domain='frequency'),
    dict(predict_null=True, window_type='kaiser', window_parameter=1.6),
    dict(cuts=True),
])
def test_point_target_analysis(kwargs):
    """ Test process that converts input point target lon/lat/ht to radar
        coordinates and computes the range/azimuth offsets of predicted
        point target location w.r.t observed point target location within
        the L1 RSLC
    """
    rslc_name = os.path.join(iscetest.data, "REE_RSLC_out17.h5")
    freq_group = 'A'
    polarization = 'HH'
    cr_llh = [-54.579586258, 3.177088785, 0.0]  # lon, lat, hgt in (deg, deg, m)

    performance_dict = slc_pt_performance(
        rslc_name,
        freq_group,
        polarization,
        cr_llh,
        **kwargs
    )

    slant_range_offset = performance_dict['range']['offset']
    azimuth_offset = performance_dict['azimuth']['offset']

    #Compare slant range offset and azimuth offset against default values
    npt.assert_(abs(slant_range_offset) < 0.1,
        f'Slant range bin offset {slant_range_offset} is larger than expected.')
    npt.assert_(abs(azimuth_offset) < 0.1,
        f'Azimuth bin offset {azimuth_offset} is larger than expected.')


# Treat warnings as test failures (internally, the PTA tool catches errors during
# processing of each corner reflector and converts them into warnings).
@pytest.mark.filterwarnings("error")
def test_nisar_csv():
    datadir = Path(iscetest.data) / "abscal"
    cr_csv = datadir / "ree_corner_reflectors_nisar.csv"
    rslc_hdf5 = datadir / "calib_slc_pass1_5mhz.h5"

    # Create a temporary file to store the JSON output of the tool.
    with NamedTemporaryFile(suffix=".json") as tmpfile:
        # Run the PTA tool.
        process_corner_reflector_csv(
            corner_reflector_csv=cr_csv,
            csv_format="nisar",
            rslc_hdf5=rslc_hdf5,
            output_json=tmpfile.name,
            freq=None,
            pol=None,
            nchip=64,
            upsample_factor=32,
            peak_find_domain="time",
            num_sidelobes=10,
            predict_null=True,
            fs_bw_ratio=1.2,
            window_type="rect",
            window_parameter=0.0,
            cuts=True,
        )

        # Read JSON output.
        results = json.load(tmpfile)

        # The result should be a list with a single item (only one corner reflector was
        # marked as valid for PTA -- the rest are too close the edge of the image to
        # accurately capture their IRFs).
        assert isinstance(results, list)
        assert len(results) == 1

        # The contents of the list should be a dict.
        cr_info = results[0]
        assert isinstance(cr_info, dict)

        # Check that the expected fields were populated.
        expected_keys = {
            "id",
            "frequency",
            "polarization",
            "elevation_angle",
            "magnitude",
            "phase",
            "range",
            "azimuth",
            "survey_date",
            "validity",
            "velocity",
        }
        assert set(cr_info.keys()) == expected_keys

        expected_rg_az_keys = {
            "index",
            "offset",
            "phase ramp",
            "resolution",
            "ISLR",
            "PSLR",
            "magnitude cut",
            "phase cut",
            "cut",
        }

        for key in ["azimuth", "range"]:
            assert set(cr_info[key].keys()) == expected_rg_az_keys

        # Check corner reflector metadata.
        assert cr_info["id"] == "CR2"
        assert cr_info["frequency"] == "A"
        assert cr_info["polarization"] == "HH"
        assert cr_info["survey_date"] == "2020-01-01T00:00:00.000000000"
        assert cr_info["validity"] == 7
        assert cr_info["velocity"] == [-1e-9, 1e-9, 0.0]

        # Rough check of range & azimuth geolocation error.
        # The units are range/azimuth samples.
        assert abs(cr_info["range"]["offset"]) < 0.01
        assert abs(cr_info["azimuth"]["offset"]) < 0.01

        # Rough check of impulse response width/ISLR/PSLR in range & azimuth.
        # The units of resolution are range/azimuth samples. ISLR/PSLR are in dB.
        assert cr_info["range"]["resolution"] < 2.0
        assert cr_info["range"]["ISLR"] < -10.0
        assert cr_info["range"]["PSLR"] < -10.0
        assert cr_info["azimuth"]["resolution"] < 2.0
        assert cr_info["azimuth"]["ISLR"] < -10.0
        assert cr_info["azimuth"]["PSLR"] < -10.0

        # We can assume that the elevation angle of this CR should be within +/- 1
        # degree with some back-of-the-envelope math:
        #  - The tilt angle of the CR is ~12.4 deg
        #  - Assume a theoretical boresight of 35.3 deg for a triangular trihedral
        #    CR incidence angle = 90 - (tilt + 35.3)
        #  - CR look angle = incidence angle - 5 deg (due to curvature of the earth)
        #  - Antenna EL angle = look angle - 37 deg (nominal mechanical boresight for
        #    NISAR)
        theta = np.deg2rad(1.0)
        assert -theta <= cr_info["elevation_angle"] <= theta


def test_uavsar_csv():
    datadir = Path(iscetest.data) / "abscal"
    cr_csv = datadir / "REE_CORNER_REFLECTORS_INFO.csv"
    rslc_hdf5 = datadir / "calib_slc_pass1_5mhz.h5"

    # Create a temporary file to store the JSON output of the tool.
    with NamedTemporaryFile(suffix=".json") as tmpfile:
        # Run the PTA tool.
        process_corner_reflector_csv(
            corner_reflector_csv=cr_csv,
            csv_format="uavsar",
            rslc_hdf5=rslc_hdf5,
            output_json=tmpfile.name,
            freq=None,
            pol=None,
            nchip=64,
            upsample_factor=32,
            peak_find_domain="time",
            num_sidelobes=10,
            predict_null=True,
            fs_bw_ratio=1.2,
            window_type="kaiser",
            window_parameter=1.6,
            cuts=False,
        )

        # Read JSON output.
        results = json.load(tmpfile)

        # The result should be a list with a single item (only the CR in the center of
        # the scene is included -- the others are too close to the edge of the image and
        # therefore filtered out from the results).
        assert isinstance(results, list)
        assert len(results) == 1

        # The contents of the list should be a dict.
        cr_info = results[0]
        assert isinstance(cr_info, dict)

        # Check that the expected fields were populated.
        expected_keys = {
            "id",
            "frequency",
            "polarization",
            "elevation_angle",
            "magnitude",
            "phase",
            "range",
            "azimuth",
        }
        assert set(cr_info.keys()) == expected_keys

        expected_rg_az_keys = {
            "index",
            "offset",
            "phase ramp",
            "resolution",
            "ISLR",
            "PSLR",
        }

        for key in ["azimuth", "range"]:
            assert set(cr_info[key].keys()) == expected_rg_az_keys

        # Check metadata.
        assert cr_info["id"] == "CR2"
        assert cr_info["frequency"] == "A"
        assert cr_info["polarization"] == "HH"

        # Rough check of range & azimuth geolocation error.
        # The units are range/azimuth samples.
        assert abs(cr_info["range"]["offset"]) < 0.01
        assert abs(cr_info["azimuth"]["offset"]) < 0.01

        # Rough check of impulse response width/ISLR/PSLR in range & azimuth.
        # The units of resolution are range/azimuth samples. ISLR/PSLR are in dB.
        assert cr_info["range"]["resolution"] < 2.0
        assert cr_info["range"]["ISLR"] < -10.0
        assert cr_info["range"]["PSLR"] < -10.0
        assert cr_info["azimuth"]["resolution"] < 2.0
        assert cr_info["azimuth"]["ISLR"] < -10.0
        assert cr_info["azimuth"]["PSLR"] < -10.0

        # Check that the elevation angle is within +/- 1 degree of antenna boresight.
        theta = np.deg2rad(1.0)
        assert -theta <= cr_info["elevation_angle"] <= theta
