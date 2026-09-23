import os

import iscetest
import pytest
from nisar.static.ephemeris import get_cropped_orbit_and_attitude

import isce3


class TestGetCroppedOrbitAndAttitude:
    @pytest.fixture
    def orbit_xml_file(self) -> str:
        return os.path.join(
            iscetest.data,
            "NISAR_ANC_L_PR_FOE_20250806T193246_20230104T061021_20230104T061655.xml",
        )

    @pytest.fixture
    def pointing_xml_file(self) -> str:
        return os.path.join(
            iscetest.data,
            "NISAR_ANC_L_PR_FRP_20250910T221957_20230104T061021_20230104T061655.xml",
        )

    def test_start_end_time(self, orbit_xml_file: str, pointing_xml_file: str):
        start_time = isce3.core.DateTime("2023-01-04T06:12:00")
        end_time = isce3.core.DateTime("2023-01-04T06:14:00")

        orbit, attitude = get_cropped_orbit_and_attitude(
            orbit_xml_file=orbit_xml_file,
            pointing_xml_file=pointing_xml_file,
            start_time=start_time,
            end_time=end_time,
        )

        assert orbit.reference_epoch == attitude.reference_epoch
        assert orbit.reference_epoch.frac == 0.0

        assert orbit.start_datetime <= start_time
        assert attitude.start_datetime <= start_time

        assert orbit.end_datetime >= end_time
        assert attitude.end_datetime >= end_time

    def test_padding(self, orbit_xml_file: str, pointing_xml_file: str):
        start_time = end_time = "2023-01-04T06:13:00"

        padding = 30.0
        orbit, attitude = get_cropped_orbit_and_attitude(
            orbit_xml_file=orbit_xml_file,
            pointing_xml_file=pointing_xml_file,
            start_time=start_time,
            end_time=end_time,
            padding=padding,
        )

        assert (orbit.end_time - orbit.start_time) >= (2 * padding)
        assert (attitude.end_time - attitude.start_time) >= (2 * padding)

    def test_min_size(self, orbit_xml_file: str, pointing_xml_file: str):
        start_time = end_time = "2023-01-04T06:13:00"

        orbit, attitude = get_cropped_orbit_and_attitude(
            orbit_xml_file=orbit_xml_file,
            pointing_xml_file=pointing_xml_file,
            start_time=start_time,
            end_time=end_time,
        )

        assert orbit.size >= 4
        assert attitude.size >= 2

    def test_bad_start_time(self, orbit_xml_file: str, pointing_xml_file: str):
        start_time = isce3.core.DateTime("2000-01-01T00:00:00")
        match = (
            f"Requested start time {start_time.isoformat()} does not fall in orbit time"
            " interval"
        )
        with pytest.raises(ValueError, match=match):
            get_cropped_orbit_and_attitude(
                orbit_xml_file=orbit_xml_file,
                pointing_xml_file=pointing_xml_file,
                start_time=start_time,
                end_time=None,
            )

    def test_bad_end_time(self, orbit_xml_file: str, pointing_xml_file: str):
        end_time = isce3.core.DateTime("2050-12-31T23:59:59")
        match = (
            f"Requested end time {end_time.isoformat()} does not fall in orbit time"
            " interval"
        )
        with pytest.raises(ValueError, match=match):
            get_cropped_orbit_and_attitude(
                orbit_xml_file=orbit_xml_file,
                pointing_xml_file=pointing_xml_file,
                start_time=None,
                end_time=end_time,
            )

    def test_bad_padding(self, orbit_xml_file: str, pointing_xml_file: str):
        padding = -10.0
        with pytest.raises(ValueError, match=f"^{padding=}, must be >= 0$"):
            get_cropped_orbit_and_attitude(
                orbit_xml_file=orbit_xml_file,
                pointing_xml_file=pointing_xml_file,
                start_time=None,
                end_time=None,
                padding=padding,
            )
