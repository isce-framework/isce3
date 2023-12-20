from __future__ import annotations

import textwrap
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

import isce3
import iscetest
import nisar
from nisar.cal import CRValidity


@contextmanager
def create_tmp_text_file(contents: str, suffix: str | None = None) -> Iterator[Path]:
    """
    A context manager that creates a temporary text file with the specified contents.

    The file is automatically removed from the file system when the context block exits.

    Parameters
    ----------
    contents : str
        The contents of the text file.
    suffix : str or None, optional
        An optional file name suffix. If None, there will be no suffix. Defaults to
        None.

    Yields
    ------
    filepath : pathlib.Path
        The file system path of the temporary file.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        filepath = Path(f.name)
        filepath.write_text(contents)
        yield filepath


@pytest.mark.parametrize("validity", [-1, 8])
def test_bad_corner_reflector_validity(validity: int):
    errmsg = "validity flag has invalid value"
    with pytest.raises(ValueError, match=errmsg):
        nisar.cal.CornerReflector(
            id="CR1",
            llh=isce3.core.LLH(0.0, 0.0, 0.0),
            elevation=0.0,
            azimuth=0.0,
            side_length=10.0,
            survey_date=isce3.core.DateTime("1970-01-01"),
            validity=validity,
            velocity=[0.0, 0.0, 0.0],
        )


class TestParseCornerReflectorCSV:
    @pytest.fixture(scope="class")
    def example_csv(self) -> Iterator[Path]:
        # The example from the Corner Reflector SIS appendix.
        contents = textwrap.dedent(
            """\
            Corner reflector ID,Latitude (deg),Longitude (deg),Height above ellipsoid (m),Azimuth (deg),Tilt / Elevation (deg),Side length (m),Survey Date,Validity,Velocity East (m/s),Velocity North (m/s),Velocity Up (m/s)
            CR1,69.721919189,-128.288391475,489.999460166,317.109,12.920,3.462,2023-06-05T17:08:00,7,0,0,0
            CR2,69.658487752,-128.484326707,489.999308966,316.925,12.377,3.462,2023-06-05T17:08:00,7,0,0,0
            CR3,69.615519185,-128.616011593,489.999186578,316.802,12.014,3.462,2023-06-05T17:08:00,7,0,0,0
            """
        )
        with create_tmp_text_file(contents, suffix=".csv") as csvfile:
            yield csvfile

    def test_parse_csv(self, example_csv: Path):
        crs = list(nisar.cal.parse_corner_reflector_csv(example_csv))

        # Check number of corner reflectors.
        assert len(crs) == 3

        # Check corner reflector IDs.
        ids = [cr.id for cr in crs]
        expected_ids = ["CR1", "CR2", "CR3"]
        npt.assert_array_equal(ids, expected_ids)

        # Check that CR latitudes & longitudes are all within 0.5 degrees of their
        # approximate expected location.
        atol = np.deg2rad(0.5)
        lats = [cr.llh.latitude for cr in crs]
        approx_lat = np.deg2rad(69.7)
        npt.assert_allclose(lats, approx_lat, atol=atol)

        lons = [cr.llh.longitude for cr in crs]
        approx_lon = np.deg2rad(-128.5)
        npt.assert_allclose(lons, approx_lon, atol=atol)

        # Check that CR heights are within 1 cm of their approximate expected location.
        heights = [cr.llh.height for cr in crs]
        approx_height = 490.0
        npt.assert_allclose(heights, approx_height, atol=0.01)

        # Check that CR azimuth & elevation angles are each within 0.5 degrees of their
        # approximate expected orientation.
        azs = [cr.azimuth for cr in crs]
        approx_az = np.deg2rad(317.0)
        npt.assert_allclose(azs, approx_az, atol=atol)

        els = [cr.elevation for cr in crs]
        approx_el = np.deg2rad(12.5)
        npt.assert_allclose(els, approx_el, atol=atol)

        # Check that CR side lengths all match their expected value.
        side_lengths = [cr.side_length for cr in crs]
        expected_side_length = 3.462
        npt.assert_array_equal(side_lengths, expected_side_length)

        # Check survey dates.
        survey_dates = [cr.survey_date for cr in crs]
        expected_survey_date = isce3.core.DateTime("2023-06-05T17:08:00")
        npt.assert_array_equal(survey_dates, expected_survey_date)

        # Check validity flags.
        validities = [cr.validity for cr in crs]
        expected_validity = CRValidity.IPR | CRValidity.RAD_POL | CRValidity.GEOM
        npt.assert_array_equal(validities, expected_validity)

        # Check velocities.
        velocities = [cr.velocity for cr in crs]
        npt.assert_array_equal(velocities, 0.0)

    @pytest.fixture(scope="class")
    def csv_with_comments(self) -> Iterator[Path]:
        contents = textwrap.dedent(
            """\
            # This is a comment
            CR1,0.0,0.0,0.0,0.0,0.0,10.0,2000-01-01T00:00:00,7,0.0,0.0,0.0
            # This is another comment
            CR2,0.0,0.0,0.0,0.0,0.0,10.0,2000-01-01T00:00:00,7,0.0,0.0,0.0
            """
        )
        with create_tmp_text_file(contents, suffix=".csv") as csvfile:
            yield csvfile

    def test_comments(self, csv_with_comments: Path):
        crs = list(nisar.cal.parse_corner_reflector_csv(csv_with_comments))
        ids = [cr.id for cr in crs]
        assert ids == ["CR1", "CR2"]

    @pytest.fixture(scope="class")
    def empty_csv(self) -> Iterator[Path]:
        with create_tmp_text_file("", suffix=".csv") as csvfile:
            yield csvfile

    def test_empty_csv(self, empty_csv: Path):
        crs = list(nisar.cal.parse_corner_reflector_csv(empty_csv))
        assert crs == []

    @pytest.fixture(scope="class")
    def bad_format_csv(self) -> Iterator[Path]:
        # A CSV in UAVSAR format (missing NISAR-specific metadata).
        contents = textwrap.dedent(
            """\
            Corner reflector ID,Latitude (deg),Longitude (deg),Height above ellipsoid (m),Azimuth (deg),Tilt / Elevation (deg),Side length (m)
            CR1,69.721919189,-128.288391475,489.999460166,317.109,12.920,3.462
            CR2,69.658487752,-128.484326707,489.999308966,316.925,12.377,3.462
            CR3,69.615519185,-128.616011593,489.999186578,316.802,12.014,3.462
            """
        )
        with create_tmp_text_file(contents, suffix=".csv") as csvfile:
            yield csvfile

    def test_bad_format(self, bad_format_csv: Path):
        errmsg = "error parsing NISAR corner reflector CSV file"
        with pytest.raises(RuntimeError, match=errmsg):
            next(nisar.cal.parse_corner_reflector_csv(bad_format_csv))

    def test_csv_not_found(self):
        csvfile = "/this/is/not/a/path.csv"
        assert not Path(csvfile).exists()

        errmsg = "corner reflector CSV file not found"
        with pytest.raises(FileNotFoundError, match=errmsg):
            next(nisar.cal.parse_corner_reflector_csv(csvfile))


class TestGetLatestCRDataBeforeEpoch:
    @pytest.fixture(scope="class")
    def crs(self) -> list[nisar.cal.CornerReflector]:
        contents = textwrap.dedent(
            """\
            # -- CR1 --
            CR1,0.0,0.0,0.0,0.0,0.0,10.0,2020-01-01T00:00:00,0,0.0,0.0,0.0
            # -- CR2 --
            CR2,0.0,0.0,0.0,0.0,0.0,10.0,2021-01-01T00:00:00,0,0.0,0.0,0.0
            CR2,0.0,0.0,0.0,0.0,0.0,10.0,2021-06-01T00:00:00,0,0.0,0.0,0.0
            CR2,0.0,0.0,0.0,0.0,0.0,10.0,2022-01-01T00:00:00,0,0.0,0.0,0.0
            CR2,0.0,0.0,0.0,0.0,0.0,10.0,2022-06-01T00:00:00,0,0.0,0.0,0.0
            CR2,0.0,0.0,0.0,0.0,0.0,10.0,2023-01-01T00:00:00,0,0.0,0.0,0.0
            CR2,0.0,0.0,0.0,0.0,0.0,10.0,2023-06-01T00:00:00,0,0.0,0.0,0.0
            # -- CR3 --
            CR3,0.0,0.0,0.0,0.0,0.0,10.0,2023-06-01T00:00:00,0,0.0,0.0,0.0
            # -- CR4 --
            CR4,0.0,0.0,0.0,0.0,0.0,10.0,2023-06-01T00:00:00.000001,0,0.0,0.0,0.0
            """
        )
        with create_tmp_text_file(contents, suffix=".csv") as csvfile:
            crs = list(nisar.cal.parse_corner_reflector_csv(csvfile))
            yield crs

    def test_epoch0(self, crs: list[nisar.cal.CornerReflector]):
        epoch = isce3.core.DateTime("1970-01-01")
        latest_crs = list(nisar.cal.get_latest_cr_data_before_epoch(crs, epoch))
        assert latest_crs == []

    def test_epoch1(self, crs: list[nisar.cal.CornerReflector]):
        epoch = isce3.core.DateTime("2020-01-02")
        latest_crs = list(nisar.cal.get_latest_cr_data_before_epoch(crs, epoch))

        ids = [cr.id for cr in latest_crs]
        assert ids == ["CR1"]

        survey_dates = [cr.survey_date for cr in latest_crs]
        assert survey_dates == [isce3.core.DateTime("2020-01-01")]

    def test_epoch2(self, crs: list[nisar.cal.CornerReflector]):
        epoch = isce3.core.DateTime("2021-12-31T23:59:59.999999")
        latest_crs = list(nisar.cal.get_latest_cr_data_before_epoch(crs, epoch))

        ids = [cr.id for cr in latest_crs]
        assert ids == ["CR1", "CR2"]

        survey_dates = [cr.survey_date for cr in latest_crs]
        assert survey_dates == [
            isce3.core.DateTime("2020-01-01"),
            isce3.core.DateTime("2021-06-01"),
        ]

    def test_epoch3(self, crs: list[nisar.cal.CornerReflector]):
        epoch = isce3.core.DateTime("2023-06-01")
        latest_crs = list(nisar.cal.get_latest_cr_data_before_epoch(crs, epoch))

        ids = [cr.id for cr in latest_crs]
        assert ids == ["CR1", "CR2", "CR3"]

        survey_dates = [cr.survey_date for cr in latest_crs]
        assert survey_dates == [
            isce3.core.DateTime("2020-01-01"),
            isce3.core.DateTime("2023-06-01"),
            isce3.core.DateTime("2023-06-01"),
        ]

    def test_empty_iterable(self):
        crs = []
        epoch = isce3.core.DateTime("9999-12-31T23:59:59.999")
        latest_crs = list(nisar.cal.get_latest_cr_data_before_epoch(crs, epoch))
        assert latest_crs == []


class TestGetValidCRs:
    @pytest.fixture(scope="class")
    def crs(self) -> list[nisar.cal.CornerReflector]:
        contents = textwrap.dedent(
            """\
            CR1,0.0,0.0,0.0,0.0,0.0,10.0,2000-01-01T00:00:00,0,0.0,0.0,0.0
            CR1,0.0,0.0,0.0,0.0,0.0,10.0,2000-01-01T00:00:00,1,0.0,0.0,0.0
            CR1,0.0,0.0,0.0,0.0,0.0,10.0,2000-01-01T00:00:00,2,0.0,0.0,0.0
            CR1,0.0,0.0,0.0,0.0,0.0,10.0,2000-01-01T00:00:00,3,0.0,0.0,0.0
            CR1,0.0,0.0,0.0,0.0,0.0,10.0,2000-01-01T00:00:00,4,0.0,0.0,0.0
            CR1,0.0,0.0,0.0,0.0,0.0,10.0,2000-01-01T00:00:00,5,0.0,0.0,0.0
            CR1,0.0,0.0,0.0,0.0,0.0,10.0,2000-01-01T00:00:00,6,0.0,0.0,0.0
            CR1,0.0,0.0,0.0,0.0,0.0,10.0,2000-01-01T00:00:00,7,0.0,0.0,0.0
            """
        )
        with create_tmp_text_file(contents, suffix=".csv") as csvfile:
            crs = list(nisar.cal.parse_corner_reflector_csv(csvfile))
            yield crs

    def test_no_flags(self, crs: list[nisar.cal.CornerReflector]):
        valid_crs = nisar.cal.get_valid_crs(crs)
        validity_codes = [int(cr.validity) for cr in valid_crs]
        assert validity_codes == [1, 2, 3, 4, 5, 6, 7]  # everything except 0

    def test_single_flag(self, crs: list[nisar.cal.CornerReflector]):
        valid_crs = nisar.cal.get_valid_crs(crs, flags=CRValidity.RAD_POL)
        validity_codes = [int(cr.validity) for cr in valid_crs]
        assert validity_codes == [2, 3, 6, 7]

    def test_multiple_flags(self, crs: list[nisar.cal.CornerReflector]):
        flags = CRValidity.IPR | CRValidity.GEOM
        valid_crs = nisar.cal.get_valid_crs(crs, flags=flags)
        validity_codes = [int(cr.validity) for cr in valid_crs]
        assert validity_codes == [1, 3, 4, 5, 6, 7]  # everything except 0 & 2

    def test_empty_iterable(self):
        crs = []
        valid_crs = list(nisar.cal.get_valid_crs(crs))
        assert valid_crs == []


@pytest.fixture
def ree_corner_reflectors_nisar_csv() -> Path:
    return Path(iscetest.data) / "abscal/ree_corner_reflectors_nisar.csv"


def test_parse_and_filter_corner_reflector_csv(ree_corner_reflectors_nisar_csv: Path):
    observation_datetime = isce3.core.DateTime("2021-12-31")
    validity_flags = CRValidity.RAD_POL
    crs = list(
        nisar.cal.parse_and_filter_corner_reflector_csv(
            ree_corner_reflectors_nisar_csv, observation_datetime, validity_flags,
        )
    )

    # Check corner reflector IDs.
    ids = [cr.id for cr in crs]
    assert ids == ["CR1", "CR2", "CR3", "CR5", "CR6", "CR7"]

    # Check survey dates.
    survey_dates = [cr.survey_date for cr in crs]
    expected_survey_date = isce3.core.DateTime("2020-01-01")
    npt.assert_array_equal(survey_dates, expected_survey_date)

    # Check validity flags.
    validities = [cr.validity for cr in crs]
    expected_validities = [2, 7, 2, 7, 7, 7]
    npt.assert_array_equal(validities, expected_validities)


@pytest.fixture
def sample_nisar_csv() -> Path:
    return Path(iscetest.data) / "abscal/NISAR_ANC_CORNER_REFLECTORS_001.csv"


def test_parse_sample_nisar_csv(sample_nisar_csv: Path):
    crs = list(nisar.cal.parse_corner_reflector_csv(sample_nisar_csv))
    # check that strings are parsed without issues due to spaces.
    npt.assert_(crs[0].id == "N01K")
    npt.assert_(crs[0].survey_date == isce3.core.DateTime("2021-06-04"))
