from __future__ import annotations

from textwrap import dedent

import pytest
import yamale
from nisar.static.runconfig import (
    default_runconfig_file,
    dump_runconfig_to_str,
    get_runconfig_params,
    validate_runconfig,
)

from ..util import create_tmp_text_file


def minimal_runconfig() -> str:
    """A simple runconfig with only the minimum parameters defined."""

    return dedent(
        """\
        runconfig:
          groups:
            dynamic_ancillary_file_group:
              dem_raster_file: /path/to/dem.tiff
              water_mask_raster_file: /path/to/water_mask.tiff
              orbit_xml_file: /path/to/orbit.xml
              pointing_xml_file: /path/to/pointing.xml
        """
    )


def full_runconfig() -> str:
    """A runconfig with all required and optional fields populated."""

    runconfig = default_runconfig_file().read_text()

    files = {
        "dem_raster_file": "/path/to/dem.tiff",
        "water_mask_raster_file": "/path/to/water_mask.tiff",
        "orbit_xml_file": "/path/to/orbit.xml",
        "pointing_xml_file": "/path/to/pointing.xml",
    }

    for key, val in files.items():
        runconfig = runconfig.replace(f"{key}:", f"{key}: {val}")

    return runconfig


def good_runconfigs() -> list[str]:
    """A list of example runconfigs that should pass validation checks."""

    # A runconfig for use with ALOS-1 data.
    alos_runconfig = dedent(
        """\
        runconfig:
          groups:
            dynamic_ancillary_file_group:
              dem_raster_file: /path/to/dem.tiff
              water_mask_raster_file: /path/to/water_mask.tiff
              orbit_xml_file: /path/to/orbit.xml
              pointing_xml_file: /path/to/pointing.xml

            primary_executable:
              mission_id: ALOS
              platform_name: ALOS-1
              instrument_name: PALSAR

            processing:
              radar_grid:
                look_side: right
                wavelength: 0.236
        """
    )

    # A runconfig with datetimes enclosed in quotes.
    quoted_datetimes_runconfig = dedent(
        """\
        runconfig:
          groups:
            dynamic_ancillary_file_group:
              dem_raster_file: /path/to/dem.tiff
              water_mask_raster_file: /path/to/water_mask.tiff
              orbit_xml_file: /path/to/orbit.xml
              pointing_xml_file: /path/to/pointing.xml

            primary_executable:
              validity_start_datetime: '1999-12-31T23:59:59'

            processing:
              ephemeris:
                start_time: '2025-10-01T12:00:00.000000000'
                end_time: '2025-10-01T12:10:00.000000000'
        """
    )

    # A runconfig with datetimes *not* enclosed in quotes.
    unquoted_datetimes_runconfig = dedent(
        """\
        runconfig:
          groups:
            dynamic_ancillary_file_group:
              dem_raster_file: /path/to/dem.tiff
              water_mask_raster_file: /path/to/water_mask.tiff
              orbit_xml_file: /path/to/orbit.xml
              pointing_xml_file: /path/to/pointing.xml

            primary_executable:
              validity_start_datetime: 1999-12-31T23:59:59

            processing:
              ephemeris:
                start_time: 2025-10-01T12:00:00.000000000
                end_time: 2025-10-01T12:10:00.000000000
        """
    )

    return [
        minimal_runconfig(),
        full_runconfig(),
        alos_runconfig,
        quoted_datetimes_runconfig,
        unquoted_datetimes_runconfig,
    ]


def bad_runconfigs() -> list[str]:
    """A list of example runconfigs that should fail validation checks."""

    # An empty string.
    empty_runconfig = ""

    # The default runconfig (with required parameters missing).
    default_runconfig = default_runconfig_file().read_text()

    # A runconfig with an additional parameter not found in the schema.
    extra_param_runconfig = dedent(
        """\
        runconfig:
          groups:
            dynamic_ancillary_file_group:
              dem_raster_file: /path/to/dem.tiff
              water_mask_raster_file: /path/to/water_mask.tiff
              orbit_xml_file: /path/to/orbit.xml
              pointing_xml_file: /path/to/pointing.xml

            banana: 123
        """
    )

    # A runconfig with invalid track & frame IDs (must be 3 digits or fewer).
    invalid_track_frame_id_runconfig = dedent(
        """\
        runconfig:
          groups:
            dynamic_ancillary_file_group:
              dem_raster_file: /path/to/dem.tiff
              water_mask_raster_file: /path/to/water_mask.tiff
              orbit_xml_file: /path/to/orbit.xml
              pointing_xml_file: /path/to/pointing.xml

            geometry:
              relative_orbit_number: 1000
              frame_number: 1001
        """
    )

    # A runconfig with negative geo grid spacing.
    negative_spacing_runconfig = dedent(
        """\
        runconfig:
          groups:
            dynamic_ancillary_file_group:
              dem_raster_file: /path/to/dem.tiff
              water_mask_raster_file: /path/to/water_mask.tiff
              orbit_xml_file: /path/to/orbit.xml
              pointing_xml_file: /path/to/pointing.xml

            processing:
              geo_grid:
                posting:
                  x: 10.0
                  y: -5.0
        """
    )

    return [
        empty_runconfig,
        default_runconfig,
        extra_param_runconfig,
        invalid_track_frame_id_runconfig,
        negative_spacing_runconfig,
    ]


class TestValidateRunconfig:
    @pytest.mark.parametrize("runconfig", good_runconfigs())
    def test_good_runconfigs(self, runconfig: str):
        with create_tmp_text_file(runconfig, ".yml") as runconfig_file:
            validate_runconfig(runconfig_file)

    @pytest.mark.parametrize("runconfig", bad_runconfigs())
    def test_bad_runconfigs(self, runconfig: str):
        with create_tmp_text_file(runconfig, ".yml") as runconfig_file:
            with pytest.raises(yamale.yamale_error.YamaleError):
                validate_runconfig(runconfig_file)


class TestGetRunconfigParams:
    def test_minimal_runconfig(self):
        with create_tmp_text_file(minimal_runconfig(), ".yml") as runconfig_file:
            params = get_runconfig_params(runconfig_file)

            groups = params["runconfig"]["groups"]
            files = groups["dynamic_ancillary_file_group"]
            processing = groups["processing"]

            assert files["dem_raster_file"] == "/path/to/dem.tiff"
            assert files["water_mask_raster_file"] == "/path/to/water_mask.tiff"
            assert files["orbit_xml_file"] == "/path/to/orbit.xml"
            assert files["pointing_xml_file"] == "/path/to/pointing.xml"

            # Test that some of the default values were populated correctly.
            assert processing["geo_grid"]["top_left"]["x"] is None
            assert processing["dem"]["interp_method"] == "biquintic"
            assert processing["radar_grid"]["bounding_box"]["min_height"] == -500.0
            assert processing["topo"]["lines_per_block"] == 1024

    def test_override_defaults(self):
        runconfig = dedent(
            """\
            runconfig:
              groups:
                dynamic_ancillary_file_group:
                  dem_raster_file: /path/to/dem.tiff
                  water_mask_raster_file: /path/to/water_mask.tiff
                  orbit_xml_file: /path/to/orbit.xml
                  pointing_xml_file: /path/to/pointing.xml

                processing:
                  geo_grid:
                    top_left:
                      x: 1000.0

                  dem:
                    interp_method: bilinear

                  radar_grid:
                    bounding_box:
                      min_height: 0.0

                  topo:
                    lines_per_block: 2048
            """
        )

        with create_tmp_text_file(runconfig, ".yml") as runconfig_file:
            params = get_runconfig_params(runconfig_file)

            processing = params["runconfig"]["groups"]["processing"]

            # Test that the default values were overwritten.
            assert processing["geo_grid"]["top_left"]["x"] == 1000.0
            assert processing["dem"]["interp_method"] == "bilinear"
            assert processing["radar_grid"]["bounding_box"]["min_height"] == 0.0
            assert processing["topo"]["lines_per_block"] == 2048


def test_dump_runconfig_to_str():
    files = {
        "dem_raster_file": "/path/to/dem.tiff",
        "water_mask_raster_file": "/path/to/water_mask.tiff",
        "orbit_xml_file": "/path/to/orbit.xml",
        "pointing_xml_file": "/path/to/pointing.xml",
    }
    params = {"runconfig": {"groups": {"dynamic_ancillary_file_group": files}}}

    # XXX: Dumping to YAML causes keys to be sorted alphabetically.
    runconfig = dedent(
        """\
        runconfig:
          groups:
            dynamic_ancillary_file_group:
              dem_raster_file: /path/to/dem.tiff
              orbit_xml_file: /path/to/orbit.xml
              pointing_xml_file: /path/to/pointing.xml
              water_mask_raster_file: /path/to/water_mask.tiff
        """
    )

    assert dump_runconfig_to_str(params) == runconfig
