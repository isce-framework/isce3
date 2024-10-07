from __future__ import annotations

import os
import sys

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from io import StringIO
from pathlib import Path

import numpy as np
import shapely.wkt
import yamale
from ruamel.yaml import YAML
from yamale.yamale_error import YamaleError
from yamale.schema.validationresults import ValidationResult

from nisar.products.readers import SLC
from nisar.workflows.helpers import WORKFLOW_SCRIPTS_DIR

DEFAULT_YAML_DIR = os.path.join(WORKFLOW_SCRIPTS_DIR, "defaults")
SCHEMA_YAML_DIR = os.path.join(WORKFLOW_SCRIPTS_DIR, "schemas")


# Use a custom help message formatter to improve readability by increasing the
# indentation of parameter descriptions to accommodate longer parameter names.
def help_formatter(prog):
    return argparse.HelpFormatter(prog, max_help_position=60)


def init_argparse():
    required_group_title = "required parameters"
    process_group_title = "workflow options"
    filepath_group_title = "runconfig filepaths"
    geocoding_group_title = "geocoding options"
    parser = argparse.ArgumentParser(
        prog=__package__ + ".dumpconfig",
        argument_default=argparse.SUPPRESS,
        formatter_class=help_formatter,
    )

    # Add arguments
    subparsers = parser.add_subparsers(dest="workflow", required=True)

    # Parser for geocoded commands:
    geo_parse = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        add_help=False
    )
    required_group = geo_parse.add_argument_group(
        required_group_title,
        "Arguments required for the workflow. All others are optional."
    )
    process_group = geo_parse.add_argument_group(process_group_title)
    filepath_group = geo_parse.add_argument_group(filepath_group_title)
    geometry_group = geo_parse.add_argument_group(geocoding_group_title)
    required_group.add_argument(
        "rslc_file",
        type=Path,
        help="The filepath to the RSLC input product for the workflow.",
    )
    required_group.add_argument(
        "-d",
        "--dem",
        dest="dem_file",
        type=Path,
        required=True,
        help="The filepath to the DEM file for the workflow.",
    )
    filepath_group.add_argument(
        "-o",
        "--output-file",
        type=Path,
        required=False,
        help="The filepath to the output file for the workflow.",
    )
    filepath_group.add_argument(
        "-t",
        "--tec",
        dest="tec_file",
        type=Path,
        required=False,
        help="The filepath to the TEC file for the workflow.",
    )
    filepath_group.add_argument(
        "--orbit",
        dest="orbit_file",
        type=Path,
        required=False,
        help="The filepath to the orbit file for the workflow.",
    )
    geometry_group.add_argument(
        "-e",
        "--epsg",
        type=int,
        required=False,
        help=(
            "The EPSG to run the workflow with. If not given, will default to the "
            "EPSG code of the DEM. Cannot be used in conjunction with "
            "--generate-epsg."
        ),
    )
    geometry_group.add_argument(
        "--generate-epsg",
        action="store_true",
        required=False,
        help=(
            "If given, an EPSG code will be determined for the tile based on the "
            "centerpoint of the input RSLC. A UTM EPSG is determined for mid-latitudes "
            "between -60 to 60 degrees, polar stereographic Antarctic for south of "
            "-60 degrees and polar stereographic north for the latitudes north of 60 "
            "respectively. Cannot be used in conjunction with --epsg."
        ),
    )
    geometry_group.add_argument(
        "--a-spacing",
        type=Decimal,
        nargs=2,
        required=False,
        metavar=("X_SPACING", "Y_SPACING"),
        help=(
            "The desired pixel spacing, in units designated by the workflow EPSG code "
            "of frequency A in the output product, in order of [x_spacing, y_spacing]. "
            "If not given, will use the default spacing behavior for the source "
            "runconfig file. The units of this argument should be the same as the SRS "
            "defined by the given EPSG code."
        ),
    )
    geometry_group.add_argument(
        "--b-spacing",
        type=Decimal,
        nargs=2,
        required=False,
        metavar=("X_SPACING", "Y_SPACING"),
        help=(
            "The desired pixel spacing, in units designated by the workflow EPSG code "
            "of frequency B in the output product, in order of [x_spacing, y_spacing]. "
            "If not given, will use the default spacing behavior for the source "
            "runconfig file. The units of this argument should be the same as the SRS "
            "defined by the given EPSG code."
        ),
    )
    geometry_group.add_argument(
        "--x-snap",
        type=Decimal,
        required=False,
        help=(
            "The X snap value. The start and end x coordinates of the output grid will "
            "be rounded down and up respectively to the nearest multiple of this "
            "value. Must be an integer multiple of the x_spacing values given. "
            "If not given, will use the default snap behavior for the source "
            "runconfig file. The units of this argument should be the same as the SRS "
            "defined by the given EPSG code."
        ),
    )
    geometry_group.add_argument(
        "--y-snap",
        type=Decimal,
        required=False,
        help=(
            "The Y snap value. The start and end y coordinates of the output grid will "
            "be rounded down and up respectively to the nearest multiple of this "
            "value. Must be an integer multiple of the y_spacing values given. "
            "If not given, will use the default snap behavior for the source "
            "runconfig file. The units of this argument should be the same as the SRS "
            "defined by the given EPSG code."
        ),
    )
    geometry_group.add_argument(
        "--nisar-defaults",
        action="store_true",
        required=False,
        help=(
            "If given, all geospatial parameters (spacing and snap values) not "
            "provided by the user will be filled with default values in meters that "
            "are suited to the workflow and bandwidth of the frequencies in the RSLC "
            "product. Only valid for products with 5, 10, 20, or 77 MHz bandwidths for "
            "all frequencies. This is suggested for use in conjunction with the "
            "--generate-epsg parameter."
        ),
    )
    geometry_group.add_argument(
        "--top-left",
        metavar=("X_ABS", "Y_ABS"),
        type=Decimal,
        required=False,
        nargs=2,
        help=(
            "The top left corner of the image, in X, Y order, in the same units as "
            "the EPSG. If not given, the default workflow behavior will be used. "
            "If given with snap values, the workflow will automatically snap this "
            "corner location to the snap values given."
        ),
    )
    geometry_group.add_argument(
        "--bottom-right",
        metavar=("X_ABS", "Y_ABS"),
        type=Decimal,
        required=False,
        nargs=2,
        help=(
            "The bottom right corner of the image, in X, Y order, in the same units "
            "as the EPSG. If not given, the default workflow behavior will be used. "
            "If given with snap values, the workflow will automatically snap this "
            "corner location to the snap values given."
        ),
    )
    process_group.add_argument(
        "--out-runconfig",
        type=Path,
        nargs='?',
        default=None,
        help=(
            "The path to output the runconfig file to. Will print to console if none "
            "given."
        ),
    )
    process_group.add_argument(
        "--source-runconfig",
        dest="source_runconfig_file",
        type=Path,
        required=False,
        help=(
            "The filepath to the runconfig to use as default. "
            "If none given, will use the NISAR default runconfig."
        ),
    )
    process_group.add_argument(
        "--validate",
        action="store_true",
        default=False,
        required=False,
        help=(
            "If given, validate the output runconfig against the schema and print all "
            "validation errors."
        ),
    )

    gslc_sub = subparsers.add_parser(
        "gslc",
        parents=[geo_parse],
        formatter_class=help_formatter,
        help="Create a GSLC runconfig.",
    )

    # This method enables adding an argument to an argument group in a parent parser of
    # a subparser without adding it to that same group in other subparsers.
    geocoding_group = next(
        filter(
            lambda group: group.title == geocoding_group_title,
            gslc_sub._action_groups,
        ),
        None,
    )
    geo_mutex = geocoding_group.add_mutually_exclusive_group()
    geo_mutex.add_argument(
        "--no-flattening",
        dest="flatten",
        action="store_false",
        default=argparse.SUPPRESS,
        help=(
            "If given, deactivate flattening in the output runconfig. If neither "
            "--no-flattening nor -flattening is given, use the default "
            "behavior for the source runconfig."
        ),
    )
    geo_mutex.add_argument(
        "--flatten",
        dest="flatten",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "If given, activate flattening in the output runconfig. If neither "
            "--no-flattening nor -flattening is given, use the default "
            "behavior for the source runconfig."
        ),
    )
    
    subparsers.add_parser(
        "gcov",
        parents=[geo_parse],
        formatter_class=help_formatter,
        help="Create a GCOV runconfig.",
    )

    return parser


@dataclass(frozen=True)
class GeocodingGeometryDefaults:
    x_posting: Decimal
    y_posting: Decimal
    x_snap: Decimal
    y_snap: Decimal


def get_geocoding_geometry_defaults(
    workflow: str,
    bandwidth: int,
) -> GeocodingGeometryDefaults:
    """
    Return the default geocoding geometry parameters for the given dataset and
    bandwidth.

    Parameters
    ----------
    bandwidth : int
        The bandwidth.
    workflow : str
        The name of the workflow, all lowercase - one of {"gslc", "gcov"}

    Returns
    -------
    GeocodingGeometryDefaults
        The default geometric parameters for geocoding. Includes posting and snap values
        in x and y.
    """
    if bandwidth not in {5, 20, 40, 77}:
        raise ValueError(
            f"Bandwidth {bandwidth} MHz not recognized. Recognized values: "
            "[5, 20, 40, 77]"
        )
    x_posting: Decimal
    y_posting: Decimal
    snap: Decimal

    if workflow == "gcov":
        snap = Decimal(0)
        if bandwidth == 5:
            x_posting = y_posting = Decimal(100)
        if bandwidth in {20, 77}:
            x_posting = y_posting = Decimal(20)
        if bandwidth == 40:
            x_posting = y_posting = Decimal(10)
    elif workflow == "gslc":
        snap = Decimal(0)
        y_posting = Decimal(5)
        if bandwidth == 5:
            x_posting = Decimal(40)
        elif bandwidth == 20:
            x_posting = Decimal(10)
        elif bandwidth == 40:
            x_posting = Decimal(5)
        elif bandwidth == 77:
            x_posting = Decimal(2.5)
    
    return GeocodingGeometryDefaults(
        x_posting=x_posting,
        y_posting=y_posting,
        x_snap=snap,
        y_snap=snap,
    )


@dataclass
class NamedMapping:
    """
    A wrapper around a Mapping object ("group") that keeps track of the names of it and
    its' children.

    Fields
    ------
    group: Mapping
        The mapping around which this class is wrapped.
    name: str
        The name of this group relative to the root group. Defaults to "".
    """
    group: Mapping
    name: str = ""

    def __getitem__(self, item: str):
        try:
            val = self.group[item]
        except KeyError as err:
            raise KeyError(
                f'Could not find item or group "{item}" in parent group "{self.name}".'
            ) from err

        if isinstance(val, Mapping):
            prefix = self.name + "." if self.name != "" else ""
            return NamedMapping(name=prefix + item, group=val)
        
        return val
    
    def __setitem__(self, key: str, value) -> None:
        self.group[key] = value


def dumpconfig_gslc_gcov(
    workflow: str,
    rslc_file: os.PathLike,
    dem_file: os.PathLike,
    tec_file: os.PathLike | None = None,
    *,
    orbit_file: os.PathLike | None = None,
    output_file: os.PathLike | None = None,
    source_runconfig_file: os.PathLike | None = None,
    epsg: int | None = None,
    a_spacing: Sequence[Decimal | float] | None = None,
    b_spacing: Sequence[Decimal | float] | None = None,
    x_snap: float | Decimal | None = None,
    y_snap: float | Decimal | None = None,
    top_left: Sequence[Decimal | float] | None = None,
    bottom_right: Sequence[Decimal | float] | None = None,
    use_gpu: bool = False,
    flatten: bool | None = None,
    access_internet: bool = False,
    generate_epsg: bool = False,
    nisar_defaults: bool = False,
    **kwargs,
) -> str:
    """
    Output a GSLC or GCOV runconfig using the given inputs; return it as a string.

    Alternative default EPSG behavior:
        If no EPSG value is given and generate_epsg is True, calculate EPSG at the
        center of the bounding box.

    Alternative default snap and spacing value behaviors:
        If nisar_defaults is True, then any snap or spacing value set to None will be
        calculated as a sensible default for the workflow at the bandwidth of the input
        product frequencies. Only valid for products with 5, 10, 20, or 77 MHz
        bandwidths for all frequencies. These defaults are defined in
        `get_geocoding_geometry_defaults()`.

    Parameters
    ----------
    workflow : str
        The name of the workflow, all lowercase - one of {"gslc", "gcov"}
    rslc_file : path-like
        The RSLC HDF5 filepath
    dem_file : path-like
        The filepath of the DEM file
    tec_file : path-like, optional
        The filepath of the TEC file, or None. Defaults to None.
    orbit_file : path-like or None, optional
        The filepath of an external orbit XML file to use for the workflow, or None. If
        None, the orbit inside the input RSLC will be used for processing.
        Defaults to None.
    output_file : path-like or None, optional
        The filepath of the output product, or None. If None, the workflow will output
        to {workflow}.h5 in the local directory. Defaults to None.
    source_runconfig_file : path-like or None, optional
        The filepath of the runconfig to base the output on, or None. If None, the
        default runconfig for the workflow will be used. Defaults to None.
    epsg : int or None, optional
        The EPSG code to use for output geogrid definition, or None. If None, the EPSG
        will be left as the default in the file.
        Defaults to None.
    a_spacing : sequence of 2 {Decimal or float} or None, optional
        The spacing of the output geocoded grid for frequency A data, in the same units
        as the EPSG projection, in [x spacing, y spacing] format. Defaults to None.
    b_spacing : sequence of 2 {Decimal or float} or None, optional
        The spacing of the output geocoded grid for frequency B data, in the same units
        as the EPSG projection, in [x spacing, y spacing] format. Defaults to None.
    x_snap : Decimal or float or None, optional
        The snap value in the x dimension, or None. Defaults to None.
    y_snap : Decimal or float or None, optional
        The snap value in the y dimension, or None. Defaults to None.
    top_left :  sequence of 2 {Decimal or float} or None, optional
        The top left corner of the image, in [x, y] order, in the same units as the
        EPSG, or None. Defaults to None.
    bottom_right :  sequence of 2 {Decimal or float} or None, optional
        The bottom right corner of the image, in [x, y] order, in the same units as the
        EPSG, or None. Defaults to None.
    use_gpu : bool, optional
        If True, activate GPU in the runconfig. Defaults to False.
    flatten : bool or None, optional
        Whether or not to flatten the GSLC output. If None, use the default behavior for
        the workflow. Not used for GCOV. Defaults to None.
    access_internet : bool, optional
        If True, activate internet access functionality in the runconfig.
        Defaults to False.
    generate_epsg : bool, optional
        If True, an EPSG is determined based on the tile containing the center of the
        input RSLC bounding polygon on ground. Defaults to False.
    nisar_defaults : bool, optional
        If True, provide sensible default snap and spacing values in meters for any snap
        or spacing value not given.
        Defaults to False.

    Returns
    -------
    str
        The contents of the output, formatted as a string.
    """
    
    if workflow not in ["gcov", "gslc"]:
        raise ValueError(
            f"Workflow name {workflow} not in", '{"gslc", "gcov"}',
            "- only these values are recognized by dumpconfig_geocoded."
        )

    if (workflow == "gcov") and (flatten is not None):
        raise ValueError("`flatten` should be None for GCOV products.")
    
    if generate_epsg and (epsg is not None):
        raise ValueError(
            "Mutually exclusive parameters: if epsg is given, generate_epsg must "
            "be False."
        )

    if source_runconfig_file is None:
        runconfig_filepath = (Path(DEFAULT_YAML_DIR) / f"{workflow}.yaml").resolve()
    else:
        runconfig_filepath = Path(source_runconfig_file).expanduser().resolve()

    rslc_file = Path(rslc_file).expanduser().resolve()
    rslc = SLC(hdf5file=os.fspath(rslc_file))

    geom_defaults: GeocodingGeometryDefaults

    if "A" in rslc.frequencies:
        if nisar_defaults:
            geom_defaults = get_geocoding_geometry_defaults(
                workflow=workflow,
                bandwidth=int(
                    np.round(rslc.getSwathMetadata('A').processed_range_bandwidth / 1e6)
                ),
            )
            if a_spacing is None:
                a_spacing = (geom_defaults.x_posting, geom_defaults.y_posting)
            else:
                if len(a_spacing) != 2:
                    raise ValueError(
                        f"a_spacing given with length {len(a_spacing)}; must have two "
                        "items in order of x_spacing, y_spacing."
                    )
    else:
        if a_spacing is not None:
            raise ValueError("a_spacing given for an RSLC product with no frequency A.")
    a_spacing = None if a_spacing is None else (
        Decimal(a_spacing[0]), Decimal(a_spacing[1])
    )

    if "B" in rslc.frequencies:
        if nisar_defaults:
            geom_defaults = get_geocoding_geometry_defaults(
                workflow=workflow,
                bandwidth=int(
                    np.round(rslc.getSwathMetadata('B').processed_range_bandwidth / 1e6)
                ),
            )
            if b_spacing is None:
                b_spacing = (geom_defaults.x_posting, geom_defaults.y_posting)
            else:
                if len(b_spacing) != 2:
                    raise ValueError(
                        f"b_spacing given with length {len(b_spacing)}; must have two "
                        "items in order of x_spacing, y_spacing."
                    )
    else:
        if b_spacing is not None:
            raise ValueError("b_spacing given for an RSLC product with no frequency B.")
    b_spacing = None if b_spacing is None else (
        Decimal(b_spacing[0]), Decimal(b_spacing[1])
    )

    yaml = YAML()
    yaml.indent(mapping=4, sequence=4, offset=4)
    with open(runconfig_filepath, mode="r") as file_in:
        output_yaml = yaml.load(file_in)
    
    root = NamedMapping(group=output_yaml)
    groups = root["runconfig"]["groups"]

    # Quick and dirty check that the source runconfig file is the correct sort of
    # runconfig.
    runconfig_type = groups["primary_executable"]["product_type"]
    if runconfig_type != workflow.upper():
        raise ValueError(
            f"Provided base runconfig has product_type {runconfig_type}. "
            f"Runconfig must have {workflow.upper()} product_type."
        )

    groups["input_file_group"]["input_file_path"] = os.fspath(rslc_file)

    dynamic_ancillary_group = groups["dynamic_ancillary_file_group"]
    
    dem_file = Path(dem_file).expanduser().resolve()
    dynamic_ancillary_group["dem_file"] = os.fspath(dem_file)

    if tec_file is not None:
        tec_file = Path(tec_file).expanduser().resolve()
        dynamic_ancillary_group["tec_file"] = os.fspath(tec_file)

    if orbit_file is not None:
        orbit_file = Path(orbit_file).expanduser().resolve()
        dynamic_ancillary_group["orbit_file"] = os.fspath(orbit_file)
    
    product_path_group = groups["product_path_group"]

    if output_file is not None:
        output_file = Path(output_file).expanduser().resolve()
        product_path_group["sas_output_file"] = os.fspath(output_file)
    
    geocode_group = groups["processing"]["geocode"]
    radar_grid_cubes_group = groups["processing"]["radar_grid_cubes"]

    if x_snap is None and nisar_defaults:
        x_snap = geom_defaults.x_snap
    elif x_snap is not None:
        geocode_group["x_snap"] = float(x_snap)
        radar_grid_cubes_group["x_snap"] = float(x_snap)

    if y_snap is None and nisar_defaults:
        y_snap = geom_defaults.y_snap
    elif y_snap is not None:
        geocode_group["y_snap"] = float(y_snap)
        radar_grid_cubes_group["y_snap"] = float(y_snap)
    
    if top_left is not None:
        if len(top_left) != 2:
            raise ValueError("top_left should either have exactly 2 values or be None.")
        top_left_group = geocode_group["top_left"]
        top_left_group["x_abs"] = float(top_left[0])
        top_left_group["y_abs"] = float(top_left[1])

    if bottom_right is not None:
        if len(bottom_right) != 2:
            raise ValueError(
                "bottom_right should either have exactly 2 values or be None."
            )
        bottom_right_group = geocode_group["bottom_right"]
        bottom_right_group["x_abs"] = float(bottom_right[0])
        bottom_right_group["y_abs"] = float(bottom_right[1])

    # Check that the snap values given are integer multiples of the spacing values for
    # all frequencies for which a spacing value has been defined.
    if a_spacing is not None:
        check_snap_vs_spacing(
            frequency="A",
            spacing=a_spacing,
            x_snap=None if x_snap is None else Decimal(x_snap),
            y_snap=None if y_snap is None else Decimal(y_snap),
        )
    if b_spacing is not None:
        check_snap_vs_spacing(
            frequency="B",
            spacing=b_spacing,
            x_snap=None if x_snap is None else Decimal(x_snap),
            y_snap=None if y_snap is None else Decimal(y_snap),
        )

    if epsg is not None:
            geocode_group["output_epsg"] = epsg
            radar_grid_cubes_group["output_epsg"] = epsg
    # EPSG default:
    # Based off of PLAnT-ISCE3: Get the EPSG of the center of the bounding
    # box.
    elif epsg is None and generate_epsg:
        # Get the parent RSLC bounding polygon
        polygon = rslc.identification.boundingPolygon
        centroid = shapely.centroid(shapely.wkt.loads(polygon))

        # Get the EPSG of the centerpoint
        default_epsg = point_to_epsg(lon=centroid.x, lat=centroid.y)
        
        geocode_group["output_epsg"] = default_epsg
        radar_grid_cubes_group["output_epsg"] = default_epsg

    if a_spacing is not None:
        x_spacing_a, y_spacing_a = a_spacing
        a_spacing_group = geocode_group["output_posting"]["A"]
        a_spacing_group["x_posting"] = float(x_spacing_a)
        a_spacing_group["y_posting"] = float(y_spacing_a)
    if b_spacing is not None:
        x_spacing_b, y_spacing_b = b_spacing
        b_spacing_group = geocode_group["output_posting"]["B"]
        b_spacing_group["x_posting"] = float(x_spacing_b)
        b_spacing_group["y_posting"] = float(y_spacing_b)

    if workflow == "gcov":
        geocode_group["apply_shadow_masking"] = False
    
    if (workflow == "gslc") and (flatten is not None):
        groups["processing"]["flatten"] = flatten

    groups["worker"]["internet_access"] = access_internet
    groups["worker"]["gpu_enabled"] = use_gpu

    # To check against the schema, output the YAML object into a string. Parse this
    # into a yamale data object.
    string_stream = StringIO()
    yaml.dump(output_yaml, string_stream)
    yaml_string = string_stream.getvalue()
    return yaml_string


def point_to_epsg(lon: float, lat: float) -> int:
    """
    Return the EPSG code of the UTM tile or polar stereographic zone containing the
    given lon/lat coordinates.

    Parameters
    ----------
    lon : float
        The longitude coordinate.
    lat : float
        The longitude coordinate.

    Returns
    -------
    int
        The EPSG code.

    Notes
    -----
    This approximates the boundary of the polar stereographic zones at +- 60 degrees
    latitude
    """
    if lon >= 180.0:
        lon = lon - 360.0
    if lat >= 60.0:
        return 3413
    elif lat <= -60.0:
        return 3031
    elif lat > 0:
        return 32601 + int(np.round((lon + 177) / 6.0))
    elif lat < 0:
        return 32701 + int(np.round((lon + 177) / 6.0))
    raise ValueError(
        'Could not determine projection for {0},{1}'.format(lat, lon))


def check_snap_vs_spacing(
    frequency: str,
    spacing: tuple[Decimal, Decimal] | None,
    x_snap: Decimal | None,
    y_snap: Decimal | None,
) -> None:
    """
    Check that the snap values for a given frequency, on a given geogrid, are an integer
    multiple of the associated spacing values for that grid.

    Parameters
    ----------
    frequency : str
        The name of the frequency, either "A" or "B".
    spacing : tuple[Decimal, Decimal]
        The spacing of the geogrid, in (x_spacing, y_spacing) format.
    x_snap : Decimal | None
        The x snap value for the geogrid, or None.
    y_snap : Decimal | None
        The y snap value for the geogrid, or None.

    Raises
    ------
    ValueError
        If the x or y snap values are not integer multiples of their associated spacing
        value.
    """
    if spacing is None:
        return

    x_spacing, y_spacing = spacing

    # Using the Decimal object ensures that the modulo operator works correctly for
    # floating point arithmetic.
    if x_snap is not None and x_snap % x_spacing != Decimal(0.0):
        raise ValueError(
            f"x snap value {x_snap} is not an integer multiple of Frequency "
            f"{frequency} x spacing value {x_spacing}"
        )
    if y_snap is not None and y_snap % y_spacing != Decimal(0.0):
        raise ValueError(
            f"y snap value {y_snap} is not an integer multiple of Frequency "
            f"{frequency} y spacing value {y_spacing}"
        )


def validate_runconfig(
    workflow: str,
    runconfig_contents: str,
    *,
    verbose: bool = True,
):
    """
    Validate the given runconfig file against the given workflow against its schema
    and print output errors.

    Parameters
    ----------
    workflow : str
        The workflow, all lowercase - one of {"focus", "rslc", "gslc", "gcov", "insar"}
    runconfig_contents : str
        The contents of the runconfig to be checked.
    verbose : bool
        If True, print the errors to the screen. Defaults to True.
    """
    # Parse yaml_str into a yamale data object.
    data_yaml = yamale.make_data(content=runconfig_contents)

    # Get the schema path and parse it into a yamale schema object.
    schema_yaml_filepath = os.path.join(SCHEMA_YAML_DIR, f"{workflow}.yaml")
    schema_yaml = yamale.make_schema(schema_yaml_filepath)

    # Run validation and output all validation errors.
    try:
        results: list[ValidationResult] = yamale.validate(schema_yaml, data_yaml)
    # If validation failed, catch the error.
    except YamaleError as err:
        results = err.results
        # If verbose, print out all errors associated with the file.
        if verbose:
            print(
                f"{workflow.upper()} schema found YAML validation errors in dumpconfig "
                "output:"
            )
            for result in results:
                for error in result.errors:
                    print(f"\t{error}")
        # Return the errors.
        return results
    
    # If validation succeeded, return an empty list.
    return []


def main():
    parser = init_argparse()
    args_parsed = parser.parse_args(sys.argv[1:])
    workflow = args_parsed.workflow

    if workflow in ["gslc", "gcov"]:
        config = dumpconfig_gslc_gcov(**vars(args_parsed))

        if args_parsed.out_runconfig is not None:
            out_runconfig = Path(args_parsed.out_runconfig).expanduser().resolve()
            with open(out_runconfig, 'w') as file:
                file.write(config)

        else:
            print(config)
            print()

        if args_parsed.validate:
            validate_runconfig(workflow=workflow, runconfig_contents=config)


if __name__ == "__main__":
    main()
