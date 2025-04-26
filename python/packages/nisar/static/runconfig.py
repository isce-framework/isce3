from __future__ import annotations

import os
import re
from io import StringIO
from pathlib import Path

import numpy as np
import yamale
import yamale.schema
from ruamel.yaml import YAML

import isce3
from nisar.workflows.helpers import WORKFLOW_SCRIPTS_DIR, deep_update

from .typing import RunConfigDict


def get_yamale_schema() -> yamale.schema.Schema:
    """ """
    schema_path = Path(WORKFLOW_SCRIPTS_DIR) / "schemas/static.yaml"
    return yamale.make_schema(schema_path, parser="ruamel")


def validate_runconfig(runconfig: os.PathLike | str) -> None:
    """ """
    schema = get_yamale_schema()
    data = yamale.make_data(runconfig, parser="ruamel")
    yamale.validate(schema, data)


def parse_runconfig(runconfig: os.PathLike | str) -> RunConfigDict:
    """ """
    yaml = YAML(typ="safe")
    with Path(runconfig).open(mode="r") as f:
        return yaml.load(f)


def get_runconfig_params(
    runconfig: os.PathLike | str,
    *,
    validate: bool = True,
) -> RunConfigDict:
    """ """
    # Validate user runconfig against schema.
    if validate:
        validate_runconfig(runconfig)

    # Parse default runconfig file.
    default_runconfig_path = Path(WORKFLOW_SCRIPTS_DIR) / "defaults/static.yaml"
    defaults_runconfig_dict = parse_runconfig(default_runconfig_path)

    # Parse user runconfig file.
    runconfig_dict = parse_runconfig(runconfig)

    # Recursively overwrite defaults with user-specified parameters.
    deep_update(defaults_runconfig_dict, runconfig_dict)

    return defaults_runconfig_dict


def is_valid_composite_release_id(composite_release_id: str) -> bool:
    """
    JPL D-102255
    """
    regex = re.compile(
        r"^(?P<environment>[ADPTS])(?P<phase>[0-3])(?P<major>(\d{2})(?P<minor>\d)"
        r"(?P<patch>\d)$"
    )
    return regex.match(composite_release_id) is not None


def approximately_integer_valued(f: float) -> bool:
    """ """
    return np.isclose(f, np.round(f))


def validate_production_runconfig_params(
    *,
    relative_orbit_number: int,
    frame_number: int,
    geo_grid: isce3.product.GeoGridParameters,
    product_doi: str,
    processing_center: str,
    dem_source: str,
    water_mask_source: str,
) -> None:
    """ """
    if not (1 <= relative_orbit_number <= 173):
        raise ValueError  # FIXME
    if not (1 <= frame_number <= 176):
        raise ValueError  # FIXME

    if not (
        isce3.core.is_utm(geo_grid.epsg) or isce3.core.is_polar_stereo(geo_grid.epsg)
    ):
        raise ValueError  # FIXME

    # ... granule ID ...
    if not approximately_integer_valued(geo_grid.spacing_x):
        raise ValueError  # FIXME
    if not approximately_integer_valued(geo_grid.spacing_y):
        raise ValueError  # FIXME

    for s in [
        product_doi,
        processing_center,
        dem_source,
        water_mask_source,
    ]:
        if s == "(NOT SPECIFIED)":
            raise ValueError  # FIXME


def dump_runconfig_to_str(params: RunConfigDict) -> str:
    """ """
    yaml = YAML(typ="safe")
    string_stream = StringIO()
    yaml.dump(params, string_stream)
    return string_stream.getvalue()
