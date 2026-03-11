from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Union

import yamale
import yamale.schema
from nisar.workflows.helpers import WORKFLOW_SCRIPTS_DIR, deep_update
from ruamel.yaml import YAML

RunConfigScalar = Union[str, bool, int, float, None]
RunConfigList = Sequence["RunConfigValue"]
RunConfigDict = Mapping[str, "RunConfigValue"]
RunConfigValue = Union[RunConfigScalar, RunConfigList, RunConfigDict]


def get_yamale_schema() -> yamale.schema.Schema:
    """Get a Yamale schema to validate NISAR Static Layers run configuration files."""
    schema_path = Path(WORKFLOW_SCRIPTS_DIR) / "schemas/static.yaml"
    return yamale.make_schema(schema_path, parser="ruamel")


def validate_runconfig(runconfig: os.PathLike | str) -> None:
    """
    Validate a NISAR Static Layers run configuration file against the schema.

    Parameters
    ----------
    runconfig : path-like
        The filesystem path of the run configuration YAML file.

    Raises
    ------
    YamaleError
        If the input file is not a valid NISAR Static Layers run configuration YAML
        file.
    """
    schema = get_yamale_schema()
    data = yamale.make_data(runconfig, parser="ruamel")
    yamale.validate(schema, data)


def parse_runconfig(runconfig: os.PathLike | str) -> RunConfigDict:
    """
    Parse a run configuration YAML file.

    Parameters
    ----------
    runconfig : path-like
        The filesystem path of the run configuration YAML file.

    Returns
    -------
    dict
        A nested dict of parameter values.
    """
    yaml = YAML(typ="safe")
    with Path(runconfig).open(mode="r") as f:
        return yaml.load(f)


def default_runconfig_file() -> Path:
    """Get the filepath of the default Static Layers runconfig YAML file."""
    return Path(WORKFLOW_SCRIPTS_DIR) / "defaults/static.yaml"


def get_runconfig_params(
    runconfig: os.PathLike | str,
    *,
    validate: bool = True,
) -> RunConfigDict:
    """
    Parse a runconfig file, optionally validate it, and populate default values.

    Parameters
    ----------
    runconfig : path-like
        The filesystem path of the run configuration YAML file.
    validate : bool, optional
        If True, validate the file against the NISAR Static Layers run configuration
        file schema. Defaults to True.

    Returns
    -------
    dict
        A nested dict of parameter values.
    """
    # Validate user runconfig against schema.
    if validate:
        validate_runconfig(runconfig)

    # Parse default runconfig file.
    defaults_runconfig_dict = parse_runconfig(default_runconfig_file())

    # Parse user runconfig file.
    runconfig_dict = parse_runconfig(runconfig)

    # Recursively overwrite defaults with user-specified parameters.
    deep_update(defaults_runconfig_dict, runconfig_dict)

    return defaults_runconfig_dict


def dump_runconfig_to_str(params: RunConfigDict) -> str:
    """
    Dump a nested dict of runconfig parameters into a string in YAML syntax.

    Parameters
    ----------
    params : dict
        The input dict of run configuration parameters.

    Returns
    -------
    str
        A string representation of the input parameter dict in YAML syntax.
    """
    yaml = YAML(typ="safe")

    # Force block style instead of flow style.
    yaml.default_flow_style = False

    string_stream = StringIO()
    yaml.dump(params, string_stream)
    return string_stream.getvalue()
