from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from typing import TypedDict

import h5py
import iscetest
import nisar
import numpy as np
import pytest
from nisar.products import DatasetSpec, ProductSpec
from nisar.workflows.static import run_static_layers_workflow

import isce3

from ..util import create_tmp_text_file


class ParamDict(TypedDict):
    """Dict of configuration parameters for each test case."""

    dem_raster_file: Path
    water_mask_raster_file: Path
    orbit_xml_file: Path
    pointing_xml_file: Path
    geo_grid: isce3.product.GeoGridParameters


# Test case covering a region of Antarctica in Polar Stereographic projection.
def antartica_params() -> ParamDict:
    prefix = Path(iscetest.data)
    dem_raster_file = prefix / "DEM_antarctica_track29_frame130_small.tiff"
    water_mask_raster_file = prefix / "WATERMASK_antarctica_track29_frame130_small.tiff"
    orbit_xml_file = (
        prefix
        / "NISAR_ANC_L_PR_FOE_20250806T193731_20230105T054735_20230105T055408.xml"
    )
    pointing_xml_file = (
        prefix
        / "NISAR_ANC_L_PR_FRP_20250910T222200_20230105T054735_20230105T055408.xml"
    )

    geo_grid = isce3.product.GeoGridParameters(
        start_x=299440.0,
        start_y=-661600.0,
        spacing_x=10.0,
        spacing_y=-5.0,
        width=128,
        length=256,
        epsg=3031,
    )

    return dict(
        dem_raster_file=dem_raster_file,
        water_mask_raster_file=water_mask_raster_file,
        orbit_xml_file=orbit_xml_file,
        pointing_xml_file=pointing_xml_file,
        geo_grid=geo_grid,
    )


# Test case covering a region of Fiji and containing an antimeridian crossing.
def fiji_params() -> ParamDict:
    prefix = Path(iscetest.data)
    dem_raster_file = prefix / "DEM_fiji_track15_frame97_small.vrt"
    water_mask_raster_file = prefix / "WATERMASK_fiji_track15_frame97_small.vrt"
    orbit_xml_file = (
        prefix
        / "NISAR_ANC_L_PR_FOE_20250806T193246_20230104T061021_20230104T061655.xml"
    )
    pointing_xml_file = (
        prefix
        / "NISAR_ANC_L_PR_FRP_20250910T221957_20230104T061021_20230104T061655.xml"
    )

    geo_grid = isce3.product.GeoGridParameters(
        start_x=808280.0,
        start_y=8123560.0,
        spacing_x=80.0,
        spacing_y=-80.0,
        width=257,
        length=256,
        epsg=32760,
    )

    return dict(
        dem_raster_file=dem_raster_file,
        water_mask_raster_file=water_mask_raster_file,
        orbit_xml_file=orbit_xml_file,
        pointing_xml_file=pointing_xml_file,
        geo_grid=geo_grid,
    )


@contextmanager
def make_tmp_runconfig_file(
    output_hdf5_file: os.PathLike | str,
    dem_raster_file: os.PathLike | str,
    water_mask_raster_file: os.PathLike | str,
    orbit_xml_file: os.PathLike | str,
    pointing_xml_file: os.PathLike | str,
    geo_grid: isce3.product.GeoGridParameters,
) -> Generator[Path, None, None]:
    """
    A context manager that creates a temporary Static Layers runconfig file.

    Creates a run configuration file for the NISAR Static Layers workflow with the
    specified parameters. The file is automatically removed from the file system when
    the context block exits.

    Parameters
    ----------
    output_hdf5_file : path-like
        The path to the output Static Layers HDF5 file.
    dem_raster_file : path-like
        The path to the input DEM raster file.
    water_mask_raster_file : path-like
        The path to the input water mask raster file.
    orbit_xml_file : path-like
        The path to the input orbit ephemeris XML file.
    pointing_xml_file : path-like
        The path to the input radar pointing XML file.
    geo_grid : isce3.product.GeoGridParameters
        The geocoded coordinate grid on which the layers in the output product will be
        computed.

    Yields
    ------
    pathlib.Path
        The path to the runconfig file.
    """
    contents = dedent(
        f"""\
        runconfig:
          groups:
            dynamic_ancillary_file_group:
              dem_raster_file: {dem_raster_file}
              water_mask_raster_file: {water_mask_raster_file}
              orbit_xml_file: {orbit_xml_file}
              pointing_xml_file: {pointing_xml_file}

            product_path_group:
              output_hdf5_file: {output_hdf5_file}

            processing:
              geo_grid:
                epsg: {geo_grid.epsg}
                top_left:
                  x: {geo_grid.start_x}
                  y: {geo_grid.start_y}
                bottom_right:
                  x: {geo_grid.end_x}
                  y: {geo_grid.end_y}
                posting:
                  x: {abs(geo_grid.spacing_x)}
                  y: {abs(geo_grid.spacing_y)}
        """
    )
    with create_tmp_text_file(contents, suffix=".yml") as f:
        yield f


def list_dataset_names(hdf5_file: h5py.File) -> list[str]:
    """
    List the full paths of each dataset in an HDF5 file.

    Parameters
    ----------
    hdf5_file : h5py.File
        The input HDF5 file.

    Returns
    -------
    list of str
        The full path of each dataset in the file.
    """
    names = []

    def append_dataset_name(name: str) -> None:
        # The `h5py.Group.visit()` method doesn't seem to prefix each name a '/', but
        # our product specs generally include this leading character, so it's useful to
        # ensure that it's prepended here for consistency with the spec.
        if not name.startswith("/"):
            name = "/" + name
        if isinstance(hdf5_file[name], h5py.Dataset):
            names.append(name)

    hdf5_file.visit(append_dataset_name)
    return names


def validate_dataset_against_spec(
    dataset: h5py.Dataset, dataset_spec: DatasetSpec
) -> None:
    """
    Check by assertion that the datatype and attributes of a dataset match the spec.

    Parameters
    ----------
    dataset : h5py.Dataset
        A dataset in a NISAR product.
    dataset_spec : nisar.products.DatasetSpec
        The dataset specification from the product spec XML.
    """
    # Check that the datasets's datatype matches the spec.
    # For string-valued datasets, if the spec doesn't expect a particular string length,
    # just check that the datatype is bytes. Otherwise, the datatype should exactly
    # match the spec.
    if dataset_spec.dtype == np.dtype("S"):
        assert np.issubdtype(dataset.dtype, np.bytes_)
    else:
        assert dataset.dtype == dataset_spec.dtype

    # Check that the dataset's attributes match what's in the spec.
    # For 'projection' datasets, the expected attributes vary based on the type of
    # projection -- the spec includes only the expected attributes for UTM projections.
    # For all other datasets, the dataset attributes should exactly match the spec
    # attributes.
    attr_names = set(dataset.attrs)
    expected_attr_names = set(dataset_spec.attrs) | {"description"}
    if not dataset.name.endswith("/projection"):
        assert attr_names == expected_attr_names


def validate_product_against_spec(
    hdf5_file: h5py.File, product_spec: ProductSpec
) -> None:
    """
    Check by assertion that a NISAR product matches its specification.

    Parameters
    ----------
    hdf5_file : h5py.File
        A NISAR product HDF5 file.
    product_spec : nisar.product.ProductSpec
        The product specification.
    """
    # Check that there aren't any missing datasets or unexpected datasets in the
    # product.
    dataset_names = list_dataset_names(hdf5_file)
    expected_dataset_names = (spec.name for spec in product_spec.iter_dataset_specs())
    assert set(dataset_names) == set(expected_dataset_names)

    # Check that there aren't any missing or unexpected global attributes in the
    # product.
    assert set(hdf5_file.attrs) == set(product_spec.global_attrs)

    # Compare each dataset in the product against the product spec.
    for dataset_spec in product_spec.iter_dataset_specs():
        dataset = hdf5_file[dataset_spec.name]
        validate_dataset_against_spec(dataset, dataset_spec)


@pytest.mark.parametrize("params", [antartica_params(), fiji_params()])
def test_static_layers_workflow(params: ParamDict):
    # Run the Static Layers workflow end-to-end and compare the contents against the
    # product spec.

    # Create a temporary HDF5 file to be cleaned up automatically upon exiting the
    # context manager.
    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        output_hdf5_file = Path(f.name)

        # Create a runconfig file and run the workflow.
        with make_tmp_runconfig_file(output_hdf5_file, **params) as runconfig_file:
            run_static_layers_workflow(runconfig_file)

        # Compare the output product against the spec.
        product_spec = nisar.products.get_product_spec("STATIC")
        with h5py.File(output_hdf5_file, mode="r") as hdf5_file:
            validate_product_against_spec(hdf5_file, product_spec)
