import argparse
import os

from pytest import fixture

from nisar.products.readers import GSLC, RSLC
from nisar.workflows.gslc_runconfig import GSLCRunConfig

import iscetest


@fixture
def gslc_object() -> GSLC:
    """A GSLC reader for the file output by the GSLC workflow test."""
    # This is the GSLC workflow test output. All tests using this fixture rely
    # therefore on the GSLC workflow test.
    gslc_file = "../../workflows/x_out.h5"

    if not os.path.isfile(gslc_file):
        raise ValueError(
            "GSLC file not found. This test depends on the GSLC workflow test - "
            "Please make sure that the workflow test has been run prior to this one. "
            "If you have modified the GSLC workflow test, please point `gslc_file` to "
            "the location of an output of the modified test."
        )

    return GSLC(hdf5file=gslc_file)


@fixture
def rslc_object() -> RSLC:
    """An RSLC reader for the RSLC file input into the GSLC workflow test."""
    rslc_file = f'{iscetest.data}/envisat.h5'

    if not os.path.isfile(rslc_file):
        raise ValueError(
            "RSLC file (envisat.h5) not found. If you have changed the name of this "
            "file, please point `rslc_file` to the location of the new RSLC test data."
        )

    return RSLC(hdf5file=rslc_file)


@fixture
def gslc_runconfig() -> GSLCRunConfig:
    test_yaml = os.path.join(iscetest.data, 'geocodeslc/test_gslc.yaml')

    # load text then substitude test directory paths since data dir is read only
    with open(test_yaml) as fh_test_yaml:
        test_yaml = fh_test_yaml.read(). \
            replace('@ISCETEST@', iscetest.data). \
            replace('@TEST_BLOCK_SZ_X@', '133'). \
            replace('@TEST_BLOCK_SZ_Y@', '1000')

    # create CLI input namespace with yaml text instead of file path
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # init runconfig object
    return GSLCRunConfig(args)


def test_gslc_source_radar_grid_params(gslc_object: GSLC, rslc_object: RSLC):
    """
    Tests that the getSourceRadarGridParameters method of the GSLC object produces
    comparable outputs to the getRadarGrid method of the parent RSLC product object.

    Parameters
    ----------
    gslc_object : GSLC
        The GSLC product reader.
    rslc_object : RSLC
        The product reader of the GSLC's parent RSLC product.
    """
    gslc_params = gslc_object.getSourceRadarGridParameters("A")
    rslc_params = rslc_object.getRadarGrid("A")

    tol = 1e-10

    # Float-valued parameters are checked within a tolerance of tol to handle
    # floating point precision issues
    start_time_diff = abs(gslc_params.sensing_start - rslc_params.sensing_start)
    assert start_time_diff < tol

    wavelength_diff = abs(gslc_params.wavelength - rslc_params.wavelength)
    assert wavelength_diff < tol

    prf_diff = abs(gslc_params.prf - rslc_params.prf)
    assert prf_diff < tol

    start_range_diff = abs(gslc_params.starting_range - rslc_params.starting_range)
    assert start_range_diff < tol

    # In particular, the range_pixel_spacing value of the two radar grid parameters
    # objects tends to differ by an extremely small amount (<1e-10)
    range_spacing_diff = abs(
        gslc_params.range_pixel_spacing - rslc_params.range_pixel_spacing
    )
    assert range_spacing_diff < tol

    assert gslc_params.lookside == rslc_params.lookside
    assert gslc_params.length == rslc_params.length
    assert gslc_params.width == rslc_params.width
    assert gslc_params.ref_epoch == rslc_params.ref_epoch


def test_gslc_product_level(gslc_object: GSLC):
    """Tests that the GSLC reader object returns the correct product level."""
    assert gslc_object.getProductLevel() == "L2"


def test_gslc_product_type(gslc_object: GSLC):
    """Tests that the GSLC reader object reaturns the correct product type."""
    assert gslc_object.identification.productType == "GSLC"


def test_gslc_coordinate_shapes(gslc_object: GSLC):
    """
    Tests that the GSLC reader object or underlying HDF5 file give the same dimensions
    to the X and Y coordinate datasets as they do to the SLC dataset.
    """
    x_coords, y_coords = gslc_object.getGeoGridCoordinateDatasets("A")
    coords_length = y_coords.shape[0]
    coords_width = x_coords.shape[0]

    data = gslc_object.getSlcDataset("A", "HH")
    data_length, data_width = data.shape

    assert data_length == coords_length
    assert data_width == coords_width


def test_gslc_coordinate_spacing(gslc_runconfig: GSLCRunConfig, gslc_object: GSLC):
    """
    Check that the spacing values of the GSLC object match what is in the runconfig.

    Parameters
    ----------
    gslc_runconfig : GSLCRunConfig
        The runconfig used to generate the GSLC product.
    gslc_object : GSLC
        The GSLC product generated with the runconfig.
    """
    cfg = gslc_runconfig.cfg
    processing_group = cfg["processing"]
    geocode_group = processing_group["geocode"]
    spacing_group = geocode_group["output_posting"]
    cfg_x_spacing = spacing_group["A"]["x_posting"]
    cfg_y_spacing = spacing_group["A"]["y_posting"]

    x_spacing, y_spacing = gslc_object.getGeoGridCoordinateSpacing("A")

    tol = 1e-10

    # Make sure the spacing recorded on the runconfig by not more than a precision error
    x_diff = abs(abs(x_spacing) - abs(cfg_x_spacing))
    assert x_diff < tol
    y_diff = abs(abs(y_spacing) - abs(cfg_y_spacing))
    assert y_diff < tol


def test_gslc_epsg(gslc_runconfig: GSLCRunConfig, gslc_object: GSLC):
    """
    Check that the EPSG value of the GSLC object matches what is in the runconfig.

    Parameters
    ----------
    gslc_runconfig : GSLCRunConfig
        The runconfig used to generate the GSLC product.
    gslc_object : GSLC
        The GSLC product generated with the runconfig.
    """
    cfg = gslc_runconfig.cfg
    processing_group = cfg["processing"]
    geocode_group = processing_group["geocode"]
    cfg_epsg = geocode_group["output_epsg"]

    epsg = gslc_object.getProjectionEpsg("A")

    assert cfg_epsg == epsg
