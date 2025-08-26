import tempfile
from pathlib import Path

import iscetest
import numpy as np
from nisar.static.water_mask import binarize_nisar_water_mask, reproject_raster

import isce3


def test_binarize_nisar_water_mask():
    # Create three arrays of water pixels (filled with zeros), non-water pixels (filled
    # with values in range [1, 200]), and invalid pixels (filled with 255).
    water = np.zeros((20, 10), dtype=np.uint8)
    nonwater = np.arange(1, 201, dtype=np.uint8).reshape(20, 10)
    invalid = np.full((20, 10), fill_value=255, dtype=np.uint8)

    # Concatenate the three arrays into a single 20x30 array.
    water_distance = np.concatenate([water, nonwater, invalid], axis=1)

    # Convert the water distance map into a binary water mask
    water_mask = binarize_nisar_water_mask(water_distance)

    # Check the datatype of the output mask.
    assert water_mask.dtype == np.uint8

    # Check the mask values.
    np.testing.assert_equal(water_mask[:, :10], 1)
    np.testing.assert_equal(water_mask[:, 10:20], 0)
    np.testing.assert_equal(water_mask[:, 20:], 255)


def test_reproject_raster():
    # A DEM raster in geodetic coordinates (EPSG:4326).
    prefix = Path(iscetest.data)
    src_file = prefix / "DEM_fiji_track15_frame97_small.vrt"

    # A grid in projected coordinates (UTM) covering a subset of the DEM.
    geo_grid = isce3.product.GeoGridParameters(
        start_x=808280.0,
        start_y=8123560.0,
        spacing_x=80.0,
        spacing_y=-80.0,
        width=257,
        length=256,
        epsg=32760,
    )

    # Create a temporary GeoTIFF file to be cleaned up automatically upon exiting the
    # context manager.
    with tempfile.NamedTemporaryFile(suffix=".tiff") as dst_file:
        # Re-project and resample the DEM raster on the output geocoded grid.
        reproject_raster(src_file, dst_file.name, geo_grid, algorithm="cubic")

        dst_raster = isce3.io.Raster(dst_file.name)

        # Check that the output raster grid matches `geo_grid`.
        assert np.isclose(dst_raster.x0, geo_grid.start_x)
        assert np.isclose(dst_raster.y0, geo_grid.start_y)
        assert np.isclose(dst_raster.dx, geo_grid.spacing_x)
        assert np.isclose(dst_raster.dy, geo_grid.spacing_y)
        assert dst_raster.width == geo_grid.width
        assert dst_raster.length == geo_grid.length

        # Ensure the raster is safely closed before the file is deleted.
        dst_raster.close_dataset()
