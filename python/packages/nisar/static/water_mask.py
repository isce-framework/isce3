from __future__ import annotations

import os

import numpy as np
from numpy.typing import ArrayLike
from osgeo import gdal

import isce3

from .util import make_scratch_file, make_scratch_gtiff, unary_transform_blockwise


def binarize_nisar_water_mask(water_distance: ArrayLike) -> np.ndarray:
    r"""
    Convert the input water distance map into a binary water mask.

    Parameters
    ----------
    water_distance : array_like
        The input water distance map, in the format specified by the NISAR Water Mask
        Product Specification\ [1]_. A value of 0 indicates a water pixel. A value of
        255 represents a no-data (invalid) pixel. Values in 1-200 represent non-water
        pixels.

    Returns
    -------
    water_mask : numpy.ndarray
        Binary mask where 1 indicates a water pixel (ocean or inland water), 0
        indicates a not-water pixel, and 255 represents a no-data (invalid) pixel.

    References
    ----------
    .. [1] J. Jung, "Water Mask Product Specification", JPL D-107710, 2024.
    """
    water_distance = np.asanyarray(water_distance)

    # Compute a binary mask where the value 1 represents (ocean or inland) water pixels
    # and the value 0 represents non-water pixels.
    water = water_distance == 0

    # Get a binary mask of invalid pixels (i.e. pixels whose value is equal to the fill
    # value of 255).
    fill_value = 255
    invalid = water_distance == fill_value

    return (water + fill_value * invalid).astype(np.uint8)


def reproject_raster(
    src: os.PathLike | str,
    dst: os.PathLike | str,
    geo_grid: isce3.product.GeoGridParameters,
    *,
    algorithm: str,
) -> None:
    """
    Re-project and resample a raster dataset to a new coordinate system and output grid.

    Parameters
    ----------
    src : path-like or str
        The file path or name of the input raster dataset.
    dst : path-like or str
        The file path or name of the output raster dataset.
    geo_grid : isce3.product.GeoGridParameters
        The output geocoded coordinate grid to re-project the input raster data onto.
    algorithm : str
        The resampling method to use. Must be one of the supported methods of `gdalwarp`
        listed here:
        https://gdal.org/en/stable/programs/gdalwarp.html#cmdoption-gdalwarp-r.
    """
    x_min, x_max = sorted((geo_grid.start_x, geo_grid.end_x))
    y_min, y_max = sorted((geo_grid.start_y, geo_grid.end_y))

    gdal.Warp(
        os.fsdecode(dst),
        os.fsdecode(src),
        outputBounds=[x_min, y_min, x_max, y_max],
        xRes=abs(geo_grid.spacing_x),
        yRes=abs(geo_grid.spacing_y),
        dstSRS=f"EPSG:{geo_grid.epsg}",
        resampleAlg=algorithm,
    )


def binarize_and_reproject_water_mask(
    water_distance_raster_file: os.PathLike | str,
    geo_grid: isce3.product.GeoGridParameters,
    *,
    scratch_dir: os.PathLike | str | None = None,
    resample_algorithm: str = "near",
) -> isce3.io.Raster:
    """
    Re-project the input water distance map and convert it to a binary mask.

    Re-project the input water distance map raster onto `geo_grid` and convert it to a
    binary mask of water/not-water pixels.

    Parameters
    ----------
    water_distance_raster_file : path-like
        The file path or name of the input water distance map file. It must be a
        GDAL-compatible raster file in the format specified by the NISAR Water Mask
        Product Specification\ [1]_. A value of 0 indicates a water pixel. A value of
        255 represents a no-data (invalid) pixel. Values in 1-200 represent non-water
        pixels.
    geo_grid : isce3.product.GeoGridParameters
        The output geocoded coordinate grid to re-project the water mask onto.
    scratch_dir : path-like or None, optional
        Directory to store intermediate files and output files created internally by
        this function. If None, a platform-specific default temporary directory will be
        used. Otherwise, it must be the file system path to an existing directory.
        Defaults to None.
    resample_algorithm : {'near', 'mode'}, optional
        Resampling method.

        'near':
          Nearest neighbor resampling. The default method.

        'mode':
          Mode resampling (selects the value which appears most often among sampled
          points).

    Returns
    -------
    isce3.io.Raster
        The re-projected binary water mask. A value of 1 indicates a water pixel (ocean
        or inland water), 0 indicates a not-water pixel, and 255 represents a no-data
        (invalid) pixel.
    """
    # Make a temporary file in the scratch directory to store the intermediate
    # re-projected water distance map raster.
    reprojected_water_distance_raster_file = make_scratch_file(
        dir_=scratch_dir,
        prefix="reprojected-water-distance-map_",
        suffix=".tif",
    )

    # Re-project the raster onto the output grid.
    reproject_raster(
        water_distance_raster_file,
        reprojected_water_distance_raster_file,
        geo_grid=geo_grid,
        algorithm=resample_algorithm,
    )

    # Open the new water distance raster and create a new raster to store the binary
    # water mask data.
    water_distance = isce3.io.Raster(str(reprojected_water_distance_raster_file))
    water_mask = make_scratch_gtiff(
        shape=(water_distance.length, water_distance.width),
        dtype=np.uint8,
        dir_=scratch_dir,
        prefix="water-mask_",
    )

    # Convert the water distance map to a binary mask.
    unary_transform_blockwise(binarize_nisar_water_mask, water_distance, water_mask)

    return water_mask
