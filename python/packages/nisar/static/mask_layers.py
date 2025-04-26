from __future__ import annotations

import os

import numpy as np
from osgeo import gdal

import isce3
from isce3.core import (
    DataInterpMethod,
    GeocodeMemoryMode,
    normalize_data_interp_method,
    normalize_geocode_memory_mode,
)
from isce3.geocode import GeocodeOutputMode
from .logging import get_logger
from .typing import Geo2RdrParamDict, Rdr2GeoParamDict, WaterMaskResampleAlgorithm
from .util import (
    create_single_band_gtiff,
    get_reference_ellipsoid,
    make_scratch_file,
    make_scratch_gtiff,
    transform_blockwise,
)


def compute_layover_shadow_mask(
    radar_grid: isce3.product.RadarGridParameters,
    orbit: isce3.core.Orbit,
    doppler: isce3.core.LUT2d,
    dem_raster: isce3.io.Raster,
    *,
    scratch_dir: os.PathLike | str | None = None,
    dem_interp_method: DataInterpMethod | str = DataInterpMethod.BIQUINTIC,
    lines_per_block: int = 1024,
    rdr2geo_params: Rdr2GeoParamDict | None = None,
) -> isce3.io.Raster:
    """ """
    if rdr2geo_params is None:
        rdr2geo_params = {}

    layover_shadow_mask = make_scratch_gtiff(
        shape=(radar_grid.length, radar_grid.width),
        dtype=np.uint8,
        dir_=scratch_dir,
        prefix="layover_shadow_mask",
    )

    ellipsoid = get_reference_ellipsoid(dem_raster)
    epsg = dem_raster.get_epsg()
    rdr2geo = isce3.geometry.Rdr2Geo(
        radar_grid=radar_grid,
        orbit=orbit,
        ellipsoid=ellipsoid,
        doppler=doppler,
        dem_interp_method=normalize_data_interp_method(dem_interp_method),
        epsg_out=epsg,
        compute_mask=True,  # FIXME?
        lines_per_block=lines_per_block,
        **rdr2geo_params,
    )
    rdr2geo.topo(
        dem_raster=dem_raster, layover_shadow_raster=layover_shadow_mask
    )

    return layover_shadow_mask


def geocode_layover_shadow_mask(
    layover_shadow_mask: isce3.io.Raster,
    radar_grid: isce3.product.RadarGridParameters,
    orbit: isce3.core.Orbit,
    doppler: isce3.core.LUT2d,
    geo_grid: isce3.product.GeoGridParameters,
    dem_raster: isce3.io.Raster,
    *,
    scratch_dir: os.PathLike | str | None = None,
    dem_interp_method: DataInterpMethod | str = DataInterpMethod.BIQUINTIC,
    geo2rdr_params: Geo2RdrParamDict | None = None,
    memory_mode: GeocodeMemoryMode | str = GeocodeMemoryMode.Auto,
    min_block_size: int = isce3.core.default_min_block_size,
    max_block_size: int = isce3.core.default_max_block_size,
) -> isce3.io.Raster:
    """ """
    if geo2rdr_params is None:
        geo2rdr_params = {}

    geocoded_layover_shadow_mask = make_scratch_gtiff(
        shape=(geo_grid.length, geo_grid.width),
        dtype=np.uint8,
        dir_=scratch_dir,
        prefix="geocoded_layover_shadow_mask",
    )

    # XXX: Float32
    geocode = isce3.geocode.GeocodeFloat32()
    geocode.orbit = orbit
    geocode.ellipsoid = get_reference_ellipsoid(dem_raster)
    geocode.doppler = doppler
    geocode.data_interpolator = DataInterpMethod.NEAREST
    geocode.geogrid(
        x_start=geo_grid.start_x,
        y_start=geo_grid.start_y,
        x_spacing=geo_grid.spacing_x,
        y_spacing=geo_grid.spacing_y,
        width=geo_grid.width,
        length=geo_grid.length,
        epsg=geo_grid.epsg,
    )

    if "threshold" in geo2rdr_params:
        geocode.threshold_geo2rdr = geo2rdr_params["threshold"]
    if "maxiter" in geo2rdr_params:
        geocode.numiter_geo2rdr = geo2rdr_params["maxiter"]

    # XXX: Add azimuth & range correction LUTs?
    geocode.geocode(
        radar_grid=radar_grid,
        input_raster=layover_shadow_mask,
        output_raster=geocoded_layover_shadow_mask,
        dem_raster=dem_raster,
        output_mode=GeocodeOutputMode.INTERP,
        # az_time_correction=az_correction,
        # slant_range_correction=srg_correction,
        memory_mode=normalize_geocode_memory_mode(memory_mode),
        min_block_size=min_block_size,
        max_block_size=max_block_size,
        dem_interp_method=normalize_data_interp_method(dem_interp_method),
    )

    return geocoded_layover_shadow_mask


def compute_geocoded_layover_shadow_mask(
    radar_grid: isce3.product.RadarGridParameters,
    orbit: isce3.core.Orbit,
    doppler: isce3.core.LUT2d,
    geo_grid: isce3.product.GeoGridParameters,
    dem_raster: isce3.io.Raster,
    *,
    scratch_dir: os.PathLike | str | None = None,
    dem_interp_method: DataInterpMethod | str = DataInterpMethod.BIQUINTIC,
    lines_per_block: int = 1024,
    rdr2geo_params: Rdr2GeoParamDict | None = None,
    geo2rdr_params: Geo2RdrParamDict | None = None,
    memory_mode: GeocodeMemoryMode | str = GeocodeMemoryMode.Auto,
    min_block_size: int = isce3.core.default_min_block_size,
    max_block_size: int = isce3.core.default_max_block_size,
) -> isce3.io.Raster:
    """ """
    logger = get_logger()

    logger.info("Computing layover/shadow mask in radar coordinates.")
    radar_grid_layover_shadow_mask = compute_layover_shadow_mask(
        radar_grid=radar_grid,
        orbit=orbit,
        doppler=doppler,
        dem_raster=dem_raster,
        scratch_dir=scratch_dir,
        dem_interp_method=dem_interp_method,
        lines_per_block=lines_per_block,
        rdr2geo_params=rdr2geo_params,
    )

    logger.info("Geocoding layover/shadow mask.")
    geo_grid_layover_shadow_mask = geocode_layover_shadow_mask(
        layover_shadow_mask=radar_grid_layover_shadow_mask,
        radar_grid=radar_grid,
        orbit=orbit,
        doppler=doppler,
        geo_grid=geo_grid,
        dem_raster=dem_raster,
        scratch_dir=scratch_dir,
        dem_interp_method=dem_interp_method,
        geo2rdr_params=geo2rdr_params,
        memory_mode=memory_mode,
        min_block_size=min_block_size,
        max_block_size=max_block_size,
    )

    return geo_grid_layover_shadow_mask  # FIXME?


def reproject_raster_file(
    src: os.PathLike | str,
    dst: os.PathLike | str,
    output_geo_grid: isce3.product.GeoGridParameters,
    *,
    algorithm: int | str,
) -> None:
    """
    FIXME

    Parameters
    ----------
    raster_file : str or path-like
        ...
    geo_grid : isce3.product.GeoGridParameters
        ...
    scratch_dir : path-like or None, optional
        ...
    algorithm : {'mode', 'near'}, optional
        ...

    """
    if not (output_geo_grid.spacing_x > 0.0):
        raise ValueError  # FIXME
    if not (output_geo_grid.spacing_y < 0.0):
        raise ValueError  # FIXME

    gdal.Warp(
        os.fsdecode(dst),
        os.fsdecode(src),
        outputBounds=[
            output_geo_grid.start_x,
            output_geo_grid.end_y,
            output_geo_grid.end_x,
            output_geo_grid.start_y,
        ],
        xRes=output_geo_grid.spacing_x,
        yRes=-output_geo_grid.spacing_y,  # Note that `spacing_y` is negated here.
        dstSRS=f"EPSG:{output_geo_grid.epsg}",
        resampleAlg=algorithm,
    )


def binarize_and_reproject_water_mask(
    water_mask: isce3.io.Raster,
    geo_grid: isce3.product.GeoGridParameters,
    *,
    scratch_dir: os.PathLike | str | None = None,
    resample_algorithm: WaterMaskResampleAlgorithm = WaterMaskResampleAlgorithm.NEAR,
):
    """ """
    binary_water_mask_raster_file = make_scratch_file(
        dir_=scratch_dir,
        prefix="binary_water_mask",
        suffix=".tif",
    )
    binary_water_mask = create_single_band_gtiff(
        binary_water_mask_raster_file,
        shape=(water_mask.length, water_mask.width),
        dtype=np.uint8,
    )

    # Convert the water mask to a binary mask.
    transform_blockwise(lambda x: np.not_equal(x, 0), water_mask, binary_water_mask)

    # ...
    binary_water_mask.set_geotransform(water_mask.get_geotransform())
    binary_water_mask.set_epsg(water_mask.get_epsg())

    # Ensure changes are flushed to the dataset and close it.
    binary_water_mask.close_dataset()

    reprojected_binary_water_mask_raster_file = make_scratch_file(
        dir_=scratch_dir,
        prefix="reprojected_binary_water_mask",
        suffix=".tif",
    )

    reproject_raster_file(
        binary_water_mask_raster_file,
        reprojected_binary_water_mask_raster_file,
        output_geo_grid=geo_grid,
        algorithm=resample_algorithm,
    )

    return isce3.io.Raster(os.fsdecode(reprojected_binary_water_mask_raster_file))


# def binarize_water_mask(...) -> ...:
#     """
#
#     NISAR Water Mask Product Specification (JPL D-107710).
#
#     References
#     ----------
#     [1]:
#     """
#     ...
