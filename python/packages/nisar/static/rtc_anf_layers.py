from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np

from .util import transform_blockwise

import isce3
from isce3.core import (
    DataInterpMethod,
    GeocodeMemoryMode,
    normalize_data_interp_method,
    normalize_geocode_memory_mode,
)
from isce3.geocode import GeocodeOutputMode, normalize_geocode_output_mode
from isce3.geometry import (
    RtcAlgorithm,
    RtcAreaBetaMode,
    RtcInputTerrainRadiometry,
    RtcOutputTerrainRadiometry,
    normalize_rtc_algorithm,
    normalize_rtc_area_beta_mode,
)
from .typing import Geo2RdrParamDict
from .util import get_reference_ellipsoid, make_scratch_gtiff


class StaticRTCLayers(NamedTuple):
    """ """
    gamma0_to_beta0_factor: isce3.io.Raster
    gamma0_to_sigma0_factor: isce3.io.Raster


def compute_rtc_anf_layers(
    radar_grid: isce3.product.RadarGridParameters,
    orbit: isce3.core.Orbit,
    native_doppler: isce3.core.LUT2d,
    img_grid_doppler: isce3.core.LUT2d,
    dem_raster: isce3.io.Raster,
    geo_grid: isce3.product.GeoGridParameters,
    *,
    scratch_dir: os.PathLike | str | None = None,
    dem_interp_method: DataInterpMethod | str = DataInterpMethod.BIQUINTIC,
    geo2rdr_params: Geo2RdrParamDict | None = None,
    output_mode: GeocodeOutputMode | str = GeocodeOutputMode.AREA_PROJECTION,
    interp_method: DataInterpMethod | str = DataInterpMethod.BIQUINTIC,
    memory_mode: GeocodeMemoryMode | str = GeocodeMemoryMode.Auto,
    min_block_size: int = isce3.core.default_min_block_size,
    max_block_size: int = isce3.core.default_max_block_size,
    min_area_factor: float = np.nan,
    dem_upsample_factor: float = 2.0,
    algorithm: RtcAlgorithm | str = RtcAlgorithm.RTC_AREA_PROJECTION,
    area_beta_mode: RtcAreaBetaMode | str | None = RtcAreaBetaMode.AUTO,
) -> StaticRTCLayers:
    """ """
    if geo2rdr_params is None:
        geo2rdr_params = {}

    geocode = isce3.geocode.GeocodeCFloat32()
    geocode.orbit = orbit
    geocode.doppler = img_grid_doppler
    geocode.native_doppler = native_doppler
    geocode.ellipsoid = get_reference_ellipsoid(dem_raster)
    geocode.data_interpolator = normalize_data_interp_method(interp_method)
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

    dummy_slc = make_scratch_gtiff(
        shape=(radar_grid.length, radar_grid.width),
        dtype=np.complex64,
        dir_=scratch_dir,
        prefix="dummy_slc",
    )

    dummy_geocoded_cov = make_scratch_gtiff(
        shape=(geo_grid.length, geo_grid.width),
        dtype=np.float32,
        dir_=scratch_dir,
        prefix="dummy_geocoded_cov",
    )

    beta0_to_gamma0_factor = make_scratch_gtiff(
        shape=(geo_grid.length, geo_grid.width),
        dtype=np.float32,
        dir_=scratch_dir,
        prefix="beta0_to_gamma0_factor",
    )

    gamma0_to_sigma0_factor = make_scratch_gtiff(
        shape=(geo_grid.length, geo_grid.width),
        dtype=np.float32,
        dir_=scratch_dir,
        prefix="gamma0_to_sigma0_factor",
    )

    # Pass a dummy output RTC raster to avoid GeocodeCov creating this internally as an
    # in-memory raster, which could cause the process to run out-of-memory.
    dummy_output_rtc = make_scratch_gtiff(
        shape=(radar_grid.length, radar_grid.width),
        dtype=np.float32,
        dir_=scratch_dir,
        prefix="dummy_output_rtc",
    )

    # XXX: Pass in input layover/shadow mask?
    geocode.geocode(
        radar_grid=radar_grid,
        input_raster=dummy_slc,
        output_raster=dummy_geocoded_cov,
        dem_raster=dem_raster,
        output_mode=normalize_geocode_output_mode(output_mode),
        flag_apply_rtc=True,
        input_terrain_radiometry=RtcInputTerrainRadiometry.BETA_NAUGHT,
        output_terrain_radiometry=RtcOutputTerrainRadiometry.GAMMA_NAUGHT,
        rtc_min_value_db=min_area_factor,
        rtc_upsampling=dem_upsample_factor,
        rtc_algorithm=normalize_rtc_algorithm(algorithm),
        out_geo_rtc=beta0_to_gamma0_factor,
        rtc_area_beta_mode=normalize_rtc_area_beta_mode(area_beta_mode),
        out_geo_rtc_gamma0_to_sigma0=gamma0_to_sigma0_factor,
        # az_time_correction=az_correction,
        # slant_range_correction=srg_correction,
        output_rtc=dummy_output_rtc,
        memory_mode=normalize_geocode_memory_mode(memory_mode),
        min_block_size=min_block_size,
        max_block_size=max_block_size,
        dem_interp_method=normalize_data_interp_method(dem_interp_method),
    )

    gamma0_to_beta0_factor = make_scratch_gtiff(
        shape=(geo_grid.length, geo_grid.width),
        dtype=np.float32,
        dir_=scratch_dir,
        prefix="gamma0_to_beta0_factor",
    )

    transform_blockwise(np.reciprocal, beta0_to_gamma0_factor, gamma0_to_beta0_factor)

    return StaticRTCLayers(
        gamma0_to_beta0_factor=gamma0_to_beta0_factor,
        gamma0_to_sigma0_factor=gamma0_to_sigma0_factor,
    )
