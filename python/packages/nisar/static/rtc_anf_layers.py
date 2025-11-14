from __future__ import annotations

import os
from typing import Any

import numpy as np

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

from .util import get_reference_ellipsoid, make_scratch_gtiff


def compute_rtc_anf_layers(
    radar_grid: isce3.product.RadarGridParameters,
    orbit: isce3.core.Orbit,
    native_doppler: isce3.core.LUT2d,
    img_grid_doppler: isce3.core.LUT2d,
    geo_grid: isce3.product.GeoGridParameters,
    dem_raster: isce3.io.Raster,
    *,
    scratch_dir: os.PathLike | str | None = None,
    dem_interp_method: DataInterpMethod | str = DataInterpMethod.BIQUINTIC,
    geo2rdr_params: dict[str, Any] | None = None,
    output_mode: GeocodeOutputMode | str = GeocodeOutputMode.AREA_PROJECTION,
    interp_method: DataInterpMethod | str = DataInterpMethod.BIQUINTIC,
    memory_mode: GeocodeMemoryMode | str = GeocodeMemoryMode.Auto,
    min_block_size: int = isce3.core.default_min_block_size,
    max_block_size: int = isce3.core.default_max_block_size,
    min_area_factor: float = np.nan,
    dem_upsample_factor: float = 2.0,
    algorithm: RtcAlgorithm | str = RtcAlgorithm.RTC_AREA_PROJECTION,
    area_beta_mode: RtcAreaBetaMode | str | None = RtcAreaBetaMode.AUTO,
) -> tuple[isce3.io.Raster, isce3.io.Raster]:
    """
    Compute area normalization factors (ANF) for radiometric terrain correction (RTC).

    Parameters
    ----------
    radar_grid : isce3.product.RadarGridParameters
        Azimuth time / slant range coordinate grid on which to compute the initial
        layover/shadow mask prior to geocoding. Its footprint on the ground should span
        the extent of `geo_grid`.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center over a time interval that spans
        the azimuth time extent of `radar_grid`.
    native_doppler : isce3.core.LUT2d
        The Doppler centroid of the radar signal data, in hertz, expressed as a function
        of azimuth time, in seconds relative to the reference epoch of `orbit`, and
        slant range, in meters. Note that this should be the native Doppler of the data
        acquisition, which may in general be different than the Doppler associated with
        the radar grid that the focused image was projected onto.
    img_grid_doppler : isce3.core.LUT2d
        The Doppler centroid, in hertz, of the radar grid, expressed as a function of
        azimuth time, in seconds relative to the reference epoch of `orbit`, and slant
        range, in meters. Note that this should be the Doppler associated with the image
        grid, which may in general be different from the native Doppler of the acquired
        echo data.
    geo_grid : isce3.product.GeoGridParameters
        The geocoded coordinate grid on which to compute the output RTC ANF layers.
    dem_raster : isce3.io.Raster
        A DEM raster spanning the output geocoded coordinate grid. Need not be in the
        same coordinate reference system as `geo_grid`.
    scratch_dir : path-like or None, optional
        Directory to store intermediate files and output files created internally by
        this function. If None, a platform-specific default temporary directory will be
        used. Otherwise, it must be the file system path to an existing directory.
        Defaults to None.
    dem_interp_method : isce3.core.DataInterpMethod or str, optional
        Interpolation method used to resample the DEM data. Defaults to biquintic
        interpolation.
    geo2rdr_params : dict or None, optional
        An optional dict of parameters configuring the behavior of the root-finding
        routine used in geo2rdr (Newton-Raphson implementation). The following keys are
        supported:

        'threshold':
          Absolute azimuth time convergence tolerance, in seconds. Defaults to 1e-8.

        'maxiter':
          Maximum number of Newton-Raphson iterations. Defaults to 50.
    output_mode : isce3.geocode.GeocodeOutputMode or str, optional
        Algorithm used for geocoding. Defaults to the area projection algorithm.
    interp_method : isce3.core.DataInterpMethod or str, optional
        Interpolation method used for geocoding. Only relevant when `output_mode` is set
        to interpolation. Ignored otherwise. Defaults to biquintic interpolation.
    memory_mode : isce3.core.GeocodeMemoryMode, optional
        Block processing mode to use for geocoding. The default value is internally
        defined.
    min_block_size : int, optional
        Minimum block size for geocoding, per thread, in bytes. The block size is chosen
        dynamically based on the image dimensions to partition work uniformly across
        available threads, bounded by `min_block_size` and `max_block_size`. The number
        of threads may be controlled by the `OMP_NUM_THREADS` environment variable. Must
        be <= `max_block_size`. Defaults to 33,554,432 (32 MiB).
    max_block_size : int, optional
        Maximum block size for geocoding, per thread, in bytes. The block size is chosen
        dynamically based on the image dimensions to partition work uniformly across
        available threads, bounded by `min_block_size` and `max_block_size`. The number
        of threads may be controlled by the `OMP_NUM_THREADS` environment variable. Must
        be >= `min_block_size`. Defaults to 268,435,456 (256 MB).
    min_area_factor : float, optional
        Minimum pixel area threshold for RTC, in dB. If set to NaN, no minimum is
        enforced, which may lead to division by very small values and result in infinite
        or inaccurate outputs. Defaults to NaN.
    dem_upsample_factor : float, optional
        DEM upsampling factor used for RTC. Defaults to 2.
    algorithm : isce3.geometry.RtcAlgorithm or str, optional
        Algorithm used for RTC. Defaults to the area projection algorithm.
    area_beta_mode : isce3.geometry.RtcAreaBetaMode or str or None, optional
        Method for estimating the area of the beta naught reference surface. The default
        value is internally defined.

    Returns
    -------
    gamma0_to_beta0_factor : isce3.io.Raster
        RTC scaling factor to normalize backscatter coefficients from gamma0 to sigma0,
        accounting for local terrain.
    gamma0_to_sigma0_factor : isce3.io.Raster
        RTC scaling factor to normalize backscatter coefficients from gamma0 to beta0,
        accounting for local terrain.
    """
    if geo2rdr_params is None:
        geo2rdr_params = {}

    geocode = isce3.geocode.GeocodeFloat32()
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

    # XXX: The input SLC and output geocoded covariance rasters are required arguments
    # but aren't used in the computation of any of the output layers of interest here,
    # so simply pass in dummy rasters.
    dummy_slc = make_scratch_gtiff(
        shape=(radar_grid.length, radar_grid.width),
        dtype=np.float32,
        dir_=scratch_dir,
        prefix="dummy-slc_",
    )
    dummy_geocoded_cov = make_scratch_gtiff(
        shape=(geo_grid.length, geo_grid.width),
        dtype=np.float32,
        dir_=scratch_dir,
        prefix="dummy-geocoded-cov_",
    )

    gamma0_to_beta0_factor = make_scratch_gtiff(
        shape=(geo_grid.length, geo_grid.width),
        dtype=np.float32,
        dir_=scratch_dir,
        prefix="gamma0-to-beta0-factor_",
    )

    gamma0_to_sigma0_factor = make_scratch_gtiff(
        shape=(geo_grid.length, geo_grid.width),
        dtype=np.float32,
        dir_=scratch_dir,
        prefix="gamma0-to-sigma0-factor_",
    )

    # XXX: Pass in dummy output RTC rasters to avoid GeocodeCov creating these
    # internally as in-memory rasters, which could cause the process to run
    # out-of-memory.
    dummy_output_rtc = make_scratch_gtiff(
        shape=(radar_grid.length, radar_grid.width),
        dtype=np.float32,
        dir_=scratch_dir,
        prefix="dummy-output-rtc_",
    )
    dummy_output_rtc_sigma = make_scratch_gtiff(
        shape=(radar_grid.length, radar_grid.width),
        dtype=np.float32,
        dir_=scratch_dir,
        prefix="dummy-output-rtc-sigma_",
    )

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
        out_geo_rtc=gamma0_to_beta0_factor,
        rtc_area_beta_mode=normalize_rtc_area_beta_mode(area_beta_mode),
        out_geo_rtc_gamma0_to_sigma0=gamma0_to_sigma0_factor,
        output_rtc=dummy_output_rtc,
        output_rtc_sigma=dummy_output_rtc_sigma,
        memory_mode=normalize_geocode_memory_mode(memory_mode),
        min_block_size=min_block_size,
        max_block_size=max_block_size,
        dem_interp_method=normalize_data_interp_method(dem_interp_method),
    )

    return gamma0_to_beta0_factor, gamma0_to_sigma0_factor
