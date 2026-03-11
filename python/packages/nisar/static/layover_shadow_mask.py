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
from isce3.geocode import GeocodeOutputMode

from .logging import get_logger
from .util import get_reference_ellipsoid, make_scratch_gtiff


def compute_layover_shadow_mask(
    radar_grid: isce3.product.RadarGridParameters,
    orbit: isce3.core.Orbit,
    img_grid_doppler: isce3.core.LUT2d,
    dem_raster: isce3.io.Raster,
    *,
    scratch_dir: os.PathLike | str | None = None,
    dem_interp_method: DataInterpMethod | str = DataInterpMethod.BIQUINTIC,
    lines_per_block: int = 1024,
    rdr2geo_params: dict[str, Any] | None = None,
) -> isce3.io.Raster:
    """
    Compute a mask of layover and/or shadow pixels on a radar coordinate grid.

    Parameters
    ----------
    radar_grid : isce3.product.RadarGridParameters
        Azimuth time / slant range coordinate grid on which to compute the
        layover/shadow mask.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center over a time interval that spans
        the azimuth time extent of `radar_grid`.
    img_grid_doppler : isce3.core.LUT2d
        The Doppler centroid, in hertz, of the radar grid, expressed as a function of
        azimuth time, in seconds relative to the orbit epoch, and slant range, in
        meters. Note that this should be the Doppler associated with the image grid,
        which may in general be different from the native Doppler of the acquired echo
        data.
    dem_raster : isce3.io.Raster
        Raster of topographic height, in meters above the reference ellipsoid of the
        raster's coordinate reference system, covering a region that spans the footprint
        of the input `radar_grid`.
    scratch_dir : path-like or None, optional
        Directory to store intermediate files and output files created internally by
        this function. If None, a platform-specific default temporary directory will be
        used. Otherwise, it must be the file system path to an existing directory.
        Defaults to None.
    dem_interp_method : isce3.core.DataInterpMethod or str, optional
        Interpolation method used to resample the DEM data. Defaults to biquintic
        interpolation.
    lines_per_block : int, optional
        Maximum block size, in number of radar grid lines, to use in the topo algorithm.
        Smaller block sizes reduce memory utilization, but may increase processing time.
        Defaults to 1024.
    rdr2geo_params : dict or None, optional
        An optional dict of parameters configuring the behavior of the root-finding
        routine used in rdr2geo (Newton-Raphson implementation). The following keys are
        supported:

        'threshold':
          Absolute slant range convergence tolerance, in meters. Defaults to 0.05.

        'maxiter':
          Maximum number of primary Newton-Raphson iterations. Defaults to 25.

        'extraiter':
          Maximum number of secondary iterations. Defaults to 10.

    Returns
    -------
    isce3.io.Raster
        The output layover/shadow mask in radar coordinates. A value of 0 indicates a
        sample not affected by layover or shadow, 1 indicates a sample affected by
        shadow, 2 indicates a sample affected by layover, and 3 indicates a sample
        affected by both layover and shadow.
    """
    if rdr2geo_params is None:
        rdr2geo_params = {}

    layover_shadow_mask = make_scratch_gtiff(
        shape=(radar_grid.length, radar_grid.width),
        dtype=np.uint8,
        dir_=scratch_dir,
        prefix="layover-shadow-mask_",
    )

    rdr2geo = isce3.geometry.Rdr2Geo(
        radar_grid=radar_grid,
        orbit=orbit,
        ellipsoid=get_reference_ellipsoid(dem_raster),
        doppler=img_grid_doppler,
        dem_interp_method=normalize_data_interp_method(dem_interp_method),
        epsg_out=dem_raster.get_epsg(),
        compute_mask=True,
        lines_per_block=lines_per_block,
        **rdr2geo_params,
    )
    rdr2geo.topo(dem_raster=dem_raster, layover_shadow_raster=layover_shadow_mask)

    return layover_shadow_mask


def geocode_layover_shadow_mask(
    layover_shadow_mask: isce3.io.Raster,
    radar_grid: isce3.product.RadarGridParameters,
    orbit: isce3.core.Orbit,
    img_grid_doppler: isce3.core.LUT2d,
    geo_grid: isce3.product.GeoGridParameters,
    dem_raster: isce3.io.Raster,
    *,
    scratch_dir: os.PathLike | str | None = None,
    dem_interp_method: DataInterpMethod | str = DataInterpMethod.BIQUINTIC,
    geo2rdr_params: dict[str, Any] | None = None,
    memory_mode: GeocodeMemoryMode | str = GeocodeMemoryMode.Auto,
    min_block_size: int = isce3.core.default_min_block_size,
    max_block_size: int = isce3.core.default_max_block_size,
) -> isce3.io.Raster:
    """
    Re-project the input layover/shadow mask to a geocoded coordinate grid.

    The mask data is interpolated onto the output grid using nearest neighbor
    interpolation.

    Parameters
    ----------
    layover_shadow_mask : isce3.io.Raster
        The input layover/shadow mask raster.
    radar_grid : isce3.product.RadarGridParameters
        Azimuth time / slant range coordinate grid on which the input layover/shadow
        mask is sampled. Its footprint on the ground should span the extent of
        `geo_grid`.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center over a time interval that
        contains the observation time of each point in `geo_grid`.
    img_grid_doppler : isce3.core.LUT2d
        The Doppler centroid, in hertz, of the radar grid, expressed as a function of
        azimuth time, in seconds relative to the reference epoch of `orbit`, and slant
        range, in meters. Note that this should be the Doppler associated with the image
        grid, which may in general be different from the native Doppler of the acquired
        echo data.
    geo_grid : isce3.product.GeoGridParameters
        The output geocoded coordinate grid to re-project the layover/shadow mask onto.
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

    Returns
    -------
    isce3.io.Raster
        The output layover/shadow mask in geocoded coordinates.
    """
    if geo2rdr_params is None:
        geo2rdr_params = {}

    geocoded_layover_shadow_mask = make_scratch_gtiff(
        shape=(geo_grid.length, geo_grid.width),
        dtype=np.uint8,
        dir_=scratch_dir,
        prefix="geocoded-layover-shadow-mask_",
    )

    # XXX: There's currently no way to create a "Geocode" object that operates directly
    # on uint8 data. Instead, the way this is done in the InSAR geocoding workflow is to
    # create a `GeocodeFloat32` object, resulting in conversion of the input mask values
    # to float32 during geocoding (and then conversion back to uint8 when they're
    # written to the output raster). This shouldn't cause any loss of fidelity since the
    # mask values are small and we're using nearest neighbor interpolation.
    geocode = isce3.geocode.GeocodeFloat32()
    geocode.orbit = orbit
    geocode.ellipsoid = get_reference_ellipsoid(dem_raster)
    geocode.doppler = img_grid_doppler
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

    geocode.geocode(
        radar_grid=radar_grid,
        input_raster=layover_shadow_mask,
        output_raster=geocoded_layover_shadow_mask,
        dem_raster=dem_raster,
        output_mode=GeocodeOutputMode.INTERP,
        memory_mode=normalize_geocode_memory_mode(memory_mode),
        min_block_size=min_block_size,
        max_block_size=max_block_size,
        dem_interp_method=normalize_data_interp_method(dem_interp_method),
    )

    return geocoded_layover_shadow_mask


def compute_geocoded_layover_shadow_mask(
    radar_grid: isce3.product.RadarGridParameters,
    orbit: isce3.core.Orbit,
    img_grid_doppler: isce3.core.LUT2d,
    geo_grid: isce3.product.GeoGridParameters,
    dem_raster: isce3.io.Raster,
    *,
    scratch_dir: os.PathLike | str | None = None,
    dem_interp_method: DataInterpMethod | str = DataInterpMethod.BIQUINTIC,
    lines_per_block: int = 1024,
    rdr2geo_params: dict[str, Any] | None = None,
    geo2rdr_params: dict[str, Any] | None = None,
    memory_mode: GeocodeMemoryMode | str = GeocodeMemoryMode.Auto,
    min_block_size: int = isce3.core.default_min_block_size,
    max_block_size: int = isce3.core.default_max_block_size,
) -> isce3.io.Raster:
    """
    Compute a mask of layover and/or shadow pixels on a geocoded coordinate grid.

    The mask is first computed in radar (azimuth time / slant range) coordinates, and
    then re-projected onto a geocoded coordinate grid.

    Parameters
    ----------
    radar_grid : isce3.product.RadarGridParameters
        Azimuth time / slant range coordinate grid on which to compute the initial
        layover/shadow mask prior to geocoding. Its footprint on the ground should span
        the extent of `geo_grid`.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center over a time interval that spans
        the azimuth time extent of `radar_grid`.
    img_grid_doppler : isce3.core.LUT2d
        The Doppler centroid, in hertz, of the radar grid, expressed as a function of
        azimuth time, in seconds relative to the reference epoch of `orbit`, and slant
        range, in meters. Note that this should be the Doppler associated with the image
        grid, which may in general be different from the native Doppler of the acquired
        echo data.
    geo_grid : isce3.product.GeoGridParameters
        The geocoded coordinate grid on which to compute the output layover/shadow mask.
    dem_raster : isce3.io.Raster
        Raster of topographic height, in meters above the reference ellipsoid of the
        raster's coordinate reference system, covering a region that spans the footprint
        of the input `radar_grid`. Need not be in the same coordinate reference system
        as `geo_grid`.
    scratch_dir : path-like or None, optional
        Directory to store intermediate files and output files created internally by
        this function. If None, a platform-specific default temporary directory will be
        used. Otherwise, it must be the file system path to an existing directory.
        Defaults to None.
    dem_interp_method : isce3.core.DataInterpMethod or str, optional
        Interpolation method used to resample the DEM data. Defaults to biquintic
        interpolation.
    lines_per_block : int, optional
        Maximum block size, in number of radar grid lines, to use in the topo algorithm.
        Smaller block sizes reduce memory utilization, but may increase processing time.
        Defaults to 1024.
    rdr2geo_params : dict or None, optional
        An optional dict of parameters configuring the behavior of the root-finding
        routine used in rdr2geo (Newton-Raphson implementation). The following keys are
        supported:

        'threshold':
          Absolute slant range convergence tolerance, in meters. Defaults to 0.05.

        'maxiter':
          Maximum number of primary Newton-Raphson iterations. Defaults to 25.

        'extraiter':
          Maximum number of secondary iterations. Defaults to 10.
    geo2rdr_params : dict or None, optional
        An optional dict of parameters configuring the behavior of the root-finding
        routine used in geo2rdr (Newton-Raphson implementation). The following keys are
        supported:

        'threshold':
          Absolute azimuth time convergence tolerance, in seconds. Defaults to 1e-8.

        'maxiter':
          Maximum number of Newton-Raphson iterations. Defaults to 50.
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

    Returns
    -------
    isce3.io.Raster
        The output layover/shadow mask in geocoded coordinates. A value of 0 indicates a
        sample not affected by layover or shadow, 1 indicates a sample affected by
        shadow, 2 indicates a sample affected by layover, and 3 indicates a sample
        affected by both layover and shadow.
    """
    logger = get_logger()

    logger.info("Computing layover/shadow mask in radar coordinates.")
    radar_grid_layover_shadow_mask = compute_layover_shadow_mask(
        radar_grid=radar_grid,
        orbit=orbit,
        img_grid_doppler=img_grid_doppler,
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
        img_grid_doppler=img_grid_doppler,
        geo_grid=geo_grid,
        dem_raster=dem_raster,
        scratch_dir=scratch_dir,
        dem_interp_method=dem_interp_method,
        geo2rdr_params=geo2rdr_params,
        memory_mode=memory_mode,
        min_block_size=min_block_size,
        max_block_size=max_block_size,
    )

    return geo_grid_layover_shadow_mask
