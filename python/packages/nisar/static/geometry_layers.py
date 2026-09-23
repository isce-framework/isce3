from __future__ import annotations

import os
from typing import Any

import numpy as np

import isce3
from isce3.core import (
    DataInterpMethod,
    normalize_data_interp_method,
    normalize_look_side,
)
from isce3.io import Raster

from .util import make_scratch_gtiff


def compute_geometry_layers(
    geo_grid: isce3.product.GeoGridParameters,
    dem_raster: Raster,
    orbit: isce3.core.Orbit,
    native_doppler: isce3.core.LUT2d,
    look_side: isce3.core.LookSide | str,
    wavelength: float,
    *,
    scratch_dir: os.PathLike | str | None = None,
    dem_interp_method: DataInterpMethod | str = DataInterpMethod.BIQUINTIC,
    geo2rdr_params: dict[str, Any] | None = None,
) -> tuple[Raster, Raster, Raster, Raster]:
    """
    Compute DEM, LOS unit vector, and LIA on a geocoded coordinate grid.

    Compute static geometry layers, including the re-projected digital elevation model
    (DEM), target-to-platform line-of-sight (LOS) unit vector, and local incidence angle
    (LIA) on the `geo_grid`.

    Parameters
    ----------
    geo_grid : isce3.product.GeoGridParameters
        The geocoded coordinate grid on which to compute each of the output layers.
    dem_raster : isce3.io.Raster
        A DEM raster spanning the output geocoded coordinate grid. Need not be in the
        same coordinate reference system as `geo_grid`.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center over a time interval that
        contains the observation time of each point in `geo_grid`.
    native_doppler : isce3.core.LUT2d
        The Doppler centroid of the radar signal data, in hertz, expressed as a function
        of azimuth time, in seconds relative to the reference epoch of `orbit`, and
        slant range, in meters. Note that this should be the native Doppler of the data
        acquisition, which may in general be different than the Doppler associated with
        the radar grid that the focused image was projected onto.
    look_side : isce3.core.LookSide or {'left', 'right'}
        The look direction of the sensor (left-looking or right-looking).
    wavelength : float
        The radar central wavelength, in meters.
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

        'delta_range':
          Step size for computing the numerical partial derivative of Doppler w.r.t
          range, in meters. Defaults to 10.

    Returns
    -------
    reprojected_dem : isce3.io.Raster
        The height of each point in `geo_grid` above the reference ellipsoid of
        `dem_raster` (not necessarily the same as the vertical datum of `geo_grid`), in
        the same units as `dem_raster`, obtained by re-projecting and resampling the
        input DEM data to the output coordinate grid.
    los_east, los_north : isce3.io.Raster
        East and north components of the target-to-platform line-of-sight unit vector at
        each point in `geo_grid`, at the time when the Doppler centroid of the beam
        crossed the target location, in an East-North-Up (ENU) coordinate system with
        its origin at the target location, oriented such that the Up component is normal
        to the surface of the reference ellipsoid.
    local_inc_angle : isce3.io.Raster
        The local incidence angle, defined as the angle between the target-to-platform
        line-of-sight vector and the topographic normal vector, in degrees, at each
        point in `geo_grid`.
    """
    look_side = normalize_look_side(look_side)
    dem_interp_method = normalize_data_interp_method(dem_interp_method)

    if geo2rdr_params is None:
        geo2rdr_params = {}
    geo2rdr_params = isce3.geometry.Geo2RdrParams(**geo2rdr_params)

    def make_output_raster(prefix: str) -> Raster:
        return make_scratch_gtiff(
            shape=(geo_grid.length, geo_grid.width),
            dtype=np.float32,
            dir_=scratch_dir,
            prefix=prefix,
        )

    reprojected_dem = make_output_raster("reprojected-dem_")
    los_east = make_output_raster("los-east_")
    los_north = make_output_raster("los-north_")
    local_inc_angle = make_output_raster("local-inc-angle_")

    # XXX: `grid_doppler` is a required argument, but isn't used in the computation of
    # any of the output layers of interest here, so just pass a dummy value.
    isce3.geogrid.get_radar_grid(
        lookside=look_side,
        wavelength=wavelength,
        dem_raster=dem_raster,
        geogrid=geo_grid,
        orbit=orbit,
        native_doppler=native_doppler,
        grid_doppler=isce3.core.LUT2d(),
        dem_interp_method=dem_interp_method,
        geo2rdr_params=geo2rdr_params,
        interpolated_dem_raster=reprojected_dem,
        los_unit_vector_x_raster=los_east,
        los_unit_vector_y_raster=los_north,
        local_incidence_angle_raster=local_inc_angle,
    )

    return reprojected_dem, los_east, los_north, local_inc_angle
