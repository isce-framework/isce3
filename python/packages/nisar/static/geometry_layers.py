from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np

import isce3
from isce3.core import DataInterpMethod, normalize_data_interp_method
from .typing import Geo2RdrParamDict
from .util import make_scratch_gtiff


class StaticGeometryLayers(NamedTuple):
    """ """
    height_above_ellipsoid: isce3.io.Raster
    line_of_sight_x: isce3.io.Raster
    line_of_sight_y: isce3.io.Raster
    local_incidence_angle: isce3.io.Raster

"""
 Parameters
 ----------
 lookside : isce3.core.LookSide
     Look side
 wavelength : double
     Wavelength
 dem_raster : isce3.io.Raster
     DEM raster
 geogrid : isce3.product.GeoGridParameters
     Output layers geogrid
 orbit : isce3.core.Orbit
     Orbit
 native_doppler : isce3.core.LUT2d
     Native image Doppler
 grid_doppler : isce3.core.LUT2d
     Grid Doppler
 dem_interp_method :  isce3::core::dataInterpMethod
     DEM interpolation method (default: biquintic)
 geo2rdr_params : double, optional
     Geo2rdr parameters
 interpolated_dem_raster : isce3.io.Raster, optional
     Interpolated DEM raster
 slant_range_raster : isce3.io.Raster, optional
     Slant-range (in meters) cube raster
 azimuth_time_raster : isce3.io.Raster, optional
     Azimuth time (in seconds relative to orbit epoch) cube raster
 incidence_angle_raster : isce3.io.Raster, optional
     Incidence angle (in degrees wrt ellipsoid normal at target)
     cube raster
 los_unit_vector_x_raster : isce3.io.Raster, optional
     LOS (target-to-sensor) unit vector X cube raster
 los_unit_vector_y_raster : isce3.io.Raster, optional
     LOS (target-to-sensor) unit vector Y cube raster
 along_track_unit_vector_x_raster : isce3.io.Raster, optional
     Along-track unit vector X raster
 along_track_unit_vector_y_raster : isce3.io.Raster, optional
     Along-track unit vector Y raster
 elevation_angle_raster : isce3.io.Raster, optional
     Elevation angle (in degrees wrt geodedic nadir) cube raster
 ground_track_velocity_raster : isce3.io.Raster, optional
     Ground-track velocity raster
 local_incidence_angle_raster : isce3.io.Raster, optional
     Local-incidence angle raster
 projection_angle_raster : isce3.io.Raster, optional
     Projection angle raster
 simulated_radar_brightness_raster : isce3.io.Raster, optional
     Simulated radar brightness raster
"""

def compute_geometry_layers(
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
) -> StaticGeometryLayers:
    """ """
    dem_interp_method = normalize_data_interp_method(dem_interp_method)

    if geo2rdr_params is None:
        geo2rdr_params = {}
    geo2rdr_params = isce3.geometry.Geo2RdrParams(**geo2rdr_params)

    def make_output_raster(prefix: str) -> isce3.io.Raster:
        return make_scratch_gtiff(
            shape=(geo_grid.length, geo_grid.width),
            dtype=np.float32,
            dir_=scratch_dir,
            prefix=prefix,
        )

    height_above_ellipsoid = make_output_raster("height_above_ellipsoid")
    line_of_sight_x = make_output_raster("line_of_sight_x")
    line_of_sight_y = make_output_raster("line_of_sight_y")
    local_incidence_angle = make_output_raster("local_incidence_angle")

    isce3.geogrid.get_radar_grid(
        lookside=radar_grid.lookside,
        wavelength=radar_grid.wavelength,
        dem_raster=dem_raster,
        geogrid=geo_grid,
        orbit=orbit,
        native_doppler=native_doppler,
        grid_doppler=img_grid_doppler,
        dem_interp_method=dem_interp_method,
        geo2rdr_params=geo2rdr_params,
        interpolated_dem_raster=height_above_ellipsoid,
        los_unit_vector_x_raster=line_of_sight_x,
        los_unit_vector_y_raster=line_of_sight_y,
        local_incidence_angle_raster=local_incidence_angle,
    )

    # XXX FIXME TODO: Are the LOS unit vectors in ENU coordinates or projected coordinates?
    return StaticGeometryLayers(
        height_above_ellipsoid=height_above_ellipsoid,
        line_of_sight_x=line_of_sight_x,
        line_of_sight_y=line_of_sight_y,
        local_incidence_angle=local_incidence_angle,
    )
