from __future__ import annotations

import itertools

import numpy as np

import isce3
from isce3.core import normalize_look_side


def infer_radar_grid_spacing_from_geo_grid(
    geo_grid: isce3.product.GeoGridParameters,
    dem: isce3.geometry.DEMInterpolator,
    orbit: isce3.core.Orbit,
    doppler: isce3.core.LUT2d,
    look_side: isce3.core.LookSide | str,
    wavelength: float,
    *,
    pts_per_side: int = 5,
    geo2rdr_params: dict | None = None,
) -> tuple[float, float]:
    """
    Estimate the max radar grid spacing necessary to avoid undersampling the `geo_grid`.

    Parameters
    ----------
    geo_grid : isce3.product.GeoGridParameters
        The input geocoded coordinate grid.
    dem : isce3.geometry.DEMInterpolator
        A digital elevation model (DEM) spanning the input grid. Need not be in the same
        coordinate reference system as `geo_grid`.
    orbit : isce3.core.Orbit
        The flight path of the radar antenna phase center over a time interval that
        contains the observation time of each point in `geo_grid`.
    doppler : isce3.core.LUT2d
        The Doppler centroid, in hertz, of the radar grid, expressed as a function of
        azimuth time, in seconds relative to the reference epoch of `orbit`, and slant
        range, in meters. Note that this should be the Doppler associated with the image
        grid, which may in general be different from the native Doppler of the acquired
        echo data.
    look_side : isce3.core.LookSide or {'left', 'right'}
        The look direction of the sensor (left-looking or right-looking).
    wavelength : float
        The radar central wavelength, in meters.
    pts_per_side : int, optional
        Side length of the NxN grid of samples used to estimate the required radar grid
        pixel spacing necessary to avoid undersampling the geocoded grid. Must be >= 2.
        Defaults to 5.
    geo2rdr_params : dict or None, optional
        An optional dict of parameters configuring the behavior of the root-finding
        routine used in geo2rdr (bracketing implementation). The following keys are
        supported:

        'tol_aztime':
          Azimuth time convergence tolerance, in seconds. Defaults to 1e-7.

        'time_start':
          Start of search interval, in seconds. Defaults to ``orbit.start_time``.

        'time_end':
          End of search interval, in seconds. Defaults to ``orbit.end_time``.

    Returns
    -------
    az_spacing : float
        The maximum required azimuth time pixel spacing, in seconds.
    rg_spacing : float
        The maximum required slant range pixel spacing, in meters.
    """
    look_side = normalize_look_side(look_side)

    if geo2rdr_params is None:
        geo2rdr_params = {}

    # Get an `isce3.core.ProjectionBase` object that represents the native spatial
    # reference system of `geo_grid`.
    proj = isce3.core.make_projection(geo_grid.epsg)

    # Given an (x, y) point in projected coordinates, get the height of the DEM at that
    # point and convert the point from projected coordinates -> LLH -> ECEF -> radar
    # coordinates (azimuth, range).
    def xy_to_rdr(x: float, y: float) -> tuple[float, float]:
        lon, lat, _ = proj.inverse((x, y, 0.0))
        height = dem.interpolate_lonlat(lon, lat)
        xyz = proj.ellipsoid.lon_lat_to_xyz((lon, lat, height))
        return isce3.geometry.geo2rdr_bracket(
            xyz=xyz,
            orbit=orbit,
            doppler=doppler,
            wavelength=wavelength,
            side=look_side,
            **geo2rdr_params,
        )

    dx = geo_grid.spacing_x
    dy = geo_grid.spacing_y

    # Compute the radar grid spacing necessary to avoid undersampling a geo grid pixel
    # as follows:
    #
    # 1. Project the four corners of the pixel centered on the point (x, y) into radar
    #    coordinates.
    # 2. Compute the bounding box of the resulting quadrilateral in radar coordinates.
    #    (NOTE: the transformation from geo -> rdr is not affine in general, but we
    #    assume that the pixel perimeter is (approximately) a quadrilateral in radar
    #    coordinates.)
    # 3. Divide the bounding box into quadrants and choose the radar grid spacing to be
    #    the dimensions of each quadrant.
    def az_rg_spacing_at_xy_point(x: float, y: float) -> tuple[float, float]:
        # Get the x & y coordinates of the four corners of the geo grid pixel.
        x0 = x - 0.5 * dx
        x1 = x + 0.5 * dx
        y0 = y - 0.5 * dy
        y1 = y + 0.5 * dy

        # Get the azimuth & slant range extents of the geo grid pixel.
        az_min, az_max = np.inf, -np.inf
        rg_min, rg_max = np.inf, -np.inf
        for xx, yy in itertools.product((x0, x1), (y0, y1)):
            az, rg = xy_to_rdr(xx, yy)
            az_min = min(az_min, az)
            az_max = max(az_max, az)
            rg_min = min(rg_min, rg)
            rg_max = max(rg_max, rg)

        # Choose the spacing to be half of the peak-to-peak extent of the bounding box.
        d_az = (az_max - az_min) / 2.0
        d_rg = (rg_max - rg_min) / 2.0

        return d_az, d_rg

    # Construct a uniformly-spaced NxN grid of points that spans the input geocoded
    # grid, where N is `pts_per_side`.
    x_coords = np.linspace(geo_grid.start_x, geo_grid.end_x, num=pts_per_side)
    y_coords = np.linspace(geo_grid.start_y, geo_grid.end_y, num=pts_per_side)

    # At each point in the coarse NxN grid, compute the largest azimuth and range
    # spacing that avoids undersampling the geocoded grid at that point. Find the
    # minimum azimuth and range spacing values from among all such points.
    min_az_spacing = np.inf
    min_rg_spacing = np.inf
    for x in x_coords:
        for y in y_coords:
            az_spacing, rg_spacing = az_rg_spacing_at_xy_point(x, y)
            min_az_spacing = min(min_az_spacing, az_spacing)
            min_rg_spacing = min(min_rg_spacing, rg_spacing)

    return min_az_spacing, min_rg_spacing
