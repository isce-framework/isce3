from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import NamedTuple

import numpy as np

import isce3
from isce3.core import normalize_look_side


class Rectangle(NamedTuple):
    """A rectangle in (x,y) coordinate space, defined by its extents."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float


def get_bounding_rectangle(pts: Iterable[tuple[float, float]]) -> Rectangle:
    """
    Get the extents of the smallest bounding box that contains a set of (x,y) points.

    Parameters
    ----------
    pts : iterable of (float, float)
        An iterable of one or more (x,y) coordinates. Must not contain NaN or (positive
        or negative) infinity values.

    Returns
    -------
    Rectangle
        The lower and upper x- and y-coordinate extents of the bounding box.

    Raises
    ------
    ValueError
        If the set of points is empty or if any points have non-finite coordinate
        values.
    """
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf

    for x, y in pts:
        if np.isnan(x) or np.isnan(y):
            raise ValueError("input points must not contain NaN values")

        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)

    if any(np.isinf(coord) for coord in [xmin, xmax, ymin, ymax]):
        raise ValueError("bounding box has one or more unbounded extents")

    return Rectangle(xmin, xmax, ymin, ymax)


def get_radar_grid_containing_az_rg_pts(
    az_rg_pts: Iterable[tuple[float, float]],
    *,
    az_spacing: float,
    rg_spacing: float,
    look_side: isce3.core.LookSide | str,
    wavelength: float,
    ref_epoch: isce3.core.DateTime,
    az_margin: float = 0.0,
    rg_margin: float = 0.0,
) -> isce3.product.RadarGridParameters:
    """
    Get a radar grid that spans a set of (azimuth,range) points.

    Parameters
    ----------
    az_rg_pts : iterable of (float, float)
        An iterable of one or more (azimuth,range) points with finite values that must
        be contained within the output radar grid.
    az_spacing : float
        Azimuth time spacing of the output grid, in seconds. Must be > 0.
    rg_spacing : float
        Slant range spacing of the output grid, in meters. Must be > 0.
    look_side : isce3.core.LookSide or {'left', 'right'}
        The look direction of the output grid (left-looking or right-looking).
    wavelength : float
        The radar central wavelength, in meters.
    ref_epoch : isce3.core.DateTime
        The reference epoch of the output grid. Azimuth time coordinates of
        points within the grid are interpreted as relative to this datetime.
    az_margin : float, optional
        Optional additional azimuth margin to add to the start and end of the
        radar grid, in seconds. Must be >= 0. Defaults to 0.
    rg_margin : float, optional
        Optional additional slant range margin to add to the near and far edges
        of the radar grid, in meters. Must be >= 0. Defaults to 0.

    Returns
    -------
    isce3.product.RadarGridParameters
        The output radar grid.

    See Also
    --------
    get_radar_grid_containing_geo_pts
    """
    if not (az_spacing > 0.0):
        raise ValueError(f"{az_spacing=}, must be > 0")
    if not (rg_spacing > 0.0):
        raise ValueError(f"{rg_spacing=}, must be > 0")

    if not (az_margin >= 0.0):
        raise ValueError(f"{az_margin=}, must be >= 0")
    if not (rg_margin >= 0.0):
        raise ValueError(f"{rg_margin=}, must be >= 0")

    # Compute the extents of the bounding box containing all of the points.
    az_min, az_max, rg_min, rg_max = get_bounding_rectangle(az_rg_pts)

    assert az_min <= az_max
    assert rg_min <= rg_max

    # Apply any additional margin to the azimuth and range extents.
    az_min -= az_margin
    az_max += az_margin
    rg_min -= rg_margin
    rg_max += rg_margin

    # Divide `num` by `den`, rounded up to the next smallest integer.
    def ceil_divide(num: float, den: float) -> int:
        return int(np.ceil(num / den))

    # Compute the number of azimuth and range samples needed to span the peak-to-peak
    # range of the points with the desired sample spacing, plus one additional sample.
    #
    # NOTE: According to ISCE3 conventions, radar grid pixel coordinates are referenced
    # to the location of the center of the pixel, so there's a half-pixel offset between
    # the coordinates of the boundary pixels and the outer extents of the radar grid
    # (see https://github-fn.jpl.nasa.gov/isce-3/isce/issues/584#issuecomment-11771.)
    # This distinction between 'pixels' and 'points' is not well-documented for users,
    # though, and sometimes causes confusion. We choose to ignore the half-pixel shift
    # here and simply add an extra row & column to the radar grid to compensate. This
    # expansion of the radar grid is unlikely to cause harm and ensures that all points
    # are contained within the radar grid regardless of the grid coordinate convention,
    # while also ensuring that the radar grid dimensions are always >= 1.
    num_az = ceil_divide(az_max - az_min, az_spacing) + 1
    num_rg = ceil_divide(rg_max - rg_min, rg_spacing) + 1

    # Construct the radar grid.
    radar_grid = isce3.product.RadarGridParameters(
        sensing_start=az_min,
        wavelength=wavelength,
        prf=1.0 / az_spacing,
        starting_range=rg_min,
        range_pixel_spacing=rg_spacing,
        lookside=normalize_look_side(look_side),
        length=num_az,
        width=num_rg,
        ref_epoch=ref_epoch,
    )

    assert radar_grid.contains(az_min, rg_min)
    assert radar_grid.contains(az_max, rg_max)

    return radar_grid


def get_radar_grid_containing_geo_pts(
    geo_pts: Iterable[tuple[float, float, float]],
    proj: isce3.core.ProjectionBase,
    *,
    az_spacing: float,
    rg_spacing: float,
    orbit: isce3.core.Orbit,
    look_side: isce3.core.LookSide | str,
    wavelength: float,
    doppler: isce3.core.LUT2d = isce3.core.LUT2d(),
    az_margin: float = 0.0,
    rg_margin: float = 0.0,
    geo2rdr_params: Mapping[str, float] | None = None,
) -> isce3.product.RadarGridParameters:
    """
    Get a radar grid that spans a set of geocoded points.

    The reference epoch of the output radar grid will be the same as the input `orbit`.

    Parameters
    ----------
    geo_pts : iterable of (float, float, float)
        An iterable of one or more points that must be contained within the output radar
        grid. Points should be specified as 3-vectors or 3-tuples of coordinates such as
        (longitude,latitude,height) or (easting,northing,height). All points should be
        in a single common coordinate system specified by the `proj` argument.
    proj : isce3.core.ProjectionBase
        A 'projection' object representing the transformation between
        longitude/latitude/height (LLH) coordinates and the coordinate system of
        `geo_pts`.
    az_spacing : float
        Azimuth time spacing of the output grid, in seconds. Must be > 0.
    rg_spacing : float
        Slant range spacing of the output grid, in meters. Must be > 0.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center over a time interval that
        includes the observation times of each point in `geo_pts`.
    look_side : isce3.core.LookSide or {'left', 'right'}
        The look direction of the output grid (left-looking or right-looking).
    wavelength : float
        The radar central wavelength, in meters.
    doppler : isce3.core.LUT2d, optional
        The Doppler centroid, in hertz, of the output grid, expressed as a function of
        azimuth time, in seconds relative to the epoch of `orbit`, and slant range, in
        meters. Defaults to zero-Doppler.
    az_margin : float, optional
        Optional additional azimuth margin to add to the start and end of the
        radar grid, in seconds. Must be >= 0. Defaults to 0.
    rg_margin : float, optional
        Optional additional slant range margin to add to the near and far edges
        of the radar grid, in meters. Must be >= 0. Defaults to 0.
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
    isce3.product.RadarGridParameters
        The output radar grid.

    See Also
    --------
    get_radar_grid_containing_az_rg_pts
    """
    if geo2rdr_params is None:
        geo2rdr_params = {}

    def _geo2rdr(geo_pt: tuple[float, float, float]) -> tuple[float, float]:
        # Convert from projected coordinates -> LLH -> ECEF -> (azimuth,range).
        llh = proj.inverse(geo_pt)
        xyz = proj.ellipsoid.lon_lat_to_xyz(llh)
        return isce3.geometry.geo2rdr_bracket(
            xyz=xyz,
            orbit=orbit,
            doppler=doppler,
            wavelength=wavelength,
            side=look_side,
            **geo2rdr_params,
        )

    # Lazily convert each item in `geo_pts` from `proj` coordinates to radar
    # coordinates.
    az_rg_pts = map(_geo2rdr, geo_pts)

    return get_radar_grid_containing_az_rg_pts(
        az_rg_pts=az_rg_pts,
        az_spacing=az_spacing,
        rg_spacing=rg_spacing,
        look_side=look_side,
        wavelength=wavelength,
        ref_epoch=orbit.reference_epoch,
        az_margin=az_margin,
        rg_margin=rg_margin,
    )


def get_bounding_radar_grid(
    geo_grid: isce3.product.GeoGridParameters,
    *,
    az_spacing: float,
    rg_spacing: float,
    orbit: isce3.core.Orbit,
    look_side: isce3.core.LookSide | str,
    wavelength: float,
    doppler: isce3.core.LUT2d = isce3.core.LUT2d(),
    min_height: float = isce3.core.GLOBAL_MIN_HEIGHT,
    max_height: float = isce3.core.GLOBAL_MAX_HEIGHT,
    pts_per_edge: int = 11,
    az_margin: float = 0.0,
    rg_margin: float = 0.0,
    geo2rdr_params: Mapping[str, float] | None = None,
):
    """
    Get a radar grid that spans an input geocoded grid over a specified height range.

    The output radar grid is constructed such that it contains points sampled along each
    of the four edges of `geo_grid` at both the specified minimum and maximum height
    bounds.

    Parameters
    ----------
    geo_grid : isce3.product.GeoGridParameters
        The input geocoded coordinate grid.
    az_spacing : float
        Azimuth time spacing of the output grid, in seconds. Must be > 0.
    rg_spacing : float
        Slant range spacing of the output grid, in meters. Must be > 0.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center over a time interval that
        includes the observation times of each point in `geo_grid` at each height
        between `min_height` and `max_height`.
    look_side : isce3.core.LookSide or {'left', 'right'}
        The look direction of the output grid (left-looking or right-looking).
    wavelength : float
        The radar central wavelength, in meters.
    doppler : isce3.core.LUT2d
        The Doppler centroid, in hertz, of the output grid, expressed as a function of
        azimuth time, in seconds relative to the epoch of `orbit`, and slant range, in
        meters. Defaults to zero-Doppler.
    min_height : float, optional
        A lower bound on the height of targets within the input `geo_grid`, in meters
        above the reference ellipsoid associated with the spatial reference system of
        `geo_grid`. Must be <= `max_height`. Defaults to
        ``isce3.core.GLOBAL_MIN_HEIGHT``.
    max_height : float, optional
        An upper bound on the height of targets within the input `geo_grid`, in meters
        above the reference ellipsoid associated with the spatial reference system of
        `geo_grid`. Must be >= `min_height`. Defaults to
        ``isce3.core.GLOBAL_MAX_HEIGHT``.
    pts_per_edge : int, optional
        The number of perimeter points to sample along each edge of the input `geo_grid`
        at each height bound. Must be >= 2. Defaults to 11.
    az_margin : float, optional
        Optional additional azimuth margin to add to the start and end of the
        radar grid, in seconds. Must be >= 0. Defaults to 0.
    rg_margin : float, optional
        Optional additional slant range margin to add to the near and far edges
        of the radar grid, in meters. Must be >= 0. Defaults to 0.
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
    isce3.product.RadarGridParameters
        The output radar grid.

    See Also
    --------
    get_geo_perimeter_wkt
    """
    if not (max_height >= min_height):
        raise ValueError(
            f"max_height must be >= min_height, got {max_height=} and {min_height=}"
        )

    if pts_per_edge < 2:
        raise ValueError(f"{pts_per_edge=}, must be >= 2")

    # NOTE: geo2rdr is not an affine transformation, so the set of points within
    # `geo_grid` is, in general, not a convex set when transformed to (azimuth,range)
    # coordinate space. Therefore, it's not sufficient to just compute the smallest
    # radar grid that contains the four corners of `geo_grid` -- we need to ensure that
    # the entire perimeter of `geo_grid` is contained within the output radar grid.
    # As a concession to computational feasibility, we sample a finite number of points
    # along each edge of `geo_grid` and choose the output radar grid such that it
    # contains those points.
    #
    # Get a generator that yields points sampled uniformly along each of the four edges
    # of `geo_grid` at both the lower and upper height bounds. The total number of
    # points is 8n-8, where n is the number of points per edge.
    def boundary_pts() -> Iterator[tuple[float, float, float]]:
        # Get x & y coordinates uniformly sampled along the extents of the `geo_grid`.
        xcoords = np.linspace(geo_grid.start_x, geo_grid.end_x, num=pts_per_edge)
        ycoords = np.linspace(geo_grid.start_y, geo_grid.end_y, num=pts_per_edge)

        for z in [min_height, max_height]:
            # Yield points sampled along the top and bottom edges.
            for y in [geo_grid.start_y, geo_grid.end_y]:
                for x in xcoords:
                    yield x, y, z

            # Yield points sampled along the left and right edges (excluding the four
            # corners).
            for x in [geo_grid.start_x, geo_grid.end_x]:
                for y in ycoords[1:-1]:
                    yield x, y, z

    # Get an `isce3.core.ProjectionBase` object that represents the spatial reference
    # system of `geo_grid`, where the first two coordinates of the projected coordinate
    # system are the `geo_grid` x & y coordinates, and the third coordinate is assumed
    # to be height above ellipsoid, in meters.
    proj = isce3.core.make_projection(geo_grid.epsg)

    return get_radar_grid_containing_geo_pts(
        geo_pts=boundary_pts(),
        proj=proj,
        az_spacing=az_spacing,
        rg_spacing=rg_spacing,
        orbit=orbit,
        look_side=look_side,
        wavelength=wavelength,
        doppler=doppler,
        az_margin=az_margin,
        rg_margin=rg_margin,
        geo2rdr_params=geo2rdr_params,
    )
