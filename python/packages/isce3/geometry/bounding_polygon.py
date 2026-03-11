from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import shapely
from numpy.typing import ArrayLike
from osgeo import ogr

import isce3
from isce3.geometry import DEMInterpolator

Point2D = tuple[float, float]
Point3D = tuple[float, float, float]


def wrap_to_interval(val: float, lower: float, upper: float) -> float:
    """
    Wrap a value to the range [lower, upper) using modular arithmetic.

    Parameters
    ----------
    val : float
        Value to wrap.
    lower : float
        Lower endpoint of the interval (inclusive).
    upper : float
        Upper endpoint of the interval (exclusive). Must be > `lower`.

    Returns
    -------
    float
        Wrapped value in the range [lower, upper).
    """
    if not (upper > lower):
        raise ValueError(f"{upper=} must be greater than {lower=}")

    # Python's `%` (modulo) operator computes the remainder with the same sign as the
    # right-hand side operand, so the result of the modulo operation below is in the
    # range [0, b-a), where b is the upper endpoint and a is the lower endpoint. Then,
    # if we add a, the result will be in the range [a, b). See
    # https://docs.python.org/3/reference/expressions.html#binary-arithmetic-operations.
    return (val - lower) % (upper - lower) + lower


def unwrap_degrees(angles: ArrayLike) -> np.ndarray:
    """
    Unwrap a sequence of angles in degrees.

    Unwraps the input angles such that the absolute differences between adjacent
    elements are never greater than 180 degrees by adding a multiple of 360 degrees to
    each element.

    Parameters
    ----------
    angles : array_like
        The input sequence of angles, in degrees.

    Returns
    -------
    numpy.ndarray
        The unwrapped angles, in degrees.
    """
    # `numpy.unwrap` doesn't correctly unwrap inputs in degrees prior to NumPy 1.21.0.
    # See https://github.com/numpy/numpy/pull/16987.
    # ISCE3 currently supports NumPy >=1.20.
    return np.rad2deg(np.unwrap(np.deg2rad(angles)))


def unwrap_longitudes(llhs: Iterable[Point3D]) -> Iterator[Point3D]:
    """
    Unwrap longitude coordinates of a sequence of LLH points.

    Parameters
    ----------
    llhs : iterable of (float, float, float)
        An ordered series of (longitude, latitude, height) triplets, with longitudes
        specified in degrees.

    Returns
    -------
    iterator of (float, float, float)
        An iterator over the input sequence with longitude values unwrapped. The order
        of the points is preserved.
    """
    lons, lats, heights = map(list, zip(*llhs))
    return zip(unwrap_degrees(lons), lats, heights)


def create_ogr_polygon(
    exterior_vertices: Iterable[Point2D] | Iterable[Point3D],
) -> ogr.Geometry:
    """
    Create a polygon with the specified exterior vertices as an OGR Geometry object.

    Parameters
    ----------
    exterior_vertices : iterable of (float, float) or iterable of (float, float, float)
        The vertices of the exterior boundary of the polygon. An ordered series of 2D or
        3D points. The final point should not be identical to the first -- the boundary
        ring will be automatically closed.

    Returns
    -------
    osgeo.ogr.Geometry
        The output polygon.
    """
    # Create the exterior ring of the polygon.
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for vertex in exterior_vertices:
        ring.AddPoint(*vertex)

    # Ensure the ring is closed by appending the first vertex to the end of the list of
    # points in the ring.
    ring.CloseRings()

    # Create the output polygon object.
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)

    return polygon


def make_geo_grid_bounding_polygon(
    geo_grid: isce3.product.GeoGridParameters,
    dem: DEMInterpolator = DEMInterpolator(),
    *,
    pts_per_edge: int = 11,
) -> ogr.Geometry:
    """
    Get the perimeter of a geocoded grid as an LLH polygon.

    Construct a polygon whose corners consist of (longitude, latitude, height) points
    sampled along the perimeter of `geo_grid`.

    The polygon is convex with counterclockwise winding order on the ellipsoid. The
    first boundary vertex is the upper-left corner of the grid. The polygon is closed
    (i.e. the first and last point in the boundary are the same).

    In the resulting polygon, longitude and latitude coordinates are in degrees and
    height is in meters above the reference ellipsoid of the input digital elevation
    model (DEM). Longitude values are in the range [-180, 180], except that they may
    extend outside this range in the case where `geo_grid` spans the antimeridian.
    Latitude values are in the range [-90, 90].

    Parameters
    ----------
    geo_grid : isce3.product.GeoGridParameters
        The input geocoded coordinate grid.
    dem : isce3.geometry.DEMInterpolator, optional
        A DEM spanning the input grid. Need not be in the same coordinate reference
        system as `geo_grid`. Defaults to a zero-height DEM w.r.t the WGS 84 ellipsoid.
    pts_per_edge : int, optional
        The number of perimeter points to sample along each edge of the input
        `geo_grid`. Must be >= 2. Defaults to 11.

    Returns
    -------
    osgeo.ogr.Geometry
        A polygon bounding the input geocoded grid, in geodetic coordinates.

    See Also
    --------
    get_dem_boundary_polygon
    get_geo_perimeter_wkt
    """
    if pts_per_edge < 2:
        raise ValueError(f"{pts_per_edge = }, must be >= 2")

    # Get a 'projection' object that represents the native spatial reference system of
    # `geo_grid`.
    proj = isce3.core.make_projection(geo_grid.epsg)

    # Convert an (x, y) point in projected coordinates to an LLH point on the DEM
    # surface.
    def xy_to_llh(x: float, y: float) -> Point3D:
        # Transform from projected coordinates to geodetic coordinates.
        # XXX: `ProjectionBase.inverse()` doesn't necessarily guarantee that the
        # resulting longitude is in [-pi, pi] or that the resulting latitude is in
        # [-pi/2, pi/2].
        lon_rad, lat_rad, _ = proj.inverse((x, y, 0.0))
        lon_rad = wrap_to_interval(lon_rad, lower=-np.pi, upper=np.pi)
        if not (-0.5 * np.pi <= lat_rad <= 0.5 * np.pi):
            raise ValueError(
                "geo_grid latitude coordinates must be within the interval"
                f" [-pi/2, pi/2], got latitude={lat_rad} (radians)"
            )

        # Interpolate the DEM height. In general, the DEM grid may be in a different
        # coordinate system than the input (x, y) points, so use `interpolate_lonlat()`,
        # which internally converts (lon, lat) points to the native DEM coordinates,
        # instead of `interpolate_xy()`.
        height = dem.interpolate_lonlat(lon_rad, lat_rad)

        # Convert to degrees.
        lon_deg, lat_deg = np.rad2deg((lon_rad, lat_rad))

        return lon_deg, lat_deg, height

    # The transformation from projected coordinates to LLH coordinates is not affine in
    # general, so it's not sufficient to just use the four corners of `geo_grid` as the
    # corners of the bounding polygon -- we need to ensure that the entire perimeter of
    # `geo_grid` is contained within the polygon. As a concession to computational
    # feasibility, we'll sample a finite number of points along each edge of `geo_grid`
    # to form the perimeter of the output polygon.

    # Yield (x, y) points sampled uniformly along the perimeter of `geo_grid` in native
    # coordinates, in counterclockwise order, starting from the upper-left corner.
    def boundary_pts() -> Iterator[Point2D]:
        # Get x & y coordinates uniformly sampled along the extents of the `geo_grid`.
        xcoords = np.linspace(geo_grid.start_x, geo_grid.end_x, num=pts_per_edge)
        ycoords = np.linspace(geo_grid.start_y, geo_grid.end_y, num=pts_per_edge)

        # Left edge from top to bottom (including both endpoints).
        x0 = xcoords[0]
        for y in ycoords:
            yield x0, y

        # Bottom edge from left to right (excluding the left endpoint).
        y1 = ycoords[-1]
        for x in xcoords[1:]:
            yield x, y1

        # Right edge from bottom to top (excluding the bottom endpoint).
        x1 = xcoords[-1]
        for y in ycoords[:-1][::-1]:
            yield x1, y

        # Top edge from right to left (excluding both endpoints).
        y0 = ycoords[0]
        for x in xcoords[1:-1][::-1]:
            yield x, y0

    # Get points sampled along the perimeter of `geo_grid` in LLH coordinates. Unwrap
    # longitude values to handle antimeridian crossings.
    perimeter_points_llh = (xy_to_llh(x, y) for (x, y) in boundary_pts())
    perimeter_points_llh = list(unwrap_longitudes(perimeter_points_llh))

    # Construct a `LinearRing` from the perimeter points so we can ensure that the
    # output polygon is simple (i.e. not self-intersecting) and has the correct winding
    # order.
    ring = shapely.LinearRing(perimeter_points_llh)

    # Check that the exterior of the polygon is a valid LinearRing (that is, it does not
    # cross or touch itself). This could happen, for example, if `geo_grid` extends
    # beyond +/-90 degrees latitude, which might result in a self-intersecting shape on
    # the ellipsoid (see
    # https://github.com/isce-framework/isce3/pull/35#discussion_r2178276599).
    if not ring.is_valid:
        raise RuntimeError("polygon boundary is invalid")

    # Ensure that the polygon has counterclockwise (CCW) winding order on the ellipsoid.
    # Reverse the order of points along the exterior boundary if necessary. This could
    # happen, for example, if the geo grid orientation is not north-up/west-left, or if
    # the projection doesn't preserve the north-south ordering and east-west ordering of
    # the points.
    if not ring.is_ccw:
        perimeter_points_llh = perimeter_points_llh[::-1]
        assert shapely.LinearRing(perimeter_points_llh).is_ccw

    return create_ogr_polygon(perimeter_points_llh)
