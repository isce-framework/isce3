from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from osgeo import ogr

import isce3
from isce3.geometry import DEMInterpolator


# TODO:
#  - Check that each point is within the polygon
#  - Check that polygon has clockwise orientation
#  - Check the number of points
#  - Check that lon/lat is in degrees
#  - Check that an exception is thrown if `pts_per_edge` < 2


def make_geo_grid_bounding_polygon(
    geo_grid: isce3.product.GeoGridParameters,
    dem: DEMInterpolator = DEMInterpolator(),
    *,
    pts_per_edge: int = 11,
) -> ogr.Geometry:
    """
    Get the perimeter of a geocoded grid as an LLH polygon.

    Construct a polygon whose corners consist of (longitude, latitude, height) points
    sampled along the perimeter of `geo_grid`. The polygon has counterclockwise
    orientation (in the native coordinates of `geo_grid`), starting from the upper-left
    corner of the grid.

    Geodetic longtiude and latitude are measured in degrees; height is measured in
    meters above the reference ellipsoid of the digital elevation model (DEM).

    The polygon is closed (i.e. the first and last point in the polygon perimeter ring
    are the same).

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
    ogr.Geometry
        A polygon bounding the input geocoded grid, in LLH coordinates.

    See Also
    --------
    get_geo_perimeter_wkt
    """
    if pts_per_edge < 2:
        raise ValueError(f"pts_per_edge must be >= 2, got {pts_per_edge=}")

    # Get a 'projection' object that represents the native spatial reference system of
    # `geo_grid`.
    proj = isce3.core.make_projection(geo_grid.epsg)

    # NOTE: The transformation from `proj` to LLH coordinates is not necessarily an
    # affine transformation, so the set of points within `geo_grid` is, in general, not
    # a convex set in LLH coordinate space. Therefore, it's not sufficient to just
    # use the four corners of `geo_grid` as the corners of the bounding polygon -- we
    # need to ensure that the entire perimeter of `geo_grid` is contained within the
    # polygon. As a concession to computational feasibility, we sample a finite number
    # of points along each edge of `geo_grid` to form the perimeter of the output
    # polygon.
    #
    # Yield (x, y) points sampled uniformly along the perimeter of `geo_grid` in native
    # coordinates, in counterclockwise order, starting from the upper-left corner.
    def boundary_pts() -> Iterator[tuple[float, float]]:
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

    # Create a ring of LLH points sampled along the perimeter of `geo_grid`.
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in boundary_pts():
        lon_rad, lat_rad, _ = proj.inverse([x, y, 0.0])
        height = dem.interpolate_lonlat(lon_rad, lat_rad)
        lon_deg, lat_deg = np.rad2deg([lon_rad, lat_rad])
        ring.AddPoint(lon_deg, lat_deg, height)

    # Ensure the ring is closed by appending the first point to the end of the list of
    # points in the ring.
    ring.CloseRings()

    # Create the output polygon object.
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)

    return polygon
