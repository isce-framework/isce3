from __future__ import annotations

import itertools
import os
from collections.abc import Iterator

import numpy as np
import pytest
import shapely

import isce3
import iscetest
from isce3.geometry import DEMInterpolator
from isce3.geometry.bounding_polygon import wrap_to_interval
from isce3.io import Raster
from isce3.product import GeoGridParameters


def test_wrap_to_interval():
    assert np.isclose(wrap_to_interval(-181.0, lower=-180.0, upper=180.0), 179.0)
    assert np.isclose(wrap_to_interval(2 * np.pi, lower=-np.pi, upper=np.pi), 0.0)
    assert np.isclose(wrap_to_interval(np.pi / 2, lower=-np.pi, upper=np.pi), np.pi / 2)
    assert np.isclose(wrap_to_interval(0.0, lower=1.0, upper=10.0), 9.0)
    assert np.isclose(wrap_to_interval(11.0, lower=1.0, upper=10.0), 2.0)


def get_raster_grid(raster: isce3.io.Raster) -> GeoGridParameters:
    """
    Get the sample coordinate grid of the input raster.

    Parameters
    ----------
    raster : isce3.io.Raster
        The input raster.

    Returns
    -------
    isce3.product.GeoGridParameters
        The coordinate grid that the raster was sampled on.
    """
    return GeoGridParameters(
        start_x=raster.x0,
        start_y=raster.y0,
        spacing_x=raster.dx,
        spacing_y=raster.dy,
        width=raster.width,
        length=raster.length,
        epsg=raster.get_epsg(),
    )


def winnipeg() -> tuple[GeoGridParameters, DEMInterpolator]:
    # A DEM raster in geodetic coordinates (EPSG:4326).
    dem_raster_path = os.path.join(iscetest.data, "winnipeg_dem.tif")
    dem_raster = Raster(dem_raster_path)
    geo_grid = get_raster_grid(dem_raster)
    dem = DEMInterpolator(dem_raster)
    return geo_grid, dem


def antarctica() -> tuple[GeoGridParameters, DEMInterpolator]:
    # A DEM raster in Antarctic Polar Stereographic coordinates (EPSG:3031).
    # Note: it does not contain the pole.
    dem_raster_path = os.path.join(iscetest.data, "dem_south_pole.tif")
    dem_raster = Raster(dem_raster_path)
    geo_grid = get_raster_grid(dem_raster)
    dem = DEMInterpolator(dem_raster)
    return geo_grid, dem


def antimeridian() -> tuple[GeoGridParameters, DEMInterpolator]:
    # A geodetic coordinate grid that spans the antimeridian and the equator.
    geo_grid = GeoGridParameters(
        start_x=179.0,
        start_y=1.0,
        spacing_x=0.01,
        spacing_y=-0.01,
        width=201,
        length=201,
        epsg=4326,
    )
    dem = DEMInterpolator()
    return geo_grid, dem


def north_down() -> tuple[GeoGridParameters, DEMInterpolator]:
    # A UTM coordinate grid with the Y axis flipped relative to the usual
    # north-up/west-left orientation. A counterclockwise ring in pixel coordinates on
    # the grid will be clockwise in map coordinates.
    geo_grid = GeoGridParameters(
        start_x=0.0,
        start_y=0.0,
        spacing_x=10.0,
        spacing_y=-10.0,
        width=1001,
        length=1001,
        epsg=32611,
    )
    geo_grid.spacing_y = 10.0  # Eww, gross!
    dem = DEMInterpolator()
    return geo_grid, dem


def get_xy_coords(geo_grid: GeoGridParameters) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the set of x and y coordinates of the input geocoded grid.

    Parameters
    ----------
    geo_grid : isce3.product.GeoGridParameters
        The input geocoded coordinate grid.

    Returns
    -------
    x_coords, y_coords : numpy.ndarray
        The x and y coordinates of the grid.
    """
    x_start = geo_grid.start_x
    y_start = geo_grid.start_y
    x_spacing = geo_grid.spacing_x
    y_spacing = geo_grid.spacing_y

    x_coords = x_start + (0.5 * x_spacing) + x_spacing * np.arange(geo_grid.width)
    y_coords = y_start + (0.5 * y_spacing) + y_spacing * np.arange(geo_grid.length)

    return x_coords, y_coords


def iter_geo_grid_llh_points(
    geo_grid: GeoGridParameters,
    dem: DEMInterpolator,
    *,
    step: int = 1,
) -> Iterator[shapely.Point]:
    """
    Iterate over points within a geocoded grid in LLH coordinates.

    Parameters
    ----------
    geo_grid : isce3.product.GeoGridParameters
        The input geocoded coordinate grid.
    dem : isce3.geometry.DEMInterpolator
        A DEM spanning the input grid.
    step : int, optional
        The stride between consecutive elements. Defaults to 1.

    Yields
    ------
    shapely.Point
        A point within the grid, in geodetic coordinates (longitude, latitude, height).
        Longitude and latitude coordinates are specified in degrees. Height is in meters
        w.r.t the vertical datum of the input `dem`.
    """
    x_coords, y_coords = get_xy_coords(geo_grid)
    proj = isce3.core.make_projection(geo_grid.epsg)
    xy_points = itertools.product(x_coords, y_coords)
    for x, y in itertools.islice(xy_points, None, None, step):
        height = dem.interpolate_xy(x, y)
        lon_rad, lat_rad, _ = proj.inverse((x, y, 0.0))
        lon_deg, lat_deg = np.rad2deg((lon_rad, lat_rad))
        yield shapely.Point(lon_deg, lat_deg, height)


class TestMakeGeoGridBoundingPolygon:
    @pytest.mark.parametrize(
        "geo_grid,dem", [winnipeg(), antarctica(), antimeridian(), north_down()]
    )
    def test_contains_geo_grid(self, geo_grid: GeoGridParameters, dem: DEMInterpolator):
        ogr_polygon = isce3.geometry.make_geo_grid_bounding_polygon(
            geo_grid,
            dem,
            pts_per_edge=101,
        )
        polygon = shapely.wkt.loads(ogr_polygon.ExportToWkt())

        # Iterate over every 10th point in `geo_grid` in LLH coordinates.
        llh_points = iter_geo_grid_llh_points(geo_grid, dem, step=10)

        assert all(polygon.contains(llh) for llh in llh_points)

    @pytest.mark.parametrize(
        "geo_grid,dem", [winnipeg(), antarctica(), antimeridian(), north_down()]
    )
    def test_counter_clockwise(self, geo_grid: GeoGridParameters, dem: DEMInterpolator):
        ogr_polygon = isce3.geometry.make_geo_grid_bounding_polygon(geo_grid, dem)
        polygon = shapely.wkt.loads(ogr_polygon.ExportToWkt())
        assert shapely.is_ccw(polygon.boundary)

    def test_latitude_overflow(self):
        # A geodetic grid with latitude coordinates extending past -90 degrees. The
        # bounding polygon of such a grid is self-intersecting (see
        # https://github.com/isce-framework/isce3/pull/35#discussion_r2178276599).
        geo_grid = isce3.product.GeoGridParameters(
            start_x=-118.0,
            start_y=-88.0,
            spacing_x=1.0,
            spacing_y=-1.0,
            width=10,
            length=5,
            epsg=4326,
        )
        dem = DEMInterpolator()

        match = (
            "^geo_grid latitude coordinates must be within the interval"
            r" \[-pi/2, pi/2\]"
        )
        with pytest.raises(ValueError, match=match):
            isce3.geometry.make_geo_grid_bounding_polygon(geo_grid, dem)

    @pytest.mark.parametrize("pts_per_edge", [2, 11])
    def test_num_points(self, pts_per_edge: int):
        geo_grid, dem = winnipeg()

        ogr_polygon = isce3.geometry.make_geo_grid_bounding_polygon(
            geo_grid,
            dem,
            pts_per_edge=pts_per_edge,
        )
        polygon = shapely.wkt.loads(ogr_polygon.ExportToWkt())

        n = len(polygon.exterior.coords)
        assert n == (4 * pts_per_edge - 3)

    def test_bad_pts_per_edge(self):
        geo_grid, dem = winnipeg()
        with pytest.raises(ValueError, match="^pts_per_edge = 1, must be >= 2$"):
            isce3.geometry.make_geo_grid_bounding_polygon(
                geo_grid,
                dem,
                pts_per_edge=1,
            )
