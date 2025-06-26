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
from isce3.io import Raster
from isce3.product import GeoGridParameters


def south_pole_dem() -> Raster:
    dem_raster_path = os.path.join(iscetest.data, "dem_south_pole.tif")
    return Raster(dem_raster_path)


def winnipeg_dem() -> Raster:
    dem_raster_path = os.path.join(iscetest.data, "winnipeg_dem.tif")
    return Raster(dem_raster_path)


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
    @pytest.mark.parametrize("dem_raster", [south_pole_dem(), winnipeg_dem()])
    def test_contains_geo_grid(self, dem_raster: Raster):
        geo_grid = get_raster_grid(dem_raster)
        dem = DEMInterpolator(dem_raster)

        ogr_polygon = isce3.geometry.make_geo_grid_bounding_polygon(
            geo_grid,
            dem,
            pts_per_edge=101,
        )
        polygon = shapely.wkt.loads(ogr_polygon.ExportToWkt())

        # Iterate over every 10th point in `geo_grid` in LLH coordinates.
        llh_points = iter_geo_grid_llh_points(geo_grid, dem, step=10)

        assert all(polygon.contains(llh) for llh in llh_points)

    @pytest.mark.parametrize("dem_raster", [south_pole_dem(), winnipeg_dem()])
    def test_counter_clockwise(self, dem_raster: Raster):
        geo_grid = get_raster_grid(dem_raster)
        dem = DEMInterpolator(dem_raster)

        ogr_polygon = isce3.geometry.make_geo_grid_bounding_polygon(geo_grid, dem)
        polygon = shapely.wkt.loads(ogr_polygon.ExportToWkt())

        assert shapely.is_ccw(polygon.boundary)

    @pytest.mark.parametrize("pts_per_edge", [2, 11])
    def test_num_points(self, pts_per_edge: int):
        dem_raster = south_pole_dem()
        geo_grid = get_raster_grid(dem_raster)
        dem = DEMInterpolator(dem_raster)

        ogr_polygon = isce3.geometry.make_geo_grid_bounding_polygon(
            geo_grid,
            dem,
            pts_per_edge=pts_per_edge,
        )
        polygon = shapely.wkt.loads(ogr_polygon.ExportToWkt())

        n = len(polygon.exterior.coords)
        assert n == (4 * pts_per_edge - 3)

    def test_bad_pts_per_edge(self):
        dem_raster = south_pole_dem()
        geo_grid = get_raster_grid(dem_raster)
        dem = DEMInterpolator(dem_raster)

        with pytest.raises(ValueError, match="^pts_per_edge = 1, must be >= 2$"):
            isce3.geometry.make_geo_grid_bounding_polygon(
                geo_grid,
                dem,
                pts_per_edge=1,
            )
