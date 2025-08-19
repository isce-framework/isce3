from __future__ import annotations

import itertools
import os
from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np
import pytest

import isce3
import iscetest
import nisar
from isce3.geometry.bounding_radar_grid import (
    get_bounding_rectangle,
    get_radar_grid_containing_az_rg_pts,
    get_radar_grid_containing_geo_pts,
)


class TestGetBoundingRectangle:
    def test_one_point(self):
        pts = [(0.0, 1.0)]
        bbox = get_bounding_rectangle(pts)
        assert bbox.xmin == 0.0
        assert bbox.ymin == 1.0
        assert bbox.xmax == 0.0
        assert bbox.ymax == 1.0

    def test_two_points(self):
        pts = [(0.0, 1.0), (2.0, 3.0)]
        bbox = get_bounding_rectangle(pts)
        assert bbox.xmin == 0.0
        assert bbox.ymin == 1.0
        assert bbox.xmax == 2.0
        assert bbox.ymax == 3.0

    def test_random_points(self):
        rng = np.random.default_rng(seed=1234)
        xcoords = rng.random(size=100)
        ycoords = rng.random(size=100)
        bbox = get_bounding_rectangle(zip(xcoords, ycoords))
        assert bbox.xmin == np.min(xcoords)
        assert bbox.ymin == np.min(ycoords)
        assert bbox.xmax == np.max(xcoords)
        assert bbox.ymax == np.max(ycoords)

    def test_empty(self):
        regex = "^bounding box has one or more unbounded extents$"
        with pytest.raises(ValueError, match=regex):
            get_bounding_rectangle([])

    def test_nan(self):
        pts = [(0.0, 1.0), (2.0, 3.0), (4.0, np.nan)]
        regex = "^input points must not contain NaN values$"
        with pytest.raises(ValueError, match=regex):
            get_bounding_rectangle(pts)

    @pytest.mark.parametrize("val", [np.inf, -np.inf])
    def test_inf(self, val: float):
        pts = [(0.0, 1.0), (2.0, 3.0), (4.0, val)]
        regex = "^bounding box has one or more unbounded extents$"
        with pytest.raises(ValueError, match=regex):
            get_bounding_rectangle(pts)


class TestGetRadarGridContainingAzRgPts:
    @pytest.fixture
    def params(self) -> dict[str, Any]:
        # Some keyword parameters to be used in the below test cases.
        return dict(
            az_rg_pts=[(0.0, 800e3), (10.0, 900e3)],
            wavelength=0.1,
            az_spacing=1.0,
            rg_spacing=10.0,
            look_side=isce3.core.LookSide.Left,
            ref_epoch=isce3.core.DateTime("2000-01-01T00:00:00"),
        )

    def test_no_margin(self, params: Mapping[str, Any]):
        radar_grid = get_radar_grid_containing_az_rg_pts(**params)

        # Check that each point is contained within the radar grid.
        assert all(radar_grid.contains(*pt) for pt in params["az_rg_pts"])

        # Check points just outside each edge of the radar grid.
        assert not radar_grid.contains(-1.0, radar_grid.mid_range)
        assert not radar_grid.contains(11.0, radar_grid.mid_range)
        assert not radar_grid.contains(radar_grid.sensing_mid, 799_990.0)
        assert not radar_grid.contains(radar_grid.sensing_mid, 900_010.0)

        # Check other radar grid metadata.
        assert np.isclose(radar_grid.prf, 1.0 / params["az_spacing"])
        assert radar_grid.range_pixel_spacing == params["rg_spacing"]
        assert radar_grid.lookside == params["look_side"]
        assert radar_grid.wavelength == params["wavelength"]
        assert radar_grid.ref_epoch == params["ref_epoch"]

    def test_margin(self, params: Mapping[str, Any]):
        radar_grid = get_radar_grid_containing_az_rg_pts(
            **params,
            az_margin=1.0,
            rg_margin=100.0,
        )

        # For each of the four boundary edges of the radar grid, check whether or not
        # the grid contained a point that should be slightly *inside* the boundary and a
        # point that should be slightly *outside* the boundary, considering the extra
        # azimuth/range margin:

        # lower azimuth bound
        assert radar_grid.contains(-1.0, radar_grid.mid_range)
        assert not radar_grid.contains(-2.0, radar_grid.mid_range)

        # upper azimuth bound
        assert radar_grid.contains(11.0, radar_grid.mid_range)
        assert not radar_grid.contains(12.0, radar_grid.mid_range)

        # lower range bound
        assert radar_grid.contains(radar_grid.sensing_mid, 799_900.0)
        assert not radar_grid.contains(radar_grid.sensing_mid, 799_890.0)

        # upper range bound
        assert radar_grid.contains(radar_grid.sensing_mid, 900_100.0)
        assert not radar_grid.contains(radar_grid.sensing_mid, 900_110.0)

    @pytest.mark.parametrize("az_spacing", [-1.0, 0.0, np.nan])
    def test_bad_spacing(self, params: MutableMapping[str, Any], az_spacing: float):
        del params["az_spacing"]
        with pytest.raises(ValueError, match=f"^{az_spacing=}, must be > 0$"):
            get_radar_grid_containing_az_rg_pts(**params, az_spacing=az_spacing)

    @pytest.mark.parametrize("az_margin", [-1.0, np.nan])
    def test_bad_margin(self, params: Mapping[str, Any], az_margin: float):
        with pytest.raises(ValueError, match=f"^{az_margin=}, must be >= 0$"):
            get_radar_grid_containing_az_rg_pts(**params, az_margin=az_margin)


@pytest.fixture
def winnipeg_rslc() -> nisar.products.readers.RSLC:
    rslc_hdf5 = os.path.join(iscetest.data, "winnipeg.h5")
    return nisar.products.readers.RSLC(hdf5file=rslc_hdf5)


@pytest.fixture
def winnipeg_dem() -> isce3.geometry.DEMInterpolator:
    dem_tiff = os.path.join(iscetest.data, "winnipeg_dem.tif")
    dem_raster = isce3.io.Raster(dem_tiff)
    return isce3.geometry.DEMInterpolator(dem_raster)


def compare_radar_grids(
    grid1: isce3.product.RadarGridParameters,
    grid2: isce3.product.RadarGridParameters,
    *,
    az_tol: float = 1e-3,
    rg_tol: float = 1e-3,
) -> None:
    r"""
    Check that two radar grids are approximately equal.

    Use assertions to check that the two input radar grids have approximately the same
    azimuth bounds, to within :math:`\pm` `az_tol`, and have approximately the same
    range bounds, to within :math:`\pm` `rg_tol`. All other radar grid metadata must be
    identical.

    Parameters
    ----------
    grid1, grid2 : isce3.product.RadarGridParameters
        The input radar grids.
    az_tol : float, optional
        Absolute tolerance for comparing the azimuth extents of the radar grids, in
        seconds. Defaults to 1e-3.
    rg_tol : float, optional
        Absolute tolerance for comparing the range extents of the radar grids, in
        meters. Defaults to 1e-3.
    """
    assert np.isclose(grid1.sensing_start, grid2.sensing_start, rtol=0.0, atol=az_tol)
    assert np.isclose(grid1.sensing_stop, grid2.sensing_stop, rtol=0.0, atol=az_tol)
    assert np.isclose(grid1.starting_range, grid2.starting_range, rtol=0.0, atol=rg_tol)
    assert np.isclose(grid1.end_range, grid2.end_range, rtol=0.0, atol=rg_tol)
    assert grid1.prf == grid2.prf
    assert grid1.range_pixel_spacing == grid2.range_pixel_spacing
    assert grid1.wavelength == grid2.wavelength
    assert grid1.lookside == grid2.lookside
    assert grid1.ref_epoch == grid2.ref_epoch


# EPSG:32614 is the code for UTM zone 14N, which contains the city of Winnipeg.
@pytest.mark.parametrize("epsg", [4326, 32614])
def test_get_radar_grid_containing_geo_pts(
    winnipeg_rslc: nisar.products.readers.RSLC,
    winnipeg_dem: isce3.geometry.DEMInterpolator,
    epsg: int,
):
    # This test starts with an existing radar grid, geocodes points along the
    # perimeter of the grid, uses `get_radar_grid_containing_geo_pts()` to get a new
    # radar grid containing those points, and checks that the new grid approximately
    # matches the original radar grid.

    # Get the radar grid and orbit metadata from the input RSLC product.
    radar_grid = winnipeg_rslc.getRadarGrid(frequency="A")
    orbit = winnipeg_rslc.getOrbit()

    # Get the radar grid coordinate spacing.
    az_spacing = 1.0 / radar_grid.prf
    rg_spacing = radar_grid.range_pixel_spacing

    # Get four points along the edges of the radar grid.
    az_rg_pts = [
        (radar_grid.sensing_start, radar_grid.mid_range),
        (radar_grid.sensing_stop, radar_grid.mid_range),
        (radar_grid.sensing_mid, radar_grid.starting_range),
        (radar_grid.sensing_mid, radar_grid.end_range),
    ]

    # Get the 'projection' object associated with the input EPSG code.
    proj = isce3.core.make_projection(epsg)

    def _rdr2geo(az: float, rg: float) -> tuple[float, float, float]:
        # Convert from radar coordinates -> ECEF -> LLH -> projected coordinates.
        xyz = isce3.geometry.rdr2geo_bracket(
            aztime=az,
            slant_range=rg,
            orbit=orbit,
            side=radar_grid.lookside,
            doppler=0.0,
            wavelength=radar_grid.wavelength,
            dem=winnipeg_dem,
        )
        llh = proj.ellipsoid.xyz_to_lon_lat(xyz)
        return proj.forward(llh)

    # Lazily transform the radar grid boundary points from (azimuth,range) coordinates
    # to projected coordinates.
    geo_pts = itertools.starmap(_rdr2geo, az_rg_pts)

    # Get the output radar grid containing the geocoded points (with a small margin
    # to ensure that the original radar grid is strictly contained within the output
    # radar grid).
    out_radar_grid = get_radar_grid_containing_geo_pts(
        geo_pts=geo_pts,
        proj=proj,
        az_spacing=az_spacing,
        rg_spacing=rg_spacing,
        orbit=orbit,
        look_side=radar_grid.lookside,
        wavelength=radar_grid.wavelength,
        az_margin=1e-3 * az_spacing,
        rg_margin=1e-3 * rg_spacing,
    )

    # Check that the input & output radar grids are approximately equivalent (to
    # within +/- 1 azimuth/range sample).
    compare_radar_grids(
        out_radar_grid,
        radar_grid,
        az_tol=az_spacing,
        rg_tol=rg_spacing,
    )

    # Check that the output radar grid reference epoch matches the input orbit
    # object's reference epoch.
    assert out_radar_grid.ref_epoch == orbit.reference_epoch

    # Check that each of the original perimeter points falls within the output radar
    # grid bounds.
    assert all(out_radar_grid.contains(*pt) for pt in az_rg_pts)


def check_radar_grid_contains_geo_pt(
    radar_grid: isce3.product.RadarGridParameters,
    geo_pt: tuple[float, float, float],
    *,
    proj: isce3.core.ProjectionBase,
    orbit: isce3.core.Orbit,
    doppler: isce3.core.LUT2d,
) -> None:
    """
    Check by assertion that a radar grid contains a geocoded point.

    Parameters
    ----------
    radar_grid : isce3.product.RadarGridParameters
        The radar grid.
    geo_pt : (float, float, float)
        The point to check. A point in a geodetic or projected coordinate system
        specified by the `proj` argument.
    proj : isce3.core.ProjectionBase
        A 'projection' object representing the transformation between
        longitude/latitude/height (LLH) coordinates and the coordinate system of
        `geo_pt`.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center. Used for transforming the
        input point into radar coordinates.
    doppler : isce3.core.LUT2d
        The Doppler centroid, in hertz, of the radar grid, expressed as a function of
        azimuth time, in seconds relative to the epoch of `orbit`, and slant range, in
        meters. Used for transforming the input point into radar coordinates.
    """
    llh = proj.inverse(geo_pt)
    xyz = proj.ellipsoid.lon_lat_to_xyz(llh)
    az, rg = isce3.geometry.geo2rdr_bracket(
        xyz=xyz,
        orbit=orbit,
        doppler=doppler,
        wavelength=radar_grid.wavelength,
        side=radar_grid.lookside,
    )
    assert radar_grid.contains(az, rg)


class TestGetBoundingRadarGrid:
    @pytest.fixture
    def geo_grid(
        self,
        winnipeg_dem: isce3.geometry.DEMInterpolator,
    ) -> isce3.product.GeoGridParameters:
        # Get the geocoded grid that the DEM raster is sampled on.
        geo_grid = isce3.product.GeoGridParameters(
            start_x=winnipeg_dem.x_start,
            start_y=winnipeg_dem.y_start,
            spacing_x=winnipeg_dem.delta_x,
            spacing_y=winnipeg_dem.delta_y,
            width=winnipeg_dem.width,
            length=winnipeg_dem.length,
            epsg=winnipeg_dem.epsg_code,
        )

        # Slice `geo_grid` to obtain just a sub-block that is within the azimuth extents
        # of the Winnipeg RSLC orbit (determined empirically). Otherwise, geo2rdr will
        # fail to converge for some points.
        return geo_grid[200:, :500]

    def test_dem(
        self,
        winnipeg_rslc: nisar.products.readers.RSLC,
        winnipeg_dem: isce3.geometry.DEMInterpolator,
        geo_grid: isce3.product.GeoGridParameters,
    ):
        # Extract the radar grid and orbit metadata from the input RSLC. Assume
        # zero-Doppler for NISAR image products.
        radar_grid = winnipeg_rslc.getRadarGrid(frequency="A")
        orbit = winnipeg_rslc.getOrbit()
        doppler = isce3.core.LUT2d()

        # Compute height statistics of the input DEM, including the min & max heights.
        # Statistics are stored internally in the `DEMInterpolator` object.
        winnipeg_dem.compute_min_max_mean_height()

        out_radar_grid = isce3.geometry.get_bounding_radar_grid(
            geo_grid=geo_grid,
            az_spacing=1.0 / radar_grid.prf,
            rg_spacing=radar_grid.range_pixel_spacing,
            orbit=orbit,
            look_side=radar_grid.lookside,
            wavelength=radar_grid.wavelength,
            min_height=winnipeg_dem.min_height,
            max_height=winnipeg_dem.max_height,
        )

        proj = isce3.core.make_projection(winnipeg_dem.epsg_code)

        # Get the set of x & y coordinates of points in the `geo_grid`.
        xcoords = np.linspace(geo_grid.start_x, geo_grid.end_x, num=geo_grid.width + 1)
        ycoords = np.linspace(geo_grid.start_y, geo_grid.end_y, num=geo_grid.length + 1)

        # Check that every point in the input DEM is contained within the output radar
        # grid.
        for x, y in itertools.product(xcoords, ycoords):
            z = winnipeg_dem.interpolate_xy(x, y)
            check_radar_grid_contains_geo_pt(
                radar_grid=out_radar_grid,
                geo_pt=(x, y, z),
                proj=proj,
                orbit=orbit,
                doppler=doppler,
            )

        # Check that the other radar grid metadata matches the original input radar
        # grid.
        assert out_radar_grid.prf == radar_grid.prf
        assert out_radar_grid.range_pixel_spacing == radar_grid.range_pixel_spacing
        assert out_radar_grid.wavelength == radar_grid.wavelength
        assert out_radar_grid.lookside == radar_grid.lookside
        assert out_radar_grid.ref_epoch == radar_grid.ref_epoch

    def test_antimeridian_crossing(self):
        # An orbit segment spanning a frame that crosses the antimeridian.
        orbit_xml_path = os.path.join(
            iscetest.data,
            "NISAR_ANC_L_PR_FOE_20250806T193246_20230104T061021_20230104T061655.xml",
        )
        orbit = nisar.products.readers.orbit.load_orbit_from_xml(orbit_xml_path)

        # Geo grid parameters for a NISAR frame that spans the antimeridian, with 80m
        # spacing.
        geo_grid = isce3.product.GeoGridParameters(
            start_x=591840.0,
            start_y=8272800.0,
            spacing_x=80.0,
            spacing_y=-80.0,
            width=4167,
            length=3987,
            epsg=32760,
        )

        # Get a radar grid that contains the geo grid.
        # (The azimuth/range spacing and wavelength aren't relevant to the test. They're
        # just assigned arbitrarily-chosen NISAR-like dummy values.)
        radar_grid = isce3.geometry.get_bounding_radar_grid(
            geo_grid=geo_grid,
            az_spacing=0.0006,
            rg_spacing=6.0,
            orbit=orbit,
            look_side="left",
            wavelength=0.24,
        )

        # Get the latitude coordinate (in degrees) of the mid-point of the geo grid.
        x_mid = geo_grid.start_x + 0.5 * geo_grid.spacing_x * geo_grid.width
        y_mid = geo_grid.start_y + 0.5 * geo_grid.spacing_y * geo_grid.length
        proj = isce3.core.make_projection(geo_grid.epsg)
        _, lat_mid, _ = np.rad2deg(proj.inverse((x_mid, y_mid, 0.0)))

        # Check that a pair of points just west and just east of the antimeridian are
        # both contained within the radar grid.
        lonlat_proj = isce3.core.make_projection(4326)
        zero_doppler = isce3.core.LUT2d()
        for lon in [-179.99, 179.99]:
            check_radar_grid_contains_geo_pt(
                radar_grid=radar_grid,
                geo_pt=(lon, lat_mid, 0.0),
                proj=lonlat_proj,
                orbit=orbit,
                doppler=zero_doppler,
            )

    def test_pole_crossing(self):
        # An orbit segment near the South pole.
        orbit_xml_path = os.path.join(
            iscetest.data,
            "NISAR_ANC_L_PR_FOE_20250806T193731_20230105T054735_20230105T055408.xml",
        )
        orbit = nisar.products.readers.orbit.load_orbit_from_xml(orbit_xml_path)

        # A geo grid that contains the South pole.
        geo_grid = isce3.product.GeoGridParameters(
            start_x=-1.0,
            start_y=-88.0,
            spacing_x=0.001,
            spacing_y=-0.002,
            width=2001,
            length=2001,
            epsg=4326,
        )

        # Get a radar grid that contains the geo grid.
        # (The azimuth/range spacing and wavelength aren't relevant to the test. They're
        # just assigned arbitrarily-chosen NISAR-like dummy values.)
        radar_grid = isce3.geometry.get_bounding_radar_grid(
            geo_grid=geo_grid,
            az_spacing=0.0006,
            rg_spacing=6.0,
            orbit=orbit,
            look_side="left",
            wavelength=0.24,
        )

        # Check that the radar grid contains the pole.
        lonlat_proj = isce3.core.make_projection(4326)
        zero_doppler = isce3.core.LUT2d()
        check_radar_grid_contains_geo_pt(
            radar_grid=radar_grid,
            geo_pt=(0.0, -90.0, 0.0),
            proj=lonlat_proj,
            orbit=orbit,
            doppler=zero_doppler,
        )

    def test_bad_height_range(
        self,
        winnipeg_rslc: nisar.products.readers.RSLC,
        geo_grid: isce3.product.GeoGridParameters,
    ):
        orbit = winnipeg_rslc.getOrbit()
        min_height, max_height = 1.0, 0.0
        regex = (
            f"^max_height must be >= min_height, got {max_height=} and {min_height=}$"
        )
        with pytest.raises(ValueError, match=regex):
            isce3.geometry.get_bounding_radar_grid(
                geo_grid=geo_grid,
                az_spacing=1.0,
                rg_spacing=1.0,
                orbit=orbit,
                look_side=isce3.core.LookSide.Left,
                wavelength=0.1,
                min_height=min_height,
                max_height=max_height,
            )

    def test_bad_pts_per_edge(
        self,
        winnipeg_rslc: nisar.products.readers.RSLC,
        geo_grid: isce3.product.GeoGridParameters,
    ):
        orbit = winnipeg_rslc.getOrbit()
        pts_per_edge = 1
        with pytest.raises(ValueError, match=f"^{pts_per_edge=}, must be >= 2$"):
            isce3.geometry.get_bounding_radar_grid(
                geo_grid=geo_grid,
                az_spacing=1.0,
                rg_spacing=1.0,
                orbit=orbit,
                look_side=isce3.core.LookSide.Left,
                wavelength=0.1,
                pts_per_edge=pts_per_edge,
            )
