from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

import isce3
from isce3.geometry import infer_radar_grid_spacing_from_geo_grid


def make_linear_orbit(
    initial_position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    times: Iterable[float],
    epoch: isce3.core.DateTime = isce3.core.DateTime("2000-01-01"),
) -> isce3.core.Orbit:
    """
    Construct an `Orbit` representing a linear flight path with fixed velocity.

    Parameters
    ----------
    initial_position : (float, float, float)
        The position of the platform, in meters, in the Earth-centered, Earth-fixed
        (ECEF) coordinate system at time t=0.
    velocity : (float, float, float)
        The velocity of the platform, in meters per second, in the same coordinate
        system as `initial_position`.
    times : iterable of float
        The timepoints at which to sample the orbit state vectors, in seconds since
        `epoch`.
    epoch : isce3.core.DateTime, optional
        The UTC date and time of the timepoint t=0. Defaults to
        `DateTime('2000-01-01')`.

    Returns
    -------
    isce3.core.Orbit
        The resulting orbit.
    """
    initial_position = np.asarray(initial_position)
    velocity = np.asarray(velocity)

    state_vectors = []
    for t in times:
        datetime = epoch + isce3.core.TimeDelta(t)
        position = initial_position + velocity * t
        state_vector = isce3.core.StateVector(datetime, position, velocity)
        state_vectors.append(state_vector)

    return isce3.core.Orbit(state_vectors, epoch)


@pytest.mark.parametrize("case", [1, 2])
def test_infer_radar_grid_spacing_from_geo_grid(case: int):
    # Construct a lat/lon grid centered on the intersection of the prime meridian and
    # the equator. In ECEF coordinates, the center of the grid is the point [R, 0, 0],
    # where R is the equatorial radius of the ellipsoid. The grid extents are small
    # relative to R, so the ellipsoid surface can be considered approximately flat
    # within the grid.

    # The EPSG code and corresponding reference ellipsoid of the lat/lon grid.
    epsg = 4326
    ellipsoid = isce3.core.WGS84_ELLIPSOID

    # Dimensions of the grid.
    width = 100
    length = 200

    # Approximate x & y spacing of the grid, in meters.
    x_spacing = 10.0
    y_spacing = 5.0

    # Longitude & latitude spacing of the grid, in degrees.
    lon_spacing = np.rad2deg(np.arctan(x_spacing / ellipsoid.a))
    lat_spacing = np.rad2deg(np.arctan(y_spacing / ellipsoid.r_north(0.0)))

    geo_grid = isce3.product.GeoGridParameters(
        start_x=width * -0.5 * lon_spacing,
        start_y=length * 0.5 * lat_spacing,
        spacing_x=lon_spacing,
        spacing_y=-lat_spacing,
        width=width,
        length=length,
        epsg=epsg,
    )

    # Construct a zero-height DEM w.r.t the WGS 84 ellipsoid.
    dem = isce3.geometry.DEMInterpolator(height=0.0, epsg=epsg)

    # Incidence angle, in radians.
    theta = np.deg2rad(37.0)

    # Platform altitude, in meters.
    h = 750e3

    # Platform speed, in meters per second.
    v = 7.5e3

    if case == 1:
        # Northward traveling, left-looking radar. The azimuth direction should be
        # aligned with the y axis of the grid and the range direction should be aligned
        # with the x axis of the grid (assuming zero squint).
        orbit = make_linear_orbit(
            initial_position=(ellipsoid.a + h, h * np.tan(theta), 0.0),
            velocity=[0.0, 0.0, v],
            times=np.linspace(-10.0, 10.0, num=21),
        )
        look_side = "left"
    elif case == 2:
        # Eastward traveling, right-looking radar. The azimuth direction should be
        # aligned with the x axis of the grid and the range direction should be aligned
        # with the y axis of the grid (assuming zero squint).
        orbit = make_linear_orbit(
            initial_position=[ellipsoid.a + h, 0.0, h * np.tan(theta)],
            velocity=[0.0, v, 0.0],
            times=np.linspace(-10.0, 10.0, num=21),
        )
        look_side = "right"
    else:
        # Should be unreachable.
        assert False

    # Use zero-Doppler (zero-squint) geometry. In this case, the wavelength shouldn't
    # have any effect on the results.
    doppler = isce3.core.LUT2d()
    dummy_wavelength = 0.25

    # Compute the azimuth and range spacing via geo2rdr.
    az_spacing, rg_spacing = infer_radar_grid_spacing_from_geo_grid(
        geo_grid=geo_grid,
        dem=dem,
        orbit=orbit,
        doppler=doppler,
        look_side=look_side,
        wavelength=dummy_wavelength,
    )

    # Compute the approximate expected range and azimuth spacing, given the simplified
    # geometry of the scene. Note that the ground track velocity is approximately equal
    # to the platform velocity since the flight track is linear and the ellipsoid
    # surface is nearly flat within `geo_grid`.
    if case == 1:
        expected_rg_spacing = x_spacing * np.sin(theta)
        expected_az_spacing = y_spacing / v
    elif case == 2:
        expected_rg_spacing = y_spacing * np.sin(theta)
        expected_az_spacing = x_spacing / v
    else:
        assert False

    # Compare the computed values against the expected values. Multiply the expected
    # values by 0.5 to match the internal logic of
    # `infer_radar_grid_spacing_from_geo_grid()`.
    np.testing.assert_allclose(rg_spacing, 0.5 * expected_rg_spacing, rtol=1e-3)
    np.testing.assert_allclose(az_spacing, 0.5 * expected_az_spacing, rtol=1e-3)
