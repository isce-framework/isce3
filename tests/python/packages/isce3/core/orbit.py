from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

import isce3
from isce3.core.orbit import count_sign_changes


class TestCountSignChanges:
    def test_empty(self):
        assert count_sign_changes([]) == 0

    @pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
    def test_single(self, value: float):
        assert count_sign_changes([value]) == 0

    @pytest.mark.parametrize("fill_value", [-1.0, 0.0, 1.0])
    def test_all_constant(self, fill_value: float):
        x = np.full(100, fill_value=fill_value)
        assert count_sign_changes(x) == 0

    def test_linspace(self):
        x = np.linspace(-1.0, 1.0, num=11)
        assert count_sign_changes(x) == 1

    def test_sinusoid(self):
        x = np.linspace(-10.0, 10.0, num=101)
        y = np.sin(x)
        assert count_sign_changes(y) == 7

    def test_signed_zeros(self):
        assert count_sign_changes([0.0, -0.0]) == 1


# Like `itertools.count`, but supports non-number types.
# See https://github.com/python/cpython/issues/90732#issuecomment-1093943716.
def count(start=0, step=1) -> Iterable:
    n = start
    while True:
        yield n
        n += step


def simulate_orbit_from_llhs(
    llhs: Iterable[isce3.core.LLH],
    dt: isce3.core.TimeDelta,
    *,
    ellipsoid: isce3.core.Ellipsoid = isce3.core.WGS84_ELLIPSOID,
) -> isce3.core.Orbit:
    """
    Simulate orbit state vectors from longitude/latitude/height (LLH) coordinates.

    The orbit velocity samples are approximated using finite differences between the
    position vectors.

    Parameters
    ----------
    llhs : iterable of isce3.core.LLH
        A sequence of orbit position vectors, specified in LLH coordinates.
    dt : isce3.core.TimeDelta
        Spacing between orbit state vectors.
    ellipsoid : isce3.core.Ellipsoid, optional
        The reference ellipsoid. Defaults to the WGS 84 ellipsoid.

    Returns
    -------
    isce3.core.Orbit
        The simulated orbit.
    """
    # An unbounded iterator yielding consecutive datetimes with uniform spacing.
    start_datetime = isce3.core.DateTime("1970-01-01T00:00:00")
    datetimes = count(start=start_datetime, step=dt)

    # Get orbit positions and velocities in ECEF coordinates.
    positions = np.asarray([ellipsoid.lon_lat_to_xyz(llh.to_vec3()) for llh in llhs])
    velocities = np.gradient(positions, dt.total_seconds(), axis=0)

    state_vectors = [
        isce3.core.StateVector(*args) for args in zip(datetimes, positions, velocities)
    ]
    return isce3.core.Orbit(state_vectors)


class TestOrbitPassDirection:
    def test_enum_values(self):
        assert str(isce3.core.OrbitPassDirection.ASCENDING) == "ascending"
        assert str(isce3.core.OrbitPassDirection.DESCENDING) == "descending"

    def test_ascending(self):
        # Sample LLH coordinates with ascending latitude and fixed longitude/height.
        lon = 0.0
        lats = np.deg2rad(np.linspace(-30.0, 30.0, num=201))
        height = 700e3
        llhs = (isce3.core.LLH(lon, lat, height) for lat in lats)

        # 10 second spacing between state vectors.
        dt = isce3.core.TimeDelta(seconds=10.0)

        # Simulate a roughly elliptical orbit at fixed height above the spheroid.
        orbit = simulate_orbit_from_llhs(llhs, dt)

        assert isce3.core.get_orbit_pass_direction(orbit) == "ascending"

    def test_descending(self):
        # Sample LLH coordinates with descending latitude and fixed longitude/height.
        lon = np.pi / 2.0
        lats = np.deg2rad(np.linspace(90.0, 30.0, num=201))
        height = 700e3
        llhs = (isce3.core.LLH(lon, lat, height) for lat in lats)

        # 10 second spacing between state vectors.
        dt = isce3.core.TimeDelta(seconds=10.0)

        # Simulate a roughly elliptical orbit at fixed height above the spheroid.
        orbit = simulate_orbit_from_llhs(llhs, dt)

        assert isce3.core.get_orbit_pass_direction(orbit) == "descending"

    def test_mixed(self):
        # Sample LLH coordinates with varying latitude and fixed longitude/height. The
        # latitude coordinate is initially ascending and then descending.
        lon = np.pi
        x = np.linspace(-1.0, 0.5, num=201)
        lats = np.pi / 2.0 * np.cos(x)
        height = 700e3
        llhs = (isce3.core.LLH(lon, lat, height) for lat in lats)

        # 10 second spacing between state vectors.
        dt = isce3.core.TimeDelta(seconds=10.0)

        # Simulate a roughly elliptical orbit at fixed height above the spheroid.
        orbit = simulate_orbit_from_llhs(llhs, dt)

        assert isce3.core.get_orbit_pass_direction(orbit) == "ascending"

    def test_multiple_periods(self):
        # Sample LLH coordinates with fixed longitude/height but sinusoidally varying
        # latitude. The sampling interval contains multiple orbital periods.
        lon = np.pi
        x = np.linspace(0.0, 10.0, num=1001)
        lats = np.pi / 2.0 * np.sin(x)
        height = 700e3
        llhs = (isce3.core.LLH(lon, lat, height) for lat in lats)

        # 10 second spacing between state vectors.
        dt = isce3.core.TimeDelta(seconds=10.0)

        # Simulate a roughly elliptical orbit at fixed height above the spheroid.
        orbit = simulate_orbit_from_llhs(llhs, dt)

        with pytest.raises(isce3.core.AmbiguousOrbitPassDirection):
            isce3.core.get_orbit_pass_direction(orbit)
