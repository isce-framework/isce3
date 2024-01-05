import os
import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import shapely
from numpy.random import default_rng
from numpy.typing import ArrayLike

import isce3
import iscetest
from isce3.cal.corner_reflector import (
    cr_to_enu_rotation,
    enu_to_cr_rotation,
    normalize_vector,
)
import nisar


def wrap(phi: ArrayLike) -> np.ndarray:
    """Wrap the input angle (in radians) to the interval [-pi, pi)."""
    phi = np.asarray(phi)
    return np.mod(phi + np.pi, 2.0 * np.pi) - np.pi


def angle_between(u: ArrayLike, v: ArrayLike, *, degrees: bool = False) -> float:
    """Measure the angle between two vectors."""
    u, v = map(normalize_vector, [u, v])
    theta = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
    if degrees:
        theta = np.rad2deg(theta)
    return theta


class TestParseTriangularTrihedralCornerReflectorCSV:
    def test_parse_csv(self):
        # Parse CSV file containing 3 Northeast-looking corner reflectors.
        csv = Path(iscetest.data) / "abscal/REE_CORNER_REFLECTORS_INFO.csv"
        crs = list(isce3.cal.parse_triangular_trihedral_cr_csv(csv))

        # Check the number of corner reflectors.
        assert len(crs) == 3

        # Check that CR latitudes & longitudes are within ~1 degree of their approximate
        # expected location.
        atol = np.sin(np.deg2rad(1.0))
        lats = [cr.llh.latitude for cr in crs]
        approx_lat = np.deg2rad(69.5)
        npt.assert_allclose(np.abs(wrap(lats - approx_lat)), 0.0, atol=atol)
        lons = [cr.llh.longitude for cr in crs]
        approx_lon = np.deg2rad(-128.5)
        npt.assert_allclose(np.abs(wrap(lons - approx_lon)), 0.0, atol=atol)

        # Check that CR heights are within 1 meter of their approximate expected location.
        heights = [cr.llh.height for cr in crs]
        approx_height = 490.0
        npt.assert_allclose(heights, approx_height, atol=1.0)

        # Check that CR azimuth & elevation angles are each within ~1 degree of their
        # approximate expected orientation.
        azs = [cr.azimuth for cr in crs]
        approx_az = np.deg2rad(317.0)
        npt.assert_allclose(np.abs(wrap(azs - approx_az)), 0.0, atol=atol)
        els = [cr.elevation for cr in crs]
        approx_el = np.deg2rad(12.5)
        npt.assert_allclose(np.abs(wrap(els - approx_el)), 0.0, atol=atol)

        # Check that CR side lengths all match their expected value.
        side_lengths = [cr.side_length for cr in crs]
        expected_side_length = 3.4629120649497214
        npt.assert_array_equal(side_lengths, expected_side_length)

    def test_empty_csv(self):
        with tempfile.NamedTemporaryFile() as f:
            csv = f.name
            crs = isce3.cal.parse_triangular_trihedral_cr_csv(csv)
            assert len(list(crs)) == 0


class TestCRToENURotation:

    sin = lambda x: np.sin(np.deg2rad(x))
    cos = lambda x: np.cos(np.deg2rad(x))

    # The test is parameterized as follows:
    # The first two parameters (az,el) are the orientation angles of the corner
    # reflector, in degrees.
    # The remaining three parameters (x,y,z) are ENU unit vectors that are expected to
    # be aligned with the three legs of the corner reflector.
    @pytest.mark.parametrize(
        "az,el,x,y,z",
        [
            (0, 0, [cos(-45), sin(-45), 0], [cos(45), sin(45), 0], [0, 0, 1]),
            (45, 0, [0, -1, 0], [1, 0, 0], [0, 0, 1]),
            (0, 90, [0, -sin(45), cos(45)], [0, -sin(-45), cos(-45)], [-1, 0, 0]),
            (
                270,
                45,
                [cos(45), sin(45) * cos(45), sin(45) * sin(45)],
                [-sin(45), cos(45) * cos(45), cos(45) * sin(45)],
                [0, -sin(45), cos(45)],
            ),
            (
                -45,
                -90,
                [cos(-45) ** 2, sin(-45) * cos(-45), sin(-45)],
                [cos(135) * cos(-45), sin(135) * cos(-45), sin(-45)],
                [cos(45), sin(45), 0],
            ),
        ],
    )
    def test_cr_to_enu(self, az, el, x, y, z):
        # # Convert degrees to radians.
        az, el = np.deg2rad([az, el])

        # Get rotation (quaternion) from CR-intrinsic coordinates to ENU coordinates.
        q = cr_to_enu_rotation(az=az, el=el)

        # Compare the rotated basis vectors to the expected ENU vectors.
        atol = 1e-6
        assert angle_between(q.rotate([1.0, 0.0, 0.0]), x) < atol
        assert angle_between(q.rotate([0.0, 1.0, 0.0]), y) < atol
        assert angle_between(q.rotate([0.0, 0.0, 1.0]), z) < atol

    # The test is parameterized as follows:
    # The first two parameters (az,el) are defined as in the previous test.
    # The remaining three parameters (e,n,u) are unit vectors in the corner
    # reflector-intrinsic coordinate system that are expected to align with the E,N,U
    # basis vectors.
    @pytest.mark.parametrize(
        "az,el,e,n,u",
        [
            (0, 0, [cos(45), sin(45), 0], [cos(135), sin(135), 0], [0, 0, 1]),
            (45, 0, [0, 1, 0], [-1, 0, 0], [0, 0, 1]),
            (0, 90, [0, 0, -1], [-sin(45), cos(45), 0], [-sin(-45), cos(-45), 0]),
            (
                270,
                45,
                [cos(-45), sin(-45), 0],
                [cos(45) * sin(45), sin(45) * cos(-45), -sin(45)],
                [cos(45) * cos(-45), sin(45) * cos(45), -sin(-45)],
            ),
            (
                -45,
                -90,
                [cos(-45) ** 2, sin(-45) * cos(-45), sin(45)],
                [cos(135) * cos(45), sin(135) * cos(45), sin(45)],
                [cos(-135), sin(-135), 0],
            ),
        ],
    )
    def test_enu_to_cr(self, az, el, e, n, u):
        # Convert degrees to radians.
        az, el = np.deg2rad([az, el])

        # Get rotation (quaternion) from CR-intrinsic coordinates to ENU coordinates.
        q = enu_to_cr_rotation(az=az, el=el)

        # Compare the rotated basis vectors to the expected ENU vectors.
        atol = 1e-6
        assert angle_between(q.rotate([1.0, 0.0, 0.0]), e) < atol
        assert angle_between(q.rotate([0.0, 1.0, 0.0]), n) < atol
        assert angle_between(q.rotate([0.0, 0.0, 1.0]), u) < atol

    def test_az_and_el(self):
        # Randomly sample azimuth angles in [-pi, pi) and elevation angles in [0, pi/2).
        rng = default_rng(seed=1234)
        n = 100
        azimuths = rng.uniform(-np.pi, np.pi, size=n)
        elevations = rng.uniform(0.0, np.pi / 2, size=n)

        for az, el in zip(azimuths, elevations):
            q = cr_to_enu_rotation(az=az, el=el)

            # Rotate the corner reflector boresight vector to ENU coordinates. Compute
            # the angle in the E-N plane of the vector from the E-axis, measured
            # clockwise. This should match the azimuth angle of the corner reflector.
            boresight = np.array([1, 1, 0]) / np.sqrt(2)
            x = q.rotate(boresight)
            theta = np.arctan2(-x[1], x[0])
            assert np.abs(wrap(theta - az)) < 1e-6

            # Rotate the corner reflector Z-axis to ENU coordinates. Compute the angle
            # between the rotated vector and the U-axis. This should match the elevation
            # angle of the corner reflector.
            z = q.rotate([0.0, 0.0, 1.0])
            phi = np.arccos(z[2])
            assert np.abs(wrap(phi - el)) < 1e-6

    def test_roundtrip(self):
        # Randomly sample azimuth angles in [-pi, pi) and elevation angles in [0, pi/2).
        rng = default_rng(seed=1234)
        n = 100
        azimuths = rng.uniform(-np.pi, np.pi, size=n)
        elevations = rng.uniform(0.0, np.pi / 2, size=n)

        # Test that a roundtrip rotation (CR -> ENU -> CR) is identity.
        identity = isce3.core.Quaternion([1.0, 0.0, 0.0, 0.0])
        for az, el in zip(azimuths, elevations):
            q_cr2enu = cr_to_enu_rotation(az=az, el=el)
            q_enu2cr = enu_to_cr_rotation(az=az, el=el)
            assert (q_enu2cr * q_cr2enu).is_approx(identity)


def test_get_target_observation_time_and_elevation():
    datadir = Path(iscetest.data)

    # Get simulated RSLC data containing a single point target.
    rslc_hdf5 = datadir / "REE_RSLC_out17.h5"
    rslc = nisar.products.readers.SLC(hdf5file=os.fspath(rslc_hdf5))

    orbit = rslc.getOrbit()
    attitude = rslc.getAttitude()
    radar_grid = rslc.getRadarGrid(frequency="A")

    # The CSV contains a single corner reflector.
    cr_csv = datadir / "REE_CR_INFO_out17.csv"
    cr = list(isce3.cal.parse_triangular_trihedral_cr_csv(cr_csv))[0]

    # Estimate target zero-Doppler UTC datetime and elevation angle.
    az_datetime, el_angle = isce3.cal.get_target_observation_time_and_elevation(
        target_llh=cr.llh,
        orbit=orbit,
        attitude=attitude,
        wavelength=radar_grid.wavelength,
        look_side=radar_grid.lookside,
    )

    # The target is located approximately in the center of the radar grid.
    expected_az = radar_grid.ref_epoch + isce3.core.TimeDelta(radar_grid.sensing_mid)
    expected_el = 0.0

    assert az_datetime.is_close(expected_az, tol=isce3.core.TimeDelta(seconds=1e-3))
    assert np.isclose(el_angle, expected_el, atol=1e-6)


def test_get_crs_in_polygon():
    # Create a rectangular lon/lat polygon.
    lon0, lon1 = -2.0, 2.0
    lat0, lat1 = -0.5, 0.5
    lonlat_polygon = shapely.Polygon(
        [
            (lon0, lat0),
            (lon0, lat1),
            (lon1, lat1),
            (lon1, lat0),
            (lon0, lat0),
        ]
    )

    # Corner reflector longitudes & latitudes in degrees:
    #  - The first four CRs are contained within the polygon
    #  - The next two CRs are on the border of the polygon
    #  - The final two CRs are slightly outside the polygon
    eps = 1e-6
    cr_lonlats = [
        (0.0, 0.0),
        (360.0, 0.0),
        (1.0, 0.0),
        (0.0, 0.5 - eps),
        (0.0, 0.5),
        (2.0, 0.5),
        (2.1 - eps, 0.0),
        (2.1 + eps, 0.0),
    ]

    # Make a list of corner reflectors with unique IDs, one at each lon/lat location.
    crs = [
        isce3.cal.TriangularTrihedralCornerReflector(
            id=f"CR{i}",
            llh=isce3.core.LLH(np.deg2rad(lon), np.deg2rad(lat), 0.0),
            elevation=0.0,
            azimuth=0.0,
            side_length=1.0,
        )
        for i, (lon, lat) in enumerate(cr_lonlats)
    ]

    cr_ids = [cr.id for cr in crs]

    # Get a list of CRs within the polygon bounds. The resulting list should contain
    # only the first 4 CRs.
    filtered_crs = isce3.cal.get_crs_in_polygon(crs, lonlat_polygon)
    filtered_cr_ids = [cr.id for cr in filtered_crs]
    assert filtered_cr_ids == cr_ids[:4]

    # Get a list of CRs inside or within 0.1 degrees of the polygon. The resulting list
    # should contain all except the last CR.
    filtered_crs = isce3.cal.get_crs_in_polygon(crs, lonlat_polygon, buffer=0.1)
    filtered_cr_ids = [cr.id for cr in filtered_crs]
    assert filtered_cr_ids == cr_ids[:-1]
