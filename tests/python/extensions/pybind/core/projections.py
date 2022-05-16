#!/usr/bin/env python3

import numpy as np
import numpy.testing as npt
import isce3.ext.isce3 as isce3


def test_lonlat():
    proj = isce3.core.LonLat()

    assert proj.code == 4326


def test_geocent():
    proj = isce3.core.Geocent()

    assert proj.code == 4978

    # Get reference ellipsoid semi-major and semi-minor axis lengths.
    a = proj.ellipsoid.a
    b = a * np.sqrt(1.0 - proj.ellipsoid.e2)

    test_data = [
        {"llh": [0.0, 0.0, 0.0], "xyz": [a, 0.0, 0.0]},  # Origin
        {"llh": [0.5 * np.pi, 0.0, 0.0], "xyz": [0.0, a, 0.0]},  # Equator 90E
        {"llh": [-0.5 * np.pi, 0.0, 0.0], "xyz": [0.0, -a, 0.0]},  # Equator 90W
        {"llh": [np.pi, 0.0, 0.0], "xyz": [-a, 0.0, 0.0]},  # Equator Dateline
        {"llh": [0.0, 0.5 * np.pi, 0.0], "xyz": [0.0, 0.0, b]},  # North Pole
        {"llh": [0.0, -0.5 * np.pi, 0.0], "xyz": [0.0, 0.0, -b]},  # South Pole
    ]

    for d in test_data:
        npt.assert_allclose(proj.forward(d["llh"]), d["xyz"], atol=1e-6)
        npt.assert_allclose(proj.inverse(d["xyz"]), d["llh"], atol=1e-6)


def test_utm():
    # UTM North
    for zone in range(1, 61):
        epsg = 32600 + zone
        proj = isce3.core.UTM(epsg)

        assert proj.code == epsg

        # Test origin of each UTM zone
        xyz = [500_000.0, 0.0, 0.0]
        llh = [np.deg2rad(-177.0 + 6.0 * (zone - 1)), 0.0, 0.0]
        npt.assert_allclose(proj.forward(llh), xyz, atol=1e-6)
        npt.assert_allclose(proj.inverse(xyz), llh, atol=1e-6)

    # UTM South
    for zone in range(1, 61):
        epsg = 32700 + zone
        proj = isce3.core.UTM(epsg)

        assert proj.code == epsg

        # Test origin of each UTM zone
        xyz = [500_000.0, 10_000_000.0, 0.0]
        llh = [np.deg2rad(-177.0 + 6.0 * (zone - 1)), 0.0, 0.0]
        npt.assert_allclose(proj.forward(llh), xyz, atol=1e-6)
        npt.assert_allclose(proj.inverse(xyz), llh, atol=1e-6)


def test_polarstereo():
    # North Pole
    epsg = 3413
    proj = isce3.core.PolarStereo(epsg)

    assert proj.code == epsg

    xyz = [0.0, 0.0, 0.0]
    llh = [0.0, 0.5 * np.pi, 0.0]
    npt.assert_allclose(proj.forward(llh), xyz, atol=1e-6)
    npt.assert_allclose(proj.inverse(xyz), llh, atol=1e-6)

    # South Pole
    epsg = 3031
    proj = isce3.core.PolarStereo(epsg)

    assert proj.code == epsg

    xyz = [0.0, 0.0, 0.0]
    llh = [0.0, -0.5 * np.pi, 0.0]
    npt.assert_allclose(proj.forward(llh), xyz, atol=1e-6)
    npt.assert_allclose(proj.inverse(xyz), llh, atol=1e-6)


def test_cea():
    proj = isce3.core.CEA()

    assert proj.code == 6933

    # XXX Copy-paste job from the C++ unit test - not sure where this data
    # originates from
    test_data = [
        {
            "llh": [-1.397694375733237e00, 8.496490909249732e-01, 6.397636527923552e03],
            "xyz": [-7.726813212349523e06, 5.503591184289403e06, 6.397636527923552e03],
        },
        {
            "llh": [1.513020264912829e00, -1.352352581363516e-01, 7.616334412360351e03],
            "xyz": [8.364364324888950e06, -9.855597820966323e05, 7.616334412360351e03],
        },
        {
            "llh": [-8.456729268427137e-01, 1.255355912492460e00, 7.246008662493211e03],
            "xyz": [-4.675096972488437e06, 6.976944015207697e06, 7.246008662493211e03],
        },
    ]

    for d in test_data:
        npt.assert_allclose(proj.forward(d["llh"]), d["xyz"], atol=1e-6)
        npt.assert_allclose(proj.inverse(d["xyz"]), d["llh"], atol=1e-6)


# Dummy projection to test inheritance. Forward/inverse transformations are
# identity.
class IdentityProjection(isce3.core.ProjectionBase):
    def __init__(self):
        super().__init__(code=-1)

    def forward(self, llh):
        return llh

    def inverse(self, xyz):
        return xyz


def test_inheritance():
    proj = IdentityProjection()

    assert proj.code == -1

    xyz = llh = np.random.uniform(-90.0, 90.0, size=3)

    npt.assert_allclose(proj.forward(llh), xyz, atol=1e-6)
    npt.assert_allclose(proj.inverse(xyz), llh, atol=1e-6)


def test_make_projection():
    # LonLat
    proj = isce3.core.make_projection(4326)
    assert type(proj) == isce3.core.LonLat

    # Geocent
    proj = isce3.core.make_projection(4978)
    assert type(proj) == isce3.core.Geocent

    # UTM North
    for zone in range(1, 61):
        proj = isce3.core.make_projection(32600 + zone)
        assert type(proj) == isce3.core.UTM

    # UTM South
    for zone in range(1, 61):
        proj = isce3.core.make_projection(32700 + zone)
        assert type(proj) == isce3.core.UTM

    # UPS North
    proj = isce3.core.make_projection(3413)
    assert type(proj) == isce3.core.PolarStereo

    # UPS South
    proj = isce3.core.make_projection(3031)
    assert type(proj) == isce3.core.PolarStereo

    # EASE-Grid 2.0
    proj = isce3.core.make_projection(6933)
    assert type(proj) == isce3.core.CEA
