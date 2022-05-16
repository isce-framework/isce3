#!/usr/bin/env python3
import numpy as np
from numpy.linalg import norm
import numpy.testing as npt
import isce3.ext.isce3 as isce3
import pytest

np.random.seed(12345)  # I've got the same combination on my luggage!

def test_access():
    R = np.eye(3)
    basis = isce3.core.Basis(R[0], R[1], R[2])
    basis.x0 == R[0]
    basis.x1 == R[1]
    basis.x2 == R[2]

def test_unitary():
    # Unit vectors but not orthogonal.
    x0 = np.random.normal(scale=100, size=3)
    x1 = np.random.normal(scale=100, size=3)
    x2 = np.random.normal(scale=100, size=3)
    x0, x1, x2 = x0 / norm(x0), x1 / norm(x1), x2 / norm(x2)
    assert abs(x0.dot(x1)) > 0, "Test fixture anomaly.  numpy.random changed?"
    with pytest.raises(Exception):
        isce3.core.Basis(x0, x1, x2)

    # Orthogonal but not unit vectors.
    x0 = np.random.normal(scale=100, size=3)
    tmp = np.random.normal(scale=100, size=3)
    x1 = tmp - tmp.dot(x0) * x0 / norm(x0)**2
    x2 = np.cross(x0, x1)
    assert norm(x0) > 1.0, "Test fixture anomaly.  numpy.random changed?"
    with pytest.raises(Exception):
        isce3.core.Basis(x0, x1, x2)

    # Okay, let's try a valid construction.
    x0, x1, x2 = x0 / norm(x0), x1 / norm(x1), x2 / norm(x2)
    basis = isce3.core.Basis(x0, x1, x2)

    # Now that we have an orthonormal basis, make sure it doesn't scale (change
    # the norm) when transforming coordinates.
    for i in range(10):
        vin = np.random.normal(size=3)
        vout = basis.project(vin)
        assert np.isclose(norm(vin), norm(vout))
        vout = basis.combine(vin)
        assert np.isclose(norm(vin), norm(vout))

def test_tcn():
    # geocentric
    p = [0, 0, 1.0]
    v = [1, 0, 0.0]
    tcn = isce3.core.Basis(position=p, velocity=v)
    assert np.allclose([1, 0, 0], tcn.x0)
    assert np.allclose([0, -1, 0], tcn.x1)
    assert np.allclose([0, 0, -1], tcn.x2)

    # geodetic
    ell = isce3.core.Ellipsoid()
    h = 740e3
    # same lon/lat, different height
    lon, lat = 0.34, 0.56
    p0 = ell.lon_lat_to_xyz([lon, lat, 0.0])
    p1 = ell.lon_lat_to_xyz([lon, lat, h])
    tcn = isce3.core.geodetic_tcn(p1, v, ell)
    assert np.isclose(h, (p0 - p1).dot(tcn.x2))

def test_factored_ypr():
    # body == tcn == world orientation
    q = isce3.core.Quaternion(1, 0, 0, 0)
    x = np.array([0, 0, -7e6])
    v = np.array([1, 0, 0])
    # geocentric
    angles = isce3.core.factored_ypr(q, x, v)
    assert np.isclose(angles.yaw, 0)
    assert np.isclose(angles.pitch, 0)
    assert np.isclose(angles.roll, 0)
    # geodetic, no difference at equator or poles
    angles = isce3.core.factored_ypr(q, x, v, isce3.core.Ellipsoid())
    assert np.isclose(angles.yaw, 0)
    assert np.isclose(angles.pitch, 0)
    assert np.isclose(angles.roll, 0)

def test_asarray():
    # orthogonal basis where B != B.transpose()
    x = np.array([1, 0,  1.0]) / np.sqrt(2)
    y = np.array([1, 0, -1.0]) / np.sqrt(2)
    z = np.array([0, 1,  0.0])
    # Check conversion to matrix has expected order.
    b = isce3.core.Basis(x, y, z).asarray()
    bx = b[:,0]
    assert np.isclose(bx.dot(x), 1.0)
