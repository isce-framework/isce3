#!/usr/bin/env python3

import numpy as np
import numpy.testing as npt

import pybind_isce3 as isce3

# constants for tolerances
RTOL = 1e-8
ATOL = 1e-10

def test_identity():
    R = np.eye(3)
    ypr = isce3.core.EulerAngles(R)
    assert np.allclose(np.zeros(3), [ypr.yaw, ypr.pitch, ypr.roll])

    m = isce3.core.EulerAngles(0, 0, 0).to_rotation_matrix()
    assert np.allclose(np.eye(3), m)


def test_ypr():
    assert np.allclose(isce3.core.EulerAngles(np.pi, 0, 0).to_rotation_matrix(),
        [[-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]])
    assert np.allclose(isce3.core.EulerAngles(0, 0, np.pi).to_rotation_matrix(),
        [[1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]])


def test_rotmat():
    ea = isce3.core.EulerAngles(
        [[-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]])
    assert np.allclose([np.pi, 0, 0], [ea.yaw, ea.pitch, ea.roll])

    ea = isce3.core.EulerAngles(
        [[1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]])
    assert np.allclose([0, 0, np.pi], [ea.yaw, ea.pitch, ea.roll])


def test_to_quaternion():
    ea = isce3.core.EulerAngles(0, 0, 0)
    q = ea.to_quaternion()
    npt.assert_allclose(q(), [1,0,0,0], rtol=RTOL, atol=ATOL)


def test_is_approx():
    ea1 = isce3.core.EulerAngles(0, 0, 0)
    ea2 = isce3.core.EulerAngles(*(ea1() + 1e-3))
    npt.assert_(not ea1.is_approx(ea2))


def test_rotate():
    ea = isce3.core.EulerAngles(0, 0, 0)
    v3 = [-1, 0, 0]
    npt.assert_allclose(ea.rotate(v3), v3, rtol=RTOL, atol=ATOL)


def test_mul():
    ea1 = isce3.core.EulerAngles(np.pi, 0, 0)
    ea2 = ea1 * ea1
    npt.assert_allclose(ea2(), [0, 0, 0], rtol=RTOL, atol=ATOL)
    ea1 *= ea1
    npt.assert_allclose(ea1(), [0, 0, 0], rtol=RTOL, atol=ATOL)

    
def test_add():
    ea1 = isce3.core.EulerAngles(np.pi, 0, 0)
    ea2 = ea1 + ea1
    npt.assert_allclose(ea2(), [2*np.pi, 0, 0], rtol=RTOL, atol=ATOL)
    ea1 += ea1
    npt.assert_allclose(ea1(), [2*np.pi, 0, 0], rtol=RTOL, atol=ATOL)


def test_sub():
    ea1 = isce3.core.EulerAngles(np.pi, 0, 0)
    ea2 = ea1 - ea1
    npt.assert_allclose(ea2(), [0, 0, 0], rtol=RTOL, atol=ATOL)
    ea1 -= ea1
    npt.assert_allclose(ea1(), [0, 0, 0], rtol=RTOL, atol=ATOL)
