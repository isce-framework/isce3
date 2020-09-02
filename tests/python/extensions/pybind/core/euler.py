#!/usr/bin/env python3

import numpy as np
import pybind_isce3 as isce3

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
