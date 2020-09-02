#!/usr/bin/env python3

import numpy as np
import pybind_isce3 as isce3


def test_quaternion():
    q = isce3.core.Quaternion(1, 0, 0, 0)
    assert np.allclose([1, 0, 0, 0], [q.w, q.x, q.y, q.z])
    assert np.allclose(np.eye(3), q.to_rotation_matrix())

    ypr = isce3.core.EulerAngles(0, 0, 0)
    q = isce3.core.Quaternion(ypr)
    assert np.allclose([1, 0, 0, 0], [q.w, q.x, q.y, q.z])
