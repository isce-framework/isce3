#!/usr/bin/env python3

import numpy as np
import pybind_isce3 as isce3

def test_sv():
    t = isce3.core.DateTime(2020, 8, 4)
    pos = [0, 0, 1.]
    vel = [1, 0, 0.]
    sv = isce3.core.StateVector(t, pos, vel)
    assert sv.datetime == t
    assert np.allclose(sv.position, pos)
    assert np.allclose(sv.velocity, vel)
