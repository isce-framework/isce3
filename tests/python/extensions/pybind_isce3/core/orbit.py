#!/usr/bin/env python3

import numpy.testing as npt
import pybind_isce3 as isce

def test_load_h5():
    from iscetest import data
    from os import path
    f = path.join(data, "envisat.h5")
    o = isce.core.Orbit.load_from_h5(f, "/science/LSAR/SLC/metadata/orbit")

    o.time_at(0)
    o.position_at(0)
    o.velocity_at(0)
