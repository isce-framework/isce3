#!/usr/bin/env python3
# NOTE most kernels are tested in interp1d.py
import pybind_isce3 as isce3
import numpy as np

def test_azimuth_acf():
    L = 1.0
    acf = isce3.core.AzimuthKernel(L)
    # some obvious values
    assert np.isclose(acf(0), 1.0)
    assert np.isclose(acf(L), 0.0)
    assert np.isclose(acf(1.0001 * L), 0.0)
    assert acf(0.9999 * L) > 0.0
    # golden value computed by hand
    assert np.isclose(acf(0.1 * L), 0.946)
