#!/usr/bin/env python3

import numpy as np

import pybind_isce3 as isce
import iscetest

def test_LUT2d():
    # Create LUT2d obj
    xvec = yvec = np.arange(-5.01, 5.01, 0.25)
    xx, yy = np.meshgrid(xvec, xvec)
    M = np.sin(xx*xx + yy*yy)
    method = isce.core.DataInterpMethod.BIQUINTIC
    lut2d = isce.core.LUT2d(xvec, yvec, M, "biquintic")
    assert lut2d.interp_method == method
    # try ctor with enum method
    lut2d = isce.core.LUT2d(xvec, yvec, M, method)
    assert lut2d.interp_method == method
    # check data accessor
    assert np.allclose(M, lut2d.data)

    # Load reference data
    f_ref = iscetest.data + 'interpolator/data.txt'
    d_refs = np.loadtxt(f_ref)
    
    # Loop over test points and check for error
    error = 0
    for d_ref in d_refs:
        z_test = lut2d.eval(d_ref[0], d_ref[1])
        error += (d_ref[5] - z_test)**2

    n_pts = d_refs.shape[0]
    assert error/n_pts < 0.058, f'pybind LUT2d failed: {error} > 0.058'
    
# end of file
