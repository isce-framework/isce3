#!/usr/bin/env python3

import iscetest
import journal
import numpy as np
import pytest

import isce3.ext.isce3 as isce


def test_LUT2d():
    # Create LUT2d obj
    xvec = yvec = np.arange(-5.01, 5.01, 0.25)
    xx, yy = np.meshgrid(xvec, xvec)
    M = np.sin(xx * xx + yy * yy)
    method = isce.core.DataInterpMethod.BIQUINTIC
    lut2d = isce.core.LUT2d(xvec, yvec, M, "biquintic")
    assert lut2d.interp_method == method
    # try ctor with enum method
    lut2d = isce.core.LUT2d(xvec, yvec, M, method)
    assert lut2d.interp_method == method
    # check data accessor
    assert np.allclose(M, lut2d.data)
    # check endpoints
    assert lut2d.x_start == xvec[0]
    assert lut2d.y_start == yvec[0]
    assert lut2d.x_end == xvec[-1]
    assert lut2d.y_end == yvec[-1]

    # Load reference data
    f_ref = iscetest.data + "interpolator/data.txt"
    d_refs = np.loadtxt(f_ref)

    # Loop over test points and check for error
    error = 0
    for d_ref in d_refs:
        z_test = lut2d.eval(d_ref[0], d_ref[1])
        error += (d_ref[5] - z_test) ** 2

    n_pts = d_refs.shape[0]
    assert error / n_pts < 0.058, f"pybind LUT2d failed: {error} > 0.058"

    # check that we can set ref_value
    lut = isce.core.LUT2d()
    assert lut.ref_value == 0.0
    lut.ref_value = 1.0
    assert lut.ref_value == 1.0
    lut = isce.core.LUT2d(2.0)
    assert lut.ref_value == 2.0

    # Call vectorized vs x
    lut2d.eval(d_refs[0,0], d_refs[:,1])
    # Call vectorized vs y and x
    lut2d.eval(d_refs[:,0], d_refs[:,1])

    # Call vectorized with size mismatch
    with pytest.raises(ValueError):
        lut2d.eval(d_refs[:10,0], d_refs[:,1])


def test_bounds_error():
    """Test that out-of-bounds evaluation raises exceptions.

    Regression test for bug where vectorized .eval crashes Python instead of
    raising a catchable exception when bounds_error=True.
    """
    # Create a simple LUT with known bounds: 10 points in x, 2 points in y
    x = np.linspace(0, 5, 10)
    y = np.linspace(10, 20, 2)
    z = np.vstack((np.linspace(100, 200, 10), np.linspace(100, 200, 10)))
    lut2d = isce.core.LUT2d(x, y, z)
    # Default is bounds_error=True
    assert lut2d.bounds_error is True

    y = 15.0
    x = 2.0
    lut2d.eval(y, x)  # y=1.0 is out of bounds (valid: 10-20)

    # Test scalar out-of-bounds
    # pyre C++ journal raises ApplicationError when libjournal bindings are loaded,
    # but falls back to RuntimeError (via std::runtime_error) when they aren't.
    x_oob = 200.0
    with pytest.raises((RuntimeError, journal.ApplicationError)):
        lut2d.eval(y, x_oob)

    # Test vectorized out-of-bounds (should raise, not crash)
    with pytest.raises((RuntimeError, journal.ApplicationError)):
        lut2d.eval(y, np.array([x_oob, x_oob]))
    with pytest.raises(IndexError):
        lut2d.eval([y, y], np.array([x_oob, x_oob]))

    # Test vectorized, all in-bounds
    result = lut2d.eval(y, np.array([1.0, 2.0, 3.0]))
    assert result.shape == (3,)

    # Test with bounds_error=False to avoid raising exceptions
    lut2d.bounds_error = False
    result = lut2d.eval(y, x_oob)  # Should clamp and return value
    assert result == 200.0
    result = lut2d.eval(y, np.array([x_oob, x_oob]))  # Should clamp and return values
    assert np.allclose(result, np.array([200, 200]))
