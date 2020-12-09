#!/usr/bin/env python3
import numpy as np
import pybind_isce3 as isce3

def test_presum_weights():
    n = 31
    rng = np.random.default_rng(12345)
    t = np.linspace(0, 9, n) + rng.normal(0.1, size=n)
    t.sort()
    tout = 4.5
    L = 1.0
    acor = isce3.core.AzimuthKernel(L)
    offset, w = isce3.focus.get_presum_weights(acor, t, tout)
    i = slice(offset, offset + len(w))
    assert all(abs(t[i] - tout) <= L)

