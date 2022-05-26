#!/usr/bin/env python3
import numpy as np
import isce3.ext.isce3 as isce3


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


def test_fill_weights():
    lut = {
        123: np.array([1, 2, 3.]),
        456: np.array([4, 5, 6.]),
        789: np.array([7, 8, 9.]),
    }
    nr = 1000
    ids = np.random.choice(list(lut.keys()), nr)
    weights = isce3.focus.fill_weights(ids, lut)
    i = 0
    # This function just accelerates this particular dict lookup.
    assert all(weights[:, i] == lut[ids[i]])
