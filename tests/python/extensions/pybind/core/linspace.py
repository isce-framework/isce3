#!/usr/bin/env python3

import numpy as np
import numpy.testing as npt
import isce3.ext.isce3 as isce

def test_linspace():
    first = 0.
    spacing = 1.
    size = 10
    x = isce.core.Linspace(first, spacing, size)

    npt.assert_almost_equal(x.first, first)
    npt.assert_almost_equal(x.spacing, spacing)
    assert(x.size == size)

def test_element_access():
    first = 0.
    spacing = 1.
    size = 10
    x = isce.core.Linspace(first, spacing, size)

    npt.assert_almost_equal(x[0], first)
    npt.assert_almost_equal(x[1], first + spacing)
    npt.assert_almost_equal(x[5], first + spacing * 5)

def test_subinterval():
    first = 0.
    spacing = 1.
    size = 10
    x1 = isce.core.Linspace(first, spacing, size)

    start = 3
    stop = 8
    x2 = x1[start:stop]

    npt.assert_almost_equal(x2.first, first + spacing * start)
    npt.assert_almost_equal(x2.last, first + spacing * (stop - 1))
    npt.assert_almost_equal(x2.size, stop - start)

def test_array():
    x = isce.core.Linspace(0., 1., 10)
    arr = np.array(x)

    assert(len(arr) == len(x))

    for i in range(len(x)):
        npt.assert_almost_equal(arr[i], x[i])

def test_comparison():
    x1 = isce.core.Linspace(0., 1., 10)
    x2 = isce.core.Linspace(0., 1., 10)
    x3 = isce.core.Linspace(1., 1., 10)

    assert(x1 == x2)
    assert(x1 != x3)

def test_setfirst():
    x = isce.core.Linspace(0., 1., 10)

    new_first = 10.
    x.first = new_first

    npt.assert_almost_equal(x.first, new_first)
    npt.assert_almost_equal(x[0], new_first)

def test_setspacing():
    x = isce.core.Linspace(0., 1., 10)

    new_spacing = 2.
    x.spacing = new_spacing

    npt.assert_almost_equal(x.spacing, new_spacing)
    npt.assert_almost_equal(x[1] - x[0], new_spacing)

def test_resize():
    x = isce.core.Linspace(0., 1., 10)

    new_size = 15
    x.resize(new_size)

    assert(x.size == new_size)

def test_search():
    x = isce.core.Linspace(0., 1., 10)

    npt.assert_almost_equal(x.search(-1.), 0)
    npt.assert_almost_equal(x.search(2.5), 3)
    npt.assert_almost_equal(x.search(11.), 10)

    x.spacing = -1.

    npt.assert_almost_equal(x.search(1.), 0)
    npt.assert_almost_equal(x.search(-2.5), 3)
    npt.assert_almost_equal(x.search(-11.), 10)
