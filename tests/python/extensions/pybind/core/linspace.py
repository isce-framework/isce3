#!/usr/bin/env python3

import numpy as np
import numpy.testing as npt
import pytest
import isce3.ext.isce3 as isce

NUMPY_MAJOR_VERSION = int(np.__version__.split('.')[0])

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

    step = 3
    x3 = x1[start:stop:step]

    npt.assert_almost_equal(x3.first, first + spacing * start)
    npt.assert_almost_equal(x3.spacing, step * x1.spacing)
    npt.assert_almost_equal(x3.size, 1 + (stop - 1 - start) // step)


def test_array():
    x = isce.core.Linspace(0., 1., 10)
    arr = np.array(x)

    assert(len(arr) == len(x))

    for i in range(len(x)):
        npt.assert_almost_equal(arr[i], x[i])

    arr2 = np.asarray(x, dtype="float64")
    npt.assert_allclose(arr, arr2)

    # The behavior of the copy keyword changed in numpy 2.x
    if NUMPY_MAJOR_VERSION >= 2:
        # Object has no storage, so a copy is always required and copy=False
        # should raise an exception.
        with pytest.raises(ValueError):
            np.array(x, copy=False)
        # copy=None should be okay since copy is allowed.
        np.array(x, copy=None)
    else:
        # In numpy 1.x copy=False is like the new copy=None
        np.array(x, copy=False)
        # And copy=None is not implemented.


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

def test_dtype():
    x = isce.core.Linspace(0., 1., 10)
    assert x.dtype == np.dtype("float64")

def test_shape():
    n = 10
    x = isce.core.Linspace(0., 1., n)
    assert x.shape == (n,)
