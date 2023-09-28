#!/usr/bin/env python3
import numpy as np
import numpy.testing as npt
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


def test_apply_weights():
    # Test data.  Use prime sizes to mess with threading.
    t = np.array([-2.2, -1.1, -0.1, 0.9, 1.9]) / 1910.
    fd = np.linspace(800, 900, 49999)
    N = lambda: np.random.normal(size=(len(t), len(fd)))
    np.random.seed(12345)
    weights = N()
    raw = (N() + 1j * N()).astype("c8")

    # Python calculation we want to speed up
    deramp = np.exp(-2j * np.pi * t[:, None] * fd[None, :])
    desired = (weights * deramp * raw).sum(axis=0)

    # check C++ gives the same answer (to single precision)
    out = np.zeros(len(fd), dtype="c8")
    isce3.focus.apply_presum_weights(out, t, fd, weights, raw)
    npt.assert_allclose(out, desired, rtol=1e-5, atol=1e-5)


def test_compute_ids_from_mask():
    mask = np.zeros((499, 4), dtype=bool)
    for j in range(4):
        mask[j:, j] = True

    ids = isce3.focus.compute_ids_from_mask(mask)

    expected = np.zeros(mask.shape[0], dtype='int64')
    expected[0] = 1
    expected[1] = 3
    expected[2] = 7
    expected[3:] = 15

    npt.assert_array_equal(ids, expected)


def test_unique_ids():
    ids = np.array([1, 1, 1, 3, 3, 3, 2, 2, 0, 0, 0, 0, 0, 0])
    desired = np.unique(ids)
    out = isce3.focus.get_unique_ids(ids)
    npt.assert_array_equal(out, desired)
