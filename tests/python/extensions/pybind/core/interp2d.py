#!python3
import numpy as np
import numpy.testing as npt
import pytest
from scipy.interpolate import RegularGridInterpolator
import isce3.ext.isce3.core as m


@pytest.fixture
def interp_args():
    np.random.seed(12345)
    z = np.random.randn(51, 73)

    nt = 1001
    t = np.zeros((nt, 2))
    t[:,0] = np.random.uniform(0, z.shape[1] - 1, nt)
    t[:,1] = np.random.uniform(0, z.shape[0] - 1, nt)
    return z, t


def test_interp2d(interp_args):
    # Most functionality already tested in 1d case.  Make sure that we've done
    # the 2d indexing correctly by comparing with scipy bilinear.
    z, t = interp_args
    kernelx = kernely = m.BartlettKernel(2)

    zt = m.interp2d(kernelx, kernely, z, t)

    # scipy puts axes slowest first--sensible but at odds with typical image
    # processing convention (x ~ fast axis)
    y = np.arange(z.shape[0]).astype(float)
    x = np.arange(z.shape[1]).astype(float)
    itp = RegularGridInterpolator((y, x), z, method="linear")
    zt_ref = itp(t[:, ::-1])

    npt.assert_allclose(zt, zt_ref)


def test_big_kernel(interp_args):
    # Make sure we don't crash or anything when using kernels larger than our
    # stack array limit of 32
    z, t = interp_args
    kernelx = kernely = m.KnabKernel(40, 0.83)
    zt = m.interp2d(kernelx, kernely, z, t)
