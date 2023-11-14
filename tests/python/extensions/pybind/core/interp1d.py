#!/usr/bin/env python3
import numpy as np
import numpy.testing as npt
import pytest
import isce3.ext.isce3.core as m
try:
    import pytest_benchmark
except:
    pytest_benchmark = None

def test_bindings():
    # define a custom kernel in Python
    class MyKernel(m.Kernel):
        def __call__(self, x):
            hw = self.width / 2
            return np.sinc(x) if abs(x) < hw else 0

    kernels = [
        MyKernel(8),
        m.LinearKernel(),
        m.BartlettKernel(4),
        m.KnabKernel(9, 0.75),
        m.NFFTKernel(4, 1024, 2048),
    ]

    # Create metakernels, including casting double->float
    kernel = kernels[-1]
    metakernels = [
        m.TabulatedKernel(kernel, 2048),
        m.TabulatedKernelF32(kernel, 2048),
        m.ChebyKernel(kernel, 11),
        m.ChebyKernelF32(kernel, 11)
    ]

    # Check that we can eval kernels (scalar)
    t = np.linspace(-5, 5, 201)
    for kernel in (kernels + metakernels):
        x = np.asarray([kernel(ti) for ti in t])
        assert np.sum(np.abs(x)**2) > 0

    # Check interp1d
    i = 10
    t += i
    for kernel in (kernels + metakernels):
        for dtype in ('f4', 'f8', 'c8', 'c16'):
            # Delta function to recover underlying kernel.
            x = np.zeros(2*i+1, dtype)
            x[i] = 1
            # Scalar
            xt = np.asarray([m.interp1d(kernel, x, ti) for ti in t])
            assert np.sum(np.abs(xt)**2) > 0
            # Vectorized
            xt = m.interp1d(kernel, x, t)
            assert np.sum(np.abs(xt)**2) > 0


def test_big_kernel():
    # Check code path for Kernel.width() > 32 where interp1d uses heap memory.
    width = 40
    kernel = m.KnabKernel(width, 0.83)
    n = 1024
    z = np.random.randn(n) + 1j * np.random.randn(n)
    t = np.random.rand(n) * n
    zt = m.interp1d(kernel, z, t)


# This test is useful for benchmarking
@pytest.fixture
def speed_test_args():
    kernel = m.TabulatedKernelF32(m.KnabKernel(9, 0.83), 2048)
    n = 32768
    ni = 2**20  # 1M
    ti = np.random.rand(ni) * n
    z = np.empty(n, dtype="c8")
    return kernel, z, ti


@pytest.mark.skipif(pytest_benchmark is None, reason="needs benchmark plugin")
def test_speed(speed_test_args, benchmark):
    benchmark(m.interp1d, *speed_test_args)
