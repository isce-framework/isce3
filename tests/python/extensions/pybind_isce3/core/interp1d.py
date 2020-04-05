#!/usr/bin/env python3
import numpy as np
import numpy.testing as npt
import pybind_isce3.core as m

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
    # Complex type has to match kernel type.  Real data can be either.
    f4 = ('f4', 'f8', 'c8')
    f8 = ('f4', 'f8', 'c16')
    for kernel in (kernels + metakernels):
        dtypes = f8 if isinstance(kernel, m.Kernel) else f4
        for dtype in dtypes:
            # Delta function to recover underlying kernel.
            x = np.zeros(2*i+1, dtype)
            x[i] = 1
            # Scalar
            xt = np.asarray([m.interp1d(kernel, x, ti) for ti in t])
            assert np.sum(np.abs(xt)**2) > 0
            # Vectorized
            xt = m.interp1d(kernel, x, t)
            assert np.sum(np.abs(xt)**2) > 0
