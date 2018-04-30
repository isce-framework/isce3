#!/usr/bin/env python3

def test_CythonInterface():
    from isceextension import pyPoly1d
    import numpy.testing as npt
        
    refpoly = pyPoly1d(order=5,
                       mean=0.,
                       norm=1.)

    ##Check that np.size works
    assert refpoly.coeffs.size == 6

    ##Assign np.array and check value
    refpoly.coeffs[0:3] = 1
    npt.assert_array_equal(refpoly.coeffs[0:3],1.0)

def test_Constant():
    from isceextension import pyPoly1d

    refval = 10.0
    for ii in range(1,6):
        pp = pyPoly1d(order=0.,
                    mean=ii*1.0,
                    norm=ii*ii*1.0)
        pp.coeffs[0] = refval
        assert(pp.eval(ii*10.0) == refval)


def test_MeanShift():
    from isceextension import pyPoly1d
    import numpy.testing as npt

    def getPoly():
        refpoly = pyPoly1d(order=2,
                        mean=0.,
                        norm=1.)
        refpoly.coeffs[:] = [0., 1., 0.]
        return refpoly

    refpoly = getPoly()

    for ii in range(5):
        newpoly = getPoly()
        newpoly.mean = 0.5 * ii * ii

        npt.assert_almost_equal(refpoly.eval(2.0*ii), newpoly.eval(2.0*ii+0.5*ii*ii),
                    decimal=12)

def test_NormShift():
    from isceextension import pyPoly1d
    import numpy.testing as npt

    def getPoly():
        refpoly = pyPoly1d(order=2,
                        mean=0.,
                        norm=1.)
        refpoly.coeffs[:] = [0., 0., 1,]
        return refpoly

    refpoly = getPoly()
    for ii in range(1,6):
        newpoly = getPoly()
        newpoly.norm = ii * ii * 1.

        npt.assert_almost_equal(refpoly.eval(2.5), newpoly.eval(2.5*ii*ii),
                decimal=12)

def test_NumpyPolyval():
    from isceextension import pyPoly1d
    import numpy as np

    for ii in range(6):
        refpoly = pyPoly1d(order = min( int(np.random.rand(1)[0] * 10), 4),
                           mean = np.random.rand(1)[0]*5,
                           norm = np.random.rand(1) + 1.0)
        refpoly.coeffs[:] = np.random.rand(refpoly.order+1)

        x = np.random.rand(3) * 8
        cpppoly = [refpoly(y) for y in x]
        nppoly = np.polyval(refpoly.coeffs[::-1], (x-refpoly.mean)/refpoly.norm)

        np.testing.assert_almost_equal(cpppoly, nppoly, decimal=10)
