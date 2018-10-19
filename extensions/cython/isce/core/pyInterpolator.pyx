#cython: language_level=3
#
# Author: Joshua Cohen, Bryan V. Riel
# Copyright 2017-2018
#

cimport cython
import numpy as np
cimport numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector
from Interpolator cimport *
from Matrix cimport Matrix, valarray

cdef numpyToMatrix(np.ndarray[double, ndim=2] a, Matrix[double] & b):
    """
    Utility function to copy double or single precision numpy array to isce::core::Matrix.
    """
    cdef int i, j
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b.data()[i*a.shape[1] + j] = a[i,j]
    return


cdef class pyInterpolator:
    """
    Cython class for creating and calling all Interpolator classes.
    """
    cdef int order

    def __init__(self, method='bilinear', order=6):

        # Validate the method
        assert method in ('bilinear', 'bicubic', 'spline'), \
            'Unsupported interpolation method'
        self.method = method

        # Validate the order (for spline only)
        assert order > 2 and order < 21, 'Invalid interpolation order'
        self.order = order
        
    def interpolate(self, double x, double y, np.ndarray[double, ndim=2] z):
        """
        Interpolate at specified coordinate.
        """
        # Convert numpy array to isce::core::Matrix
        cdef Matrix[double] zmat = Matrix[double](z.shape[0], z.shape[1])
        numpyToMatrix(z, zmat)

        # Dynamically create interpolation object
        cdef Interpolator[double] * c_interp;
        if self.method == 'bilinear':
            c_interp = new BilinearInterpolator[double]()
        elif self.method == 'bicubic':
            c_interp = new BicubicInterpolator[double]()
        elif self.method == 'spline':
            c_interp = new Spline2dInterpolator[double](self.order)

        # Call interpolator
        cdef double value = c_interp.interpolate(x, y, zmat)

        # Done
        del c_interp
        return value


    #def quadInterpolate(self, x, y, double xintp):
    #    cdef int i, N
    #    N = x.size
    #    cdef valarray[double] x_array = valarray[double](N)
    #    cdef valarray[double] y_array = valarray[double](N)
    #    for i in range(N):
    #        x_array[i] = x[i]
    #        y_array[i] = y[i]
    #    return self.c_interp.quadInterpolate(x_array, y_array, xintp)

    #def akima(self, double x, double y, np.ndarray[np.float32_t, ndim=2] c):
    #    cdef Matrix[float] cmat = Matrix[float](c.shape[0], c.shape[1])
    #    numpyToMatrixFloat(c, cmat)
    #    return self.c_interp.akima(x, y, cmat)

# end of file
