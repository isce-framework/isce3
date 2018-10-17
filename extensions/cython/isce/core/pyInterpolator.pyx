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
from Interpolator cimport Interpolator
from Matrix cimport Matrix, valarray

cdef numpyToMatrix(np.ndarray[np.float64_t, ndim=2] a,
                   Matrix[double] & b):
    cdef int i, j
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b.data()[i*a.shape[1] + j] = a[i,j]
    return

cdef numpyToMatrixFloat(np.ndarray[np.float32_t, ndim=2] a,
                        Matrix[float] & b):
    cdef int i, j
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b.data()[i*a.shape[1] + j] = a[i,j]
    return

cdef class pyInterpolator:
    cdef Interpolator *c_interp
    cdef bool __owner

    def __cinit__(self):
        self.c_interp = new Interpolator()
        self.__owner = True
    def __dealloc__(self):
        if self.__owner:
            del self.c_interp

    # Note no static binder since we'll never need to pass any particular Interpolator object
    # around...

    def bilinear(self, double a, double b, np.ndarray[np.float64_t, ndim=2] c):
        cdef Matrix[double] cmat = Matrix[double](c.shape[0], c.shape[1])
        numpyToMatrix(c, cmat)
        return self.c_interp.bilinear[double](a, b, cmat)

    #def bicubic(self, double a, double b, np.ndarray[np.float64_t, ndim=2] c):
    #    cdef Matrix[double] cmat = Matrix[double](c.shape[0], c.shape[1])
    #    numpyToMatrix(c, cmat)
    #    return self.c_interp.bicubic[double](a, b, cmat)

    #def interp_2d_spline(self, double a, double b, np.ndarray[np.float64_t, ndim=2] dat,
    #                     int degree):
    #    cdef Matrix[double] mat = Matrix[double](dat.shape[0], dat.shape[1])
    #    numpyToMatrix(dat, mat)
    #    return self.c_interp.interp_2d_spline(degree, mat, a, b)

    def quadInterpolate(self, x, y, double xintp):
        cdef int i, N
        N = x.size
        cdef valarray[double] x_array = valarray[double](N)
        cdef valarray[double] y_array = valarray[double](N)
        for i in range(N):
            x_array[i] = x[i]
            y_array[i] = y[i]
        return self.c_interp.quadInterpolate(x_array, y_array, xintp)

    #def akima(self, double x, double y, np.ndarray[np.float32_t, ndim=2] c):
    #    cdef Matrix[float] cmat = Matrix[float](c.shape[0], c.shape[1])
    #    numpyToMatrixFloat(c, cmat)
    #    return self.c_interp.akima(x, y, cmat)

# end of file
