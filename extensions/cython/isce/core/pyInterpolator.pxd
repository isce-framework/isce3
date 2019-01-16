#cython: language_level=3
#
# Author: Joshua Cohen, Bryan V. Riel
# Copyright 2017-2018
#

cimport numpy as np
from libcpp.string cimport string
from Matrix cimport Matrix

cdef Matrix[double] numpyToMatrix(np.ndarray[double, ndim=2])

cdef class pyInterpolator:
    cdef int order
    cdef string method

# end of file
