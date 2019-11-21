#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from Poly2d cimport Poly2d

cdef class pyPoly2d:
    cdef Poly2d *c_poly2d
    cdef bool __owner

# end of file 
