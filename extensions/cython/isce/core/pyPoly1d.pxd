#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from Poly1d cimport Poly1d

cdef class pyPoly1d:
    cdef Poly1d *c_poly1d
    cdef bool __owner

