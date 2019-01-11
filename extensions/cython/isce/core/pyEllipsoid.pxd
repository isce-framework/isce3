#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from Ellipsoid cimport Ellipsoid

cdef class pyEllipsoid:
    cdef Ellipsoid *c_ellipsoid
    cdef bool __owner

# end of file
