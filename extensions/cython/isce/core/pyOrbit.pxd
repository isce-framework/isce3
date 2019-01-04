#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from Orbit cimport Orbit

cdef class pyOrbit:
    cdef Orbit *c_orbit
    cdef bool __owner


# end of file
