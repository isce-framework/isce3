#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from Orbit cimport Orbit, orbitInterpMethod

cdef class pyOrbit:
    cdef Orbit *c_orbit
    cdef bool __owner

    methods = { 'hermite': orbitInterpMethod.HERMITE_METHOD,
                'sch' :  orbitInterpMethod.SCH_METHOD,
                'legendre': orbitInterpMethod.LEGENDRE_METHOD}

# end of file
