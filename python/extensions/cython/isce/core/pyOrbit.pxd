#cython: language_level=3

from Orbit cimport Orbit

cdef class pyOrbit:
    cdef Orbit c_orbit
