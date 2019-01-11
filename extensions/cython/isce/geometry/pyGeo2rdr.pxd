#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

from libcpp cimport bool
from libcpp.string cimport string
from Geo2rdr cimport Geo2rdr
from Orbit cimport orbitInterpMethod

cdef class pyGeo2rdr:
    cdef Geo2rdr * c_geo2rdr
    cdef bool __owner

# end of file
