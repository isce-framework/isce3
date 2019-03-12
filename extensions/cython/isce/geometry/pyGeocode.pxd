#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2019
#

from libcpp cimport bool
from Geocode cimport Geocode

cdef class pyGeocode:
    cdef Geocode * c_geocode
    cdef bool __owner

# end of file
