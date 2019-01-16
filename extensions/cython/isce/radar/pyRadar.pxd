#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Radar cimport Radar

cdef class pyRadar:
    cdef Radar * c_radar
    cdef bool __owner

# end of file
