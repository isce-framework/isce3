#cython: language_level=3
#
# Author: Bryan Riel, Heresh Fattahi
# Copyright 2017-2019
#

from libcpp cimport bool
from Crossmul cimport Crossmul

cdef class pyCrossmul:
    cdef Crossmul * c_crossmul
    cdef bool __owner

# end of file
