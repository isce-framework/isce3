#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from Pegtrans cimport Pegtrans


cdef class pyPegtrans:
    cdef Pegtrans *c_pegtrans
    cdef bool __owner

# end of file
