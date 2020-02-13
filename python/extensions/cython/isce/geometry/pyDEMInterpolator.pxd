#cython: language_level=3
#
# Author: Piyush Agram
# Copyright 2017-2018
#

from DEMInterpolator cimport DEMInterpolator
from libcpp cimport bool

cdef class pyDEMInterpolator:
    cdef DEMInterpolator *c_deminterp
    cdef bool __owner

# end of file
