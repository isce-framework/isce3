#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from Position cimport Position

cdef class pyPosition:
    cdef Position *c_position
    cdef bool __owner
