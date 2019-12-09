#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from Peg cimport Peg

cdef class pyPeg:
    cdef Peg *c_peg
    cdef bool __owner

