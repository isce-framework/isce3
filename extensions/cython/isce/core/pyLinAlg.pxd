#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from LinAlg cimport LinAlg

cdef class pyLinAlg:
    cdef LinAlg *c_linAlg
    cdef bool __owner

