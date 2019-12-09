#cython: language_level=3
#
# Author: Piyush Agram
# Copyright 2017-2018
#

from libcpp cimport bool
from DateTime cimport DateTime

cdef class pyDateTime:
    cdef DateTime * c_datetime
    cdef bool __owner
