#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2018
#

from libcpp cimport bool
from Attitude cimport Attitude

cdef class pyAttitude:
    cdef Attitude * c_attitude
    cdef bool __owner

# end of file
