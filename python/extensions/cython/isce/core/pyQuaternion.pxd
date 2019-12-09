#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2018
#

from libcpp cimport bool
from Quaternion cimport Quaternion

cdef class pyQuaternion:
    cdef Quaternion * c_quaternion
    cdef bool __owner

# end of file
