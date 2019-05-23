#cython: language_level=3
#
# Author: Piyush Agram
# Copyright 2017-2018
#

cdef class pyDEMInterpolator:
    cdef Topo * c_deminterp
    cdef bool __owner

# end of file
