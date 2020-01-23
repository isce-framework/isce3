#cython: language_level=3
#
# Author: Bryan V. Riel, Liang Yu
# Copyright 2017-2019
#

from cuTopo cimport Topo

cdef class pyTopo:
    cdef Topo * c_topo
    cdef bool __owner

# end of file
