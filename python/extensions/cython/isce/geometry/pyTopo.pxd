#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Topo cimport Topo
from Interpolator cimport dataInterpMethod

cdef class pyTopo:
    cdef Topo * c_topo
    cdef bool __owner

# end of file
