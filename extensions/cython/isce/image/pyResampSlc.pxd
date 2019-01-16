#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017
#

from libcpp cimport bool
from ResampSlc cimport ResampSlc

cdef class pyResampSlc:
    cdef ResampSlc * c_resamp
    cdef bool __owner
    
# end of file
