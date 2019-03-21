#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Product cimport Product
from libcpp cimport bool

cdef class pyProduct:

    # C++ class pointers
    cdef Product * c_product
    cdef bool __owner

    # Cython class members
    cdef pySwath py_swathA
    cdef pyMetadata py_metadata
    
# end of file
