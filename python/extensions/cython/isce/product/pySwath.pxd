#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2019
#

from Swath cimport Swath

cdef class pySwath:
    """
    Cython wrapper for isce::product::Swath.

    Args:
        None

    Return:
        None
    """
    # C++ class
    cdef Swath * c_swath
    cdef bool __owner
    
# end of file
