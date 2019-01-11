#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Basis cimport Basis

cdef class pyBasis:
    cdef Basis c_basis

# end of file
