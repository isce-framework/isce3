#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Identification cimport Identification

cdef class pyIdentification:
    cdef Identification c_identification

# end of file
